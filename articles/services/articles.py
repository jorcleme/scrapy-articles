import pymongo.errors
import requests
import re
import os
import json
import aiohttp
import asyncio
import logging
import warnings
import time
import pymongo
from hashlib import sha256
from uuid import uuid4
from datetime import date
from typing import List, Optional, Dict, Any, Union, TypeVar, Sequence, TypedDict
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, field_serializer
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_text_splitters import HTMLHeaderTextSplitter
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(filename=".env"))


def get_article_links_after_spidering():
    uri = (
        os.environ.get("MONGO_URI")
        .replace("<username>", os.environ.get("MONGODB_APP_USER"))
        .replace("<password>", os.environ.get("MONGODB_APP_PASS"))
    )
    client = pymongo.MongoClient(uri)
    article_link_collection = client["smb_documents"]["article_links"]
    links = article_link_collection.find({})
    return links


class Revision(BaseModel):
    """
    Represents a revision of an article.

    Attributes:
        revision (float): The revision number.
        publish_date (Optional[date]): The date when the revision was published.
        comments (Optional[str]): Any comments or notes about the revision.
    """

    revision: float = 1.0
    publish_date: Optional[date] = None
    comments: Optional[str] = "Initial Release"

    @field_serializer("publish_date", when_used="always")
    @classmethod
    def serialize_publish_date(cls, value: date) -> str:
        """
        Serialize the publish date value into a string format.

        Args:
            value (date): The publish date value to be serialized.

        Returns:
            str: The serialized publish date in the format "%Y-%m-%d".
        """
        return value.strftime("%Y-%m-%d")


class ArticleDict(TypedDict, total=False):
    series: str
    title: str
    document_id: str
    category: str
    url: str
    objective: str
    applicable_devices: List[dict]
    intro: Optional[str]
    steps: List[dict]
    revision_history: Optional[List[Revision]]
    type: str


class Article:
    """
    Represents an article.

    Attributes:
        name (str): The name of the article series.
        title (str): The title of the article.
        document_id (str): The ID of the article document.
        url (str): The URL of the article.
        category (str): The category of the article.
        objective (str): The objective of the article.
        applicable_devices (List): The list of applicable devices for the article.
        intro (Optional[str]): The introduction of the article. (default: None)
        steps (List): The list of steps in the article. (default: [])
        revision_history (Optional[List[Revision]]): The revision history of the article. (default: None)
    """

    def __init__(
        self,
        *,
        name: str,
        title: str,
        document_id: str,
        url: str,
        category: str,
        objective: str,
        applicable_devices: List,
        intro: Optional[str] = None,
        steps: List = [],
        revision_history: Optional[List[Revision]] = None,
    ):
        self.series = name
        self.title = title
        self.document_id = document_id
        self.url = url
        self.category = category
        self.objective = objective
        self.applicable_devices = applicable_devices
        self.intro = intro
        self.steps = steps
        self.revision_history = revision_history
        self.type = self.__class__.__name__

    def add_step(
        self,
        section: str,
        step_num: int,
        text: str,
        src: Optional[str] = None,
        alt: Optional[str] = None,
        video_src: Optional[str] = None,
        note: Optional[str] = None,
        emphasized_text: Optional[List[str]] = None,
        emphasized_tags: Optional[List[str]] = None,
    ):
        """
        Adds a step to the article.

        Args:
            section (str): The section of the step.
            step_num (int): The step number.
            text (str): The text of the step.
            src (Optional[str]): The source of the step. (default: None)
            alt (Optional[str]): The alternative text for the step. (default: None)
            video_src (Optional[str]): The video source for the step. (default: None)
            note (Optional[str]): A note for the step. (default: None)
            emphasized_text (Optional[List[str]]): The list of emphasized text in the step. (default: None)
            emphasized_tags (Optional[List[str]]): The list of emphasized tags in the step. (default: None)
        """
        self.steps.append(
            {
                "section": section,
                "step_num": step_num,
                "text": text,
                "src": src,
                "alt": alt,
                "video_src": video_src,
                "note": note,
                "emphasized_text": emphasized_text,
                "emphasized_tags": emphasized_tags,
            }
        )

    def set_steps(self, steps):
        """
        Sets the steps of the article.

        Args:
            steps (List): The list of steps.
        """
        self.steps = steps

    def to_dict(self) -> ArticleDict:
        """
        Converts the article to a dictionary.

        Returns:
            dict: The dictionary representation of the article.
        """
        revisions = []
        if self.revision_history:
            for revision in self.revision_history:
                revisions.append(revision.model_dump())
        return {
            "series": self.series,
            "title": self.title,
            "document_id": self.document_id,
            "category": self.category,
            "url": self.url,
            "objective": self.objective,
            "applicable_devices": self.applicable_devices,
            "intro": self.intro,
            "steps": self.steps,
            "revision_history": revisions if self.revision_history else None,
            "type": self.type,
        }


class ArticleParser:
    """A class that parses HTML to extract Cisco SMB articles."""

    def __init__(self) -> None:
        self.headers = ["h1", "h2", "h3", "h4", "h5", "h6"]
        self.logger = logging.getLogger(self.__class__.__name__)
        # logging.basicConfig(level=logging.INFO)
        self.logger.setLevel(logging.INFO)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_file_path = os.path.join(
            dir_path, "logs", f"{self.__class__.__name__.lower()}.log"
        )
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf8")
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def parse(self, soup: BeautifulSoup, url: str, series: str) -> Article:
        """
        Parse the given HTML soup and extract the necessary information to create an Article object.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object representing the HTML content.
            url (str): The URL of the article.
            series (str): The series of the article.

        Returns:
            Article: The parsed Article object.

        """
        self.logger.info("Starting to parse article from URL: %s", url)
        try:
            name = series
            title = self.get_title(soup)
            document_id = self.get_document_id(soup)
            category = self.get_category(soup, title)
            objective = self.get_objective(soup)
            applicable_devices = self.get_applicable_devices(soup)
            intro = self.get_intro(soup)
            steps = self.get_steps(soup)
            if len(steps) == 0:
                steps = self.parse_backup_steps(soup)
            revision_history = self.get_revision_history(soup)
            self.log_article_components(
                title,
                document_id,
                category,
                objective,
                applicable_devices,
                intro,
                steps,
                revision_history,
            )
            return Article(
                name=name,
                title=title,
                document_id=document_id,
                url=url,
                category=category,
                objective=objective,
                applicable_devices=applicable_devices,
                intro=intro,
                steps=steps,
                revision_history=revision_history,
            )
        except Exception as err:
            self.logger.error("Error parsing article from URL: %s", url)
            self.logger.error(err)
            pass

    def log_article_components(
        self,
        title,
        document_id,
        category,
        objective,
        applicable_devices,
        intro,
        steps,
        revision_history,
    ):
        self.logger.info("Title: %s", title)
        self.logger.info("Document ID: %s", document_id)
        self.logger.info("Category: %s", category)
        self.logger.info("Objective: %s", objective)
        self.logger.info("Number of Applicable Devices: %d", len(applicable_devices))
        self.logger.info("Introduction: %s", intro)
        self.logger.info("Number of Steps: %d", len(steps))
        self.logger.info("Number of Revisions: %d", len(revision_history))

    @staticmethod
    def get_category_with_llm(title: str) -> str:
        """
        Get the category for an article based on the given title.

        Args:
            title (str): The title of the article.

        Returns:
            str: The category name.

        Raises:
            None

        This method uses a language model to determine the category of an article based on its title.
        It provides a set of rules to help choose the right category, but also allows flexibility in the choice.

        The available categories are:
        1. Troubleshooting
        2. Configuration
        3. Install & Upgrade
        4. Maintain & Operate
        5. Design

        The method returns only the category name and nothing else.
        """
        template = """Based on the given title, choose a category from below.
            
            Here are some rules to help you choose the right category:
            1. If the article is about troubleshooting a problem, choose the 'Troubleshooting' category.
            2. If the article title is about configuring a feature, choose the 'Configuration' category.
            3. If the article title is about upgrading or installing firmware or Day Zero setups, choose the 'Install & Upgrade' category.
            4. If the article is an overview of a feature or a best practice guide, choose the 'Maintain & Operate' category.
            5. Most of our articles fall under the "Configuration" category but do not let this rule limit your choice. Choose the category that best fits the article.
            
            There are 5 different categories to choose from.
            
            The categories are: 1. Troubleshooting, 2. Configuration, 3. Install & Upgrade, 4. Maintain & Operate, 5. Design.
            
            Title: {title}
            
            Return only the category name and nothing else.
            """
        prompt = PromptTemplate.from_template(template)
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=2.0)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"title": title})

    @staticmethod
    def is_blank_string(string: str | None) -> bool:
        if string is None:
            True
        return bool(re.match(r"^\s*$", string))

    @staticmethod
    def format_keys(strng: str) -> str:
        # First if there is a key like "Publish Date" we want to convert it to "publish_date"
        # Then we want to remove all non-alphanumeric characters and replace spaces with underscores
        # Finally we want to convert to lowercase
        return re.sub(r"[\W_]+", "_", strng.lower())

    @staticmethod
    def is_tag(tag) -> bool:
        return isinstance(tag, Tag)

    @staticmethod
    def is_step_indicator(node: Tag) -> bool:
        pattern = re.compile(r"Step (\d+)")
        return pattern.match(node.text.strip()) is not None

    @staticmethod
    def extract_step_number(node: Tag) -> Union[int, None]:
        pattern = re.compile(r"Step (\d+)")
        match = pattern.match(node.text.strip())
        return int(match.group(1)) if match else None

    @staticmethod
    def get_title(soup: BeautifulSoup) -> str:
        title_elem = soup.find(id="fw-pagetitle")
        if title_elem:
            return title_elem.text.strip()
        else:
            title = soup.title.string
            pattern = re.compile(r"^(.*?)(?= - Cisco)")
            match = pattern.search(title)
            return match.group(1) if match else title

    @staticmethod
    def get_document_id(soup: BeautifulSoup) -> str:
        element = soup.find("div", attrs={"class": "documentId"})
        if element:
            pattern = re.compile(r"((?:smb)?\d+)", flags=re.IGNORECASE)
            document_id = pattern.search(element.text.strip())
            return (
                document_id.group(1)
                if document_id
                else sha256(uuid4().bytes).hexdigest()
            )
        return sha256(uuid4().bytes).hexdigest()

    def get_category(self, soup: BeautifulSoup, title: str) -> str:
        category_pattern = re.compile(
            r"(?P<Troubleshooting>Troubleshoot(?:ing)?)|"
            r"(?P<Configuration>(?:Configure|Configuration|Configuration Examples and TechNotes))|"
            r"(?P<InstallUpgrade>(?:Install(?:ation)?|Upgrade))|"
            r"(?P<MaintainOperate>(?:Maintain and Operate|Maintain and Operate TechNotes))|"
            r"(?P<Design>Design)",
            re.IGNORECASE,
        )
        element = soup.select_one(
            '#fw-breadcrumb > ul > li:last-child > a > span[itemprop="name"]'
        )

        if element:
            match = category_pattern.search(element.get_text(strip=True))
            if match:
                category = next(filter(None, match.groups()))
                self.logger.info(f"Determined Category: {category}")
                return category
            else:
                category = self.get_category_with_llm(title)
                self.logger.info(f"Category from LLM: {category}")
                return category
        category = self.get_category_with_llm(title)
        self.logger.info(f"Category from LLM: {category}")
        return category

    def get_objective(self, soup: BeautifulSoup) -> Union[str, None]:
        pattern = re.compile(r"^Objective:?\s?", re.IGNORECASE)
        objective_element = soup.find("h2", string=re.compile(pattern))
        if objective_element:
            objective = []
            sibling = objective_element.find_next_sibling()
            while sibling and sibling.name in {"p", "ul", "ol", "table"}:
                if sibling.name in {"ul", "ol", "table"}:
                    objective.append(str(sibling.extract()))
                elif sibling.name == "p":
                    objective.append(sibling.get_text(strip=True))
                sibling = sibling.find_next_sibling()
            objective_text = " ".join(objective)
            text = self.sanitize_text(objective_text)
            self.logger.info(f"Objective: {text}")
            return text
        self.logger.info("Objective not found.")
        return None

    @staticmethod
    def get_applicable_devices(soup: BeautifulSoup) -> List[dict]:
        pattern = re.compile(
            r"(Applicable Devices\s*\|\s*Software|Applicable\s+Switches\s*|Applicable Devices \| Software Version|Applicable Devices|Applicable Devices \| Firmware Version|Applicable Devices\s*|Applicable Devices:|Applicable Devices and Software Version|Applicable Device|Applicable Devices|Applicable Devices \| Software)\b",
            re.IGNORECASE,
        )
        element = soup.find(["h2", "h3"], string=pattern)
        contents = []

        if element:
            sibling = element.find_next_sibling()
            if sibling and sibling.name in ["ul", "ol"]:
                for li in sibling.find_all("li"):
                    match = re.search(
                        r"(.*?)\s*(?:\|\s*([\d.]+))?\s*\(Data\s*Sheet\)",
                        li.get_text(strip=True),
                    )
                    if match:
                        device_name, software_version = match.groups()
                    else:
                        device_name = li.get_text(strip=True).split("|")[0].strip()
                    software_version = re.search(r"\| (\S+)", li.get_text())
                    if software_version:
                        version = software_version.group(1)
                    else:
                        version = None
                    datasheet_link = li.find(
                        "a",
                        href=True,
                        string=lambda text: text and "Data" in text,
                    )
                    download_latest_link = li.find(
                        "a",
                        href=True,
                        string=lambda text: text
                        and re.search(
                            r"\(Download latest\)|Download latest", text, re.IGNORECASE
                        )
                        is not None,
                    )
                    contents.append(
                        {
                            "device": device_name,
                            "software": version,
                            "datasheet_link": (
                                datasheet_link["href"] if datasheet_link else None
                            ),
                            "software_link": (
                                download_latest_link["href"]
                                if download_latest_link
                                else None
                            ),
                        }
                    )
        return contents

    def get_intro(self, soup: BeautifulSoup) -> Union[str, None]:
        headers = ["h1", "h2", "h3", "h4", "h5", "h6"]
        intro = soup.find(
            ["h3", "h2"], string=re.compile(r"^Introduction", re.IGNORECASE)
        )
        intro_text = None

        if intro:
            intro_content = []
            next_intro_element = intro.find_next_sibling(["p", "ul", "table", "div"])
            while next_intro_element and next_intro_element.name not in headers:
                if next_intro_element.find(headers):
                    break
                if next_intro_element.name == "ul":
                    intro_content.append(next_intro_element.prettify())
                elif next_intro_element.name == "table":
                    table_text = " ".join(
                        [
                            cell.get_text()
                            for cell in next_intro_element.find_all(["th", "td"])
                        ]
                    )
                    intro_content.append(table_text)
                elif (
                    next_intro_element.name == "div"
                    and next_intro_element.has_attr("class")
                    and "cdt-note" in next_intro_element["class"]
                ):
                    intro_content.append(next_intro_element.prettify())
                elif next_intro_element.name == "p":
                    intro_content.append(next_intro_element.get_text(strip=True))
                next_intro_element = next_intro_element.find_next_sibling()
            intro_text = " ".join(intro_content)
        return self.sanitize_text(intro_text) if intro_text else None

    def get_steps(self, soup: BeautifulSoup) -> List:
        """
        Extracts the steps from the given BeautifulSoup object.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object representing the HTML content.

        Returns:
            List: A list of dictionaries, where each dictionary represents a step and contains the following keys:
             ```JSON
                section: The section name of the step.
                step_num: The step number.
                text: The text content of the step.
                src: The source URL of any related diagram, image, or screenshot.
                alt: The alternative text for the related diagram, image, or screenshot.
                note: Additional notes related to the step.
                emphasized_text: A list of emphasized text within the step.
                emphasized_tags: A list of HTML tags used for emphasizing text within the step.
            ```
        """

        elements = soup.find_all(["h3", "h4", "p"])
        steps = []

        for _, element in enumerate(elements):
            if not self.is_tag(element) or self.is_blank_string(element.get_text()):
                continue
            if self.is_step_indicator(element):
                step = self.process_step(element)
                if step:
                    steps.append(step)
        return steps

    def process_step(self, element: Tag):
        section, text, emphasized_text, emphasized_tags = (None, None, None, None)
        step_number = self.extract_step_number(element)
        header_elements = self.get_header_elements(element)
        section = self.get_section(element, header_elements)
        text, emphasized_text, emphasized_tags = self.get_step_text(element)
        text, note, src, alt, video_src, emphasized_text, emphasized_tags = (
            self.process_next_elements(element, text, emphasized_text, emphasized_tags)
        )
        if section and step_number and text:
            text = self.sanitize_text(text)
            if note:
                note = self.sanitize_text(note)
            if src and (alt is None or alt == ""):
                alt = "Related diagram, image, or screenshot"
            return {
                "section": section,
                "step_num": step_number,
                "text": text,
                "src": src,
                "alt": alt,
                "video_src": video_src,
                "note": note,
                "emphasized_text": emphasized_text,
                "emphasized_tags": emphasized_tags,
            }
        else:
            self.logger.error("Could not parse step (element): %s", element)
            return None

    def get_header_elements(self, element: Tag):
        header_elements = ["h6", "h5", "h4", "h3", "h2", "h1"]
        try:
            header_elements.remove(element.name)
        except ValueError:
            header_elements = ["h6", "h5", "h4", "h3", "h2", "h1"]
        return header_elements

    def get_section(self, element: Tag, header_elements: List[str]):
        header = element.find_previous(header_elements)
        if header:
            if header.get_text(strip=True).lower() == "introduction":
                try:
                    header_elements.remove(header.name)
                except ValueError:
                    header_elements = header_elements
                header = element.find_previous(header_elements)
            section = header.get_text(strip=True)
        else:
            section = element.find_previous("h2").get_text(strip=True)
        if self.is_step_indicator(header):
            section = element.find_previous("h2").get_text(strip=True)
        self.logger.info("Determined section: %s", section)
        return self.sanitize_text(section)

    def get_emphasized_text(self, element: Tag):
        emphasized_text = []
        emphasized_tags = []
        tags = ["strong", "b", "em", "i"]
        if element.children and any([child.name in tags for child in element.children]):
            emphasized_text = [
                self.sanitize_text(child.get_text(strip=True))
                for child in element.children
                if self.is_tag(child) and child.name in tags
            ]
            emphasized_tags = [
                child.name
                for child in element.children
                if self.is_tag(child) and child.name in tags
            ]
        return emphasized_text, emphasized_tags

    def get_step_text(self, element: Tag):
        text = None
        emphasized_tags = []
        emphasized_text = []
        try:
            if re.match(r"^Step\s*\d+\.\s*(.*)", string=next(element.strings, "")):
                strngs = "".join(element.strings)
                print(f"STRINGS: {strngs}")
                text = re.sub(r"^Step \d+\.?", "", strngs).strip()
        except AttributeError as e:
            self.logger.error("Error extracting step text: %s", str(e))
            self.logger.error(
                "This could be desired behavior if the step text is in a sibling element."
            )
        if text:
            emphasized_text, emphasized_tags = self.get_emphasized_text(element)
        else:
            text = ""
        return text, emphasized_text, emphasized_tags

    def process_next_elements(
        self,
        element: Tag,
        text: str,
        emphasized_text: list,
        emphasized_tags: list,
    ):
        next_element: Tag = element.find_next_sibling()
        note, src, alt, video_src = None, None, None, None
        while (
            next_element
            and next_element.name not in self.headers
            and not re.search(
                f"^Step", string=next_element.text.strip(), flags=re.IGNORECASE
            )
        ):
            if next_element.text.strip().startswith("Step") or re.search(
                r"^Step", next_element.text
            ):
                break

            if next_element.name in {"p"} and next_element.find("img"):
                img = next_element.find("img")
                if self.is_cisco_doc_img("https://www.cisco.com" + img.get("src")):
                    src = "https://www.cisco.com" + img.get("src")
                    alt = img.get("alt", "Related diagram, image, or screenshot")

            if next_element.name in {"p"} and next_element.find("a"):
                anchor = next_element.find("a")
                # check if the anchor tag has a class attribute and if it contains "show-image-alone"
                if anchor.has_attr(
                    "class"
                ) and "show-image-alone" in anchor.get_attribute_list("class"):
                    src = anchor["href"]
                    alt = "Related diagram, image, or screenshot"

            if (
                next_element.name in {"a", "img"}
                and next_element.has_attr("class")
                and any(
                    class_ in next_element.get_attribute_list("class")
                    for class_ in ["show-image-alone"]
                )
            ):
                if next_element.name == "a":
                    src = next_element.get("href")
                    alt = "Related diagram, image, or screenshot"
                elif next_element.name == "img":
                    src = next_element.get("src")
                    alt = next_element.get(
                        "alt", "Related diagram, image, or screenshot"
                    )
                if self.is_blank_string(alt):
                    alt = "Related diagram, image, or screenshot"

            if next_element.name in {"img"} and self.is_cisco_doc_img(
                "https://www.cisco.com" + next_element.get("src")
            ):
                src = "https://www.cisco.com" + next_element.get("src")
                alt = next_element.get("alt", "Related diagram, image, or screenshot")
                if self.is_blank_string(alt):
                    alt = "Related diagram, image, or screenshot"

            if (
                next_element.name in {"div"}
                and next_element.has_attr("class")
                and any(
                    note_class in next_element.get_attribute_list("class")
                    for note_class in ["cdt-note", "cdt-best-practice"]
                )
            ):
                temp_note = next_element.get_text(strip=True, separator=" ")
                if note is not None:
                    note += " " + temp_note
                else:
                    note = temp_note

            if (
                next_element.name in {"div"}
                and next_element.has_attr("class")
                and any(
                    kbd_class in next_element.get_attribute_list("class")
                    for kbd_class in ["kbd-cdt"]
                )
            ):
                text += next_element.prettify()

            if next_element.name in {
                "ul",
                "ol",
                "kbd",
                "svg",
                "table",
                "pre",
                "code",
            } or (next_element.name == "a" and not next_element.has_attr("class")):
                text += " " + next_element.prettify()

            if next_element.name in {"video"} or next_element.find("video"):
                video = next_element.find("video")
                if video:
                    video_src = video.get("src", None)
                else:
                    video_src = next_element.get("src", None)

            if next_element.name in {"iframe"} or next_element.find("iframe"):
                iframe = next_element.find("iframe")
                if iframe:
                    video_src = iframe.get("src", None)
                else:
                    video_src = next_element.get("src", None)

            if next_element.name in {"p"} and next_element.find_parent(
                "div", class_="cdt-note"
            ):
                if note is None:
                    note = next_element.get_text(strip=True)
                else:
                    note += " " + next_element.get_text(strip=True)
            if (
                next_element.name in {"p"}
                and not next_element.text.startswith("Note")
                and not next_element.find(self.headers)
            ):
                text += " " + next_element.get_text(strip=True, separator=" ")
            elif next_element.name in {"p"} and next_element.text.startswith("Note"):
                pattern = re.compile(r"[\n\t]+|Note(?:\:)")
                temp_note = re.sub(
                    pattern,
                    " ",
                    next_element.get_text(strip=True, separator=" "),
                )
                if note is not None:
                    note += " " + temp_note
                else:
                    note = temp_note

            if next_element.find_all(["strong", "b", "em", "i"]):
                emphasized_text, emphasized_tags = self.get_emphasized_text(
                    next_element
                )
            if next_element.find(["h2", "h3", "h4", "h5", "h6"]):
                break

            text = text.strip()
            next_element = next_element.find_next_sibling(self.is_tag)

        return (text, note, src, alt, video_src, emphasized_text, emphasized_tags)

    def parse_backup_steps(self, soup: BeautifulSoup):
        html = soup.prettify()
        headers = ["h1", "h2", "h3", "h4", "h5", "h6"]
        headers_to_split_on = [("h2", "Header 2")]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = html_splitter.split_text(html)
        pattern = re.compile(
            r"""
                ^Objective         |  # Match 'Objective' at the start of the string
                ^Download\ Options |  # Match 'Download Options' at the start of the string
                ^Bias-Free         |  # Match 'Bias-Free' at the start of the string
                ^Applicable\ Devices |  # Match 'Applicable Devices' at the start of the string
                ^Introduction      |  # Match 'Introduction' at the start of the string
                ^Available\ Languages   # Match 'Available Languages' at the start of the string
                """,
            re.VERBOSE,
        )
        steps = []
        step_number = 0
        for i, doc in enumerate(splits):
            content = doc.page_content
            metadata = doc.metadata
            if "Header 2" in metadata:
                if pattern.search(content):
                    continue
                if metadata["Header 2"].lower() == "introduction":
                    continue
                if metadata["Header 2"].lower() == "objective":
                    continue
                if metadata["Header 2"].lower() == "table of contents":
                    continue
                if metadata["Header 2"].lower() == "support":
                    continue
                if metadata["Header 2"].lower() == "revision history":
                    continue
                if metadata["Header 2"].lower() == "applicable devices | software":
                    continue
                if metadata["Header 2"].lower() == "applicable devices":
                    continue

                section = metadata["Header 2"]
                step_number = step_number + 1
                text = self.sanitize_text(content)
                src = None
                alt = None
                found_element = soup.find("h2", string=re.compile(section))
                if not found_element:
                    continue
                next_element = found_element.find_next_sibling(self.is_tag)
                while next_element and next_element.name not in headers:
                    print(f"next_element: {next_element}")
                    if next_element.name in ["img"]:
                        src = "https://www.cisco.com" + next_element.get("src")
                        alt = next_element.get(
                            "alt", "Related diagram, image, or screenshot"
                        )
                    if next_element.name in ["a"] and next_element.has_attr("class"):
                        if "show-image-alone" in next_element.get_attribute_list(
                            "class"
                        ):
                            src = next_element.get("href")
                            alt = "Related diagram, image, or screenshot"
                    next_element = next_element.find_next_sibling(self.is_tag)
                steps.append(
                    {
                        "section": section,
                        "step_num": step_number,
                        "text": text,
                        "src": src,
                        "alt": alt,
                        "note": None,
                    }
                )
        return steps

    def get_revision_history(self, soup: BeautifulSoup) -> List[Revision]:
        element = soup.find("div", id="eot-revision-history")
        revisions = []
        if element and element.find("table"):
            headers = list(
                map(
                    lambda x: self.format_keys(x),
                    [header.get_text(strip=True) for header in element.find_all("th")],
                )
            )
            for row in element.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) == len(headers):
                    revision = dict(
                        zip(headers, [cell.get_text(strip=True) for cell in cells])
                    )
                    parsed_time = time.strptime(revision["publish_date"], "%d-%b-%Y")
                    revision["publish_date"] = date(
                        year=parsed_time.tm_year,
                        month=parsed_time.tm_mon,
                        day=parsed_time.tm_mday,
                    )
                    revision["revision"] = float(revision["revision"])
                    revisions.append(Revision(**revision))
        return revisions

    @staticmethod
    def is_cisco_doc_img(src: str):
        return bool(
            re.match(r"https://www\.cisco\.com/c/dam/en/us/support/docs/.*", src)
        )

    @staticmethod
    def sanitize_text(text: str) -> str:
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        cleaned_text = cleaned_text.replace("\\", "")
        cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)
        return cleaned_text


default_header_template = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class ArticleScraper:
    """A class for scraping articles from a list of URLs."""

    def __init__(
        self,
        series: List[str],
        urls: Sequence[str] = (),
        requests_per_second: int = 2,
        continue_on_failure: bool = True,
        ssl_verify: bool = False,
        default_parser: str = "html.parser",
        requests_kwargs: Optional[Dict[str, Any]] = None,
        bs_get_text_kwargs: Optional[Dict[str, Any]] = None,
        bs_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ArticleScraper.

        Args:
            urls (Sequence): A list of URLs or a single URL as a string.
            series (List[str]): A list of series names.

        Raises:
            TypeError: If `urls` is not a list or a string.
        """
        if urls:
            self.urls = list(urls)
        elif isinstance(urls, str):
            self.urls = [urls]
        elif isinstance(urls, Sequence):
            self.urls = list(urls)
        else:
            raise TypeError(f"urls must be str or Sequence[str], got ({type(urls)})")
        self.series = series
        self.requests_per_second = requests_per_second
        self.continue_on_failure = continue_on_failure
        self.ssl_verify = ssl_verify
        self.default_parser = default_parser
        self.requests_kwargs = requests_kwargs or {}
        self.bs_get_text_kwargs = bs_get_text_kwargs or {}
        self.bs_kwargs = bs_kwargs or {}
        self._session = requests.Session()
        self.unwanted_attributes = {
            "id": "fw-skiplinks",
            "class": "narrow-v2",
            "class": "linksRow",
            "class": "docHeaderComponent",
            "class": "availableLanguagesList",
            "class": "disclaimers",
            "id": "download-list-container",
            "id": "skiplink-content",
            "id": "skiplink-search",
            "id": "skiplink-footer",
        }
        self.article_parser = ArticleParser()

        header_template = default_header_template.copy()
        self._session.headers.update(header_template)
        if not self.ssl_verify:
            self._session.verify = False

    @staticmethod
    def _check_parser(parser: str) -> None:
        """Check that parser is valid for bs4."""
        valid_parsers = ["html.parser", "lxml", "xml", "lxml-xml", "html5lib"]
        if parser not in valid_parsers:
            raise ValueError(
                "`parser` must be one of " + ", ".join(valid_parsers) + "."
            )

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        async with aiohttp.ClientSession() as session:
            for i in range(retries):
                try:
                    async with session.get(
                        url,
                        headers=self._session.headers,
                        ssl=None if self._session.verify else False,
                        cookies=self._session.cookies.get_dict(),
                    ) as response:
                        return await response.text()
                except aiohttp.ClientConnectionError as e:
                    if i == retries - 1:
                        raise
                    else:
                        logger.warning(
                            f"Error fetching {url} with attempt "
                            f"{i + 1}/{retries}: {e}. Retrying..."
                        )
                        await asyncio.sleep(cooldown * backoff**i)
        raise ValueError("retry count exceeded")

    async def _fetch_with_rate_limit(
        self, url: str, semaphore: asyncio.Semaphore
    ) -> str:
        async with semaphore:
            try:
                return await self._fetch(url)
            except Exception as e:
                if self.continue_on_failure:
                    logger.warning(
                        f"Error fetching {url}, skipping due to"
                        f" continue_on_failure=True"
                    )
                    return ""
                logger.exception(
                    f"Error fetching {url} and aborting, use continue_on_failure=True "
                    "to continue loading urls after encountering an error."
                )
                raise e

    async def fetch_all(self, urls: List[str]) -> Any:
        """Fetch all urls concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            from tqdm.asyncio import tqdm_asyncio

            return await tqdm_asyncio.gather(
                *tasks, desc="Fetching pages", ascii=True, mininterval=1
            )
        except ImportError:
            warnings.warn("For better logging of progress, `pip install tqdm`")
            return await asyncio.gather(*tasks)

    def scrape(self):
        """Scrape the articles from the list of urls."""
        soups = self.scrape_all(self.urls)
        for i, soup in enumerate(soups):
            url = self.urls[i]
            series = self.series[i]
            self.remove_unwanted_elements_by_attrs(soup, self.unwanted_attributes)
            self.remove_unwanted_tags(soup)
            yield self.article_parser.parse(soup, url, series)

    def scrape_all(
        self, urls: List[str], parser: Union[str, None] = None
    ) -> List[BeautifulSoup]:
        """Fetch all urls, then return soups for all results."""
        from bs4 import BeautifulSoup

        results = asyncio.run(self.fetch_all(urls))
        final_results = []
        for i, result in enumerate(results):
            url = urls[i]
            if parser is None:
                if url.endswith(".xml"):
                    parser = "xml"
                else:
                    parser = self.default_parser
                self._check_parser(parser)
            final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))

        return final_results

    @staticmethod
    def remove_unwanted_tags(
        soup: BeautifulSoup,
        tags: Optional[List[str]] = None,
    ):
        if tags is None:
            tags = [
                "nav",
                "aside",
                "form",
                "header",
                "noscript",
                "canvas",
                "footer",
                "script",
                "style",
                "cdc-header",
                "cdc-footer",
            ]
        for tag in soup(tags):
            tag.decompose()

    @staticmethod
    def remove_unwanted_elements_by_attrs(soup: BeautifulSoup, attrs: Dict[str, str]):
        for k, v in attrs.items():
            element = soup.find(attrs={k: v})
            if element:
                element.decompose()


def convert_series_to_product_family(abbreviation: str) -> str:
    product_family_name_map = {
        "Catalyst-1200": "Cisco Catalyst 1200 Series Switches",
        "Catalyst-1300": "Cisco Catalyst 1300 Series Switches",
        "CBS220": "Cisco Business 220 Series Smart Switches",
        "CBS250": "Cisco Business 250 Series Smart Switches",
        "CBS350": "Cisco Business 350 Series Managed Switches",
        "switches-350-family": "Cisco 350 Series Managed Switches",
        "switches-350x-family": "Cisco 350X Series Stackable Managed Switches",
        "switches-550x-family": "Cisco 550X Series Stackable Managed Switches",
        "routers-100-family": "RV100 Product Family",
        "routers-160-family": "RV160 VPN Router",
        "routers-260-family": "RV260 VPN Router",
        "wireless-mesh-100-200-series": "Cisco Business Wireless AC",
        "wireless-mesh-100-AX-series": "Cisco Business Wireless AX",
        "routers-320-family": "RV320 Product Family",
        "routers-340-family": "RV340 Product Family",
    }
    return product_family_name_map.get(abbreviation, abbreviation)


def upload_articles_to_db(articles: List[ArticleDict]):
    uri = (
        os.environ.get("MONGO_URI")
        .replace("<username>", os.environ.get("MONGODB_APP_USER"))
        .replace("<password>", os.environ.get("MONGODB_APP_PASS"))
    )
    client = pymongo.MongoClient(uri)
    db = client["smb_documents"]
    articles_collection = db["articles"]
    product_fam_collection = db["product_families"]

    for article in articles:
        series = article["series"]
        pf = product_fam_collection.find_one({"name": series})
        for device in article["applicable_devices"]:
            if device["software_link"] is None:
                if pf:
                    device["software_link"] = pf["software_url"]
            if device["datasheet_link"] is None:
                if pf:
                    device["datasheet_link"] = pf["datasheet_url"][0]

    article_ids = []

    for article in articles:
        series = article["series"]
        pf = product_fam_collection.find_one({"name": series})
        if not pf:
            logger.info(f"Product Family {series} not found in the database.")
            continue
        existing_article = articles_collection.find_one(
            {"document_id": article["document_id"]}
        )
        if existing_article:
            logger.info(
                f"Article {article['document_id']} already exists in the database."
            )
            articles_collection.update_one(
                {"_id": existing_article["_id"]}, {"$addToSet": {"series": pf["_id"]}}
            )
            article_ids.append(existing_article["_id"])
        else:
            try:
                article["series"] = [pf["_id"]]
                article_id = articles_collection.insert_one(article).inserted_id
                logger.info(f"Article {article_id} inserted into the database.")
                article_ids.append(article_id)
            except pymongo.errors.WriteError as e:
                logger.error(
                    f"Error inserting article {article['document_id']} into the database. {e}"
                )
                continue
            except pymongo.errors.OperationFailure as e:
                logger.error(
                    f"Error inserting article {article['document_id']} into the database. {e}"
                )
                continue


def process_links():
    links = get_article_links_after_spidering()
    urls = []
    series = []
    for link in links:
        if "url" in link and "family" in link:
            urls.append(link["url"])
            series.append(convert_series_to_product_family(link["family"]))
    return urls, series


def run_scraper():
    urls, series = process_links()
    scraper = ArticleScraper(
        series=series,
        urls=urls,
    )
    articles = [article.to_dict() for article in scraper.scrape()]
    upload_articles_to_db(articles)
    return articles


run_scraper()
