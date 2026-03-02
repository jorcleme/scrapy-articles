import asyncio
import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import date
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict, TypeVar, Union
from uuid import uuid4

import aiohttp
import html2text
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_serializer, field_validator

logger = logging.getLogger(__name__)

CURRENT_FILE = Path(__file__).resolve()
ARTICLES_DIR = CURRENT_FILE.parent.parent
SERVICES_DIR = CURRENT_FILE.parent
DATA_DIR = ARTICLES_DIR / "data"

load_dotenv(find_dotenv(filename=".env"))


# Enhanced logging configuration
def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logger with proper configuration and error handling."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, mode="a", encoding="utf8")
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Failed to set up file logging: {e}")

    return logger


logger = setup_logger(__name__)


class ArticleCategory(Enum):
    """Standardized article categories."""

    TROUBLESHOOTING = "Troubleshooting"
    CONFIGURATION = "Configuration"
    INSTALL_UPGRADE = "Install & Upgrade"
    MAINTAIN_OPERATE = "Maintain & Operate"
    DESIGN = "Design"


@dataclass
class ParsingConfig:
    """Configuration for article parsing behavior."""

    use_llm_fallback: bool = False
    llm_timeout: float = 30.0
    max_retries: int = 3
    default_category: ArticleCategory = ArticleCategory.CONFIGURATION
    skip_empty_steps: bool = False
    default_bs_parser: str = "html.parser"


class LinksDict(BaseModel):
    """Type-safe representation of scraped links."""

    url: str
    family: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if not v or not isinstance(v, str) or not v.startswith("http"):
            raise ValueError("Invalid URL")
        return v


class ArticleStep(BaseModel):
    section: str = Field(
        ..., description="The section header of this particular set of steps."
    )
    step_num: int = Field(
        ...,
        description="The step number within the section usually prepended with the word 'Step'.",
    )
    text: str = Field(
        ...,
        description="The actual text content of the step. This is all of the text from the beginning of this step until the next step.",
    )
    src: Optional[str] = Field(
        None, description="An optional image URL if this step has an associated image."
    )
    alt: Optional[str] = Field(
        None, description="The alt text for the image if an image is present."
    )
    note: Optional[str] = Field(
        None,
        description="An optional note associated with the step. Usually prepended with 'Note:'",
    )
    video_src: Optional[str] = Field(
        None, description="An optional video URL if this step has an associated video."
    )
    emphasized_text: Optional[List[str]] = Field(
        None, description="A list of emphasized text within the step (bold, italic)."
    )
    emphasized_tags: Optional[List[str]] = Field(
        None,
        description="A list of HTML tags used for emphasizing text within the step.",
    )


class ArticleSteps(BaseModel):
    steps: List[ArticleStep]


def fetch_links_from_fs() -> List[LinksDict]:
    links = DATA_DIR / "links.json"
    with open(links, "r+", encoding="utf8") as f:
        data = json.load(f)
        return [LinksDict(**item) for item in data]


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


class Article(BaseModel):
    """Improved Article class with better validation."""

    series: str
    title: str
    document_id: str
    category: str
    url: str
    objective: Optional[str] = None
    applicable_devices: List[Dict[str, Any]] = Field(default_factory=list)
    intro: Optional[str] = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    revision_history: List[Revision] = Field(default_factory=list)
    type: str = "article"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
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
            "revision_history": [r.model_dump() for r in self.revision_history],
            "type": self.type,
        }


class InterfaceContext(Enum):
    """Supported interface contexts for article steps."""

    WEB_UI = "web_ui"
    CLI = "cli"


class ArticleParser:
    """A class that parses HTML to extract Cisco SMB articles."""

    def __init__(self, config: Optional[ParsingConfig] = None) -> None:
        self.config = config or ParsingConfig()
        self.logger = setup_logger(
            self.__class__.__name__,
            CURRENT_FILE.parent / "logs" / f"{self.__class__.__name__.lower()}.log",
        )
        self.headers = ["h6", "h5", "h4", "h3", "h2", "h1"]

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
        # Clear article separator
        self.logger.info("=" * 80)
        self.logger.info(f"🔄 PARSING ARTICLE: {url}")
        self.logger.info("=" * 80)

        try:
            title = self._get_title(soup)
            document_id = self._get_document_id(soup)
            category = self._get_category(soup, title)
            objective = self._get_objective(soup)
            applicable_devices = self._get_applicable_devices(soup)
            intro = self._get_intro(soup)
            steps = self._get_steps(soup)
            revision_history = self._get_revision_history(soup)

            # Concise summary log
            self.log_article_summary(
                title,
                document_id,
                category,
                len(steps),
                len(applicable_devices),
                len(revision_history),
            )

            return Article(
                series=series,
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
            self.logger.error(f"❌ Error parsing article from URL: {url}")
            self.logger.error(f"Error details: {str(err)}")
            pass

    def log_article_summary(
        self,
        title,
        document_id,
        category,
        steps_count,
        devices_count,
        revisions_count,
    ):
        """Log a concise summary of the parsed article components."""
        self.logger.info(f"📋 Title: {title}")
        self.logger.info(f"🆔 Doc ID: {document_id} | 📂 Category: {category}")
        self.logger.info(
            f"📝 Steps: {steps_count} | 📱 Devices: {devices_count} | 🔄 Revisions: {revisions_count}"
        )
        self.logger.info("✅ Article parsing completed successfully")

    @staticmethod
    def _get_category_with_llm(title: str) -> str:
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
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), timeout=2.0, model="gpt-5"
        )
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
        if not node.text:
            return False

        text = node.text.strip()
        step_pattern = re.compile(r"^Step \d+", re.IGNORECASE)

        if node.name in ["h3", "h4"]:
            # Accept h3 and h4 elements that start with "Step X"
            return step_pattern.match(text) is not None
        elif node.name == "p":
            # For paragraphs, be more strict:
            # 1. Must start with "Step X." (with period)
            # 2. Must NOT contain embedded h4 elements (avoid complex paragraphs)
            step_with_period = re.compile(r"^Step \d+\.", re.IGNORECASE)
            if step_with_period.match(text):
                # Check if this paragraph contains embedded h4 elements
                embedded_h4s = node.find_all("h4")
                if len(embedded_h4s) == 0:
                    return True

        return False

    @staticmethod
    def extract_step_number(node: Tag) -> Union[int, None]:
        if not node.text:
            return None

        text = node.text.strip()
        # Look for "Step X" or "Step X." patterns
        pattern = re.compile(r"Step (\d+)", re.IGNORECASE)
        match = pattern.match(text)
        return int(match.group(1)) if match else None

    def _get_title(self, soup: BeautifulSoup) -> str:
        title_elem = soup.find(id="fw-pagetitle")
        if title_elem:
            return title_elem.text.strip()
        else:
            title = soup.title.string
            pattern = re.compile(r"^(.*?)(?= - Cisco)")
            match = pattern.search(title)
            return match.group(1) if match else title

    def _get_document_id(self, soup: BeautifulSoup) -> str:
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

    def _get_category(self, soup: BeautifulSoup, title: str) -> str:
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
                return category
            else:
                category = self._get_category_with_llm(title)
                return category
        category = self._get_category_with_llm(title)
        return category

    def _get_objective(self, soup: BeautifulSoup) -> Optional[str]:

        def extract_objective_text(element: Tag) -> str:
            objective_text = []
            sibling = element.find_next_sibling()
            while sibling and sibling.name in ["p", "ul", "ol", "table"]:
                if sibling.name in ["ul", "ol", "table"]:
                    objective_text.app(str(sibling.extract()))
                elif sibling.name in ["p"]:
                    objective_text.append(sibling.get_text(strip=True))

                sibling = sibling.find_next_sibling()

            objective = " ".join(objective_text)
            objective = self.sanitize_text(objective)
            return objective

        pattern = r"^Objective:?\s?"

        if objective_element := soup.find(
            "h2", string=re.compile(pattern, re.IGNORECASE)
        ):
            return extract_objective_text(objective_element)
        elif objective_element := soup.find(name="h2", id="objective"):
            return extract_objective_text(objective_element)
        else:
            return None

    def _get_applicable_devices(self, soup: BeautifulSoup) -> List[dict]:
        pattern = re.compile(
            r"(Applicable Devices\s*\|\s*Software|Applicable\s+Switches\s*|Applicable Devices\s*\|\s*Software Version|Applicable Devices|Applicable Devices \| Firmware Version|Applicable Devices\s*|Applicable Devices:|Applicable Devices and Software Version|Applicable Device|Applicable Devices|Applicable Devices \| Software)\b",
            re.IGNORECASE,
        )
        element = soup.find(["h2", "h3"], string=pattern)
        applicable_devices = []

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
                    applicable_devices.append(
                        {
                            "device": device_name,
                            "software": version,
                            "datasheet_url": (
                                datasheet_link["href"] if datasheet_link else None
                            ),
                            "software_url": (
                                download_latest_link["href"]
                                if download_latest_link
                                else None
                            ),
                        }
                    )
        return applicable_devices

    def _get_intro(self, soup: BeautifulSoup) -> Union[str, None]:
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

    def _get_steps(self, soup: BeautifulSoup) -> List:
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
        # Try traditional parsing first
        elements = soup.find_all(["h3", "h4", "p"])
        steps = []

        for _, element in enumerate(elements):
            if not self.is_tag(element) or self.is_blank_string(element.get_text()):
                continue
            elif self.is_step_indicator(element):
                step = self._process_step(element)
                if step:
                    steps.append(step)

        # Return default if nothing found
        return steps

    def _process_step(self, element: Tag):
        section, text, emphasized_text, emphasized_tags = (None, None, None, None)
        step_number = self.extract_step_number(element)
        header_elements = self._filter_headers(element)
        section = self._get_step_section(element, header_elements)
        text, emphasized_text, emphasized_tags = self._get_step_text(element)
        text, note, src, alt, video_src, emphasized_text, emphasized_tags = (
            self._process_next_elements(element, text, emphasized_text, emphasized_tags)
        )

        if section and step_number and text:
            text = self.sanitize_text(text)
            if note:
                note = self.sanitize_text(note)
            if src and not alt:
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
            self.logger.error("Section: %s", section)
            self.logger.error("Step Number: %s", step_number)
            self.logger.error("Text: %s", text)
            return None

    def _filter_headers(self, element: Tag):
        header_elements = ["h6", "h5", "h4", "h3", "h2", "h1"]
        try:
            header_elements.remove(element.name)
        except ValueError:
            header_elements = ["h6", "h5", "h4", "h3", "h2", "h1"]
        return header_elements

    def _get_step_section(self, element: Tag, header_elements: List[str]):
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

        if not section or self.is_blank_string(section):
            section = "General"
        return self.sanitize_text(section)

    def _get_step_emphasized_text(self, element: Tag):
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

    def _get_step_text(self, element: Tag):
        text = None
        emphasized_tags = []
        emphasized_text = []
        try:
            if re.match(r"^Step\s*\d+\.\s*(.*)", string=next(element.strings, "")):
                strngs = "".join(element.strings)
                text = re.sub(r"^Step \d+\.?", "", strngs).strip()
        except AttributeError as e:
            self.logger.error("Error extracting step text: %s", str(e))
            self.logger.error(
                "This could be desired behavior if the step text is in a sibling element."
            )
        if text:
            emphasized_text, emphasized_tags = self._get_step_emphasized_text(element)
        else:
            text = ""
        return text, emphasized_text, emphasized_tags

    def _process_next_elements(
        self,
        element: Tag,
        text: str,
        emphasized_text: list,
        emphasized_tags: list,
    ):
        next_element: Tag = element.find_next_sibling()
        note, src, alt, video_src = None, None, None, None
        elements_processed = 0
        while (
            next_element
            and next_element.name not in self.headers
            and elements_processed < 50  # Safety limit to prevent infinite loops
        ):
            elements_processed += 1

            # Only break for actual step headers (h4), not complex paragraphs containing step text
            # But be careful with paragraphs that might contain both images and step text
            if next_element.name in [
                "h3",
                "h4",
            ] and next_element.text.strip().startswith("Step"):
                break

            if next_element.name in ["div"] and any(
                id in next_element.get("id", "")
                for id in ["CDT-Tag-Container", "CDT-Tag-Container-Button"]
            ):
                break

            # For paragraphs, only break if it's clearly a step paragraph (not a complex one with images)
            if (
                next_element.name == "p"
                and next_element.text.strip().startswith("Step")
                and not next_element.find_all(
                    "img"
                )  # Don't break if it has images - process them first
                and not next_element.find_all("h4")
            ):  # Don't break if it has embedded headers
                break

            # CRITICAL FIX: Handle complex paragraphs that mix images with step headers
            # Extract images IMMEDIATELY when we encounter such elements
            if next_element.name in {"p"} and next_element.find_all("h4"):
                self.logger.info("Found complex paragraph with embedded headers :(")
                # This is a complex paragraph with embedded headers - extract images first
                images = next_element.find_all("img", recursive=True)
                if images and not src:
                    first_img = images[0]
                    if self.is_absolute_path(first_img.get("src")):
                        src = first_img.get("src")
                    else:
                        src = "https://www.cisco.com" + first_img.get("src")
                    alt = first_img.get("alt", "Related diagram, image, or screenshot")
                    if self.is_blank_string(alt):
                        alt = "Related diagram, image, or screenshot"
                # Skip processing this complex element further to avoid conflicts
                next_element = next_element.find_next_sibling(self.is_tag)
                continue

            # IMPROVED IMAGE EXTRACTION: Check for images FIRST, before other processing
            # This handles standard cases where images are in simple paragraphs
            if next_element.name in {"p"}:

                # Extract images from this element before processing other content
                images = next_element.find_all("img", recursive=True)
                if (
                    images and not src
                ):  # Only take the first image if we don't have one already
                    first_img = images[0]
                    if self.is_absolute_path(first_img.get("src")):
                        src = first_img.get("src")
                    else:
                        src = "https://www.cisco.com" + first_img.get("src")
                    alt = first_img.get("alt", "Related diagram, image, or screenshot")
                    if self.is_blank_string(alt):
                        alt = "Related diagram, image, or screenshot"

                if next_element.find(name="div", id="CDT-Tag-Container"):
                    self.logger.info(
                        "Found CDT tag container: %s",
                        next_element.find(name="div", id="CDT-Tag-Container").get("id"),
                    )
                    next_element = next_element.find_next_sibling(self.is_tag)
                    continue

            # Check for anchors with images
            if next_element.name in {"p"} and next_element.find("a") and not src:
                anchor = next_element.find("a")
                if anchor and anchor.find("img"):
                    img = anchor.find("img")
                    if self.is_absolute_path(img.get("src")):
                        src = img.get("src")
                    else:
                        src = "https://www.cisco.com" + img.get("src")
                    alt = img.get("alt", "Related diagram, image, or screenshot")

            # Check for anchors with show-image-alone class
            if next_element.name in {"p"} and next_element.find("a") and not src:
                anchor = next_element.find("a")
                if (
                    anchor
                    and anchor.has_attr("class")
                    and any(
                        _class in anchor.get_attribute_list("class")
                        for _class in ["show-image-alone"]
                    )
                ):
                    if self.is_absolute_path(anchor.get("href")):
                        src = anchor.get("href")
                    else:
                        src = "https://www.cisco.com" + anchor.get("href")
                    alt = "Related diagram, image, or screenshot"

            # Handle direct img elements
            if next_element.name in {"img"} and not src:
                if self.is_absolute_path(next_element.get("src")):
                    src = next_element.get("src")
                else:
                    src = "https://www.cisco.com" + next_element.get("src")
                alt = next_element.get("alt", "Related diagram, image, or screenshot")
                if self.is_blank_string(alt):
                    alt = "Related diagram, image, or screenshot"

            # Handle anchors and images with show-image-alone class
            if (
                next_element.name in {"a", "img"}
                and next_element.has_attr("class")
                and any(
                    class_ in next_element.get_attribute_list("class")
                    for class_ in ["show-image-alone"]
                )
                and not src
            ):
                if next_element.name == "a":
                    if self.is_absolute_path(next_element.get("href")):
                        src = next_element.get("href")
                    else:
                        src = "https://www.cisco.com" + next_element.get("href")
                    alt = "Related diagram, image, or screenshot"
                elif next_element.name == "img":
                    if self.is_absolute_path(next_element.get("src")):
                        src = next_element.get("src")
                    else:
                        src = "https://www.cisco.com" + next_element.get("src")
                    alt = next_element.get(
                        "alt", "Related diagram, image, or screenshot"
                    )
                if self.is_blank_string(alt):
                    alt = "Related diagram, image, or screenshot"

            # Handle notes in div elements
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

            # Handle bold/strong text
            if next_element.name in {"strong", "b"}:
                text += next_element.get_text(strip=True)

            # Handle keyboard shortcuts
            if (
                next_element.name in {"div"}
                and next_element.has_attr("class")
                and any(
                    kbd_class in next_element.get_attribute_list("class")
                    for kbd_class in ["kbd-cdt"]
                )
            ):
                text += next_element.prettify()

            # Handle lists, code blocks, etc.
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

            # Handle videos
            if next_element.name in {"video"} or next_element.find("video"):
                video = next_element.find("video")
                if video:
                    video_src = video.get("src", None)
                else:
                    video_src = next_element.get("src", None)

            # Handle iframes
            if next_element.name in {"iframe"} or next_element.find("iframe"):
                iframe = next_element.find("iframe")
                if iframe:
                    video_src = iframe.get("src", None)
                else:
                    video_src = next_element.get("src", None)

            # Handle notes in paragraphs
            if next_element.name in {"p"} and next_element.find_parent(
                "div", class_="cdt-note"
            ):
                if note is None:
                    note = next_element.get_text(strip=True)
                else:
                    note += " " + next_element.get_text(strip=True)

            # Handle regular paragraph text (but be more careful about headers)
            elif (
                next_element.name in {"p"}
                and not next_element.text.startswith("Note")
                and not next_element.find(self.headers)
            ):
                text += " " + next_element.get_text(strip=True, separator=" ")

            # Handle note paragraphs
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

            # Handle emphasized text
            if next_element.find_all(["strong", "b", "em", "i"]):
                emphasized_text, emphasized_tags = self._get_step_emphasized_text(
                    next_element
                )

            # CRITICAL: Break on headers that would end this step's processing
            # BUT only if we haven't found images that belong to this step
            if next_element.find(["h2", "h3", "h4", "h5", "h6"]):
                break

            text = text.strip()
            next_element = next_element.find_next_sibling(self.is_tag)

        return (text, note, src, alt, video_src, emphasized_text, emphasized_tags)

    def _get_revision_history(self, soup: BeautifulSoup) -> List[Revision]:
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
    def is_cisco_support_image(src: str):
        return bool(
            re.match(r"https://www\.cisco\.com/c/dam/en/us/support/docs/.*", src)
        )

    @staticmethod
    def is_absolute_path(src: str) -> bool:
        """
        Check if the image source string is an absolute path (starts with http/https)
        or a relative path.

        Args:
            src (str): The image source string to check

        Returns:
            bool: True if absolute path (starts with http/https), False if relative
        """
        if not src:
            return False
        return src.startswith(("http://", "https://"))

    @staticmethod
    def sanitize_text(text: str) -> str:
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        cleaned_text = cleaned_text.replace("\\", "")
        cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)
        return cleaned_text


class ArticleScraper:
    """A class for scraping articles from a list of URLs."""

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    DEFAULT_PARSER = "lxml"

    def __init__(
        self,
        urls: Sequence[str],
        series: Sequence[str],
        *,
        requests_per_second: int = 1,  # Reduced from 2 to 1 to be gentler on the server
        continue_on_failure: bool = True,
        ssl_verify: bool = False,
        parser_config: Optional[ParsingConfig] = None,
        timeout: int = 60,  # Increased timeout to 60 seconds for large pages
    ):
        """Initialize scraper with improved configuration."""
        if len(urls) != len(series):
            raise ValueError("URLs and series lists must have the same length")

        self.urls = list(urls)
        self.series = list(series)
        self.requests_per_second = requests_per_second
        self.continue_on_failure = continue_on_failure
        self.ssl_verify = ssl_verify
        self.timeout = timeout

        self.logger = setup_logger(self.__class__.__name__)
        self.parser = ArticleParser(parser_config)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS.copy())
        self.session.verify = ssl_verify

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
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i in range(retries):
                try:
                    async with session.get(
                        url,
                        headers=self.session.headers,
                        ssl=False,
                    ) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"HTTP {response.status}",
                            )
                except (
                    aiohttp.ClientConnectionError,
                    aiohttp.ServerTimeoutError,
                    asyncio.TimeoutError,
                    aiohttp.ClientResponseError,
                ) as e:
                    if i == retries - 1:
                        self.logger.error(
                            f"Failed to fetch {url} after {retries} attempts: {type(e).__name__}: {e}"
                        )
                        raise
                    else:
                        self.logger.warning(
                            f"Error fetching {url} with attempt "
                            f"{i + 1}/{retries}: {type(e).__name__}: {e}. Retrying in {cooldown * backoff**i:.1f}s..."
                        )
                        await asyncio.sleep(cooldown * backoff**i)
        raise ValueError("retry count exceeded")

    async def _fetch_with_rate_limit(
        self, url: str, semaphore: asyncio.Semaphore
    ) -> str:
        async with semaphore:
            try:
                return await self._fetch(url)
            except (
                aiohttp.ClientConnectionError,
                aiohttp.ServerTimeoutError,
                asyncio.TimeoutError,
            ) as e:
                if self.continue_on_failure:
                    logger.warning(
                        f"Timeout/connection error fetching {url}: {type(e).__name__}: {e}. Skipping due to continue_on_failure=True"
                    )
                    return ""
                logger.exception(
                    f"Timeout/connection error fetching {url}: {type(e).__name__}: {e}. Aborting - use continue_on_failure=True to continue after errors."
                )
                raise e
            except Exception as e:
                if self.continue_on_failure:
                    logger.warning(
                        f"Unexpected error fetching {url}: {type(e).__name__}: {e}. Skipping due to continue_on_failure=True"
                    )
                    return ""
                logger.exception(
                    f"Unexpected error fetching {url}: {type(e).__name__}: {e}. Aborting - use continue_on_failure=True to continue after errors."
                )
                raise e

    async def scrape_all(
        self, urls: List[str], parser: Union[str, None] = None
    ) -> List[BeautifulSoup]:
        """Fetch all urls, then return soups for all results."""
        from bs4 import BeautifulSoup

        results = await self.fetch_all(urls)
        final_results = []
        for i, result in enumerate(results):
            url = urls[i]
            if parser is None:
                if url.endswith(".xml"):
                    parser = "xml"
                else:
                    parser = self.DEFAULT_PARSER
                self._check_parser(parser)
            final_results.append(BeautifulSoup(result, parser))

        return final_results

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
                *tasks, desc="Fetching articles", ascii=True, mininterval=1
            )
        except ImportError:
            warnings.warn("For better logging of progress, `pip install tqdm`")
            return await asyncio.gather(*tasks)

    async def scrape(self):
        """Scrape the articles from the list of urls."""
        soups = await self.scrape_all(self.urls)
        for i, soup in enumerate(soups):
            url = self.urls[i]
            series = self.series[i]
            self._clean_soup(soup)
            yield self.parser.parse(soup, url, series)

    @staticmethod
    def _clean_soup(soup: BeautifulSoup):
        tags = [
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

        atts = {
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

        for tag in soup(tags):
            tag.decompose()

        for k, v in atts.items():
            element = soup.find(attrs={k: v})
            if element:
                element.decompose()


# Product family mapping
PRODUCT_FAMILY_MAPPING = {
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


def convert_series_to_product_family(abbreviation: str) -> str:
    """Convert abbreviation to full product family name."""
    return PRODUCT_FAMILY_MAPPING.get(abbreviation, abbreviation)


def process_links() -> list[tuple[str, str]]:
    """
    Build a list of (url, series_name) tuples from the scraped links so
    each url stays paired with its corresponding (converted) family/series.
    """
    links = fetch_links_from_fs()
    url_series_pairs: list[tuple[str, str]] = []
    for link in links:
        if "url" in link and "family" in link:
            url = link.url
            series_name = convert_series_to_product_family(link.family)
            url_series_pairs.append((url, series_name))
        else:
            logger.info("Link missing expected keys 'url' and/or 'family': %s", link)
    return url_series_pairs


def load_scraped_links() -> List[LinksDict]:
    """Load links from JSON file with validation."""
    links_file = DATA_DIR / "links.json"

    try:
        with links_file.open("r", encoding="utf8") as f:
            data = json.load(f)

        # Validate links
        validated_links = []
        for item in data:
            try:
                link = LinksDict(**item)
                validated_links.append(link)
            except Exception as e:
                logger.warning(f"Invalid link data: {item}, error: {e}")

        return validated_links

    except Exception as e:
        logger.error(f"Error loading links file: {e}")
        return []


def save_articles_to_filesystem(
    articles: List[Article], filename: str = "articles.json"
):
    """Save articles to filesystem with error handling."""
    try:
        output_file = DATA_DIR / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionaries
        articles_data = [article.model_dump() for article in articles if article]

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(articles_data)} articles to {output_file}")

    except Exception as e:
        logger.error(f"Error saving articles: {e}")
        raise


async def run_article_extraction(
    config: Optional[ParsingConfig] = None, output_filename: str = "articles.json"
) -> List[Article]:
    """Main scraper function with improved error handling."""
    logger.info("Starting improved article scraper")

    try:
        # Load links
        links = load_scraped_links()
        if not links:
            logger.warning("No links found to process")
            return []

        # Convert to URL-series pairs
        url_series_pairs = [
            (link.url, convert_series_to_product_family(link.family)) for link in links
        ]

        if not url_series_pairs:
            logger.warning("No valid URL-series pairs found")
            return []

        catalyst_1200_urls_and_series = []

        for url, series in url_series_pairs:
            if "Cisco Catalyst 1200 Series Switches" in series:
                catalyst_1200_urls_and_series.append((url, series))

        # Extract URLs and series
        urls, series = zip(*catalyst_1200_urls_and_series)

        # Create and run scraper
        scraper = ArticleScraper(
            urls=urls,
            series=series,
            parser_config=config,
            continue_on_failure=True,
        )

        articles = []
        async for article in scraper.scrape():
            if article:
                articles.append(article)

        save_articles_to_filesystem(articles, output_filename)

        logger.info(f"Scraping completed. Processed {len(articles)} articles")
        return articles

    except Exception as e:
        logger.error(f"Error in scraper execution: {e}")
        raise


# Main execution
if __name__ == "__main__":

    # Configure parsing behavior for full run (default: no LLM fallback)
    config = ParsingConfig(
        use_llm_fallback=False,
        llm_timeout=30.0,
        max_retries=3,
        skip_empty_steps=False,
    )

    # Run the full scraper
    try:
        articles = asyncio.run(run_article_extraction(config))
        logger.info(f"Scraping completed successfully with {len(articles)} articles")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise
