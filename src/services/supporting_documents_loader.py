import json
import requests
import re
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup, Tag
from typing import List, Iterator, Dict, Any, Optional, Literal, Union
from langchain_text_splitters import TextSplitter
from langchain.schema import Document
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb.utils.embedding_functions as embedding_functions
from lxml import etree
from config import DATA_DIR, LOGS_DIR, ADMIN_GUIDE_DATA_DIR, CLI_GUIDE_DATA_DIR
from src.constants import CollectionFactory
from src.db.chroma import VECTOR_DB


def setup_logging() -> logging.Logger:
    """Setup logging with file handler (overwrites on each run) and stream handler."""
    logger = logging.getLogger(__name__)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Set logging level
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create logs directory if it doesn't exist
    logs_dir = LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    # File handler - overwrites file on each run (mode='w')
    log_file = logs_dir / "supporting_documents_loader.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


logger = setup_logging()

load_dotenv()

MAX_TOKENS = 8192

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


class CLIParser:
    DESCRIPTION = "description"
    COMMAND_NAME = "command_name"
    TOPIC = "topic"
    SYNTAX = "syntax"
    PARAMETERS = "parameters"
    DEFAULT_CONFIGURATION = "default_configuration"
    COMMAND_MODE = "command_mode"
    USER_GUIDELINES = "user_guidelines"
    EXAMPLES = "examples"

    def __init__(self):
        pass

    @staticmethod
    def sanitize_text(text: str) -> str:
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        cleaned_text = cleaned_text.replace("\\", "")
        cleaned_text = cleaned_text.replace("#", " ")
        cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)
        return cleaned_text

    def load(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        return self._parse(soup)

    def _parse(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        cli_sections = []
        page_article_bodies = soup.find_all(
            "article", class_=("topic", "reference", "nested1")
        )
        if not page_article_bodies or len(page_article_bodies) == 0:
            print(
                "No article body found. This is usually an index page or reference page."
            )
            return cli_sections

        for article_body in page_article_bodies:
            topic_sections = article_body.find_all("section", class_=("body"))
            topic = soup.find("meta", attrs={"name": "description"}).get(
                "content", None
            )
            for section in topic_sections:
                if re.match(
                    r"^This chapter contains the following sections:",
                    section.get_text().strip(),
                ):
                    continue
                command_name = section.find_previous(class_=("title",)).get_text(
                    strip=True
                )
                if topic and topic == "Introduction":
                    cli_sections.append(
                        self._parse_intro_section(section, command_name, topic)
                    )
                else:
                    cli_sections.append(
                        self._parse_detail_section(section, command_name, topic)
                    )

        return cli_sections

    def _parse_intro_section(self, section: Tag, command_name: str, topic: str):
        sections = []
        content = {self.DESCRIPTION: [], self.COMMAND_NAME: command_name}
        for child in section.children:
            if not isinstance(child, Tag):
                continue
            sections.extend(self._extract_text_from_tag(child))
        content[self.DESCRIPTION] = list(
            filter(lambda x: x != "", content[self.DESCRIPTION])
        )
        content[self.DESCRIPTION] = list(map(self.sanitize_text, sections))
        content[self.TOPIC] = topic
        return content

    def _parse_detail_section(self, section: Tag, command_name: str, topic: str):
        sections = section.find_all("section")
        (
            description,
            syntax,
            params,
            default_config,
            command_mode,
            user_guidelines,
            examples,
        ) = (None, None, [], None, None, None, None)
        seen_params = set()
        for i, sec in enumerate(sections):
            if i == 0:
                description = self._extract_text(sec)
            elif sec.find(string=re.compile(r"^Syntax", flags=re.I)):
                syntax = self._extract_paragraphs(sec)
            elif sec.find(string=re.compile(r"^Parameters", flags=re.I)):
                params = self._extract_parameters(sec, seen_params)
            elif sec.find(string=re.compile(r"^Default Configuration", flags=re.I)):
                default_config = self._extract_text(sec)
            elif sec.find(string=re.compile(r"^Command Mode", flags=re.I)):
                command_mode = self._extract_text(sec)
            elif sec.find(string=re.compile(r"^User Guidelines", flags=re.I)):
                user_guidelines = self._extract_user_guidelines(sec)
            elif sec.find(string=re.compile(r"^Examples?", flags=re.I)):
                examples = self._extract_examples(sec)

        return {
            self.TOPIC: topic,
            self.COMMAND_NAME: command_name,
            self.DESCRIPTION: (
                self.sanitize_text(description) if description else None
            ),
            self.SYNTAX: list(map(self.sanitize_text, syntax)) if syntax else None,
            self.PARAMETERS: list(map(self.sanitize_text, params)),
            self.DEFAULT_CONFIGURATION: (
                self.sanitize_text(default_config) if default_config else None
            ),
            self.COMMAND_MODE: (
                self.sanitize_text(command_mode) if command_mode else None
            ),
            self.USER_GUIDELINES: (
                self.sanitize_text(user_guidelines) if user_guidelines else None
            ),
            self.EXAMPLES: examples,
        }

    def _extract_text_from_tag(self, tag: Tag) -> List[str]:
        if tag.name == "p":
            return [tag.get_text()]
        elif tag.name == "ul":
            return [li.get_text() for li in tag.find_all("li")]
        elif tag.name == "pre":
            return tag.get_text().split("\n")
        else:
            return [tag.get_text()]

    def _extract_paragraphs(self, section: Tag) -> List[str]:
        return [p.get_text() for p in section.find_all("p")]

    def _extract_parameters(self, section: Tag, seen_params: set) -> List[str]:
        params = []
        p = section.find("p")
        if p is not None:
            text = p.get_text().strip()
            if text not in seen_params:
                seen_params.add(text)
                params.append(text)

        ul = section.find("ul")
        if ul is not None:
            for li in ul.find_all("li"):
                text = li.get_text().strip()
                if text not in seen_params:
                    seen_params.add(text)
                    params.append(text)

        return params

    def _extract_text(self, section: Tag) -> str:
        p = section.find("p")
        return p.get_text() if p else section.get_text()

    def _extract_user_guidelines(self, section: Tag) -> str:
        list_p = section.find_all("p")
        if list_p is not None:
            return " ".join([p.get_text() for p in list_p])
        return section.get_text()

    def _extract_examples(self, section: Tag) -> List[Dict[str, Any]]:
        examples = []
        ex = {}
        description = section.find("p")
        if description:
            ex[self.DESCRIPTION] = self.sanitize_text(description.get_text())
        ul = section.find("ul")
        if ul:
            ex["commands"] = [li.get_text() for li in ul.find_all("li")]
        pre = section.find("pre")
        if pre:
            lines = pre.get_text().split("\n")
            if "commands" in ex:
                ex["commands"].extend(lines)
            else:
                ex["commands"] = lines
        if "commands" in ex:
            ex["commands"] = list(filter(lambda x: x != "", ex["commands"]))
        examples.append(ex)
        return examples


class CiscoSupportingDocumentsLoader(BaseLoader):
    """
    A class for loading Cisco supporting documents, primarily Admin Guide and CLI Guide.

    There are 2 class methods for creating an instance out of convience. \n
    The `from_url` method expects a URL to the main page of the document. \n
    The `from_file` method expects a path to a JSON file containing the documents. If the file does not exist, instance is re-created `from_url`.

    If you pass schema="cli" to the constructor, the loader will parse the CLI Guide documents into a suitable schema for Database storage. This uses the CLIParser class.
    """

    def __init__(
        self,
        paths: List[str],
        schema: Optional[Literal["cli"]] = None,
        doc_type: Optional[str] = None,
    ) -> None:
        self.paths = paths
        self.schema = schema
        self.doc_type = doc_type
        self.documents: List[Union[Document, Dict[str, Any]]] = []

    @classmethod
    def from_url(
        cls,
        url: str,
        schema: Optional[Literal["cli"]] = None,
        doc_type: Optional[str] = None,
    ) -> "CiscoSupportingDocumentsLoader":
        html = cls._make_request(url)
        soup = BeautifulSoup(html, "html.parser")
        paths = cls._extract_paths(soup)
        return cls(paths=paths, schema=schema, doc_type=doc_type)

    @classmethod
    def from_datasheet(cls, url: str) -> "CiscoSupportingDocumentsLoader":
        return cls(paths=[url], schema=None, doc_type="Datasheet")

    @property
    def default_parser(self) -> str:
        return "html.parser"

    @default_parser.setter
    def default_parser(self, parser: str) -> None:
        self._check_parser(parser)
        self.default_parser = parser

    @staticmethod
    def _check_parser(parser: str) -> None:
        """Check that parser is valid for bs4."""
        valid_parsers = ["html.parser", "lxml", "xml", "lxml-xml", "html5lib"]
        if parser not in valid_parsers:
            raise ValueError(
                "`parser` must be one of " + ", ".join(valid_parsers) + "."
            )

    @staticmethod
    def _make_request(url: str) -> str:
        res = requests.get(url)
        res.encoding = "utf-8"
        res.raise_for_status()
        return res.text

    @staticmethod
    def _extract_paths(soup: BeautifulSoup) -> str:
        toc = soup.select("ul#bookToc > li > a")
        links = [f"https://www.cisco.com{link.get('href')}" for link in toc]
        return links

    @staticmethod
    def sanitize_text(text: str) -> str:
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        cleaned_text = cleaned_text.replace("\\", "")
        cleaned_text = cleaned_text.replace("#", " ")
        cleaned_text = re.sub(r"([^\w\s])\1*", r"\1", cleaned_text)
        return cleaned_text

    @staticmethod
    def _build_metadata(soup: BeautifulSoup, url: str, **kwargs) -> Dict[str, str]:
        """Build metadata from BeautifulSoup output.
        Args:
            soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML.
            url (str): The URL of the source.
            **kwargs: Additional keyword arguments.
        Returns:
            Dict[str, str]: The metadata dictionary containing the extracted information.
        """
        dom = etree.HTML(str(soup), parser=None)
        metadata = {"source": url}
        if title := soup.find("meta", attrs={"name": "description"}):
            metadata["title"] = title.get("content", "Chapter not found.")
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")
        if concept := soup.find("meta", attrs={"name": "concept"}):
            metadata["concept"] = concept.get("content", "No concept found.")
        if topic := kwargs.get("topic"):
            metadata["topic"] = topic
        if published_date := dom.xpath('//*[@id="documentInfo"]/dl/dd'):
            metadata["published_date"] = published_date[0].text.strip()
        if document_id := soup.find("meta", attrs={"name": "documentId"}):
            metadata["document_id"] = document_id.get(
                "content", "No document ID found."
            )
        metadata["doc_id"] = str(uuid.uuid4())
        return metadata

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Union[Document, Dict[str, Any]]]:
        for path in self.paths:
            yield from self._fetch(path)

    def _fetch(self, path: str):
        print(f"Fetching document from {path}")
        html = self._make_request(path)
        soup = BeautifulSoup(html, "html.parser")

        if self.schema is not None and self.schema == "cli":
            parser = CLIParser()
            data = parser.load(soup)
            metadata = [self._build_metadata(soup, path) for _ in data]
            # merge metadata and data
            for d, m in zip(data, metadata):
                if self.doc_type:
                    m["doc_type"] = self.doc_type
                merged = {**d, **m}
                self.documents.append(merged)
                yield merged
        else:
            data = self._parse(soup)
            metadata = [
                self._build_metadata(soup, path, topic=d["topic"]) for d in data
            ]
            ids = [str(uuid.uuid4()) for _ in range(len(data))]
            for d, m, i in zip(data, metadata, ids):
                if re.match(
                    r"^This chapter contains the following sections:", d["text"]
                ):
                    continue
                if self.doc_type:
                    m["doc_type"] = self.doc_type
                document = Document(page_content=d["text"], metadata=m, id=i)
                self.documents.append(document)
                yield document

    def _parse(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """This method is useful for parsing the content of the Admin Guide or CLI Guide into a large chunk of text and topics suitable for LCEL Documents"""
        topic_sections = soup.find_all("section", class_=("body"))

        return [
            {
                "topic": f"{section.find_previous('h2', class_='title').get_text(strip=True) if section.find_previous('h2', class_='title') else ''} {section.find_previous_sibling('h3', class_='title').get_text(strip=True) if section.find_previous_sibling('h3', class_='title') else ''}".strip(),
                "text": self.sanitize_text(section.get_text()),
            }
            for section in topic_sections
        ]

    def save_to_json(self, path: Union[Path, str]) -> None:
        try:
            if isinstance(path, str):
                path = Path(path)

            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(
                    [
                        doc.model_dump() if isinstance(doc, Document) else doc
                        for doc in self.documents
                    ],
                    f,
                    indent=2,
                    ensure_ascii=True,
                )
        except Exception as e:
            logger.error(f"An error occurred while dumping the JSON file: {e}")

    @classmethod
    def from_file(
        cls, path: Union[Path, str], url: str, schema: Optional[Literal["cli"]] = None
    ) -> "CiscoSupportingDocumentsLoader":
        try:
            if isinstance(path, str):
                path = Path(path)
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            _self = cls(paths=[], schema=schema)
            _self.documents = [
                (
                    Document.model_validate(doc)
                    if "type" in doc and doc["type"] == "Document"
                    else doc
                )
                for doc in data
            ]
            return _self
        except (json.JSONDecodeError, FileNotFoundError):
            logger.info(
                f"\n"
                f"File not found at {path}."
                f"Recreating the data from URL {url}."
                f"Returning cls.from_url(url, schema=schema)"
            )
            return cls.from_url(url, schema=schema)
        except Exception as e:
            logger.info(
                f"An error occurred while loading the JSON file: {e}\n"
                f"Returning cls.from_url(url, schema=schema)"
            )
            return cls.from_url(url, schema=schema)


######################
# Loads the documents into ChromaDB
# Chunks must be > 8192 tokens
######################

FAMILIES = {
    "Cisco Business 220 Series Smart Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbss/CBS220/Adminstration-Guide/cbs-220-admin-guide.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbss/CBS220/CLI-Guide/b_220CLI.html",
    },
    "Cisco Business 250 Series Smart Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/Administration-Guide/cbs-250-ag.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-250-cli.html",
    },
    "Cisco Business 350 Series Managed Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/Administration-Guide/cbs-350.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-350-cli-.html",
    },
    "Cisco 350 Series Managed Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/350xseries/2_5_7/Administration/tesla-350-550.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-350-cli-.html",
    },
    "Cisco 350X Series Stackable Managed Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/350xseries/2_5_7/Administration/tesla-350-550.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-350-cli-.html",
    },
    "Cisco 550X Series Stackable Managed Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/350xseries/2_5_7/Administration/tesla-350-550.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/lan/csbms/CBS_250_350/CLI/cbs-350-cli-.html",
    },
    "Cisco Catalyst 1200 Series Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/campus-lan-switches-access/Catalyst-1200-and-1300-Switches/Admin-Guide/catalyst-1200-admin-guide.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/campus-lan-switches-access/Catalyst-1200-and-1300-Switches/cli/C1200-cli.html",
    },
    "Cisco Catalyst 1300 Series Switches": {
        "admin_guide": "https://www.cisco.com/c/en/us/td/docs/switches/campus-lan-switches-access/Catalyst-1200-and-1300-Switches/Admin-Guide/catalyst-1300-admin-guide.html",
        "cli_guide": "https://www.cisco.com/c/en/us/td/docs/switches/campus-lan-switches-access/Catalyst-1200-and-1300-Switches/cli/C1300-cli.html",
    },
}


def load_documents(loader: CiscoSupportingDocumentsLoader) -> List[Document]:
    if len(loader.documents) == 0:
        return loader.load()
    return loader.documents


def add_document_type(docs: List[Union[Document, Dict[str, Any]]], doc_type: str):
    documents = []
    for doc in docs:
        if isinstance(doc, Document):
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
        elif isinstance(doc, dict):
            # Handle dictionary format documents
            if "metadata" not in doc:
                doc["metadata"] = {}
            doc["metadata"]["doc_type"] = doc_type
            documents.append(doc)
        else:
            logger.warning(f"Unexpected document type: {type(doc)}")
            documents.append(doc)
    return documents


def process_and_insert_documents(
    url: str, collection_name: str, doc_type: str, series_name: str
):
    schema = None
    if doc_type in ["Admin_Guide"]:
        file_path = f"{ADMIN_GUIDE_DATA_DIR}/{collection_name}.json"
    elif doc_type in ["Cli_Guide"]:
        schema = "cli"
        file_path = f"{CLI_GUIDE_DATA_DIR}/{collection_name}.json"
    else:
        file_path = f"{DATA_DIR}/documents/{collection_name}.json"
    loader = CiscoSupportingDocumentsLoader.from_file(file_path, url)
    docs = load_documents(loader)
    docs = add_document_type(docs, doc_type)
    loader.save_to_json(file_path)
    insert_docs_to_chroma(url, docs, collection_name, series_name)


def insert_docs_to_chroma(
    url: str, documents: List[Document], collection_name: str, series_name: str
):
    print(f"Loaded {len(documents)} documents from {collection_name}.json")
    texts = [doc.page_content for doc in documents]  # Langchain Document Schema
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    metadatas = [doc.metadata for doc in documents]

    if len(texts) != len(ids):
        logger.warning(
            f"Length of ids and content do not match (ids={len(ids)}, content={len(texts)}). Generating new Ids..."
        )
        ids = [str(uuid.uuid4()) for _ in texts]

    collection_metadata = {"home_url": url, "name": series_name.title()}

    try:
        VECTOR_DB.client.delete_collection(collection_name)
    except ValueError as err:
        logger.info(
            f"Collection {collection_name} does not exist. Creating collection...."
        )
    # Explicitly setting embedding function, which is the default so its not required
    collection = VECTOR_DB.client.create_collection(
        name=collection_name,
        metadata=collection_metadata,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(),
    )

    for id, text, metadata in zip(ids, texts, metadatas):
        try:
            collection.add(ids=[id], metadatas=[metadata], documents=[text])
        except ValueError as err:
            logger.error(err)
            continue


def prepare_and_run():
    for series, sources in FAMILIES.items():
        print(f"Processing for {series} assets...")
        admin_guide_url = sources["admin_guide"]
        cli_guide_url = sources["cli_guide"]

        admin_guide_collection_name = CollectionFactory.retrieve_collection_name(
            "admin_guide", series
        )
        cli_guide_collection_name = CollectionFactory.retrieve_collection_name(
            "cli_guide", series
        )

        process_and_insert_documents(
            admin_guide_url, admin_guide_collection_name, "Admin_Guide", series
        )
        process_and_insert_documents(
            cli_guide_url, cli_guide_collection_name, "Cli_Guide", series
        )
    query = "Radius and DUO Auth"
    query_collection(query)


def run():
    prepare_and_run()


def reset_vector_db():
    return VECTOR_DB.reset()


def query_collection(query: str):
    try:
        for collection in VECTOR_DB.client.list_collections():
            print(collection)
            docs = VECTOR_DB.client.get_collection(collection.name).query(
                query_texts=[query],
                include=["metadatas", "documents", "distances"],
            )
            print(f"Query: {query}\n\nResults: {docs}")

    except Exception as e:
        logger.error(f"An error occurred while querying the collection: {e}")
        return None


if __name__ == "__main__":
    run()
