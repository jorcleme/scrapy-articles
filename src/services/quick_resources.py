"""Scrapes the quick resources from the website and saves them to a json file."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator, model_validator
import requests

from config import DATA_DIR

# Setup logger
logger = logging.getLogger(__name__)


class NestedResource(BaseModel):
    """Model for a nested resource item (device-specific link).

    Represents individual device links within a dropdown/grouped resource.
    Example: {"device": "Catalyst 1300", "href": "https://..."}
    """

    device: str = Field(..., description="Device model or name")
    href: str = Field(..., description="URL to the resource")

    @field_validator("href")
    @classmethod
    def validate_href(cls, v: str) -> str:
        """Validate that href is not empty and is a valid URL."""
        if not v or not isinstance(v, str):
            raise ValueError("href must be a non-empty string")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"href must be a valid URL: {v}")
        return v


class ResourceItem(BaseModel):
    """Model for a single resource item.

    Can be either:
    1. A regular link: {"id": "QuickStartGuide", "href": "https://..."}
    2. A grouped dropdown: {"id": "AdminGuides", "nested_resources": [...]}
    """

    id: str = Field(..., description="Resource identifier (camelCase)")
    href: Optional[str] = Field(
        None, description="Direct link URL (for regular resources)"
    )
    nested_resources: Optional[List[NestedResource]] = Field(
        None,
        description="List of nested device-specific resources (for dropdown groups)",
    )

    @model_validator(mode="after")
    def validate_resource_type(self) -> "ResourceItem":
        """Ensure exactly one of href or nested_resources is provided."""
        has_href = self.href is not None
        has_nested = (
            self.nested_resources is not None and len(self.nested_resources) > 0
        )

        if not has_href and not has_nested:
            raise ValueError(
                "ResourceItem must have either 'href' or 'nested_resources'"
            )

        if has_href and has_nested:
            raise ValueError(
                "ResourceItem cannot have both 'href' and 'nested_resources'"
            )

        return self

    @field_validator("href")
    @classmethod
    def validate_href(cls, v: Optional[str]) -> Optional[str]:
        """Validate href if provided."""
        if v is not None:
            if not v.startswith(("http://", "https://")):
                raise ValueError(f"href must be a valid URL: {v}")
        return v


class QuickResourceItem(BaseModel):
    """Model for a complete quick resource entry for a product series.

    Represents all scraped resources for a single Cisco product page.
    Example:
    {
        "series": "Cisco Catalyst 1300 Series",
        "page": "https://...",
        "description": "...",
        "resources": [...]
    }
    """

    series: str = Field(..., description="Product series name")
    page: str = Field(..., description="Product support page URL")
    description: Optional[str] = Field(None, description="Page meta description")
    resources: List[ResourceItem] = Field(
        default_factory=list, description="List of all available resources"
    )

    @field_validator("page")
    @classmethod
    def validate_page_url(cls, v: str) -> str:
        """Validate that page URL is valid."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"page must be a valid URL: {v}")
        return v

    @field_validator("resources")
    @classmethod
    def validate_resources_not_empty(cls, v: List[ResourceItem]) -> List[ResourceItem]:
        """Warn if resources list is empty."""
        if not v:
            logger.warning("QuickResourceItem has empty resources list")
        return v


class QuickResourcesScraper:
    """Scraper for Cisco Quick Resources pages.

    This class handles scraping quick resource links from Cisco support pages,
    supporting both legacy HTML structure (#flexContainer) and modern structure
    (.resource-button-wrapper).

    Attributes:
        urls: List of Cisco support page URLs to scrape
        timeout: Request timeout in seconds
        output_path: Path to save JSON output
        session: Requests session for HTTP operations
        results: List of scraped and validated QuickResourceItem objects
    """

    # Default URLs for Cisco support pages
    DEFAULT_URLS = [
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS220.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS250.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS350.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/Catalyst-1200.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/Catalyst-1300.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/wireless-mesh-100-200-series.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/wireless-mesh-100-AX-series.html?cachemode=refresh",
    ]

    # Default headers for HTTP requests
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Parser for BeautifulSoup
    DEFAULT_PARSER = "html.parser"

    def __init__(
        self,
        urls: Optional[List[str]] = None,
        *,
        timeout: int = 10,
        output_path: Optional[Path] = None,
    ):
        """Initialize the QuickResourcesScraper.

        Args:
            urls: List of URLs to scrape. Defaults to DEFAULT_URLS.
            timeout: Request timeout in seconds. Defaults to 10.
            output_path: Path to save JSON output. Defaults to DATA_DIR/quick_resources/quick_resources.json.
        """
        self.urls = urls or self.DEFAULT_URLS
        self.timeout = timeout
        self.output_path = output_path or (
            DATA_DIR / "quick_resources" / "quick_resources.json"
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup HTTP session
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

        # Results storage
        self._results: List[QuickResourceItem] = []

    @property
    def results(self) -> List[QuickResourceItem]:
        """Get the scraped results.

        Returns:
            List of QuickResourceItem objects
        """
        return self._results

    @results.setter
    def results(self, value: List[QuickResourceItem]) -> None:
        """Set the results.

        Args:
            value: List of QuickResourceItem objects
        """
        self._results = value

    def scrape_all(self) -> List[QuickResourceItem]:
        """Scrape all configured URLs and return validated results.

        Returns:
            List[QuickResourceItem]: List of validated quick resource items
        """
        self.logger.info(f"Starting to scrape {len(self.urls)} pages")

        for url in self.urls:
            try:
                self.logger.info(f"Fetching page: {url}")
                quick_resource = self._scrape_single_page(url)

                if quick_resource:
                    self._results.append(quick_resource)

            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch {url}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error processing {url}: {e}")
                continue

        self.logger.info(f"Successfully scraped {len(self._results)} product series")
        return self._results

    def save_to_file(self, path: Optional[Path] = None) -> None:
        """Save scraped results to JSON file.

        Args:
            path: Optional custom path. Uses self.output_path if not provided.
        """
        output_path = path or self.output_path

        try:
            # Convert Pydantic models to dicts, excluding None values
            results_dict = [
                item.model_dump(exclude_none=True) for item in self._results
            ]
            dumped = json.dumps(results_dict, indent=4, skipkeys=True)

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with output_path.open("w", encoding="utf8") as file:
                file.write(dumped)

            self.logger.info(f"Successfully saved resources to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save resources: {e}")
            raise

    def scrape_and_save(self) -> List[QuickResourceItem]:
        """Convenience method to scrape all pages and save results.

        Returns:
            List[QuickResourceItem]: List of validated quick resource items
        """
        results = self.scrape_all()
        self.save_to_file()
        return results

    def _scrape_single_page(self, url: str) -> Optional[QuickResourceItem]:
        """Scrape a single support page.

        Args:
            url: URL of the page to scrape

        Returns:
            QuickResourceItem if successful, None otherwise
        """
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, self.DEFAULT_PARSER)

        # Extract metadata
        series = self._extract_meta(soup, "og:title")
        page_url = self._extract_meta(soup, "og:url")
        description = self._extract_meta(soup, "og:description")

        if not all([series, page_url]):
            self.logger.warning(f"Missing required metadata for {url}")
            return None

        self.logger.info(f"Processing series: {series}")

        # Determine parsing strategy based on series
        if not series:
            self.logger.warning(
                "Series name is empty, cannot determine parsing strategy"
            )
            return None

        if series == "Cisco Catalyst 1300 Series":
            resources = self._parse_modern_structure(soup, series)
        else:
            resources = self._parse_legacy_structure(soup, series)

        # Create and validate QuickResourceItem
        if resources:
            try:
                quick_resource = QuickResourceItem(
                    series=series,
                    page=page_url,
                    description=description,
                    resources=resources,
                )
                return quick_resource
            except Exception as e:
                self.logger.error(
                    f"Failed to create QuickResourceItem for {series}: {e}"
                )
                return None
        else:
            self.logger.warning(f"No resources found for {series}")
            return None

    @staticmethod
    def _extract_meta(soup: BeautifulSoup, property: str) -> Optional[str]:
        """Extract meta tag content.

        Args:
            soup: BeautifulSoup object
            property: Meta property name (e.g., 'og:title')

        Returns:
            Meta content or None
        """
        meta_tag = soup.find("meta", property=property)
        return meta_tag.get("content") if meta_tag else None

    @staticmethod
    def _normalize_key(text: str) -> str:
        """Normalize text to camelCase ID.

        Args:
            text: Text to normalize (e.g., "Quick Start Guide")

        Returns:
            Normalized camelCase string (e.g., "QuickStartGuide")
        """
        words = text.split()
        return "".join(words) if len(words) > 1 else text

    def _parse_modern_structure(
        self, soup: BeautifulSoup, series: str
    ) -> List[ResourceItem]:
        """Parse modern HTML structure (.resource-button-wrapper).

        Used for Cisco Catalyst 1300 Series and future products.

        Args:
            soup: BeautifulSoup object
            series: Product series name

        Returns:
            List of ResourceItem objects
        """
        self.logger.info(f"Processing modern structure for: {series}")
        resources: List[ResourceItem] = []

        # Find all resource button wrappers
        resource_wrappers = soup.select(".resource-button-wrapper")

        if not resource_wrappers:
            self.logger.warning(f"No resource wrappers found for {series}")
            return resources

        self.logger.debug(f"Found {len(resource_wrappers)} resource wrappers")

        for wrapper in resource_wrappers:
            # Check if this is a dropdown (details element) or a regular link
            details_element = wrapper.find("details")

            if details_element:
                # This is a dropdown with nested resources
                resource_item = self._parse_dropdown_element(details_element)
                if resource_item:
                    resources.append(resource_item)
            else:
                # This is a regular link (not a dropdown)
                resource_item = self._parse_regular_link(wrapper)
                if resource_item:
                    resources.append(resource_item)

        self.logger.info(f"Parsed {len(resources)} resources from modern structure")
        return resources

    def _parse_legacy_structure(
        self, soup: BeautifulSoup, series: str
    ) -> List[ResourceItem]:
        """Parse legacy HTML structure (#flexContainer).

        Used for CBS220, CBS250, CBS350, Catalyst 1200, and Wireless series.

        Args:
            soup: BeautifulSoup object
            series: Product series name

        Returns:
            List of ResourceItem objects
        """
        self.logger.debug(f"Processing legacy structure for: {series}")
        resources: List[ResourceItem] = []

        # Parse regular links
        targets = soup.select("#flexContainer > a")
        self.logger.debug(f"Found {len(targets)} regular link targets")

        for tag in targets:
            try:
                key = tag.find_next(class_="copy").get_text(strip=True)
                self.logger.debug(f"Processing target: {key}")

                key = self._normalize_key(key)
                href = tag.get("href")

                if href:
                    resource_item = ResourceItem(id=key, href=href)
                    resources.append(resource_item)
            except Exception as e:
                self.logger.warning(f"Failed to parse regular link: {e}")
                continue

        # Parse dropdown targets
        dropdown_targets = soup.select(
            "#flexContainer > div.flexItem > details.QSG > div#AG"
        )
        self.logger.debug(f"Found {len(dropdown_targets)} dropdown targets")

        for target in dropdown_targets:
            try:
                resource_item = self._parse_legacy_dropdown(target)
                if resource_item:
                    resources.append(resource_item)
            except Exception as e:
                self.logger.error(f"Failed to parse dropdown: {e}")
                continue

        self.logger.info(f"Parsed {len(resources)} resources from legacy structure")
        return resources

    def _parse_dropdown_element(self, details_element) -> Optional[ResourceItem]:
        """Parse a dropdown details element (modern structure).

        Args:
            details_element: BeautifulSoup details tag

        Returns:
            ResourceItem with nested_resources, or None if invalid
        """
        summary = details_element.find("summary")
        if not summary:
            self.logger.debug("Skipping details element without summary")
            return None

        # Get the category name (e.g., "Hardware Installation Guides", "Admin Guides")
        category_name = summary.get_text(strip=True)
        self.logger.debug(f"Processing dropdown category: {category_name}")

        # Convert to camelCase ID
        key = self._normalize_key(category_name)

        # Find all dropdown items (links inside the dropdown menu)
        dropdown_menu = details_element.find("div", class_="dropdown-menu")
        if not dropdown_menu:
            self.logger.debug(f"No dropdown menu found for category: {category_name}")
            return None

        dropdown_items = dropdown_menu.find_all("a", class_="dropdown-item")
        nested_resources: List[NestedResource] = []

        for item in dropdown_items:
            device = item.get_text(strip=True)
            href = item.get("href")
            if device and href:
                try:
                    nested_resources.append(NestedResource(device=device, href=href))
                    self.logger.debug(f"  Added nested resource: {device} -> {href}")
                except Exception as e:
                    self.logger.warning(f"Invalid nested resource {device}: {e}")
                    continue

        if nested_resources:
            try:
                resource_item = ResourceItem(id=key, nested_resources=nested_resources)
                self.logger.info(
                    f"Added dropdown '{key}' with {len(nested_resources)} nested resources"
                )
                return resource_item
            except Exception as e:
                self.logger.error(f"Failed to create ResourceItem for {key}: {e}")
                return None

        return None

    def _parse_regular_link(self, wrapper) -> Optional[ResourceItem]:
        """Parse a regular link wrapper (modern structure).

        Args:
            wrapper: BeautifulSoup wrapper element

        Returns:
            ResourceItem with href, or None if invalid
        """
        link = wrapper.find("a", class_="resource-button")
        if not link:
            self.logger.debug("Skipping wrapper without resource-button link")
            return None

        # Get the link text as the ID
        key = link.get_text(strip=True)
        href = link.get("href")

        self.logger.debug(f"Processing regular link: {key}")

        # Convert to camelCase ID
        key = self._normalize_key(key)

        if href:
            try:
                resource_item = ResourceItem(id=key, href=href)
                self.logger.info(f"Added regular resource: {key} -> {href}")
                return resource_item
            except Exception as e:
                self.logger.error(f"Failed to create ResourceItem for {key}: {e}")
                return None

        return None

    def _parse_legacy_dropdown(self, target) -> Optional[ResourceItem]:
        """Parse a legacy dropdown structure.

        Args:
            target: BeautifulSoup div#AG element

        Returns:
            ResourceItem with nested_resources, or None if invalid
        """
        key = target.find_previous("summary").get_text(strip=True)
        self.logger.debug(f"Processing dropdown: {key}")

        key = self._normalize_key(key)

        nested_resources: List[NestedResource] = []
        for link in target.contents[1::2]:
            try:
                self.logger.debug(f"Processing link: {link}")
                device = link.get_text(strip=True)
                href = link["href"]
                nested_resources.append(NestedResource(device=device, href=href))
            except Exception as e:
                self.logger.warning(f"Invalid nested resource: {e}")
                continue

        if nested_resources:
            try:
                resource_item = ResourceItem(id=key, nested_resources=nested_resources)
                return resource_item
            except Exception as e:
                self.logger.error(f"Failed to create ResourceItem for {key}: {e}")
                return None

        return None


# Backward compatibility: Keep old function interface
def quick_resources() -> List[QuickResourceItem]:
    """Parse Quick Resources from Cisco Support Pages (legacy wrapper).

    This function maintains backward compatibility with existing code.
    For new code, use QuickResourcesScraper class directly.

    Returns:
        List[QuickResourceItem]: List of validated quick resource items
    """
    scraper = QuickResourcesScraper()
    return scraper.scrape_and_save()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the scraper using the class
    scraper = QuickResourcesScraper()
    results = scraper.scrape_and_save()
    logger.info(f"Successfully scraped {len(results)} product series")
