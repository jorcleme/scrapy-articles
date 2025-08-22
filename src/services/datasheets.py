"""
Scrapes Cisco's Support Page Datasheets for product information.

This module provides functionality to scrape product datasheets from Cisco's
support pages and extract structured data about various networking devices
including switches, access points, and routers.

Example:
    Basic usage:
        from datasheets import main, DEFAULT_URLS
        main(DEFAULT_URLS)

    Custom URL list:
        custom_urls = [
            {
                "concept": "My Product",
                "url": "https://example.com/datasheet"
            }
        ]
        main(custom_urls)
"""

import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup, Tag
from config import LOGS_DIR, DATASHEETS_DATA_DIR

# Centralized constants with fallback import for direct-script execution
try:
    from src.constants import (
        CISCO_CATALYST_1200_SERIES,
        CISCO_CATALYST_1300_SERIES,
        CBS_110_BUSINESS_SERIES_UNMANAGED,
        CBS_220_BUSINESS_SERIES,
        CBS_250_BUSINESS_SERIES,
        CBS_350_BUSINESS_SERIES,
        CBW_100_SERIES_140AC,
        CBW_100_SERIES_145AC,
        CISCO_110_SERIES_UNMANAGED,
        CISCO_350_SERIES_MANAGED_SWITCHES,
        CISCO_350X_STACKABLE_SERIES,
        CISCO_550X_STACKABLE_SERIES,
        CATALYST_1000_SERIES,
        COMBINED_SERIES,
    )
except ImportError:  # pragma: no cover
    import sys as _sys

    _sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.constants import (
        CISCO_CATALYST_1200_SERIES,
        CISCO_CATALYST_1300_SERIES,
        CBS_110_BUSINESS_SERIES_UNMANAGED,
        CBS_220_BUSINESS_SERIES,
        CBS_250_BUSINESS_SERIES,
        CBS_350_BUSINESS_SERIES,
        CBW_100_SERIES_140AC,
        CBW_100_SERIES_145AC,
        CISCO_110_SERIES_UNMANAGED,
        CISCO_350_SERIES_MANAGED_SWITCHES,
        CISCO_350X_STACKABLE_SERIES,
        CISCO_550X_STACKABLE_SERIES,
        CATALYST_1000_SERIES,
        COMBINED_SERIES,
    )

# Constants for improved maintainability
DEFAULT_REQUEST_TIMEOUT = 30
MAX_LOG_FILE_SIZE = 1_000_000
LOG_BACKUP_COUNT = 3
USER_AGENT = "Mozilla/5.0 (compatible; ArticleCreatorBot/1.0)"

logger = logging.getLogger(__file__.split("/")[-1])


def setup_logging(
    level: Union[str, int, None] = None, log_file: Optional[str] = None
) -> None:
    """Configure logging once for this module.

    Args:
        level: Logging level (string or integer). Can be overridden by LOG_LEVEL env var.
        log_file: Path to log file. Can be overridden by LOG_FILE env var.

    Environment overrides:
        LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL (default INFO)
        LOG_FILE: path to log file (default data/datasheet_scraper.log)
    """
    if logger.handlers:
        return

    # Determine logging level
    if isinstance(level, int):
        level_value = level
    else:
        level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
        level_value = getattr(logging, level_str, logging.INFO)

    # Determine log file path
    default_log_file = str(LOGS_DIR / "datasheet_scraper.log")
    log_path = log_file or os.getenv("LOG_FILE", default_log_file)

    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Fallback to current directory if unable to create log directory
        log_path = "datasheet_scraper.log"
        print(f"Warning: Could not create log directory, using current directory: {e}")

    logger.setLevel(level_value)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level_value)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler with rotation
    try:
        file_handler = RotatingFileHandler(
            log_path, maxBytes=MAX_LOG_FILE_SIZE, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(level_value)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning("Could not create file handler: %s", e)

    # constants moved to src.constants


def normalize_model_code(text: str) -> str:
    """Normalize Cisco model codes by removing stray spaces around hyphens
    and collapsing internal whitespace.

    Args:
        text: Raw text containing model code

    Returns:
        Normalized model code string

    Example:
        'C1300-8FP- 2G' -> 'C1300-8FP-2G'

    Note:
        Only use this for model keys, not for general descriptive values.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Normalize whitespace and remove spaces around hyphens
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"\s*-\s*", "-", normalized)

    return normalized


def main(urls: List[Dict[str, str]]) -> None:
    """Main function to scrape datasheets from provided URLs.

    Args:
        urls: List of dictionaries containing 'concept' and 'url' keys
    """
    setup_logging()
    json_list: List[Dict[str, Any]] = []

    # Use a single run timestamp for the output filename
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for url_config in urls:
        # Initialize empty dict for each iteration
        smb_builder: Dict[str, Any] = {}

        # Validate input
        if not isinstance(url_config, dict) or "url" not in url_config:
            logger.warning("Invalid URL config: %s", url_config)
            continue

        url = url_config["url"]
        concept = url_config.get("concept", "Unknown")

        # Make request with error handling
        try:
            request = make_request(url=url)
            request.raise_for_status()
        except Exception as e:
            logger.error("Request failed for %s: %s", url, e)
            continue

        # Parse content with BeautifulSoup
        try:
            soup = BeautifulSoup(request.content, "html.parser")
        except Exception as e:
            logger.error("Failed to parse HTML for %s: %s", url, e)
            continue

        logger.info("Processing concept: %s", concept)

        # Process based on concept type
        series_datasheet = _process_datasheet_by_concept(soup, smb_builder, concept)

        # Create output structure
        output = {
            "family": concept,
            "features": series_datasheet,
            "datasheet_url": url,
            "last_scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        json_list.append(output)

    # Write results to timestamped file
    _write_results_to_file(json_list, run_ts)


def _process_datasheet_by_concept(
    soup: BeautifulSoup, smb_builder: Dict[str, Any], concept: str
) -> Dict[str, Any]:
    """Process datasheet based on the concept type.

    Args:
        soup: BeautifulSoup object of the parsed HTML
        smb_builder: Dictionary to store processed data
        concept: The product concept/family name

    Returns:
        Processed datasheet data
    """
    if re.search(r"Cisco Catalyst 1000 Series Switches", concept):
        sloppy_series_datasheet = process_catalyst_1000_series(soup, smb_builder)
        return transform_catalyst_1000_data(sloppy_series_datasheet)
    elif re.search(r"Cisco 110 Series Unmanaged Switches", concept):
        return process_110_series_unmanaged_switch_data(soup, smb_builder)
    elif re.search(r"Cisco 350 Series Managed Switches", concept):
        return process_300_series_managed_switch_data(soup, smb_builder)
    elif re.search(r"Cisco Business Wireless AC", concept):
        return _process_business_wireless_ac(soup, smb_builder)
    elif re.search(r"Cisco Business Wireless AX", concept):
        return _process_business_wireless_ax(soup, smb_builder)
    else:
        return parse_table(soup=soup, obj=smb_builder)


def _process_business_wireless_ac(
    soup: BeautifulSoup, smb_builder: Dict[str, Any]
) -> Dict[str, Any]:
    """Process Cisco Business Wireless AC datasheets."""
    series_datasheet: Dict[str, Any] = {}
    meta_title = soup.find("meta", attrs={"name": "title"})

    if not meta_title:
        logger.warning("No meta title found for Business Wireless AC datasheet")
        return parse_table(soup=soup, obj=smb_builder)

    sub_concept_meta = meta_title.get("content", "").strip()

    if sub_concept_meta == "Cisco Business 140AC Access Point Data Sheet":
        sub_concept = "CBW140AC"
    elif sub_concept_meta == "Cisco Business 145AC Access Point Data Sheet":
        sub_concept = "CBW145AC"
    elif sub_concept_meta == "Cisco Business 240AC Access Point Data Sheet":
        sub_concept = "CBW240AC"
    else:
        logger.warning("Unknown Business Wireless AC meta title: %s", sub_concept_meta)
        return parse_table(soup=soup, obj=smb_builder)

    series_datasheet[sub_concept] = parse_table(soup=soup, obj=smb_builder)
    return series_datasheet


def _process_business_wireless_ax(
    soup: BeautifulSoup, smb_builder: Dict[str, Any]
) -> Dict[str, Any]:
    """Process Cisco Business Wireless AX datasheets."""
    series_datasheet: Dict[str, Any] = {}
    meta_title = soup.find("meta", attrs={"name": "title"})

    if not meta_title:
        logger.warning("No meta title found for Business Wireless AX datasheet")
        return parse_table(soup=soup, obj=smb_builder)

    sub_concept_meta = meta_title.get("content", "").strip()

    if sub_concept_meta == "Cisco Business 150AX Access Point Data Sheet":
        sub_concept = "CBW150AXM"
    elif sub_concept_meta == "Cisco Business Wireless 151AXM Mesh Extender Datasheet":
        sub_concept = "CBW151AXM"
    else:
        logger.warning("Unknown Business Wireless AX meta title: %s", sub_concept_meta)
        return parse_table(soup=soup, obj=smb_builder)

    series_datasheet[sub_concept] = parse_table(soup=soup, obj=smb_builder)
    return series_datasheet


def _write_results_to_file(json_list: List[Dict[str, Any]], run_ts: str) -> None:
    """Write scraped results to a timestamped JSON file.

    Args:
        json_list: List of scraped datasheet data
        run_ts: Timestamp string for filename
    """
    try:
        DATASHEETS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATASHEETS_DATA_DIR / f"cisco_datasheets_{run_ts}.json"

        with open(out_path, "w", encoding="utf8") as file:
            json.dump(json_list, file, indent=4, ensure_ascii=True)

        logger.info("Wrote scraped datasheets to %s", out_path)
    except Exception as e:
        logger.error("Failed to write results to file: %s", e)
        raise


def process_110_series_unmanaged_switch_data(
    soup: BeautifulSoup, smb_builder: Dict[str, Any]
) -> Dict[str, Any]:
    """Process Cisco 110 Series Unmanaged Switch datasheet data.

    Args:
        soup: BeautifulSoup object of the parsed HTML
        smb_builder: Dictionary to store processed data

    Returns:
        Processed datasheet data
    """
    rows = soup.select("tbody > tr")
    desired_headers = [
        "physical_dimensions",
        "weight",
        "ports",
        "switching_capacity",
        "forwarding_capacity",
    ]

    for row in rows:
        cells = [
            cell.get_text(strip=True, separator=" ")
            for cell in row.find_all("td")
            if re.search(r".+", string=cell.get_text(strip=True, separator=" "))
        ]
        logger.debug("Row cells (110 unmanaged): %s", cells)

        determine_key = row.select("td")[0].text.strip()
        logger.debug("determine_key: %s", determine_key)

        key = (
            determine_key
            if determine_key in CISCO_110_SERIES_UNMANAGED
            else create_joined_header(determine_key.lower()).strip()
        )

        try:
            values = [
                strng.get_text(strip=True, separator=" ")
                for strng in row.select("td")[1].contents[1::2]
            ]

            if key in desired_headers:
                for value in values:
                    if ":" not in value:
                        continue
                    model, data_entry = value.split(":", 1)
                    model_norm = normalize_model_code(model.strip())
                    if model_norm in CISCO_110_SERIES_UNMANAGED:
                        if model_norm not in smb_builder:
                            smb_builder[model_norm] = {}
                        smb_builder[model_norm][key] = data_entry.strip()

            elif key == "power_over_ethernet":
                _process_poe_data(row, smb_builder)

            elif key and len(values) == 1:
                # Prefer anchor href in the value cell when present
                value_td = row.select("td")
                href = None
                if len(value_td) > 1:
                    a = value_td[1].find("a", href=True)
                    if a and a.get("href"):
                        href = a.get("href").strip()
                smb_builder[key] = href if href else values[0]
            else:
                smb_builder[key] = values

        except (IndexError, ValueError) as e:
            logger.warning("110 unmanaged parsing issue for key %s: %s", key, e)
            continue

    return smb_builder


def _process_poe_data(row: Tag, smb_builder: Dict[str, Any]) -> None:
    """Process Power over Ethernet data from table row.

    Args:
        row: BeautifulSoup Tag representing table row
        smb_builder: Dictionary to store processed data
    """
    try:
        model_names = [
            strng.get_text(strip=True, separator=" ")
            for strng in row.select("td")[1].find_all("p")
            if not re.search(r"Model Name", strng.get_text(strip=True))
        ]
        power_dedicated_to_poe = [
            strng.get_text(strip=True, separator=" ")
            for strng in row.select("td")[2].find_all("p")
            if not re.search(r"Power Dedicated to PoE", strng.get_text(strip=True))
        ]
        number_of_ports_that_support_poe = [
            strng.get_text(strip=True, separator=" ")
            for strng in row.select("td")[3].find_all("p")
            if not re.search(r"Number of PoE Ports", strng.get_text(strip=True))
        ]

        for i, model in enumerate(model_names):
            model_norm = normalize_model_code(model)
            if model_norm not in smb_builder:
                smb_builder[model_norm] = {}
            if i < len(power_dedicated_to_poe):
                smb_builder[model_norm]["power_dedicated_to_poe"] = (
                    power_dedicated_to_poe[i]
                )
            if i < len(number_of_ports_that_support_poe):
                smb_builder[model_norm]["number_of_ports_that_support_poe"] = (
                    number_of_ports_that_support_poe[i]
                )

        logger.debug("model_names: %s", model_names)
    except (IndexError, AttributeError) as e:
        logger.warning("Failed to process PoE data: %s", e)


def process_300_series_managed_switch_data(
    soup: BeautifulSoup, smb_builder: Dict[str, Any]
) -> Dict[str, Any]:
    """Process Cisco 300 Series Managed Switch datasheet data.

    Args:
        soup: BeautifulSoup object of the parsed HTML
        smb_builder: Dictionary to store processed data

    Returns:
        Processed datasheet data
    """
    headers_needing_iteration = ["Model", "Model Name"]
    rows = soup.select("tbody > tr")
    skip_count = 0

    for row in rows:
        if skip_count > 0:
            skip_count -= 1
            continue

        cells = [
            cell.get_text(strip=True, separator=" ")
            for cell in row.find_all("td")
            if re.search(r".+", string=cell.get_text(strip=True, separator=" "))
        ]
        logger.debug("Row cells (300 series managed): %s", cells)

        if len(cells) == 1:
            continue

        if len(cells) == 2:
            key = create_joined_header(cells[0])
            value = cells[1]
            if not key.startswith("-") and value:
                smb_builder[key] = value

        if any(header in cells for header in headers_needing_iteration):
            if "Power Dedicated to PoE" in cells:
                rowspan = 14
                smb_builder = parse_row_data(
                    rowspan=rowspan,
                    cells=cells,
                    row=row,
                    obj=smb_builder,
                    length=len(cells),
                )
                skip_count = rowspan
            else:
                smb_builder = parse_row_data(
                    rowspan=len(CISCO_350_SERIES_MANAGED_SWITCHES),
                    cells=cells,
                    row=row,
                    obj=smb_builder,
                    length=len(cells),
                )
                skip_count = len(CISCO_350_SERIES_MANAGED_SWITCHES)

    return smb_builder


def transform_catalyst_1000_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Catalyst 1000 data based on specific business rules.

    Transformation rules:
    1. For the "fan" property, converts 'Y' to 'No' and otherwise to 'Yes'
    2. Appends "kg" to "unit_weight"
    3. Updates the "rj-45_ports" to "x Gigabit Ethernet" format
    4. Transforms the "uplink_ports" based on provided logic
    5. Adds calculated forwarding_rate, switching_capacity, and mtbf values

    Args:
        data: Raw scraped data dictionary

    Returns:
        Transformed data dictionary
    """
    devices = {k: v for k, v in data.items() if k in CATALYST_1000_SERIES}

    for device, attributes in devices.items():
        # Transform fan property
        if "fan" in attributes:
            attributes["fan"] = "No" if attributes["fan"] == "Y" else "Yes"

        # Add kg unit to weight
        if "unit_weight" in attributes:
            attributes["unit_weight"] += " kg"

        # Transform RJ-45 ports and calculate performance metrics
        if "rj-45_ports" in attributes:
            match = re.match(r"(\d+)", attributes["rj-45_ports"])
            if match:
                num_ports = match.group(1)
                attributes["rj-45_ports"] = f"{num_ports} x Gigabit Ethernet"

                # Add performance metrics based on port count
                _add_performance_metrics(attributes, num_ports, device)

        # Transform uplink ports
        if "uplink_ports" in attributes:
            _transform_uplink_ports(attributes)

        data[device] = attributes

    return data


def _add_performance_metrics(
    attributes: Dict[str, Any], num_ports: str, device: str
) -> None:
    """Add performance metrics based on port count and device type.

    Args:
        attributes: Device attributes dictionary
        num_ports: Number of ports as string
        device: Device name for specific logic
    """
    port_count = int(num_ports)

    if port_count == 8:
        attributes["forwarding_rate"] = 14.88
        attributes["switching_capacity"] = 20.0
        attributes["mtbf"] = 2171669
    elif port_count == 16:
        attributes["mtbf"] = 2165105
        attributes["forwarding_rate"] = 26.78
        attributes["switching_capacity"] = 36.0
    elif port_count == 24:
        attributes["mtbf"] = 2026793
        if "FE" in device:
            attributes["forwarding_rate"] = 9.52
            attributes["switching_capacity"] = 12.8
        elif "uplink_ports" in attributes and re.search(
            r"\bSFP\b(?!\+)", attributes["uplink_ports"]
        ):
            attributes["forwarding_rate"] = 41.67
            attributes["switching_capacity"] = 56.0
        else:
            attributes["forwarding_rate"] = 95.23
            attributes["switching_capacity"] = 128.0
    elif port_count == 48:
        attributes["mtbf"] = 1452667
        if "FE" in device:
            attributes["forwarding_rate"] = 13.09
            attributes["switching_capacity"] = 17.6
        elif "uplink_ports" in attributes and re.search(
            r"\bSFP\b(?!\+)", attributes["uplink_ports"]
        ):
            attributes["forwarding_rate"] = 77.38
            attributes["switching_capacity"] = 104.0
        else:
            attributes["forwarding_rate"] = 130.94
            attributes["switching_capacity"] = 176.0


def _transform_uplink_ports(attributes: Dict[str, Any]) -> None:
    """Transform uplink ports description based on SFP type.

    Args:
        attributes: Device attributes dictionary
    """
    uplink_ports = attributes["uplink_ports"]
    match = re.match(r"(\d+)", uplink_ports)

    if not match:
        return

    num_ports = match.group(1)
    is_combo = "combo" in uplink_ports.lower()
    combo_suffix = " combo" if is_combo else ""

    if "SFP+" in uplink_ports:
        attributes["uplink_ports"] = f"{num_ports} 10G SFP+{combo_suffix}"
    elif "SFP" in uplink_ports:
        attributes["uplink_ports"] = f"{num_ports} Gigabit Ethernet SFP{combo_suffix}"


def process_catalyst_1000_series(
    soup: BeautifulSoup, smb_builder: Dict[str, Any]
) -> Dict[str, Any]:
    """Process Cisco Catalyst 1000 Series Switch datasheet data.

    Args:
        soup: BeautifulSoup object of the parsed HTML
        smb_builder: Dictionary to store processed data

    Returns:
        Processed datasheet data
    """

    def iterate_catalyst_table_section(
        index: int, row: Tag, array: List[List[str]]
    ) -> None:
        """Iterate through a table section and append to an array for Catalyst 1000.

        Args:
            index: Number of rows to iterate
            row: Current BeautifulSoup row Tag
            array: List to append row data to
        """
        current_row = row
        for _ in range(index):
            current_row = current_row.find_next("tr")
            if current_row is None:
                break
            new_cells = [
                cell.get_text(strip=True, separator=" ")
                for cell in current_row.contents[1::2]
                if cell.name == "td"
            ]
            if new_cells:  # Only append non-empty cell data
                array.append(new_cells)

    # Define headers for different sections
    HEADERS1 = [
        "rj-45_ports",
        "uplink_ports",
        "power_dedicated_to_poe",
        "fan",
        "unit_dimensions",
        "unit_weight",
    ]
    HEADERS2 = [
        "8-port models",
        "16-port models",
        "24-port models (1/10G uplinks)",
        "48-port models (1/10G uplinks)",
    ]

    # Initialize data collection lists
    model_info: List[List[str]] = []
    port_model_info: List[List[str]] = []
    generic_series_info: List[List[str]] = []
    management_info: List[List[str]] = []

    undesired_headers = ["Product number", "Note:", "Note", "*Note:"]

    # Process all table rows
    for row in soup.find_all("tr"):
        cells = [
            cell.get_text(strip=True, separator=" ")
            for cell in row.contents[1::2]
            if cell.name == "td"
        ]
        cells = [cell for cell in cells if re.search(r".+", cell)]
        logger.debug("Row cells (cat1k): %s", cells)

        if not cells:
            continue

        # Collect model information
        if cells[0] in CATALYST_1000_SERIES:
            model_info.append(cells)

        # Single cell rows - skip
        if len(cells) == 1:
            continue

        # Port model information section
        if len(cells) == 5 and "8-port models" in cells:
            iterate_catalyst_table_section(30, row=row, array=port_model_info)

        # Generic series information
        if (
            len(cells) == 2
            and not any(word in cells for word in CATALYST_1000_SERIES)
            and not any(word in cells for word in undesired_headers)
        ):
            generic_series_info.append(cells)

        # Management information
        desired_generic_headers = ["Management", "Standards", "RFC compliance"]
        if (len(cells) == 4 or len(cells) == 3) and any(
            word in cells for word in desired_generic_headers
        ):
            cells[0] = create_joined_header(cells[0])
            management_info.append(cells)

    # Process collected data
    header1_to_model = model_info[: len(CATALYST_1000_SERIES)]

    for array in header1_to_model:
        key = array.pop(0)
        logger.debug("cat1k table model key: %s", key)
        smb_builder[key] = dict(zip(HEADERS1, array))

    for array in port_model_info:
        if array:  # Check if array is not empty
            key = array.pop(0)
            smb_builder[key] = dict(zip(HEADERS2, array))

    for array in generic_series_info:
        if len(array) >= 2:  # Ensure we have at least key and value
            key = create_joined_header(array.pop(0))
            smb_builder[key] = array[0]

    for array in management_info:
        if array:  # Check if array is not empty
            key = array.pop(0)
            smb_builder[key] = array

    return smb_builder


def handle_table_data(
    headers_map: List[str], table_data: List[str], obj: Dict[str, Any]
) -> Dict[str, Any]:
    """Process table data based on header map and update the given object.

    Args:
        headers_map: List of headers for mapping data
        table_data: List of table data values
        obj: Object to update with the processed data

    Returns:
        Updated object dictionary

    Raises:
        ValueError: If table_data is empty or model extraction fails
    """
    if not table_data:
        raise ValueError("table_data cannot be empty")

    model = table_data.pop(0)
    model = normalize_model_code(model)

    if model not in obj:
        obj[model] = {}

    # Define conversion functions for specific data types
    convert_keys = {
        "switching_capacity": lambda x: (
            "switching_capacity",
            float(x.replace(",", "")),
        ),
        "forwarding_rate": lambda x: ("forwarding_rate", float(x.replace(",", ""))),
        "combo_ports": lambda x: ("uplink_ports", x),
        "dimensions": lambda x: ("unit_dimensions", x),
        "poe_power_budget": lambda x: ("power_dedicated_to_poe", x),
        "number_of_ports_that_support_poe": lambda x: (
            "number_of_ports_that_support_poe",
            int(re.search(r"^\d+", x).group(0)) if re.search(r"^\d+", x) else 0,
        ),
        "heat_dissipation": lambda x: ("heat_dissipation", float(x.replace(",", ""))),
        "capacity_in_millions_of_packets_per_second": lambda x: (
            "forwarding_rate",
            float(x.replace(",", "")),
        ),
        "switching_capacity_in_gigabits_per_second": lambda x: (
            "switching_capacity",
            float(x.replace(",", "")),
        ),
        "mtbf": lambda x: ("mtbf", int(x.replace(",", ""))),
    }

    # Process each data item according to its header
    for index, data in enumerate(table_data):
        if index >= len(headers_map):
            logger.warning("More data than headers for model %s", model)
            break

        key = headers_map[index]
        conversion = convert_keys.get(key)

        if conversion:
            try:
                new_key, value = conversion(data)
                obj[model][new_key] = value
            except Exception as e:
                logger.debug(
                    "Conversion issue for key %s with value %s: %s", key, data, e
                )
                # Fallback to original data if conversion fails
                obj[model][key] = data
        else:
            obj[model][key] = data

    return obj


def parse_row_data(
    rowspan: int, cells: List[str], row: Tag, obj: Dict[str, Any], length: int
) -> Dict[str, Any]:
    """Parse table row data spanning multiple rows.

    Args:
        rowspan: Number of rows to process
        cells: List of cell text content
        row: BeautifulSoup Tag representing the current row
        obj: Dictionary to store processed data
        length: Length of cells list

    Returns:
        Updated object dictionary
    """
    undesired_headers = [
        "Model",
        "Model Name",
        "Model name",
        "SKU",
        "Product Name",
        "Product name",
        "Product Ordering Number",
    ]

    # Filter out undesired headers
    filtered_cells = [cell for cell in cells if cell not in undesired_headers]

    # Calculate header slice position
    if length % 2 == 0:
        headers_slice = (math.floor(length / 2) * -1) - 1
    else:
        headers_slice = math.ceil(length / 2) * -1

    unformatted_headers = filtered_cells[headers_slice:]
    headers_map = list(dict.fromkeys(map(create_joined_header, unformatted_headers)))
    logger.debug("headers_map: %s", headers_map)

    current_row = row
    for _ in range(rowspan):
        current_row = current_row.find_next("tr")
        if current_row is None:
            logger.warning("Could not find next row, breaking early")
            break

        table_data = [
            cell.get_text(strip=True, separator=" ")
            for cell in current_row.find_all("td")
            if re.search(r".+", cell.get_text(strip=True, separator=" "))
        ]
        logger.debug("table_data: %s", table_data)

        if not table_data:
            logger.debug("Empty table data, skipping row")
            continue

        try:
            obj = handle_table_data(headers_map, table_data, obj)
        except Exception as e:
            logger.warning(
                "Failed to handle row data with headers %s and table data %s: %s",
                headers_map,
                table_data,
                e,
            )
            continue

    return obj


def parse_table(soup: BeautifulSoup, obj: Dict[str, Any]) -> Dict[str, Any]:
    """Parse generic table data from datasheet.

    Args:
        soup: BeautifulSoup object of the parsed HTML
        obj: Dictionary to store processed data

    Returns:
        Updated object dictionary
    """
    desired_titles = [
        "Model",
        "Model Name",
        "Model name",
        "SKU",
        "Product Name",
        "Product name",
        "Product Ordering Number",
        "Data rates supported",
        "General",
    ]

    # Target keys that should prefer anchor hrefs for their values
    TARGET_ANCHOR_KEYS = {
        "information_on_product_material_content_laws_and_regulations",
        "information_on_electronic_waste_laws_and_regulations_including_products_batteries_and_packaging",
        "information_on_product-material-content_laws_and_regulations",
    }

    rows = soup.select("tbody > tr")
    skip_count = 0

    for row in rows:
        if skip_count > 0:
            skip_count -= 1
            continue

        rowspan = int(row.find_next("td").get("rowspan", "0"))
        cell_data_text = [
            cell.get_text(strip=True)
            for cell in row.find_all("td")
            if re.search(r".+", cell.get_text(strip=True, separator=" "))
        ]
        logger.debug("cell data: %s", cell_data_text)

        if rowspan > 1 and any(word in cell_data_text for word in desired_titles):
            # Process multi-row section starting with desired titles
            smb_builder = parse_row_data(
                rowspan=rowspan - 1,
                cells=cell_data_text,
                row=row,
                obj=obj,
                length=len(cell_data_text),
            )
            obj.update(smb_builder)
            skip_count = rowspan - 1

        elif rowspan > 1 and not any(word in cell_data_text for word in desired_titles):
            # Process multi-row section without desired titles
            new_row = row.find_next("tr")
            if new_row is None:
                logger.warning("Expected next row but found None")
                continue

            new_unformatted_cells = [
                cell.get_text(strip=True, separator=" ")
                for cell in new_row.find_all("td")
            ]
            smb_builder = parse_row_data(
                rowspan=rowspan - 2,
                cells=new_unformatted_cells,
                row=new_row,
                obj=obj,
                length=len(new_unformatted_cells),
            )
            obj.update(smb_builder)
            skip_count = rowspan - 1

        elif rowspan == 0 and len(cell_data_text) == 2:
            # Process simple key-value pairs
            key = create_joined_header(cell_data_text[0])
            value = _extract_cell_value(row, key, TARGET_ANCHOR_KEYS)

            if value is None:
                value = cell_data_text[1]

            if not key.startswith("-") and value:
                obj[key] = value

    return obj


def _extract_cell_value(row: Tag, key: str, target_anchor_keys: set) -> Optional[str]:
    """Extract value from table cell, preferring anchor hrefs for specific keys.

    Args:
        row: BeautifulSoup Tag representing table row
        key: The key being processed
        target_anchor_keys: Set of keys that should prefer anchor hrefs

    Returns:
        Extracted value or None if extraction failed
    """
    tds = row.find_all("td")
    value_td = tds[1] if len(tds) > 1 else None

    if value_td is None:
        return None

    # Check for anchor href if this key should prefer it
    if key in target_anchor_keys:
        a = value_td.find("a", href=True)
        if a and a.get("href"):
            return a.get("href").strip()

    # Extract from paragraphs and list items
    parts = [
        el.get_text(strip=True, separator=" ")
        for el in value_td.find_all(["p", "li"])
        if re.search(r".+", el.get_text(strip=True, separator=" "))
    ]

    if parts:
        return parts[0] if len(parts) == 1 else parts

    return None


# Configuration data for datasheet scraping
DEFAULT_URLS = [
    {
        "concept": "Cisco Business 110 Series Unmanaged Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/business-110-series-unmanaged-switches/datasheet-c78-744158.html?ccid=cc001531",
    },
    {
        "concept": "Cisco Business 220 Series Smart Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/business-220-series-smart-switches/datasheet-c78-744915.html",
    },
    {
        "concept": "Cisco Business 250 Series Smart Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/business-250-series-smart-switches/nb-06-bus250-smart-switch-ds-cte-en.html",
    },
    {
        "concept": "Cisco Business 350 Series Managed Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/business-350-series-managed-switches/datasheet-c78-744156.html",
    },
    {
        "concept": "Cisco Catalyst 1000 Series Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/catalyst-1000-series-switches/nb-06-cat1k-ser-switch-ds-cte-en.html",
    },
    {
        "concept": "Cisco Catalyst 1200 Series Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/catalyst-1200-series-switches/nb-06-cat1200-ser-data-sheet-cte-en.html",
    },
    {
        "concept": "Cisco Catalyst 1300 Series Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/catalyst-1300-series-switches/nb-06-cat1300-ser-data-sheet-cte-en.html",
    },
    {
        "concept": "Cisco 350 Series Managed Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/small-business-smart-switches/data-sheet-c78-737359.html",
    },
    {
        "concept": "Cisco 350X Series Stackable Managed Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/350x-series-stackable-managed-switches/datasheet-c78-735986.html",
    },
    {
        "concept": "Cisco 550X Series Stackable Managed Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/550x-series-stackable-managed-switches/datasheet-c78-735874.html",
    },
    {
        "concept": "Cisco 250 Series Smart Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/250-series-smart-switches/datasheet-c78-737061.html",
    },
    {
        "concept": "Cisco 220 Series Smart Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/small-business-220-series-smart-plus-switches/datasheet-c78-731284.html",
    },
    {
        "concept": "Cisco 300 Series Managed Switches",
        "url": "https://www.cisco.com/c/en/us/products/collateral/switches/small-business-smart-switches/data_sheet_c78-610061.html",
    },
    {
        "concept": "Cisco Business Wireless AC",
        "url": "https://www.cisco.com/c/en/us/products/collateral/wireless/business-100-series-access-points/smb-01-bus-140ac-ap-ds-cte-en.html",
    },
    {
        "concept": "Cisco Business Wireless AC",
        "url": "https://www.cisco.com/c/en/us/products/collateral/wireless/business-100-series-access-points/smb-01-bus-145ac-ap-ds-cte-en.html",
    },
    {
        "concept": "Cisco Business Wireless AC",
        "url": "https://www.cisco.com/c/en/us/products/collateral/wireless/business-200-series-access-points/smb-01-bus-240ac-ap-ds-cte-en.html",
    },
    {
        "concept": "Cisco Business Wireless AX",
        "url": "https://www.cisco.com/c/en/us/products/collateral/wireless/business-100-series-access-points/business-access-point-ds.html",
    },
    {
        "concept": "Cisco Business Wireless AX",
        "url": "https://www.cisco.com/c/en/us/products/collateral/wireless/business-100-series-mesh-extenders/busines-mesh-extender-ds.html",
    },
]


def make_request(url: str) -> requests.Response:
    """Make HTTP request to the specified URL with appropriate headers and timeout.

    Args:
        url: Target URL to request

    Returns:
        requests.Response object

    Raises:
        requests.RequestException: If the request fails
    """
    headers = {"User-Agent": USER_AGENT}
    logger.debug("Requesting URL: %s", url)

    try:
        response = requests.get(
            url=url, timeout=DEFAULT_REQUEST_TIMEOUT, headers=headers
        )
        return response
    except requests.RequestException as e:
        logger.error("Request failed for URL %s: %s", url, e)
        raise


def create_joined_header(key: str) -> str:
    """Transform a string into a standardized header format.

    Args:
        key: Raw header text

    Returns:
        Standardized header string with underscores and normalized format

    Example:
        'Forwarding Rate (Millions of packets per second)' -> 'forwarding_rate'
    """
    if not isinstance(key, str):
        return str(key) if key is not None else ""

    # Remove parenthetical content and replace slashes
    refined_key = re.sub(r"\([^)]*\)", "", key.lower()).replace("/", " ")

    # Create underscore-separated string
    joined_header = (
        "_".join(word for word in refined_key.split() if word)
        .replace(",", "")
        .replace(":", "")
    )

    # Normalize specific header patterns
    header_mappings = {
        "forwarding_rate": [
            "forwarding_rate",
            "capacity_in_millions_of_packets",
            "capacity_in_mpps",
        ],
        "switching_capacity": ["switching_capacity"],
        "mtbf": ["mtbf"],
        "power_consumption": ["power_consumption_worst_case"],
    }

    for normalized_header, patterns in header_mappings.items():
        if any(pattern in joined_header for pattern in patterns):
            return normalized_header

    return joined_header


if __name__ == "__main__":
    main(urls=DEFAULT_URLS)
