"""Scrapes the quick resources from the website and saves them to a json file."""

import json
import os
from bs4 import BeautifulSoup
import requests
from config import DATA_DIR

cwd = os.getcwd()

CBS_220_BUSINESS_SERIES = [
    "CBS220-8T-E-2G",
    "CBS220-8P-E-2G",
    "CBS220-8FP-E-2G",
    "CBS220-16T-2G",
    "CBS220-16P-2G",
    "CBS220-24T-4G",
    "CBS220-24P-4G",
    "CBS220-24FP-4G",
    "CBS220-48T-4G",
    "CBS220-48P-4G",
    "CBS220-24T-4X",
    "CBS220-24P-4X",
    "CBS220-24FP-4X",
    "CBS220-48T-4X",
    "CBS220-48P-4X",
    "CBS220-48FP-4X",
]

CBS_250_BUSINESS_SERIES = [
    "CBS250-8T-D",
    "CBS250-8PP-D",
    "CBS250-8T-E-2G",
    "CBS250-8PP-E-2G",
    "CBS250-8P-E-2G",
    "CBS250-8FP-E-2G",
    "CBS250-16T-2G",
    "CBS250-16P-2G",
    "CBS250-24T-4G",
    "CBS250-24PP-4G",
    "CBS250-24P-4G",
    "CBS250-24FP-4G",
    "CBS250-48T-4G",
    "CBS250-48PP-4G",
    "CBS250-48P-4G",
    "CBS250-24T-4X",
    "CBS250-24P-4X",
    "CBS250-24FP-4X",
    "CBS250-48T-4X",
    "CBS250-48P-4X",
]

CBS_350_BUSINESS_SERIES = [
    "CBS350-8T-E-2G",
    "CBS350-8P-2G",
    "CBS350-8P-E-2G",
    "CBS350-8FP-2G",
    "CBS350-8FP-E-2G",
    "CBS350-8S-E-2G",
    "CBS350-16T-2G",
    "CBS350-16T-E-2G",
    "CBS350-16P-2G",
    "CBS350-16P-E-2G",
    "CBS350-16FP-2G",
    "CBS350-24T-4G",
    "CBS350-24P-4G",
    "CBS350-24FP-4G",
    "CBS350-24S-4G",
    "CBS350-48T-4G",
    "CBS350-48P-4G",
    "CBS350-48FP-4G",
    "CBS350-24T-4X",
    "CBS350-24P-4X",
    "CBS350-24FP-4X",
    "CBS350-48T-4X",
    "CBS350-48P-4X",
    "CBS350-48FP-4X",
    "CBS350-8MGP-2X",
    "CBS350-8MP-2X",
    "CBS350-24MGP-4X",
    "CBS350-12NP-4X",
    "CBS350-24NGP-4X",
    "CBS350-48NGP-4X",
    "CBS350-8XT",
    "CBS350-12XS",
    "CBS350-12XT",
    "CBS350-16XTS",
    "CBS350-24XS",
    "CBS350-24XT",
    "CBS350-24XTS",
    "CBS350-48XT-4X",
]

CATALYST_1000_SERIES = [
    "C1000-8T-2G-L",
    "C1000-8T-E-2G-L",
    "C1000-8P-2G-L",
    "C1000-8P-E-2G-L",
    "C1000-8FP-2G-L",
    "C1000-8FP-E-2G-L",
    "C1000-16T-2G-L",
    "C1000-16T-E-2G-L",
    "C1000-16P-2G-L",
    "C1000-16P-E-2G-L",
    "C1000-16FP-2G-L",
    "C1000-24T-4G-L",
    "C1000-24P-4G-L",
    "C1000-24FP-4G-L",
    "C1000-48T-4G-L",
    "C1000-48P-4G-L",
    "C1000-48FP-4G-L",
    "C1000-24T-4X-L",
    "C1000-24P-4X-L",
    "C1000-24FP-4X-L",
    "C1000-48T-4X-L",
    "C1000-48P-4X-L",
    "C1000-48FP-4X-L",
    "C1000FE-24T-4G-L",
    "C1000FE-24P-4G-L",
    "C1000FE-48T-4G-L",
    "C1000FE-48P-4G-L",
]

COMBINED_SERIES = (
    CBS_220_BUSINESS_SERIES
    + CBS_250_BUSINESS_SERIES
    + CBS_350_BUSINESS_SERIES
    + CATALYST_1000_SERIES
)


# Parses Quick Resources from Support Pages


def quick_resources():
    """This function parses Quick Resources from our Support Pages"""
    urls = [
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS220.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS250.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS350.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/wireless-mesh-100-200-series.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/wireless-mesh-100-AX-series.html?cachemode=refresh",
    ]

    refined_options = []
    for page in urls:
        response = requests.get(page, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        series = soup.find("meta", property="og:title").get("content")
        url = soup.find("meta", property="og:url").get("content")
        description = soup.find("meta", property="og:description").get("content")
        targets = soup.select("#flexContainer > a")
        print(f"targets: {targets}")
        anchors = []

        for tag in targets:
            key = tag.find_next(class_="copy").get_text(strip=True)
            print(key)
            key_list = key.split(" ")
            if key_list and len(key_list) > 1:
                key = "".join(key_list)
            anchors.append({"id": key, "href": tag.get("href")})

        dropdown_targets = soup.select(
            "#flexContainer > div.flexItem > details.QSG > div#AG"
        )
        print(f"dropdowns: {dropdown_targets}")
        if len(dropdown_targets) > 0:
            for target in dropdown_targets:
                subanchors = []
                key = target.find_previous("summary").get_text(strip=True)
                print(key)
                text = key.split(" ")
                if text and len(text) > 1:
                    key = "".join(text)
                    print(f"new key: {key}")
                for link in target.contents[1::2]:
                    print(f"link: {link}")
                    device = link.get_text(strip=True)
                    href = link["href"]
                    subanchors.append({"device": device, "href": href})

                anchors.append({"id": key, "nested_resources": subanchors})

        refined_options.append(
            {
                "series": series,
                "page": url,
                "description": description,
                "resources": anchors,
            }
        )
        dumped = json.dumps(refined_options, indent=4, skipkeys=True)
        quick_resources_path = DATA_DIR / "quick_resources" / "quick_resources.json"
        quick_resources_path.parent.mkdir(parents=True, exist_ok=True)
        with quick_resources_path.open("w+", encoding="utf8") as file:
            file.write(dumped)


quick_resources()
