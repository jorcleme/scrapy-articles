"""
Centralized constants and product lists used across the project.
This provides a single source of truth for series names, model codes,
and common filesystem paths used by scrapers and detectors.
"""

from __future__ import annotations
from typing import Literal, Optional, Dict


# Series and model lists (Switches)
CISCO_CATALYST_1200_SERIES = [
    "C1200-8T-D",
    "C1200-8T-E-2G",
    "C1200-8P-E-2G",
    "C1200-8FP-2G",
    "C1200-16T-2G",
    "C1200-16P-2G",
    "C1200-24T-4G",
    "C1200-24P-4G",
    "C1200-24FP-4G",
    "C1200-48T-4G",
    "C1200-48P-4G",
    "C1200-24T-4X",
    "C1200-24P-4X",
    "C1200-24FP-4X",
    "C1200-48T-4X",
    "C1200-48P-4X",
]

CISCO_CATALYST_1300_SERIES = [
    "C1300-8FP-2G",
    "C1300-8T-E-2G",
    "C1300-8P-E-2G",
    "C1300-16T-2G",
    "C1300-16P-2G",
    "C1300-16FP-2G",
    "C1300-24T-4G",
    "C1300-24P-4G",
    "C1300-24FP-4G",
    "C1300-48T-4G",
    "C1300-48P-4G",
    "C1300-48FP-4G",
    "C1300-16P-4X",
    "C1300-24T-4X",
    "C1300-24P-4X",
    "C1300-24FP-4X",
    "C1300-48T-4X",
    "C1300-48P-4X",
]

CBS_110_BUSINESS_SERIES_UNMANAGED = [
    "CBS110-5T-D",
    "CBS110-8T-D",
    "CBS110-8PP-D",
    "CBS110-16T",
    "CBS110-16PP",
    "CBS110-24T",
    "CBS110-24PP",
]

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

CBW_100_SERIES_140AC = ["CBW140AC-x"]

CBW_100_SERIES_145AC = ["CBW145AC-x"]

CISCO_110_SERIES_UNMANAGED = [
    "SF110D-05",
    "SF110D-08",
    "SF110D-08HP",
    "SF110D-16",
    "SF110D-16HP",
    "SF110-16",
    "SF110-24",
    "SF112-24",
    "SG110D-05",
    "SG110D-08",
    "SG110D-08HP",
    "SG110-16",
    "SG110-16HP",
    "SG112-24",
    "SG110-24",
    "SG110-24HP",
]

CISCO_350_SERIES_MANAGED_SWITCHES = [
    "SF350-08",
    "SF352-08",
    "SF352-08P",
    "SF352-08MP",
    "SF350-24",
    "SF350-24P",
    "SF350-24MP",
    "SF350-48",
    "SF350-48P",
    "SF350-48MP",
    "SG350-8PD",
    "SG350-10",
    "SG350-10P",
    "SG350-10MP",
    "SG355-10MP",
    "SG350-10SFP",
    "SG350-20",
    "SG350-28",
    "SG350-28P",
    "SG350-28MP",
    "SG350-28SFP",
    "SG350-52",
    "SG350-52P",
    "SG350-52MP",
]

CISCO_350X_STACKABLE_SERIES = [
    "SG350X-8PMD",
    "SG350X-12PMV",
    "SG350X-24P",
    "SG350X-24MP",
    "SG350X-24PD",
    "SG350X-24PV",
    "SG350X-48",
    "SG350X-48P",
    "SG350X-48MP",
    "SG350X-48PV",
    "SG350X-48P",
    "SG350XG-2F10",
    "SG350XG-24F",
    "SG350XG-24T",
    "SG350XG-48T",
    "SX350X-08",
    "SX350X-12",
    "SX350X-24F",
    "SX350X-24",
    "SX350X-52",
]

CISCO_550X_STACKABLE_SERIES = [
    "SF500-24",
    "SF500-24P",
    "SF500-24MP",
    "SF500-48",
    "SF500-48P",
    "SF500-48MP",
    "SG500-28",
    "SG500-28P",
    "SG500-28MPP",
    "SG500-52",
    "SG500-52P",
    "SG500-52MP",
    "SG500X-24",
    "SG500X-24P",
    "SG500X-24MPP",
    "SG500X-48",
    "SG500X-48P",
    "SG500X-48MP",
    "SG500XG-8F8T",
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


# Derived / combined constants
COMBINED_SERIES = (
    CBS_220_BUSINESS_SERIES
    + CBS_250_BUSINESS_SERIES
    + CBS_350_BUSINESS_SERIES
    + CATALYST_1000_SERIES
    + CBW_100_SERIES_140AC
    + CISCO_110_SERIES_UNMANAGED
    + CBW_100_SERIES_145AC
    + CISCO_110_SERIES_UNMANAGED
    + CISCO_CATALYST_1200_SERIES
    + CISCO_CATALYST_1300_SERIES
    + CISCO_350_SERIES_MANAGED_SWITCHES
    + CBS_110_BUSINESS_SERIES_UNMANAGED
)


# Concept names used across the project mapped to series lists
SERIES_BY_CONCEPT: dict[str, list[str]] = {
    # Cisco Business switches
    "Cisco Business 110 Series Unmanaged Switches": CBS_110_BUSINESS_SERIES_UNMANAGED,
    "Cisco Business 220 Series Smart Switches": CBS_220_BUSINESS_SERIES,
    "Cisco Business 250 Series Smart Switches": CBS_250_BUSINESS_SERIES,
    "Cisco Business 350 Series Managed Switches": CBS_350_BUSINESS_SERIES,
    # Legacy small business naming
    "Cisco 350 Series Managed Switches": CISCO_350_SERIES_MANAGED_SWITCHES,
    "Cisco 350X Series Stackable Managed Switches": CISCO_350X_STACKABLE_SERIES,
    "Cisco 550X Series Stackable Managed Switches": CISCO_550X_STACKABLE_SERIES,
    # Catalyst
    "Cisco Catalyst 1000 Series Switches": CATALYST_1000_SERIES,
    "Cisco Catalyst 1200 Series Switches": CISCO_CATALYST_1200_SERIES,
    "Cisco Catalyst 1300 Series Switches": CISCO_CATALYST_1300_SERIES,
    # Terse Catalyst names used elsewhere
    "Catalyst 1200 Series": CISCO_CATALYST_1200_SERIES,
    "Catalyst 1300 Series": CISCO_CATALYST_1300_SERIES,
}

# High-level category by concept (expand as needed)
CATEGORY_BY_CONCEPT: dict[str, str] = {
    **{k: "Switch" for k in SERIES_BY_CONCEPT.keys()},
    "Cisco Business Wireless AC": "Wireless",
    "Cisco Business Wireless AX": "Wireless",
}


CollectionType = Literal["admin_guide", "cli_guide", "article"]


class BaseCollections:

    _collections: Dict[str, str] = {}

    def __init__(self, collection_type: CollectionType) -> None:
        self._collection_type = collection_type

    @classmethod
    def resolve_collection_name(cls, family: str) -> Optional[str]:
        return cls._collections.get(family, None)


class AdminGuideCollectionNames(BaseCollections):

    CATALYST_1300 = "catalyst_1300_admin_guide"
    CATALYST_1200 = "catalyst_1200_admin_guide"
    CBS_220 = "cbs_220_admin_guide"
    CBS_250 = "cbs_250_admin_guide"
    CBS_350 = "cbs_350_admin_guide"
    CISCO_350 = "cisco_350_admin_guide"
    CISCO_350X = "cisco_350x_admin_guide"
    CISCO_550X = "cisco_550x_admin_guide"
    RV100 = "rv100_admin_guide"
    RV160_VPN = "cisco_rv160_vpn_admin_guide"
    RV260_VPN = "cisco_rv260_vpn_admin_guide"
    RV320 = "rv320_admin_guide"
    RV340 = "rv340_admin_guide"
    CBW_AC = "cisco_business_wireless_ac_admin_guide"
    CBW_AX = "cisco_business_wireless_ax_admin_guide"
    WAP100 = "cisco_wap100_admin_guide"
    WAP300 = "cisco_wap300_admin_guide"
    WAP500 = "cisco_wap500_admin_guide"

    _collections = {
        "Cisco Catalyst 1300 Series Switches": CATALYST_1300,
        "Cisco Catalyst 1200 Series Switches": CATALYST_1200,
        "Cisco Business 220 Series Smart Switches": CBS_220,
        "Cisco Business 250 Series Smart Switches": CBS_250,
        "Cisco Business 350 Series Managed Switches": CBS_350,
        "Cisco 350 Series Managed Switches": CISCO_350,
        "Cisco 350X Series Stackable Managed Switches": CISCO_350X,
        "Cisco 550X Series Stackable Managed Switches": CISCO_550X,
        "RV100 Product Family": RV100,
        "RV160 VPN Router": RV160_VPN,
        "RV260 VPN Router": RV260_VPN,
        "RV320 Product Family": RV320,
        "RV340 Product Family": RV340,
        "Cisco Business Wireless AC": CBW_AC,
        "Cisco Business Wireless AX": CBW_AX,
        "Cisco Small Business 100 Series Wireless Access Points": WAP100,
        "Cisco Small Business 300 Series Wireless Access Points": WAP300,
        "Cisco Small Business 500 Series Wireless Access Points": WAP500,
    }

    def __init__(self) -> None:
        super().__init__(collection_type="admin_guide")


class CLIGuideCollectionNames(BaseCollections):
    CATALYST_1300 = "catalyst_1300_cli_guide"
    CATALYST_1200 = "catalyst_1200_cli_guide"
    CBS_220 = "cbs_220_cli_guide"
    CBS_250 = "cbs_250_cli_guide"
    CBS_350 = "cbs_350_cli_guide"
    CISCO_350 = "cisco_350_cli_guide"
    CISCO_350X = "cisco_350x_cli_guide"
    CISCO_550X = "cisco_550x_cli_guide"

    _collections = {
        "Cisco Catalyst 1300 Series Switches": CATALYST_1300,
        "Cisco Catalyst 1200 Series Switches": CATALYST_1200,
        "Cisco Business 220 Series Smart Switches": CBS_220,
        "Cisco Business 250 Series Smart Switches": CBS_250,
        "Cisco Business 350 Series Managed Switches": CBS_350,
        "Cisco 350 Series Managed Switches": CISCO_350,
        "Cisco 350X Series Stackable Managed Switches": CISCO_350X,
        "Cisco 550X Series Stackable Managed Switches": CISCO_550X,
    }

    def __init__(self) -> None:
        super().__init__(collection_type="cli_guide")


class ArticleCollections(BaseCollections):
    CATALYST_1300 = "catalyst_1300_articles"
    CATALYST_1200 = "catalyst_1200_articles"
    CBS_220 = "cbs_220_articles"
    CBS_250 = "cbs_250_articles"
    CBS_350 = "cbs_350_articles"
    CISCO_350 = "cisco_350_articles"
    CISCO_350X = "cisco_350x_articles"
    CISCO_550X = "cisco_550x_articles"
    RV100 = "rv100_articles"
    RV160_VPN = "cisco_rv160_vpn_articles"
    RV260_VPN = "cisco_rv260_vpn_articles"
    RV320 = "rv320_articles"
    RV340 = "rv340_articles"
    CBW_AC = "cisco_business_wireless_ac_articles"
    CBW_AX = "cisco_business_wireless_ax_articles"
    WAP100 = "cisco_wap100_articles"
    WAP300 = "cisco_wap300_articles"
    WAP500 = "cisco_wap500_articles"

    _collections = {
        "Cisco Catalyst 1300 Series Switches": CATALYST_1300,
        "Cisco Catalyst 1200 Series Switches": CATALYST_1200,
        "Cisco Business 220 Series Smart Switches": CBS_220,
        "Cisco Business 250 Series Smart Switches": CBS_250,
        "Cisco Business 350 Series Managed Switches": CBS_350,
        "Cisco 350 Series Managed Switches": CISCO_350,
        "Cisco 350X Series Stackable Managed Switches": CISCO_350X,
        "Cisco 550X Series Stackable Managed Switches": CISCO_550X,
        "RV100 Product Family": RV100,
        "RV160 VPN Router": RV160_VPN,
        "RV260 VPN Router": RV260_VPN,
        "RV320 Product Family": RV320,
        "RV340 Product Family": RV340,
        "Cisco Business Wireless AC Series": CBW_AC,
        "Cisco Business Wireless AX Series": CBW_AX,
        "Cisco WAP100 Series": WAP100,
        "Cisco WAP300 Series": WAP300,
        "Cisco WAP500 Series": WAP500,
    }

    def __init__(self) -> None:
        super().__init__(collection_type="article")


class CollectionFactory:
    _article = ArticleCollections()
    _admin_guide = AdminGuideCollectionNames()
    _cli_guide = CLIGuideCollectionNames()

    @classmethod
    def retrieve_collection_name(
        cls, collection_type: CollectionType, family: str
    ) -> Optional[str]:
        """
        Get the collection name for a given type and product family.
        """
        if collection_type == "admin_guide":
            return cls._admin_guide.resolve_collection_name(family)
        elif collection_type == "cli_guide":
            return cls._cli_guide.resolve_collection_name(family)
        elif collection_type == "article":
            return cls._article.resolve_collection_name(family)
        return None
