from __future__ import annotations

import re
from typing import Optional, Tuple

try:
    # Prefer absolute import when used as a package
    from src.constants import SERIES_BY_CONCEPT, CATEGORY_BY_CONCEPT
except Exception:  # pragma: no cover - fallback for direct execution
    # Fallback to relative import style when run as a script
    from ..constants import SERIES_BY_CONCEPT, CATEGORY_BY_CONCEPT  # type: ignore


class DeviceDetector:
    """
    Utilities to determine product family/category and normalize device names.
    Stateless; all methods are static for easy reuse.
    """

    @staticmethod
    def family_category(concept_name: str) -> str:
        """Return high-level category for a concept/family name, else empty string."""
        return CATEGORY_BY_CONCEPT.get(concept_name, "")

    @staticmethod
    def normalize_model(model: str) -> str:
        """Normalize a model string by trimming spaces and uppercasing core code tokens."""
        m = model.strip()
        m = re.sub(r"\s+", " ", m)
        tokens = [
            t.upper() if re.search(r"[A-Za-z]\d|\d[A-Za-z]", t) else t
            for t in re.split(r"(\s+)", m)
        ]
        return "".join(tokens)

    @staticmethod
    def detect_series(concept_name: str, model: str) -> Optional[str]:
        """
        Given a concept/family name and a device model, return the canonical series
        entry from SERIES_BY_CONCEPT if it matches; else None.
        """
        series_list = SERIES_BY_CONCEPT.get(concept_name)
        if not series_list:
            return None
        normalized = DeviceDetector.normalize_model(model)
        # Direct match
        if normalized in series_list:
            return normalized
        # Loose match: remove spaces for comparison
        compact = normalized.replace(" ", "")
        for candidate in series_list:
            if candidate.replace(" ", "").upper() == compact.upper():
                return candidate
        return None

    @staticmethod
    def detect(concept_name: str, model: str) -> Tuple[str, Optional[str]]:
        """Return (category, canonical_series_model)."""
        category = DeviceDetector.family_category(concept_name)
        series_model = DeviceDetector.detect_series(concept_name, model)
        return category, series_model
