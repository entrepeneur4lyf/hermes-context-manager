"""Content normalization for semantic deduplication."""
from __future__ import annotations
import re

_NORMALIZATIONS: list[tuple[re.Pattern, str]] = [
    # ISO timestamps
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?"), "<TIMESTAMP>"),
    # Date-only
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<DATE>"),
    # UUIDs
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<UUID>"),
    # Hex hashes (7+ chars)
    (re.compile(r"\b[0-9a-f]{7,}\b", re.I), "<HEX>"),
    # Temp paths
    (re.compile(r"/(?:tmp|var/tmp)/\S+"), "<TMPPATH>"),
    # Large numbers (5+ digits)
    (re.compile(r"\b\d{5,}\b"), "<NUM>"),
]


def normalize_for_dedup(text: str) -> str:
    result = text
    for pattern, placeholder in _NORMALIZATIONS:
        result = pattern.sub(placeholder, result)
    return result
