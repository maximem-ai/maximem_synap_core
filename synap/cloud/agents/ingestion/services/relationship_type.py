"""A relationship type as the graph can store it.

The model writes the type freely ("COLLECTS_FOR", "sees / consults"). Neo4j
takes a relationship type as part of the query text, never as a parameter, so a
type with a space, a hyphen or a slash makes the write of that note's graph
invalid. ``storable`` keeps the words and drops what the query cannot hold.
"""

from __future__ import annotations

import re
from typing import Any

DEFAULT = "RELATED_TO"
_NOT_ALLOWED = re.compile(r"[^A-Z0-9]+")


def storable(raw: Any) -> str:
    """UPPERCASE letters, digits and single underscores, starting with a letter."""
    text = _NOT_ALLOWED.sub("_", str(raw or "").strip().upper()).strip("_")
    if not text:
        return DEFAULT
    return text if text[0].isalpha() else f"REL_{text}"


def readable(relationship_type: Any) -> str:
    """"COLLECTS_FOR" as words: "collects for"."""
    return str(relationship_type or "").replace("_", " ").strip().lower()
