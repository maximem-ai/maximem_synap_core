"""Literal spans (IDs, emails, phones, amounts, ISO dates, versions) pulled out of text.

Moved unchanged from ``completeness_check.py`` so the v3 checks can use them in
Core, which does not ship the Jev completeness check. ``completeness_check``
imports both names back, so the old import path still works.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

# Recall-tuned on purpose: over-finding is cheap, because a candidate the
# document lacks is only reported if Jev says it is still current.
_LITERAL_PATTERNS = (
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",   # email
    r"\+\d[\d\s()\-.]{6,}\d",                             # phone, international
    r"[$€£¥]\s?\d[\d,]*(?:\.\d{2})?",                     # money
    r"\b[A-Z][A-Z0-9]{0,5}-[A-Z0-9]+(?:-[A-Z0-9]+)*\b",   # ORD-88213-KX
    r"\b\d{4}-\d{2}-\d{2}\b",                             # ISO date
    r"\b\d+\.\d+\.\d+\b",                                 # version
)


def literal_candidates(texts: Sequence[str]) -> List[str]:
    """Distinct literal spans across the texts, in first-seen order."""
    found: Dict[str, None] = {}
    for text in texts:
        for pattern in _LITERAL_PATTERNS:
            for match in re.finditer(pattern, text):
                found.setdefault(match.group(0).strip(), None)
    return list(found)
