"""The two Jev settings and the note-cutting that every Jev question shares.

They sit here, in a file that ships, because the graph and resolution files that
ask Jev a question read them. Reading them must not import
``jev_adjudicator``, which stays in Cloud. In Core nothing asks Jev anything:
Core refuses ``JEV_API_KEY`` (shared/edition.py), so these are read by no
running code there.
"""

import os
from typing import List, Optional

# Pinned, not "jev-latest": the alias moves when TypeSafe ships a release, and
# the cut points were measured on this version.
JEV_MODEL = os.environ.get("SYNAP_ER_JEV_MODEL", "jev-1.13.0")
JEV_TIMEOUT_S = float(os.environ.get("SYNAP_ER_JEV_TIMEOUT_S", "4.0"))

# Jev reads the whole state for every question and gets less accurate as
# unrelated text grows, so a long document is cut to windows around the
# mentions being judged.
MAX_NOTE_CHARS = 6000
WINDOW_CHARS = 400


def note_around(context: Optional[str], mentions: List[str]) -> str:
    """The whole note when short, else windows around each mention."""
    text = (context or "").strip()
    if len(text) <= MAX_NOTE_CHARS:
        return text
    spans = []
    lowered = text.lower()
    for mention in mentions:
        at = lowered.find(mention.lower())
        if at >= 0:
            spans.append((max(0, at - WINDOW_CHARS), min(len(text), at + len(mention) + WINDOW_CHARS)))
    if not spans:
        return text[:MAX_NOTE_CHARS]
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return " … ".join(text[s:e] for s, e in merged)[:MAX_NOTE_CHARS]
