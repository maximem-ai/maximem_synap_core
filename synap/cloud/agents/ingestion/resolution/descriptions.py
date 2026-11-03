"""Short descriptions of registry entries, for the judges.

The extraction model says in a phrase who or what each entity is, using only
what the note states ("designer on the Data Product team; she"). The phrases
from the first few notes are kept on the registry entry, and shown beside a
candidate when a mention could not be settled on names alone: "R. Smith
uploaded the design mocks" can then be told from the Robert Smith who is a
customer contact.

A description is a customer's own data. It stays on their registry entry and
never reaches the global dictionary.

It is evidence for the judges and nothing more: no merge is made on it by
code. Both judges read it: the first one through DescribingAdjudicator (describing_adjudicator.py), on
every mention it is asked about, and the second through SecondLookAdjudicator.
A first judge that cannot see it says "same" for a bare "Ravi" whenever one
Ravi is on file, whoever the note is about.

Off unless SYNAP_ER_DESCRIPTIONS=true.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Callable, List, Optional

ABOUT_KEY = "about"          # in a registry entry's metadata_: list of phrases
MAX_PHRASES = 4              # then the entry is left alone: writes stay bounded
MAX_PHRASE_CHARS = 120
MAX_DESCRIPTION_CHARS = 400
# With profiles on (profiles.py): the maintained profile, and the phrases that
# have arrived since it was last written.
PROFILE_KEY = "profile"
NEW_FACTS_KEY = "about_new"
MAX_NEW_FACTS = 12


def descriptions_enabled() -> bool:
    return os.environ.get("SYNAP_ER_DESCRIPTIONS", "false").strip().lower() in ("true", "1", "yes")


def clean_phrase(raw: Any) -> Optional[str]:
    """One phrase from the extraction model, or None when it said nothing."""
    if not isinstance(raw, str):
        return None
    phrase = " ".join(raw.split()).strip(" .;")
    return phrase[:MAX_PHRASE_CHARS] or None


def with_phrase(existing: Any, phrase: Optional[str]) -> Optional[List[str]]:
    """*existing* plus *phrase*, or None when nothing would change."""
    phrases = [p for p in (existing or []) if isinstance(p, str)]
    if not phrase or len(phrases) >= MAX_PHRASES:
        return None
    if phrase.casefold() in {p.casefold() for p in phrases}:
        return None
    return phrases + [phrase]


def with_new_fact(metadata: Any, phrase: Optional[str]) -> Optional[List[str]]:
    """The not-yet-profiled phrases plus *phrase*, or None when it is already known."""
    metadata = metadata or {}
    waiting = [p for p in (metadata.get(NEW_FACTS_KEY) or []) if isinstance(p, str)]
    if not phrase:
        return None
    profile = metadata.get(PROFILE_KEY) or {}
    known = {p.casefold() for p in waiting} | set(profile.get("seen") or [])
    if not profile:
        known |= {p.casefold() for p in (metadata.get(ABOUT_KEY) or []) if isinstance(p, str)}
    if phrase.casefold() in known:
        return None       # said before: not a reason to rewrite the profile
    return (waiting + [phrase])[-MAX_NEW_FACTS:]


def render(metadata: Any) -> Optional[str]:
    """What a judge is shown about an entry: its profile when it has one, with
    anything learned since; otherwise the first phrases its notes gave."""
    metadata = metadata or {}
    profile = metadata.get(PROFILE_KEY) or {}
    if profile.get("summary") or profile.get("fields"):
        fields = "; ".join(f"{k}: {v}" for k, v in (profile.get("fields") or {}).items())
        waiting = [p for p in (metadata.get(NEW_FACTS_KEY) or []) if isinstance(p, str)]
        parts = [fields, profile.get("summary") or ""] + (
            ["also: " + "; ".join(waiting)] if waiting else [])
        return ". ".join(p.strip(" .") for p in parts if p)[:MAX_DESCRIPTION_CHARS] or None
    phrases = [p for p in (metadata.get(ABOUT_KEY) or []) if isinstance(p, str)]
    return "; ".join(phrases)[:MAX_DESCRIPTION_CHARS] or None


def describer(index_manager: Any) -> Callable[[str], Optional[str]]:
    """candidate key -> the description on its registry entry, if any."""

    def describe(key: str) -> Optional[str]:
        try:
            entry = index_manager.get_entry(uuid.UUID(str(key)))
        except (ValueError, AttributeError):
            return None      # an earlier mention of the same note: no entry yet
        return render(entry.metadata_) if entry is not None else None

    return describe
