"""An entity's main name follows its fullest name.

The first time an entity is seen it may be by a short form: "Ravi", "LVI",
"VW". That becomes its registry entry's name. When a later, certain mention
uses a fuller name ("Ravi Kumar"), the entry is renamed to it and the short
form becomes an alias.

One entity, one identity, through the rename:

* the registry entry keeps its id;
* its graph node is found by that id and renamed (AsyncGraphWrapper);
* its records in the vector store keep their ids. Those ids are a hash of the
  entity's name, so the entry remembers the name they were first made from
  (``first_name``), and every later write uses it for the id only.

Only a certain match or a reviewer's decision renames an entry. Off with
SYNAP_ER_UPGRADE_NAMES=false.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

FIRST_NAME_KEY = "first_name"            # in a registry entry's metadata_
VECTOR_NAME_PROPERTY = "er_vector_name"  # on an ExtractedEntity, for the storage writers
RENAMED_FROM_PROPERTY = "er_renamed_from"


def upgrade_names_enabled() -> bool:
    return os.environ.get("SYNAP_ER_UPGRADE_NAMES", "true").strip().lower() in ("true", "1", "yes")


def fuller_name(current: str, names: List[str], entity_type: Any) -> Optional[str]:
    """The name among *names* that says more than *current*, if any.

    "More" as code can see it: *current* is a bare given name or surname,
    initials, an acronym, or has fewer identifying words than the other, and
    the reverse is not true. A person's fuller name has at least two words.
    """
    from synap_er.normalize import content_tokens, is_weak_alias

    kind = getattr(entity_type, "value", entity_type)
    person = str(kind).upper() == "PERSON"
    best = None
    for name in names:
        if not name or name.strip().lower() == current.strip().lower():
            continue
        if not is_weak_alias(current, name, kind) or is_weak_alias(name, current, kind):
            continue
        words = content_tokens(name, kind)
        if person and len(words) < 2:
            continue
        if best is None or len(words) > len(content_tokens(best, kind)):
            best = name
    return best


async def upgrade_name(write_through: Any, entry: Any, names: List[str]) -> Optional[str]:
    """Rename *entry* to the fullest of *names* when one is fuller than its
    current name. Returns the new name, or None when nothing changed."""
    if entry is None or not upgrade_names_enabled():
        return None
    old = entry.canonical_name
    new = fuller_name(old, names, entry.entity_type)
    if new is None:
        return None
    aliases = [a for a in list(entry.aliases or []) if a.strip().lower() != new.strip().lower()]
    if old.strip().lower() not in {a.strip().lower() for a in aliases}:
        aliases.append(old)
    metadata = dict(entry.metadata_ or {})
    metadata.setdefault(FIRST_NAME_KEY, old)
    await write_through.update_entry(
        entry.canonical_id, {"canonical_name": new, "aliases": aliases, "metadata_": metadata}
    )
    logger.info("Entity renamed to its fuller name: %r -> %r", old, new)
    return new


def stamp_vector_name(entity: Any, entry: Any) -> None:
    """Tell the storage writers which name this entity's vector ids are made from."""
    first = (getattr(entry, "metadata_", None) or {}).get(FIRST_NAME_KEY) if entry is not None else None
    if first:
        entity.properties[VECTOR_NAME_PROPERTY] = first
