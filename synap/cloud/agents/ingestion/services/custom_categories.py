"""memory_categories.custom_categories, applied at extraction.

A custom category is not a new memory type. A memory of a custom category is
stored as ``facts`` and carries the category's canonical name in ``category``
(Postgres ``memories.category`` and the vector store's ``category`` metadata).

Off unless SYNAP_CUSTOM_CATEGORIES is on for the instance. Then only the
enabled categories count, the same rule as the compaction loader's
``_get_maca_custom_categories``; that loader is left as it was so nothing
changes with the switch off.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, List, Tuple

from synap.cloud.memory_config.enums import MemoryCategoryName
from synap.cloud.shared.instance_flags import enabled_for

_FLAG = "SYNAP_CUSTOM_CATEGORIES"
MAX_NAME_CHARS = 100  # the width of memories.category
MAX_EXAMPLES = 5

_BUILT_IN = frozenset(m.value for m in MemoryCategoryName)


@dataclass(frozen=True)
class CustomCategory:
    name: str  # canonical
    description: str
    examples: Tuple[str, ...] = ()


# Set by Stage 3 for the length of one extraction, like PROHIBITED_CONTENT, so the
# prompt builder and the accept rule see the same list without a signature change.
ACTIVE: ContextVar[Tuple[CustomCategory, ...]] = ContextVar(
    "extraction_custom_categories", default=())


def canonical_name(raw: Any) -> str:
    """strip, lowercase, runs of whitespace to one underscore, at most 100 characters."""
    return re.sub(r"\s+", "_", str(raw or "").strip().lower())[:MAX_NAME_CHARS]


def is_on(instance_id: Any) -> bool:
    return enabled_for(_FLAG, instance_id)


def _field(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def enabled_categories(raw: Any) -> Tuple[CustomCategory, ...]:
    """The enabled categories of a ``custom_categories`` list (objects or raw dicts), with
    canonical names, a built-in's name and a repeated name left out."""
    out: List[CustomCategory] = []
    seen = set(_BUILT_IN)
    for item in raw or []:
        if not _field(item, "enabled", True):
            continue
        name = canonical_name(_field(item, "name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        examples = tuple(
            str(e).strip() for e in (_field(item, "examples") or []) if str(e).strip()
        )[:MAX_EXAMPLES]
        out.append(CustomCategory(
            name=name,
            description=" ".join(str(_field(item, "description", "") or "").split()),
            examples=examples,
        ))
    return tuple(out)


def custom_categories_for(config: Any, instance_id: Any) -> Tuple[CustomCategory, ...]:
    """The instance's enabled custom categories, or () when the switch is off, the config
    is missing or the list is empty. A bad config must not stop an extraction."""
    if config is None or not is_on(instance_id):
        return ()
    try:
        return enabled_categories(config.memory_categories.custom_categories)
    except AttributeError:
        return ()


def format_for_prompt(categories: Tuple[CustomCategory, ...]) -> str:
    """One data line per category: ``- name: description (examples: "a"; "b")``.
    Data only; every sentence around the list is in the Core prompt pack."""
    lines = []
    for cat in categories:
        line = f"- {cat.name}: {cat.description}" if cat.description else f"- {cat.name}"
        if cat.examples:
            line += " (examples: " + "; ".join(
                json.dumps(e, ensure_ascii=False) for e in cat.examples) + ")"
        lines.append(line)
    return "\n".join(lines)
