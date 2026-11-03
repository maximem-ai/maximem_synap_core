"""guardrails.compliance.prohibited_memory_content, applied at extraction.

Two layers, and they are not the same strength.

1. Guidance to the model. The listed items are put into the extraction prompt
   (prompts/maca_context_builder.py). A model can still ignore it or read an
   item more narrowly than the developer meant.
2. A backstop here: after extraction, a memory whose text contains a listed
   phrase is dropped. It only catches the LITERAL phrase, compared without
   regard to case. A paraphrase, a misspelling or a related fact passes.

Off unless SYNAP_GUARDRAIL_PROHIBITED_CONTENT is on for the instance, and then
the config is read only for this one list.

memory_extraction.filters.require_entity_grounding is applied here too, behind
SYNAP_GUARDRAIL_ENTITY_GROUNDING (see ``entity_grounding_required``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Sequence, Tuple

from synap.cloud.shared.instance_flags import enabled_for

logger = logging.getLogger(__name__)

_FLAG = "SYNAP_GUARDRAIL_PROHIBITED_CONTENT"
_GROUNDING_FLAG = "SYNAP_GUARDRAIL_ENTITY_GROUNDING"


def prohibited_content_for(config: Any, instance_id: Any) -> Tuple[str, ...]:
    """The instance's prohibited items, or () when the switch is off, the config
    is missing or the list is empty. A bad config must not stop an extraction."""
    if config is None or not enabled_for(_FLAG, instance_id):
        return ()
    try:
        raw = config.guardrails.compliance.prohibited_memory_content
    except AttributeError:
        return ()
    seen: List[str] = []
    for item in raw or []:
        text = str(item).strip()
        if text and text.casefold() not in (s.casefold() for s in seen):
            seen.append(text)
    return tuple(seen)


def drop_prohibited(memories: Sequence[Any], prohibited: Sequence[str]) -> Tuple[List[Any], int]:
    """(kept memories in their order, how many were dropped)."""
    needles = [p.casefold() for p in prohibited if p and p.strip()]
    if not needles:
        return list(memories), 0
    kept = [m for m in memories
            if not any(n in (getattr(m, "content", "") or "").casefold() for n in needles)]
    return kept, len(memories) - len(kept)


def _was_set(model: Any, name: str) -> bool:
    return name in (getattr(model, "model_fields_set", None) or ())


def entity_grounding_required(config: Any, instance_id: Any) -> bool:
    """True only when the developer wrote ``require_entity_grounding: true``.

    The schema defaults the field to true, so its value alone cannot tell
    "asked for it" from "said nothing"; acting on the default would drop
    memories for every instance. Each level of the path must be one the
    developer wrote (pydantic records that), and the value must be true.
    With the switch off the config is not read at all. A config that cannot
    answer counts as not set: a bad config must not stop an extraction.
    """
    if config is None or not enabled_for(_GROUNDING_FLAG, instance_id):
        return False
    try:
        if not _was_set(config, "memory_extraction"):
            return False
        extraction = config.memory_extraction
        if not _was_set(extraction, "filters"):
            return False
        filters = extraction.filters
        return _was_set(filters, "require_entity_grounding") \
            and filters.require_entity_grounding is True
    except AttributeError:
        return False


def drop_ungrounded(
    memories: Sequence[Any],
    names: Sequence[str],
    is_named: Callable[[str, str], bool],
) -> Tuple[List[Any], int]:
    """(kept memories in their order, how many were dropped).

    A memory is grounded when it names at least one of *names* (lower case),
    judged by *is_named(name, lowered_text)*: the matcher the entity linker uses,
    passed in so there is one rule for "this memory names that entity".
    """
    kept = []
    for m in memories:
        text = (getattr(m, "content", "") or "").lower()
        if any(is_named(n, text) for n in names):
            kept.append(m)
    return kept, len(memories) - len(kept)
