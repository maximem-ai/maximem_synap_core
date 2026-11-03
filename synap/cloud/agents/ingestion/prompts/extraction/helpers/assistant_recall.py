"""Assistant-recall helpers that are logic, not prompt text: the on/off flag, the quote hint and the quote merge.

Moved byte for byte out of cloud/assistant_recall.py so Core can ship them; that module
imports them back, so Cloud sees the same function objects. The prompt text stays there.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Set

# Content that plausibly contains quoted material. Used to decide whether the
# second (quotation) call is worth making, so quote-free sessions do not pay for
# it. Deliberately loose: a false positive costs one call, a false negative
# costs a benchmark point.
_QUOTE_HINT = re.compile(
    r'["“”]|\b(writes|wrote|notes|noted|says|said|states|stated|argues|observes)\b',
    re.I,
)

def allowlisted_instances() -> Set[str]:
    """Instance ids this profile is enabled for. Empty set means off everywhere.

    ⚠ These must be instance **UUIDs**, not the ``inst_...`` external ids. The
    ingestion `Scope` carries `instance_id` as a UUID and resolving the external
    id here would need a database round trip on every extraction call. For the
    benchmark instances the mapping is:

        00000000-0000-0000-0000-0000000000a1  = inst_00000000000000a1 (LongMemEval)
        00000000-0000-0000-0000-0000000000a2  = inst_00000000000000a2 (LoCoMo)
    """
    raw = os.environ.get("SYNAP_ASSISTANT_RECALL_INSTANCES", "").strip()
    return {p.strip().lower() for p in raw.split(",") if p.strip()}

def enabled_for(instance_id: Optional[object]) -> bool:
    """True when *instance_id* is allowlisted for assistant-recall extraction."""
    if instance_id is None:
        return False
    allow = allowlisted_instances()
    return bool(allow) and str(instance_id).strip().lower() in allow

def wants_quote_pass(content: str) -> bool:
    return bool(content) and bool(_QUOTE_HINT.search(content))

def quote_memory_content(speaker: str, about: str, quote: str) -> str:
    """Render one captured quotation as a self-contained memory string."""
    who = (speaker or "").strip() or "Unattributed"
    topic = (about or "").strip()
    tail = f" (on {topic})" if topic else ""
    return f'{who} said, "{quote.strip()}"{tail}'

def merge_quote_entries(
    existing_contents: List[str], quotes: List[dict]
) -> List[dict]:
    """Turn quote-pass output into memory dicts, skipping ones already captured.

    A quote the main pass already reproduced verbatim is dropped rather than
    stored twice — duplicate memories are retrieval noise and cost tokens on
    every later fetch.
    """
    out: List[dict] = []
    for q in quotes or []:
        if not isinstance(q, dict):
            continue
        text = str(q.get("quote", "") or "").strip()
        if not text:
            continue
        if any(text in c for c in existing_contents):
            continue
        out.append(
            {
                "type": "facts",
                "content": quote_memory_content(
                    str(q.get("speaker", "") or ""),
                    str(q.get("about", "") or ""),
                    text,
                ),
                "confidence": 0.9,
            }
        )
    return out


def system_addendum(memory_type: Optional[object] = None) -> str:
    """Core's ``system_addendum``: "" for every type but facts, as in Cloud.

    For facts the addendum is prompt text, and Core has none: asking for it
    stops with the clear missing-prompt message. Cloud does not use this function;
    its own module defines the real one. Reached only through the interface of
    ``assistant_recall`` when there is no Core file for it.
    """
    name = getattr(memory_type, "value", memory_type)
    if str(name).lower() != "facts":
        return ""
    from synap.cloud.agents.ingestion.prompts.extraction import assistant_recall as text

    return str(text._RECALL_RULES)  # raises CorePromptMissing in Core
