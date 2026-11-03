"""The strict rules Core applies to a hand-written memory config.

Cloud never calls this: on ``main`` the schema drops a key it does not know
without an error, and that stays. Only the file loader (``file_loader.py``)
uses these rules.

What is checked, on the config as the developer wrote it:

* a key the schema does not know (a typo, a key at the wrong level), named in
  the message, with the closest real key when there is one;
* a key on the removed list (``REMOVED``): fields of features Core does not have,
  fields Core dropped, and fields ``main`` retired on 21 Oct 2026. The message
  says "removed" and why, instead of "unknown";
* ``storage_architecture.scoping.user_context_isolation: maca_suggests``;
* a word outside the set a field understands (``value_types.VALUE_ENUMS``): the
  message names the field, the word and the allowed words;
* the ranking weights: all three of relevance, importance and recency, or none.
  A partial set is rejected.

The schema itself is not changed and not consulted for its defaults here: the
ranking weights a retrieval reader sees are the ones the developer wrote.
"""

from __future__ import annotations

import difflib
import typing
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from . import config_schema as _schema
from . import value_types

#: The three ranking weights, and the only keys ``primary_signals`` may hold.
RANKING_WEIGHTS = ("relevance", "importance", "recency")

_MACA = "MACA is not in Core"

#: Dotted path -> why Core rejects it. ``*`` stands for one name (a memory
#: category). A list item adds no segment. A removed block is not looked into.
REMOVED: Dict[str, str] = {
    # --- Cut features ---
    "storage_architecture.scoping.allow_cross_user_retrieval": "cross-user grants are not in Core",
    "storage_architecture.scoping.allow_cross_customer_retrieval": "cross-user grants are not in Core",
    "storage_architecture.scoping.allow_cross_instance_for_subagents": "multi-agent memory sharing is not in Core",
    "storage_architecture.scoping.cross_user_recommendation": _MACA,
    "storage_architecture.scoping.cross_customer_recommendation": _MACA,
    "storage_architecture.proposed_ladder": "the scope ladder proposal comes from MACA, which is not in Core",
    "retrieval_config.anticipation": "anticipation is not in Core",
    "agent_hints.retrieval_agent.anticipation_patterns": "anticipation is not in Core",
    "consolidation": "consolidation is not in Core",
    "compaction_config.adaptive_thresholds.steering_enabled": "steering mode is not in Core, so there are no steering texts; Core compacts in the fidelity mode only",
    "compaction_config.adaptive_thresholds.trigger_at_tokens": "compaction no longer starts at a token count; the schema keeps it only so configs Cloud stored still load",
    "compaction_config.adaptive_thresholds.trigger_at_messages": "compaction no longer starts at a message count; the schema keeps it only so configs Cloud stored still load",
    "compaction_config.adaptive_thresholds.incremental_message_delta": "nothing reads it; the schema keeps it only so configs Cloud stored still load",
    "compaction_config.adaptive_thresholds.target_tokens": "retired in compaction v3, which sizes its summary from target_ratio",
    "compaction_config.adaptive_thresholds.preserve_last_n_turns": "retired in compaction v3, which keeps the turns it needs on its own",
    "memory_extraction.pii_handling": "PII management is not in Core (also removed on main, 21 Oct 2026)",
    "memory_categories.*.append_only": "its only reader was consolidation (also removed on main, 21 Oct 2026)",
    "memory_categories.custom_categories.append_only": "its only reader was consolidation (also removed on main, 21 Oct 2026)",
    "memory_categories.*.retention_days": "a Cloud-only setting that nothing reads",
    "memory_categories.custom_categories.retention_days": "a Cloud-only setting that nothing reads",
    # --- Removed from Core, and the 21 Oct 2026 removals ---
    "memory_extraction.extraction_mode": "the per-category extraction_aggressiveness covers it (also removed on main, 21 Oct 2026)",
    "memory_extraction.triggers.on_conversation_end": "it never drove anything (also removed on main, 21 Oct 2026)",
    "memory_extraction.triggers.on_explicit_signal": "it never drove anything (also removed on main, 21 Oct 2026)",
    "memory_extraction.triggers.real_time_streaming": "it never drove anything (also removed on main, 21 Oct 2026)",
    "memory_extraction.filters.deduplicate_similar": "retired on main (PR #1349, then removed from the schema on 21 Oct 2026)",
    "memory_extraction.filters.similarity_threshold": "retired on main (PR #1349); the threshold is measured, not configured",
    "memory_extraction.domain_hints.max_entities": "it was for the second entity prompt, which graph extraction v2 replaces",
    "memory_extraction.domain_hints.entity_priority": "it was for the second entity prompt, which graph extraction v2 replaces",
    "categorization": "the categorisation step was removed (also removed on main, 21 Oct 2026)",
    "metadata.template_base": "removed on main, 21 Oct 2026 (PR #1368)",
    "storage_architecture.primary_mode": "the stores come from the environment file",
    "storage_architecture.vector_config": "the vector store and embedding model come from the environment file",
    "evolution": "config self-update and version compatibility are not in Core",
    "agent_hints.retrieval_agent.ranking_guidance": "free-text ranking advice for an LLM, which retrieval in Core does not call",
    "agent_hints.ingestion_agent.extraction_scope_notes": "written by MACA for the graph prompt, which is Cloud only",
    "agent_hints.ingestion_agent.extraction_patterns": "written by MACA for the graph prompt, which is Cloud only",
    "agent_hints.ingestion_agent.extraction_examples": "written by MACA for the graph prompt, which is Cloud only",
}

#: Values that are cut features even though the key itself is valid.
REMOVED_VALUES: Dict[str, Tuple[Any, str]] = {
    "storage_architecture.scoping.user_context_isolation": ("maca_suggests", _MACA),
}

#: The memory categories are fields of the schema, not dict keys.
_CATEGORY_NAMES = ("facts", "preferences", "episodes", "emotions", "temporal_events")


def _removed_reason(path: Tuple[str, ...]) -> Optional[str]:
    """The reason when ``path`` (list indices already dropped) is on the removed list."""
    parts = list(path)
    if len(parts) == 3 and parts[0] == "memory_categories" and parts[1] in _CATEGORY_NAMES:
        parts[1] = "*"
    return REMOVED.get(".".join(parts))


def _unwrap(annotation: Any) -> Any:
    """Strip Optional/Union-with-None."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _unwrap(args[0])
    return annotation


def _child(annotation: Any) -> Tuple[str, Any]:
    """('model', cls), ('list', model_cls), ('dict', model_cls) or ('leaf', None)."""
    annotation = _unwrap(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "model", annotation
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (list, List) and args:
        inner = _unwrap(args[0])
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return "list", inner
    if origin in (dict, Dict) and len(args) == 2:
        inner = _unwrap(args[1])
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return "dict", inner
    return "leaf", None


def _dotted(path: Tuple[Any, ...]) -> str:
    out = ""
    for p in path:
        out += f"[{p}]" if isinstance(p, int) else (f".{p}" if out else str(p))
    return out or "(top level)"


Problem = Tuple[Tuple[Any, ...], str]  # (path with list indices, message)


def _walk(model: type, data: Any, path: Tuple[Any, ...], plain: Tuple[str, ...], out: List[Problem]) -> None:
    if not isinstance(data, dict):
        return  # a wrong type is reported by the schema pass
    fields = model.model_fields
    for key, value in data.items():
        key_path = path + (key,)
        key_plain = plain + (str(key),)
        reason = _removed_reason(key_plain)
        if reason is not None:
            out.append((key_path, f"{_dotted(key_path)}: this field was removed from Core. Why: {reason}"))
            continue
        if key not in fields:
            near = difflib.get_close_matches(str(key), list(fields), n=1)
            hint = f" Did you mean '{near[0]}'?" if near else ""
            where = _dotted(path) if path else "the top level"
            out.append((key_path, f"unknown key '{key}' at {where}. A typo, a key at the wrong level, "
                                  f"or a key the schema does not have.{hint}"))
            continue
        kind, inner = _child(fields[key].annotation)
        if kind == "model":
            _walk(inner, value, key_path, key_plain, out)
        elif kind == "list" and isinstance(value, list):
            for i, item in enumerate(value):
                _walk(inner, item, key_path + (i,), key_plain, out)
        elif kind == "dict" and isinstance(value, dict):
            for name, item in value.items():
                _walk(inner, item, key_path + (name,), key_plain, out)


def _check_values(raw: Dict[str, Any], out: List[Problem]) -> None:
    for dotted, (bad, why) in REMOVED_VALUES.items():
        node: Any = raw
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node == bad:
            path = tuple(dotted.split("."))
            out.append((path, f"{dotted}: the value '{bad}' was removed from Core. Why: {why}"))

    for problem in value_types.find_value_problems(raw):
        path = problem[0]
        plain = tuple(p for p in path if isinstance(p, str))
        # A field inside a removed block is reported as removed already.
        if any(_removed_reason(plain[:end]) is not None for end in range(1, len(plain))):
            continue
        out.append((path, value_types.describe(problem)))

    rc = raw.get("retrieval_config")
    ranking = rc.get("ranking") if isinstance(rc, dict) else None
    if isinstance(ranking, dict) and "primary_signals" in ranking:
        path = ("retrieval_config", "ranking", "primary_signals")
        weights = ranking["primary_signals"]
        if not isinstance(weights, dict):
            out.append((path, f"{_dotted(path)}: must list relevance, importance and recency"))
            return
        for key in weights:
            if key not in RANKING_WEIGHTS:
                out.append((path + (key,), f"{_dotted(path)}: unknown weight '{key}'. "
                                           f"The weights are {', '.join(RANKING_WEIGHTS)}."))
        missing = [w for w in RANKING_WEIGHTS if w not in weights]
        if missing:
            out.append((path, f"{_dotted(path)}: a partial set of ranking weights is not accepted "
                              f"(missing {', '.join(missing)}). Give relevance, importance and recency, "
                              f"or leave out the whole 'ranking' weights block to get 0.70, 0.20, 0.10."))


def find_unknown_keys(raw: Any) -> List[Problem]:
    """Only the keys the schema does not know. Core rejects these; Cloud warns about them.

    A field Core removed on purpose is valid in Cloud, so it is not reported here.
    """
    if not isinstance(raw, dict):
        return []
    out: List[Problem] = []
    _walk(_schema.MemoryArchitectureConfig, raw, (), (), out)
    return [p for p in out if p[1].startswith("unknown key")]


def find_problems(raw: Any) -> List[Problem]:
    """Every problem in the config as written. Empty when it is acceptable.

    Runs the unknown-key, removed-key and ranking-weight rules only. The
    schema's own type checks are a separate pass (``file_loader``).
    """
    if not isinstance(raw, dict):
        return [((), "the file must hold a YAML mapping at the top level")]
    out: List[Problem] = []
    _walk(_schema.MemoryArchitectureConfig, raw, (), (), out)
    _check_values(raw, out)
    return out
