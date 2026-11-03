"""Which config fields hold one of a fixed set of words.

The fields stay plain strings in the schema, because Cloud's stored configs may
hold a word outside the set and must keep loading. This table says which words
are understood, and one function finds the ones that are not:

* Core rejects them at load (``core_rules``), naming the field, the word and
  the allowed words.
* ``main`` logs one warning per config load (``MemoryArchitectureConfig``) and
  carries on.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Tuple, Type

from .enums import (
    ContextInjectionStyle,
    DecayFunction,
    GraphSchemaType,
    Priority,
    RetrievalModeName,
    UserContextIsolation,
)

#: Path to the field -> the enum holding the words it understands. ``*`` is one
#: list item.
VALUE_ENUMS: dict[Tuple[str, ...], Type[Enum]] = {
    ("retrieval_config", "ranking", "decay_function"): DecayFunction,
    ("retrieval_config", "default_mode"): RetrievalModeName,
    ("agent_hints", "retrieval_agent", "context_injection_style"): ContextInjectionStyle,
    ("agent_hints", "retrieval_agent", "anticipation_patterns", "*", "priority"): Priority,
    ("storage_architecture", "scoping", "user_context_isolation"): UserContextIsolation,
    ("storage_architecture", "graph_schemas", "*", "schema_type"): GraphSchemaType,
}

#: (path with list positions, the word written, the allowed words)
ValueProblem = Tuple[Tuple[Any, ...], str, Tuple[str, ...]]


def _find(node: Any, pattern: Tuple[str, ...], path: Tuple[Any, ...], enum: Type[Enum], out: List[ValueProblem]) -> None:
    head, rest = pattern[0], pattern[1:]
    if head == "*":
        if isinstance(node, list):
            for i, item in enumerate(node):
                _find(item, rest, path + (i,), enum, out)
        return
    if not isinstance(node, dict) or head not in node:
        return
    value = node[head]
    if rest:
        _find(value, rest, path + (head,), enum, out)
        return
    allowed = tuple(m.value for m in enum)
    # A non-string is the schema's own type error to report, not ours.
    if isinstance(value, str) and value not in allowed:
        out.append((path + (head,), value, allowed))


def find_value_problems(raw: Any) -> List[ValueProblem]:
    """Every field in ``raw`` (a config as a dict) that holds a word nothing understands."""
    out: List[ValueProblem] = []
    for pattern, enum in VALUE_ENUMS.items():
        _find(raw, pattern, (), enum, out)
    return out


def describe(problem: ValueProblem) -> str:
    path, value, allowed = problem
    dotted = ""
    for p in path:
        dotted += f"[{p}]" if isinstance(p, int) else (f".{p}" if dotted else str(p))
    return f"{dotted}: '{value}' is not an allowed value. Allowed values: {', '.join(allowed)}"
