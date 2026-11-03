"""References from a graph node to the memories that mention it.

The graph keeps a memory's text on a relationship, and only when the text
names both of that relationship's entities. A memory that names one entity
("Leopold wants the lane report by Thursday") is on no relationship, so a
graph walk that reaches Leopold cannot return it.

With ``SYNAP_GRAPH_NODE_MEMORIES`` set, Stage 4 also writes the ids of the
memories that mention an entity onto that entity's node, and the graph walk
fetches those memories from the memories collection for every node it reaches.
They are references, not copies: a superseded memory is gone from the
collection and so from the walk, and the collection's own scope metadata says
who may read each one.

The writer's node carries none: nearly every memory of a user mentions "the
user", so the list would be the whole collection (services/writer_entity).

Off by default. Unset, nothing is written and nothing is read.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from synap.cloud.agents.ingestion.services import writer_entity

_FLAG_ENV = "SYNAP_GRAPH_NODE_MEMORIES"
_EDGE_FLAG_ENV = "SYNAP_GRAPH_EDGE_MEMORY_IDS"

PROPERTY = "memory_ids"
MAX_PER_NODE = 500      # newest kept
MAX_FETCH = 200         # per graph walk


def enabled(instance_id: Optional[Any] = None) -> bool:
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_FLAG_ENV, instance_id)


def references(memories: Iterable[Any], entities: Sequence[Any],
               node_ids: Dict[str, str]) -> Dict[str, List[str]]:
    """{graph node id: [memory id]} for this document. *node_ids* maps an
    extraction entity id to the node it was written to."""
    skip = {e.entity_id for e in entities if writer_entity.is_fixed(e)}
    out: Dict[str, List[str]] = {}
    for memory in memories:
        for entity_id in getattr(memory, "entities_mentioned", None) or []:
            node_id = node_ids.get(entity_id)
            if node_id and entity_id not in skip:
                out.setdefault(node_id, []).append(str(memory.memory_id))
    return out


def edges_enabled(instance_id: Optional[Any] = None) -> bool:
    """SYNAP_GRAPH_EDGE_MEMORY_IDS: a relationship keeps the ids of its
    memories, as a node does, and no longer a copy of their text and
    embedding. A copy grew without limit, was rewritten whole on every note
    and outlived the memory it copied."""
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_EDGE_FLAG_ENV, instance_id)


def walk_limit() -> int:
    """The most memories one graph walk hands on. The walk keeps the closest
    70% of what it found; with references that share of 200 is 140."""
    try:
        return max(5, int(os.environ.get("SYNAP_GRAPH_WALK_LIMIT", "60")))
    except ValueError:
        return 60


def wanted(nodes: Iterable[Any], seen: Iterable[str], edges: Iterable[Any] = ()) -> List[str]:
    """The memory ids the reached *edges* and *nodes* point to that the walk
    has not already found, capped.

    Give both in order of closeness to the question. A relationship's memories
    come first: they name both of its entities. Then the nodes take turns, each
    giving its newest first, so one much-mentioned node cannot use the whole cap.
    """
    have, out = set(seen), []

    def take(memory_id) -> None:
        if memory_id and memory_id not in have:
            have.add(memory_id)
            out.append(memory_id)

    for edge in edges:
        for memory_id in reversed((getattr(edge, "properties", None) or {}).get(PROPERTY) or []):
            take(memory_id)
    lists = [list(reversed((getattr(node, "properties", None) or {}).get(PROPERTY) or []))
             for node in nodes]
    for turn in range(max((len(ids) for ids in lists), default=0)):
        for ids in lists:
            if turn < len(ids):
                take(ids[turn])
        if len(out) >= MAX_FETCH:
            break
    return out[:MAX_FETCH]
