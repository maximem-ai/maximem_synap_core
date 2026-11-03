"""Which graph node an entity is written to, when entity resolution has an opinion.

A graph node is found by ``canonical_name``, and Neo4j enforces one node per
(canonical_name, namespace). On its own that makes the spelling the identity:
two different entities written the same way share a node, and so do two
mentions nobody has been able to tell apart yet.

Adjudicated resolution gives every entity an ``er_node_key``: the registry
entry's id once it is matched or registered, or a provisional key while it is
ambiguous. The node keeps that key, and the name is only *claimed* by it. A
second entity with the same spelling and a different key gets a
disambiguated ``canonical_name`` instead of the first one's node.

The uniqueness constraint and the anchor lookup are untouched: every node
still has a unique ``canonical_name``, and every lookup here goes through it.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

NODE_KEY_PROPERTY = "er_node_key"
PROVISIONAL_PREFIX = "prov:"


def new_provisional_key() -> str:
    """Key for a mention that is not (yet) known to be any registry entry."""
    return f"{PROVISIONAL_PREFIX}{uuid4().hex}"


def is_provisional(node_key: Optional[str]) -> bool:
    return bool(node_key) and node_key.startswith(PROVISIONAL_PREFIX)


def node_name_candidates(base_name: str, entity_type: str, node_key: str) -> List[str]:
    """``canonical_name`` values to try for this entity, most natural first.

    The plain name, then the name with its type ("Paris (person)"), then the
    name with the tail of the key, which cannot collide. An unresolved mention
    skips the type form: it is not known to be a distinct entity of that type,
    only an unsettled one.
    """
    tail = node_key.rsplit(":", 1)[-1][-8:]
    if is_provisional(node_key):
        return [base_name, f"{base_name} [unresolved {tail}]"]
    return [base_name, f"{base_name} ({entity_type})", f"{base_name} [{tail}]"]


def may_use_node(node_key: str, existing_key: Optional[str]) -> bool:
    """Whether an existing node with the wanted name is this entity's node.

    Yes when it carries the same key. A node with no key was written before
    keys existed (or by the legacy path): a resolved entity adopts it, so old
    graphs keep one node per name, but an unresolved mention does not, since
    that would hand an established node to something still in doubt.
    """
    if existing_key == node_key:
        return True
    return existing_key is None and not is_provisional(node_key)
