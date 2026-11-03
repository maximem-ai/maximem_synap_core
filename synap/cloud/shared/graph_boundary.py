"""The instance boundary on graph records: what is stamped, and who may read it.

WHY THIS MODULE EXISTS
----------------------
Vector records carry `instance_id`, and the cross-instance visibility policy is
enforced on them. Neo4j nodes and relationships carried no instance at all: the
graph was isolated per instance until the namespace moved from `instance_id` to
`customer_id` (2026-03), and nothing replaced that isolation. The graph arm
therefore went quiet for any client with an `isolated` policy, and three other
features each worked around the missing property in a different way.

This module is the one place that says what a graph record carries and how a
reader decides it may see one. The writer (`graph_wrapper`), the adapter and the
retrieval gate all import from here, so they cannot disagree about spelling.

THE MODEL
---------
Entity identity is `(canonical_name, _namespace)` and the namespace is the
customer, so two instances of one client with the same customer id share ONE
node. A single scalar `instance_id` would be overwritten by whichever instance
wrote last. So:

  node / edge          `instance_ids`  a LIST of every instance that has written
                                       it. Only ever added to.
  memory dict on edge  `instance_id`   a scalar: one memory has one writer.

A node or edge is visible to a viewer when ANY of its instances is visible. A
memory is visible when ITS instance is visible. A record with neither key is
unstamped (written before this existed) and is refused under a restrictive
policy, which is the fail-closed answer; under default-open nothing is asked.

Design: docs published as "Graph Instance Boundary".
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Optional

# Reserved property names. Spelled here so the write side and the read side
# cannot disagree.
INSTANCE_IDS = "instance_ids"   # list on nodes and relationships
INSTANCE_ID = "instance_id"     # scalar on the memory dicts carried on a relationship

# The most instances one record remembers. A client has a handful; this only
# stops a runaway list.
MAX_INSTANCES_PER_RECORD = 200

_STAMP_ENV = "SYNAP_GRAPH_STAMP_INSTANCE"
_FILTER_ENV = "SYNAP_GRAPH_INSTANCE_FILTER"
_DELETE_ENV = "SYNAP_GRAPH_INSTANCE_DELETE"
_FALSY = {"0", "false", "no", "off"}


def stamping_enabled() -> bool:
    """Whether new graph records are stamped. On unless switched off.

    Additive and inert until something reads it, so it is on by default and the
    variable is only a kill switch.
    """
    return os.environ.get(_STAMP_ENV, "").strip().lower() not in _FALSY


def filter_enabled(instance_id: Optional[Any] = None) -> bool:
    """Whether the graph arm enforces the instance policy for this instance.

    Off by default and per instance (`SYNAP_GRAPH_INSTANCE_FILTER`, or
    `SYNAP_GRAPH_INSTANCE_FILTER_INSTANCES`), because it can only be turned on
    for a client once that client's existing graph has been stamped: an
    unstamped record is refused under a restrictive policy.
    """
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_FILTER_ENV, instance_id)


def delete_enabled(instance_id: Optional[Any] = None) -> bool:
    """Whether a customer-called delete uses the instance list on graph records.

    Off by default (`SYNAP_GRAPH_INSTANCE_DELETE`). With it off, such a delete
    skips the graph whenever the person also has data in another instance, as it
    always has. With it on, the delete takes this instance off the records other
    instances still use and removes the rest. Like the read filter, it should be
    on only for a client whose graph has been stamped.
    """
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_DELETE_ENV, instance_id)


def instance_token(scope: Any) -> Optional[str]:
    """The id a record is stamped with: the same string the vector metadata uses
    (`str(scope.instance_id)`), which is also the form the visibility set is
    resolved to."""
    value = getattr(scope, "instance_id", None)
    return str(value) if value else None


def node_edge_properties(scope: Any) -> Dict[str, Any]:
    """Boundary keys for a node or relationship written under *scope*."""
    if not stamping_enabled():
        return {}
    token = instance_token(scope)
    return {INSTANCE_IDS: [token]} if token else {}


def memory_properties(scope: Any) -> Dict[str, Any]:
    """Boundary keys for a memory dict carried on a relationship."""
    if not stamping_enabled():
        return {}
    token = instance_token(scope)
    return {INSTANCE_ID: token} if token else {}


def split_instance_ids(properties: Optional[Mapping[str, Any]]):
    """(properties without the instance list, the list).

    The adapter uses this so the list is UNIONED into a record and never written
    over by a `SET n += $props`, which would drop every other instance.
    """
    props = dict(properties or {})
    raw = props.pop(INSTANCE_IDS, None)
    if raw is None:
        return props, []
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    ids = []
    for item in raw:
        if item is None:
            continue
        text = str(item)
        if text and text not in ids:
            ids.append(text)
    return props, ids


def record_instances(record: Mapping[str, Any]) -> Optional[frozenset]:
    """The instances a record belongs to, or None when it is unstamped."""
    listed = record.get(INSTANCE_IDS)
    if isinstance(listed, (list, tuple, set, frozenset)) and listed:
        return frozenset(str(i) for i in listed)
    scalar = record.get(INSTANCE_ID)
    if scalar:
        return frozenset({str(scalar)})
    return None


def record_visible(record: Mapping[str, Any], allowed: Optional[Iterable[str]]) -> bool:
    """May a viewer who can read *allowed* instances see this record?

    `allowed is None` means default-open and nothing is asked. Otherwise a
    record is visible when any of its instances is allowed; an unstamped record
    is not (fail closed).
    """
    if allowed is None:
        return True
    owners = record_instances(record)
    if owners is None:
        return False
    return not owners.isdisjoint(frozenset(str(a) for a in allowed))
