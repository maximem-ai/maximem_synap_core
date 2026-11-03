"""Which graph a record belongs to, and which graphs a reader may walk.

A graph node used to be one per name per CUSTOMER, stamped with the scope of
whoever wrote it last: two users naming the same person shared one node, and
the second write took it away from the first. The same held for a
relationship and for an entity vector.

With ``SYNAP_GRAPH_PER_SCOPE`` set, a record goes to the graph of the scope
its document was written at, so a node, a relationship and an entity vector
exist once per scope by construction:

    client-scope document     the client's graph      (as before)
    customer-scope document   the customer's graph    (as before)
    user-scope document       the user's own graph
    a ladder level            that level's own graph  (its scope node id)

A reader walks their own graph and the graph of every level above them, the
same set the read filter already allows: never a sibling, never a level below.

Off by default. Unset, both functions give what the code gave before.
"""

from __future__ import annotations

from typing import Any, List, Optional

_FLAG_ENV = "SYNAP_GRAPH_PER_SCOPE"


def per_scope_enabled(instance_id: Optional[Any] = None) -> bool:
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_FLAG_ENV, instance_id)


def _legacy_graph_id(scope: Any) -> str:
    level = getattr(scope, "scope_level", None)
    if level == "client" or not getattr(scope, "customer_id", None):
        return str(scope.client_id)
    return str(scope.customer_id)


def write_scope_id(scope: Any) -> str:
    """The id of the scope a document under *scope* (an ingestion Scope) is
    written at: the graph its records go to."""
    if not per_scope_enabled(getattr(scope, "instance_id", None)):
        return _legacy_graph_id(scope)
    customer_id = getattr(scope, "customer_id", None)
    user_id = getattr(scope, "user_id", None)
    level = getattr(scope, "scope_level", None) or (
        "user" if user_id else "customer" if customer_id else "client")
    if getattr(scope, "chain", None):
        # The rung this write lands on, by the same rule that stamps node_id
        # on the record itself.
        from synap.cloud.agents.ingestion.storage.vector_wrapper import _nested_scope_metadata
        node_id = _nested_scope_metadata(scope, level).get("node_id")
        if node_id:
            return str(node_id)
    if level == "client" or not customer_id:
        return str(scope.client_id)
    if level == "user" and user_id:
        return str(user_id)
    return str(customer_id)


def _rung_id(rung: Any) -> Optional[str]:
    value = rung.get("node_id") if isinstance(rung, dict) else getattr(rung, "node_id", None)
    return str(value) if value else None


def readable_scope_ids(scope: Any, roll_up: bool = True) -> List[str]:
    """The graphs a reader at *scope* may walk, nearest first: their own, then
    each level above, ending at the client's.

    *scope* is a retrieval ScopeHierarchy (``chain_rungs``) or an ingestion
    Scope (``chain``). With a ladder the chain's node ids come first; the three
    legacy ids follow, so records written before a ladder existed, or by a
    caller that resolved no chain, are still reached. ``roll_up=False`` is the
    instance setting that keeps a user out of the customer's shared records.
    """
    client_id = str(scope.client_id)
    customer_id = getattr(scope, "customer_id", None)
    user_id = getattr(scope, "user_id", None)
    if not per_scope_enabled(getattr(scope, "instance_id", None)):
        return [str(customer_id), client_id] if customer_id else [client_id]

    ids: List[str] = []
    rungs = getattr(scope, "chain_rungs", None) or getattr(scope, "chain", None) or []
    for rung in reversed(list(rungs)):
        ids.append(_rung_id(rung))
    if user_id:
        ids.append(str(user_id))
    if customer_id and (roll_up or not user_id):
        ids.append(str(customer_id))
    ids.append(client_id)
    out: List[str] = []
    for i in ids:
        if i and i not in out:
            out.append(i)
    return out
