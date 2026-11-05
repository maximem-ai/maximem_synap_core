"""The graph arm may not return what the read filter refuses.

WHY THIS FILE EXISTS
--------------------
The read filter honours the whole ladder: `node_id IN chain`, with the flat
customer/user pair only as a transitional second arm. The graph arm never
learned any of that. Its Cypher scopes relationships by the id pair and does not
scope nodes at all, so a path that is legitimately reachable can still end on
another identity's entity, and that entity was turned into a ContextItem and
served.

Measured before this was fixed, with a positive control
in the same run because "nothing leaked" and "nothing came back" are the same
picture. Anchors belonging to one user, the real Cypher, the real registry
chain, and the real `build_scope_filter` compiled over the records that came
back:

    namespace  returned  admitted by the read filter  refused by it
    ns-a            26                          24                2
    ns-b            15                          10                5
    ns-c            31                          29                2

The refused rows carried another user's `node_id` and another user's `user_id`,
so both arms of the read filter say no, and the graph said yes.

The tests below are written against behaviour, not source text: each one drives
`_neo4j_graph_discovery` with a graph provider that returns a known mixture and
asserts on the ContextItems that come out. Every leak case carries its own
positive control in the same assertion, because a version of this file that only
checked "the sibling is absent" would pass just as happily against a traversal
that returned nothing at all.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from synap.cloud.application.client_api.retrieval_manager import (
    MACARetrievalSettings,
    RetrievalManager,
    _graph_scope_gate,
)
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL, ScopeHierarchy
from synap.providers.graph.models import Edge, GraphResult, Node

pytestmark = pytest.mark.asyncio

CLIENT = str(UUID("11111111-1111-5111-8111-111111111111"))
CUST = str(UUID("22222222-2222-5222-8222-222222222222"))
PRACTICE = str(UUID("33333333-3333-5333-8333-333333333333"))
TEAM_MINE = str(UUID("44444444-4444-5444-8444-444444444444"))
TEAM_THEIRS = str(UUID("55555555-5555-5555-8555-555555555555"))
ME = str(UUID("66666666-6666-5666-8666-666666666666"))
THEM = str(UUID("77777777-7777-5777-8777-777777777777"))

# One shared direction, so every node scores identically and nothing is dropped
# by the similarity cut. This test is about the scope gate; letting the score
# decide would make a green run mean two different things.
VEC = [1.0, 0.0, 0.0, 0.0]


def _rungs(node_ids, level_keys, roles):
    return [
        {"node_id": n, "level_key": k, "role": r, "roles": [r] if r else []}
        for n, k, r in zip(node_ids, level_keys, roles)
    ]


def _deep_scope() -> ScopeHierarchy:
    """A real five-rung ladder: client / customer / practice / care_team / user."""
    s = ScopeHierarchy(
        instance_id=str(uuid4()), client_id=CLIENT, customer_id=CUST, user_id=ME,
    )
    s.chain_rungs = _rungs(
        [CLIENT, CUST, PRACTICE, TEAM_MINE, ME],
        ["client", "customer", "practice", "care_team", "user"],
        ["root", "tenant_of_record", None, None, "identity"],
    )
    return s


def _flat_scope(user_id: str = ME) -> ScopeHierarchy:
    """No chain at all, which is every request on the default three rungs."""
    return ScopeHierarchy(
        instance_id=str(uuid4()), client_id=CLIENT, customer_id=CUST, user_id=user_id,
    )


def _entity(name: str, **props) -> Node:
    return Node(
        id=f"node-{name}",
        node_type="person",
        properties={"canonical_name": name, "entity_type": "person",
                    "entity_id": f"e-{name}", "embedding": list(VEC), **props},
    )


def _edge_with(memories, **props) -> Edge:
    import json

    return Edge(
        id=f"edge-{len(memories)}-{props.get('tag', '')}",
        from_id="node-mine", to_id="node-theirs", edge_type="KNOWS",
        properties={"memory_contents": json.dumps(memories), **props},
    )


def _memory(mid: str, content: str, **fields) -> dict:
    return {"memory_id": mid, "content": content, "memory_type": "facts",
            "confidence": 0.9, "embedding": list(VEC), **fields}


class _FakeGraph:
    """Returns a fixed result per namespace, and records what it was asked."""

    def __init__(self, by_namespace: dict):
        self._by_namespace = by_namespace
        self.calls: list[dict] = []

    async def combined_name_traverse(self, *, namespace, anchor_names, **kw):
        self.calls.append({"namespace": namespace, "anchors": list(anchor_names), **kw})
        return self._by_namespace.get(namespace, GraphResult(nodes=[], edges=[]))


def _manager(graph, maca: MACARetrievalSettings | None = None,
             visibility=None) -> RetrievalManager:
    m = RetrievalManager(vector_provider=AsyncMock(), postgres_provider=AsyncMock(),
                         graph_provider=graph)
    m._current_maca = maca or MACARetrievalSettings()
    if visibility is not None:
        m._current_visibility = visibility
    m._track_graph_queries = AsyncMock()
    return m


def _contents(items) -> set:
    return {i.content for i in items}


def _ids(items) -> set:
    return {i.item_id for i in items}


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------

async def test_an_entity_from_a_sibling_care_team_is_not_served():
    """The leak, in the shape it was observed.

    Both nodes sit in the requester's customer namespace, so the Cypher's
    namespace pin lets both through. One is filed at the requester's own care
    team, the other at a sibling team under the same practice. The read filter
    refuses the sibling: its node_id is not on the chain and its user_id is
    somebody else.
    """
    mine = _entity("mine", node_id=TEAM_MINE, customer_id=CUST, user_id=ME,
                   anc_client=CLIENT, anc_customer=CUST, anc_practice=PRACTICE,
                   anc_care_team=TEAM_MINE)
    theirs = _entity("theirs", node_id=TEAM_THEIRS, customer_id=CUST, user_id=THEM,
                     anc_client=CLIENT, anc_customer=CUST, anc_practice=PRACTICE,
                     anc_care_team=TEAM_THEIRS)
    graph = _FakeGraph({CUST: GraphResult(nodes=[mine, theirs], edges=[])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_deep_scope(), anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    served = _contents(items)
    # Positive control, in the same assertion: a run that returned nothing
    # would fail here rather than passing as "no leak".
    assert "mine (person)" in served, "the traversal returned nothing at all"
    assert "theirs (person)" not in served, "a sibling care team's entity was served"


async def test_a_memory_on_an_edge_belonging_to_another_identity_is_not_served():
    """The same question asked of the memories carried on relationship edges."""
    ok = _memory("m-ok", "mine", customer_id=CUST, user_id=ME)
    shared = _memory("m-shared", "customer shared",
                     customer_id=CUST, user_id=CUSTOMER_SCOPE_SENTINEL)
    leaked = _memory("m-leak", "theirs", customer_id=CUST, user_id=THEM)
    edge = _edge_with([ok, shared, leaked], customer_id=CUST, user_id=ME)
    anchor = _entity("mine", node_id=ME, customer_id=CUST, user_id=ME)
    graph = _FakeGraph({CUST: GraphResult(nodes=[anchor], edges=[edge])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_deep_scope(), anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    ids = _ids(items)
    assert {"m-ok", "m-shared"} <= ids, "the readable memories did not come back"
    assert "m-leak" not in ids, "another identity's memory was served"


async def test_an_entity_filed_at_a_rung_on_the_chain_is_served():
    """The other half of the boundary, which a filter can get wrong by being shut.

    A record filed at the practice, two rungs above the requester, is readable:
    it is on the chain. The flat customer/user pair cannot express that, so a
    gate built out of the pair alone would refuse it and the ladder would be
    a decoration.
    """
    practice_wide = _entity("practice fact", node_id=PRACTICE, customer_id=CUST,
                            user_id=THEM, anc_client=CLIENT, anc_customer=CUST,
                            anc_practice=PRACTICE)
    graph = _FakeGraph({CUST: GraphResult(nodes=[practice_wide], edges=[])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_deep_scope(), anchor_names=["practice fact"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    assert "practice fact (person)" in _contents(items)


async def test_the_client_namespace_is_not_gated():
    """Client-scope shared knowledge stays readable.

    Its records carry neither a customer nor a user by construction, and the
    vector arm reads the matching client-level collection with no scope
    predicate either. Gating this would be the same bug pointing the other way:
    the graph narrower than the read it mirrors.
    """
    shared = _entity("client wide")  # no customer_id, no user_id, no node_id
    graph = _FakeGraph({CLIENT: GraphResult(nodes=[shared], edges=[])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_deep_scope(), anchor_names=["client wide"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    assert "client wide (person)" in _contents(items)


async def test_a_graph_node_with_no_user_is_customer_shared_not_refused():
    """The graph omits a key where the vector store writes the sentinel.

    `upsert_entity_node` sets `user_id` only when the write had one, and the
    Cypher reads that back as `rel.user_id IS NULL`. Treating the absence as an
    unknown owner would silently delete every customer-shared entity from the
    graph arm, which is a recall regression dressed as a security fix.
    """
    shared = _entity("customer wide", customer_id=CUST)  # no user_id key at all
    graph = _FakeGraph({CUST: GraphResult(nodes=[shared], edges=[])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_deep_scope(), anchor_names=["customer wide"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    assert "customer wide (person)" in _contents(items)


async def test_roll_up_off_means_customer_shared_is_refused():
    """The sentinel translation obeys the same switch the vector read obeys."""
    shared = _entity("customer wide", customer_id=CUST)
    mine = _entity("mine", customer_id=CUST, user_id=ME)
    graph = _FakeGraph({CUST: GraphResult(nodes=[shared, mine], edges=[])})
    maca = MACARetrievalSettings(roll_up_to_customer=False)

    items = await _manager(graph, maca)._neo4j_graph_discovery(
        scope=_flat_scope(), anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    served = _contents(items)
    assert "mine (person)" in served, "the traversal returned nothing at all"
    assert "customer wide (person)" not in served


async def test_a_visibility_policy_stops_the_arm_rather_than_leaking_past_it():
    """No graph record carries `instance_id`, so the clause cannot be written.

    Some clients have an `isolated` policy. Traversing anyway would
    serve them material from instances they declared invisible, while the vector
    arm beside it pinned the instance correctly.
    """
    mine = _entity("mine", customer_id=CUST, user_id=ME)
    graph = _FakeGraph({CUST: GraphResult(nodes=[mine], edges=[])})
    m = _manager(graph, visibility=frozenset({str(uuid4())}))

    items = await m._neo4j_graph_discovery(
        scope=_flat_scope(), anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    assert items == []
    assert graph.calls == [], "the store was queried despite the policy"


# ---------------------------------------------------------------------------
# What this does to the clients on the default three rungs
# ---------------------------------------------------------------------------

async def test_a_b2c_request_loses_nothing():
    """B2C files the identity on the tenant rung, so nothing is ever a sibling.

    Asserted rather than assumed: this is the shape that broke twice before,
    once because a reader checked `role` instead of `roles`.
    """
    scope = ScopeHierarchy(instance_id=str(uuid4()), client_id=CLIENT,
                           customer_id=CUST, user_id=CUST)
    scope.chain_rungs = _rungs([CLIENT, CUST], ["client", "customer"],
                               ["root", "tenant_of_record"])
    scope.chain_rungs[1]["roles"] = ["tenant_of_record", "identity"]
    own = _entity("mine", node_id=CUST, customer_id=CUST, user_id=CUST)
    shared = _entity("shared", customer_id=CUST, user_id=CUSTOMER_SCOPE_SENTINEL)
    graph = _FakeGraph({CUST: GraphResult(nodes=[own, shared], edges=[])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=scope, anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    assert {"mine (person)", "shared (person)"} <= _contents(items)


async def test_a_single_user_customer_loses_nothing():
    """The shape the instances that actually traverse the graph are in.

    Every populated namespace on those instances holds exactly one user,
    so there is no sibling for the gate to refuse and the served set is the same
    set as before.
    """
    a = _entity("a", customer_id=CUST, user_id=ME)
    b = _entity("b", customer_id=CUST, user_id=ME)
    c = _entity("c", customer_id=CUST, user_id=CUSTOMER_SCOPE_SENTINEL)
    graph = _FakeGraph({CUST: GraphResult(nodes=[a, b, c], edges=[])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_flat_scope(), anchor_names=["a"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    assert _contents(items) == {"a (person)", "b (person)", "c (person)"}


async def test_cross_user_retrieval_still_means_cross_user():
    """The escape hatch is a MACA setting, and the gate reads it from one place.

    Every `strict` instance in that state also carries
    `allow_cross_user_retrieval=True`, so a gate that ignored it would have
    changed what those clients are served.
    """
    mine = _entity("mine", customer_id=CUST, user_id=ME)
    theirs = _entity("theirs", customer_id=CUST, user_id=THEM)
    graph = _FakeGraph({CUST: GraphResult(nodes=[mine, theirs], edges=[])})
    maca = MACARetrievalSettings(allow_cross_user_retrieval=True)

    items = await _manager(graph, maca)._neo4j_graph_discovery(
        scope=_flat_scope(), anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    assert {"mine (person)", "theirs (person)"} <= _contents(items)


async def test_the_gate_is_the_read_filter_and_not_a_copy_of_it():
    """One predicate, two stores.

    The hand-rolled pair of comparisons this replaced was described in its own
    comment as mirroring `_build_customer_scope_filter`. Mirroring is the defect:
    the read filter grew a chain arm and the copy did not. So the gate is checked
    against the read filter's own compiler over the same records rather than
    against a list of expected answers written here.
    """
    from synap.cloud.application.client_api.retrieval_manager import (
        _nested_scope_predicate,
        VISIBILITY_ALL,
    )
    from synap.cloud.shared.scope_filter import to_python_predicate

    scope, maca = _deep_scope(), MACARetrievalSettings()
    gate = _graph_scope_gate(scope, maca, CUST)
    read = to_python_predicate(_nested_scope_predicate(scope, maca, VISIBILITY_ALL, None))

    records = [
        {"node_id": TEAM_MINE, "customer_id": CUST, "user_id": ME},
        {"node_id": TEAM_THEIRS, "customer_id": CUST, "user_id": THEM},
        {"node_id": PRACTICE, "customer_id": CUST, "user_id": THEM},
        {"customer_id": CUST, "user_id": ME},
        {"customer_id": CUST, "user_id": CUSTOMER_SCOPE_SENTINEL},
        {"customer_id": str(uuid4()), "user_id": ME},
    ]
    for r in records:
        assert gate(r) == read(r), f"gate and read filter disagree on {r}"
    # And the set is not trivially all-True or all-False.
    answers = {read(r) for r in records}
    assert answers == {True, False}


# ---------------------------------------------------------------------------
# Deletion, which is the same shape of question
# ---------------------------------------------------------------------------

async def test_a_deleted_memory_is_not_served_from_a_graph_edge():
    """`soft_delete_memory_from_edges` writes a flag nothing used to read.

    The SDK delete route marks the entry `is_deleted` on every edge that carries
    it, and the Neo4j arm served it anyway. No stored entries carry the
    flag today, so this closes the gap before the
    first deletion on a graph-enabled instance rather than after it.
    """
    live = _memory("m-live", "still here", customer_id=CUST, user_id=ME)
    gone = _memory("m-gone", "deleted", customer_id=CUST, user_id=ME,
                   is_deleted=True, is_current=False)
    edge = _edge_with([live, gone], customer_id=CUST, user_id=ME)
    graph = _FakeGraph({CUST: GraphResult(nodes=[_entity("mine", customer_id=CUST,
                                                         user_id=ME)], edges=[edge])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_flat_scope(), anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    ids = _ids(items)
    assert "m-live" in ids, "the traversal returned nothing at all"
    assert "m-gone" not in ids


async def test_a_flag_that_came_back_as_a_string_still_reads_correctly():
    """Neo4j stores these as the strings "True" and "False".

    A truthiness test on "False" is true, so the naive check drops every live
    memory instead of the deleted one. Both directions are asserted, because a
    check that only proved "deleted is dropped" would pass while everything else
    was being dropped with it.
    """
    live = _memory("m-live", "still here", customer_id=CUST, user_id=ME,
                   is_deleted="False", is_current="True")
    gone = _memory("m-gone", "deleted", customer_id=CUST, user_id=ME,
                   is_deleted="True", is_current="False")
    edge = _edge_with([live, gone], customer_id=CUST, user_id=ME)
    graph = _FakeGraph({CUST: GraphResult(nodes=[_entity("mine", customer_id=CUST,
                                                         user_id=ME)], edges=[edge])})

    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_flat_scope(), anchor_names=["mine"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )

    ids = _ids(items)
    assert "m-live" in ids, "the string \"False\" was read as deleted"
    assert "m-gone" not in ids


@pytest.fixture(autouse=True)
def _every_referenced_memory_is_alive(monkeypatch):
    """The graph arm asks Postgres whether each memory on a record is alive
    (SYNAP_GRAPH_SERVE_CHECKS_LIVENESS, forced on in Core). These tests are
    about the scope gate, so every memory they reference is alive."""
    from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import SupporterRow
    from synap.cloud.shared import memory_liveness

    async def _alive(ids, session=None):
        return {str(i): SupporterRow(id=str(i)) for i in ids}

    monkeypatch.setattr(memory_liveness, "load_supporter_rows", _alive)
