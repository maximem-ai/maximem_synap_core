"""What the graph's own ChromaDB fetches ask for.

`_build_adjacency_matrix` is the other half of the graph arm: it loads the
entities, triples and memories that the BFS walks. It asked for them with
`_build_customer_scope_filter`, which names a customer and a user and has no
field for a rung between them. On a ladder of client / customer / practice /
care_team / user, every record filed at a practice or a care team was therefore
invisible to the graph while being perfectly readable through the vector arm.
The graph then answered a question about a smaller world, and nothing about that
looks broken from the outside.

Two things are pinned here, and the second matters as much as the first:

1. With a chain, the customer-level fetches carry the chain clause, so the graph
   pool is the set the read filter admits.
2. Without a chain, the emitted where-clauses are byte for byte what they were
   before, because that is every request from clients running
   the default three rungs while nested scoping is off.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from synap.cloud.application.client_api.retrieval_manager import (
    MACARetrievalSettings,
    RetrievalManager,
    _and_where,
)
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL, ScopeHierarchy

pytestmark = pytest.mark.asyncio

CLIENT = str(UUID("11111111-1111-5111-8111-111111111111"))
CUST = str(UUID("22222222-2222-5222-8222-222222222222"))
PRACTICE = str(UUID("33333333-3333-5333-8333-333333333333"))
TEAM = str(UUID("44444444-4444-5444-8444-444444444444"))
ME = str(UUID("66666666-6666-5666-8666-666666666666"))


def _rungs():
    pairs = [
        (CLIENT, "client", "root"),
        (CUST, "customer", "tenant_of_record"),
        (PRACTICE, "practice", None),
        (TEAM, "care_team", None),
        (ME, "user", "identity"),
    ]
    return [{"node_id": n, "level_key": k, "role": r, "roles": [r] if r else []}
            for n, k, r in pairs]


def _scope(with_chain: bool) -> ScopeHierarchy:
    s = ScopeHierarchy(instance_id=str(uuid4()), client_id=CLIENT,
                       customer_id=CUST, user_id=ME)
    if with_chain:
        s.chain_rungs = _rungs()
    return s


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def as_provider(self):
        v = AsyncMock()

        async def _get_all(**kw):
            self.calls.append({"collection_type": kw.get("collection_type"),
                               "scope_level": kw.get("scope_level"),
                               "filters": kw.get("filters"),
                               "limit": kw.get("limit")})
            return []

        v.get_all_from_collection = AsyncMock(side_effect=_get_all)
        return v

    def find(self, collection_type, scope_level):
        for c in self.calls:
            if c["collection_type"] == collection_type and c["scope_level"] == scope_level:
                return c
        raise AssertionError(f"no fetch for {collection_type}/{scope_level}")


async def _run(with_chain: bool):
    rec = _Recorder()
    m = RetrievalManager(vector_provider=rec.as_provider(), postgres_provider=AsyncMock())
    m._current_maca = MACARetrievalSettings()
    await m._build_adjacency_matrix(_scope(with_chain))
    return rec


async def test_with_a_chain_the_customer_fetches_carry_the_chain():
    rec = await _run(with_chain=True)

    for collection in ("entities", "triples"):
        where = rec.find(collection, "customer")["filters"]
        assert "$or" in where, f"{collection} fetch lost the chain clause"
        node_clause = where["$or"][0]
        assert node_clause == {"node_id": {"$in": [CLIENT, CUST, PRACTICE, TEAM, ME]}}


async def test_the_memories_fetch_keeps_the_chain_beside_its_own_conditions():
    """The `$or` must survive being combined with confidence and entity filters.

    `{**a, **b}` keeps one `$or` key and drops the other, and the one it drops
    is the scope predicate: the request would then read the whole collection
    filtered on nothing but confidence.
    """
    rec = await _run(with_chain=True)
    where = rec.find("memories", "customer")["filters"]

    assert set(where) == {"$and"}
    operands = where["$and"]
    assert {"confidence": {"$gte": 0.7}} in operands
    assert {"entities_mentioned": {"$ne": ""}} in operands
    assert any("$or" in o for o in operands), "the chain clause was dropped"


async def test_without_a_chain_nothing_changed():
    """Clients on the default three rungs, asserted not assumed."""
    rec = await _run(with_chain=False)
    flat = {"customer_id": CUST, "user_id": {"$in": [ME, CUSTOMER_SCOPE_SENTINEL]}}

    assert rec.find("entities", "customer")["filters"] == flat
    assert rec.find("triples", "customer")["filters"] == flat
    assert rec.find("memories", "customer")["filters"] == {
        **flat, "confidence": {"$gte": 0.7}, "entities_mentioned": {"$ne": ""},
    }
    # Client-level shared knowledge is never customer- or user-filtered, with a
    # chain or without one.
    assert rec.find("entities", "client")["filters"] == {}
    assert rec.find("memories", "client")["filters"] == {
        "confidence": {"$gte": 0.7}, "entities_mentioned": {"$ne": ""},
    }


async def test_and_where_merges_flat_clauses_and_never_drops_an_operator():
    """The combiner itself, because the leak it prevents is silent.

    A flat merge is what the call sites did before and is what keeps a
    no-chain request byte-identical. A clause carrying a top-level operator
    cannot be merged that way, and the failure is not an error: it is a
    where-clause that matches more than it should.
    """
    assert _and_where({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert _and_where({"a": 1}, None, {}) == {"a": 1}

    scoped = {"$or": [{"node_id": {"$in": ["x"]}}, {"customer_id": "c"}]}
    combined = _and_where(scoped, {"confidence": {"$gte": 0.7}})
    assert combined == {"$and": [scoped, {"confidence": {"$gte": 0.7}}]}
    # The thing that must never happen.
    assert combined != {**scoped, "confidence": {"$gte": 0.7}}
