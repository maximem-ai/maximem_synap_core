"""One user must not read another user's memories through the graph (Core walks one graph per scope).

Core turns SYNAP_GRAPH_PER_SCOPE on, so the graph arm walks the reader's own graph (named by the
user id), then the customer's graph, and the client's. This file drives the real
``RetrievalManager._neo4j_graph_discovery`` with a fake graph keyed by those ids, and asserts on
the items that come out. It needs nothing from a Cloud-only folder, so it ships to the Core tree.

Two separate defences are tested, because either one alone would hide a break in the other:

* the walk: a reader never asks for another user's graph;
* the gate: if another user's record does turn up in a graph the reader may walk (a customer graph
  that holds a record stamped with someone else's user id), it is refused.

Each test carries a positive control in the same assertion (the reader's own memory IS returned),
because "nothing leaked" and "nothing came back" look the same.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from synap.cloud.application.client_api.retrieval_manager import MACARetrievalSettings, RetrievalManager
from synap.cloud.shared.scope import ScopeHierarchy
from synap.providers.graph.models import GraphResult, Node

pytestmark = pytest.mark.asyncio

CLIENT = str(UUID("11111111-1111-5111-8111-111111111111"))
CUST = str(UUID("22222222-2222-5222-8222-222222222222"))
USER_A = str(UUID("66666666-6666-5666-8666-666666666666"))
USER_B = str(UUID("77777777-7777-5777-8777-777777777777"))
VEC = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _per_scope_on(monkeypatch):
    # Core sets this itself; set here so the file means the same thing in Cloud.
    monkeypatch.setenv("SYNAP_GRAPH_PER_SCOPE", "true")
    monkeypatch.delenv("SYNAP_GRAPH_PER_SCOPE_INSTANCES", raising=False)


def _entity(name: str, **props) -> Node:
    return Node(
        id=f"node-{name}", node_type="person",
        properties={"canonical_name": name, "entity_type": "person", "entity_id": f"e-{name}",
                    "embedding": list(VEC), **props},
    )


class _FakeGraph:
    """A graph per namespace id; records which namespaces were walked."""

    def __init__(self, by_namespace: dict):
        self._by_namespace = by_namespace
        self.walked: list[str] = []

    async def combined_name_traverse(self, *, namespace, anchor_names, **kw):
        self.walked.append(str(namespace))
        return self._by_namespace.get(str(namespace), GraphResult(nodes=[], edges=[]))


def _manager(graph) -> RetrievalManager:
    m = RetrievalManager(vector_provider=AsyncMock(), postgres_provider=AsyncMock(), graph_provider=graph)
    m._current_maca = MACARetrievalSettings()
    m._track_graph_queries = AsyncMock()
    return m


def _scope(user_id: str) -> ScopeHierarchy:
    return ScopeHierarchy(instance_id=str(uuid4()), client_id=CLIENT, customer_id=CUST, user_id=user_id)


async def _read(graph, user_id: str) -> set:
    items = await _manager(graph)._neo4j_graph_discovery(
        scope=_scope(user_id), anchor_names=["a-thing", "b-thing"],
        query_vectors=[np.array(VEC, dtype=np.float32)],
    )
    return {i.content for i in items}


def _two_users_graph() -> _FakeGraph:
    return _FakeGraph({
        USER_A: GraphResult(nodes=[_entity("a-thing", customer_id=CUST, user_id=USER_A)], edges=[]),
        USER_B: GraphResult(nodes=[_entity("b-thing", customer_id=CUST, user_id=USER_B)], edges=[]),
    })


async def test_a_memory_written_for_user_a_is_not_returned_for_user_b():
    graph = _two_users_graph()
    served = await _read(graph, USER_B)
    assert "b-thing (person)" in served, "the traversal returned nothing for B: the control failed"
    assert "a-thing (person)" not in served
    assert USER_A not in graph.walked, "user B's read walked user A's graph"


async def test_the_same_user_does_get_their_own_memory_back():
    graph = _two_users_graph()
    served = await _read(graph, USER_A)
    assert "a-thing (person)" in served
    assert "b-thing (person)" not in served
    assert USER_A in graph.walked and USER_B not in graph.walked


async def test_a_record_stamped_with_another_user_is_refused_even_in_a_graph_the_reader_may_walk():
    """The second defence: A's record sitting in the customer graph is still not B's to read."""
    graph = _FakeGraph({
        USER_B: GraphResult(nodes=[_entity("b-thing", customer_id=CUST, user_id=USER_B)], edges=[]),
        CUST: GraphResult(nodes=[_entity("a-thing", customer_id=CUST, user_id=USER_A)], edges=[]),
    })
    served = await _read(graph, USER_B)
    assert "b-thing (person)" in served, "the control failed: B's own memory did not come back"
    assert "a-thing (person)" not in served
    assert CUST in graph.walked, "the customer graph was not walked, so the gate was not exercised"


async def test_a_record_stamped_with_another_user_inside_the_readers_own_graph_is_refused():
    """A stale or mis-routed write: A's record sits in B's own graph. The gate still refuses it.

    The reader's own graph is the one place a "trust this graph" shortcut would be tempting, so it
    gets its own test: B's correctly stamped record in the same graph is the positive control.
    """
    graph = _FakeGraph({
        USER_B: GraphResult(nodes=[
            _entity("b-thing", customer_id=CUST, user_id=USER_B),
            _entity("a-thing", customer_id=CUST, user_id=USER_A),
        ], edges=[]),
    })
    served = await _read(graph, USER_B)
    assert "b-thing (person)" in served, "the control failed: B's own record did not come back"
    assert "a-thing (person)" not in served
    assert USER_B in graph.walked, "B's own graph was not walked, so the gate was not exercised"
