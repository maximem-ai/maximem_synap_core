"""One graph per scope (shared/scope_key, SYNAP_GRAPH_PER_SCOPE). Off by default."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.models.inputs import ScopeRung
from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper
from synap.cloud.agents.ingestion.storage.vector_wrapper import deterministic_entity_id, entity_id_scope
from synap.cloud.application.client_api.retrieval_manager import MACARetrievalSettings, RetrievalManager
from synap.cloud.shared import scope_key
from synap.cloud.shared.scope import ScopeHierarchy
from synap.providers.graph.models import GraphResult

CLIENT, INSTANCE, CUSTOMER = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
ANA, DEV = uuid.uuid4(), uuid.uuid4()


def _scope(user=None, level=None, chain=None):
    return Scope(client_id=CLIENT, instance_id=INSTANCE, customer_id=CUSTOMER, user_id=user,
                 scope_level=level or ("user" if user else "customer"), chain=chain)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_PER_SCOPE", "true")


def test_off_everything_is_the_customers_graph(monkeypatch):
    monkeypatch.delenv("SYNAP_GRAPH_PER_SCOPE", raising=False)
    wrapper = AsyncGraphWrapper(SimpleNamespace())
    assert wrapper._get_namespace(_scope(ANA)) == str(CUSTOMER)
    assert entity_id_scope(_scope(ANA)) == str(CUSTOMER)
    reader = ScopeHierarchy(instance_id=str(INSTANCE), client_id=str(CLIENT),
                            customer_id=str(CUSTOMER), user_id=str(ANA))
    assert scope_key.readable_scope_ids(reader) == [str(CUSTOMER), str(CLIENT)]


def test_a_document_goes_to_the_graph_of_its_own_scope(on):
    wrapper = AsyncGraphWrapper(SimpleNamespace())
    assert wrapper._get_namespace(_scope(ANA)) == str(ANA)
    assert wrapper._get_namespace(_scope(DEV)) == str(DEV)
    assert wrapper._get_namespace(_scope()) == str(CUSTOMER)
    client_scope = Scope(client_id=CLIENT, instance_id=INSTANCE, scope_level="client")
    assert wrapper._get_namespace(client_scope) == str(CLIENT)


def test_a_ladder_level_has_a_graph_of_its_own(on):
    team = uuid.uuid4()
    chain = [ScopeRung(node_id=CLIENT, level_key="client", role="root"),
             ScopeRung(node_id=CUSTOMER, level_key="company", role="tenant_of_record"),
             ScopeRung(node_id=team, level_key="team"),
             ScopeRung(node_id=ANA, level_key="person", role="identity")]
    assert scope_key.write_scope_id(_scope(ANA, level="team", chain=chain)) == str(team)
    assert scope_key.write_scope_id(_scope(ANA, level="user", chain=chain)) == str(ANA)


def test_a_reader_walks_their_own_graph_and_each_level_above(on):
    team = str(uuid.uuid4())
    reader = ScopeHierarchy(instance_id=str(INSTANCE), client_id=str(CLIENT),
                            customer_id=str(CUSTOMER), user_id=str(ANA))
    assert scope_key.readable_scope_ids(reader) == [str(ANA), str(CUSTOMER), str(CLIENT)]
    assert scope_key.readable_scope_ids(reader, roll_up=False) == [str(ANA), str(CLIENT)]
    reader.chain_rungs = [{"node_id": str(CLIENT)}, {"node_id": str(CUSTOMER)},
                          {"node_id": team}, {"node_id": str(ANA)}]
    assert scope_key.readable_scope_ids(reader) == [str(ANA), team, str(CUSTOMER), str(CLIENT)]
    # Dev's graph is on nobody's list but Dev's.
    assert str(DEV) not in scope_key.readable_scope_ids(reader)


def test_two_users_naming_one_entity_get_two_entity_vectors(on):
    ids = {deterministic_entity_id("Benedikt Aune", "person", str(CLIENT), entity_id_scope(_scope(u)))
           for u in (ANA, DEV, None)}
    assert len(ids) == 3


class _Graph:
    def __init__(self):
        self.calls = []

    async def combined_name_traverse(self, **kw):
        self.calls.append(kw)
        return GraphResult(nodes=[], edges=[], paths=[])


def _manager(graph, maca=None):
    m = RetrievalManager(vector_provider=AsyncMock(), postgres_provider=AsyncMock(), graph_provider=graph)
    m._current_maca = maca or MACARetrievalSettings()
    m._track_graph_queries = AsyncMock()
    return m


async def _walk(maca=None):
    graph = _Graph()
    reader = ScopeHierarchy(instance_id=str(INSTANCE), client_id=str(CLIENT),
                            customer_id=str(CUSTOMER), user_id=str(ANA))
    await _manager(graph, maca)._neo4j_graph_discovery(
        scope=reader, anchor_names=["Benedikt Aune"],
        query_vectors=[np.array([1.0, 0.0], dtype=np.float32)])
    return graph.calls


@pytest.mark.asyncio
async def test_walk_off_is_the_two_graphs_it_always_was(monkeypatch):
    monkeypatch.delenv("SYNAP_GRAPH_PER_SCOPE", raising=False)
    calls = await _walk()
    assert [c["namespace"] for c in calls] == [str(CUSTOMER), str(CLIENT)]
    assert calls[0]["user_id"] == str(ANA) and "across_customer" not in calls[0]


@pytest.mark.asyncio
async def test_walk_covers_own_graph_then_the_levels_above(on):
    calls = await _walk()
    assert [c["namespace"] for c in calls] == [str(ANA), str(CUSTOMER), str(CLIENT)]


@pytest.mark.asyncio
async def test_a_cross_user_reader_walks_every_graph_of_the_customer(on):
    maca = MACARetrievalSettings()
    maca.allow_cross_user_retrieval = True
    calls = await _walk(maca)
    assert calls[0]["across_customer"] is True and calls[0]["user_id"] is None
    assert calls[0]["customer_id"] == str(CUSTOMER)
    assert [c["namespace"] for c in calls] == [str(CUSTOMER), str(CLIENT)]
