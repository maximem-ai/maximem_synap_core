"""References from a graph node to the memories that mention it
(services/node_memories). Off by default."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from synap.cloud.agents.ingestion.services import node_memories as nm
from synap.cloud.agents.ingestion.services import writer_entity as we
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
from synap.cloud.shared.scope import ScopeHierarchy

VEC = [1.0, 0.0, 0.0]
SCOPE = ScopeHierarchy(instance_id=str(uuid.uuid4()), client_id=str(uuid.uuid4()),
                       customer_id=str(uuid.uuid4()), user_id=str(uuid.uuid4()))


def _memory(memory_id, mentions):
    return SimpleNamespace(memory_id=memory_id, entities_mentioned=mentions)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SYNAP_GRAPH_NODE_MEMORIES", raising=False)
    assert not nm.enabled()


def test_references_cover_one_entity_memories_and_skip_the_writer():
    boss = SimpleNamespace(entity_id="e1", properties={})
    writer = SimpleNamespace(entity_id="e2", properties={we.WRITER_PROPERTY: True})
    memories = [_memory("m1", ["e1"]), _memory("m2", ["e1", "e2"]), _memory("m3", [])]
    assert nm.references(memories, [boss, writer], {"e1": "n1", "e2": "n2"}) == {"n1": ["m1", "m2"]}


def test_wanted_leaves_out_what_the_walk_already_found():
    nodes = [SimpleNamespace(properties={nm.PROPERTY: ["m1", "m2"]}),
             SimpleNamespace(properties={nm.PROPERTY: ["m2", "m3"]}), SimpleNamespace(properties={})]
    assert nm.wanted(nodes, {"m1"}) == ["m2", "m3"]


def _manager(documents):
    vector = SimpleNamespace(_get_namespace=lambda scope, level: "ns",
                             _provider=SimpleNamespace(get=lambda namespace, ids: documents))
    return RetrievalManager(vector_provider=vector, postgres_provider=AsyncMock(), graph_provider=AsyncMock())


def _doc(doc_id, content, user_id):
    return SimpleNamespace(id=doc_id, content=content, embedding=VEC,
                           metadata={"user_id": user_id, "memory_type": "facts"})


@pytest.mark.asyncio
async def test_walk_returns_referenced_memories_the_reader_may_see(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_NODE_MEMORIES", "true")
    docs = [_doc("m1", "Halvard Finch wants the safety audit by Tuesday\n\n[Context: person:Halvard Finch]", "me"),
            _doc("m2", "Someone else's note about Halvard Finch", "other")]
    nodes = [SimpleNamespace(properties={nm.PROPERTY: ["m1", "m2"]})]
    out = await _manager(docs)._node_referenced_memories(
        SCOPE, nodes, set(), np.array(VEC, dtype=np.float32),
        lambda meta: meta.get("user_id") == "me", lambda *_: None)
    assert [(item.item_id, item.content, item.source) for _, item in out] == [
        ("m1", "Halvard Finch wants the safety audit by Tuesday", "neo4j")]


@pytest.mark.asyncio
async def test_walk_reads_nothing_when_off(monkeypatch):
    monkeypatch.delenv("SYNAP_GRAPH_NODE_MEMORIES", raising=False)
    nodes = [SimpleNamespace(properties={nm.PROPERTY: ["m1"]})]
    assert await _manager([_doc("m1", "x", "me")])._node_referenced_memories(
        SCOPE, nodes, set(), np.array(VEC, dtype=np.float32), None, lambda *_: None) == []
