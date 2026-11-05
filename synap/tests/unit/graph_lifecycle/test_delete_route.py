"""The SDK delete route tells the truth about the graph (W5, W7).

- Deleting a memory erases its graph support through the lifecycle service and
  reports what changed; a failing graph step gives still_present_in=["graph"]
  and never fully_removed=True.
- An id that is not a memory but a graph edge or entity id is erased as that
  record only, and the live memories that still say it are returned.
- The client's query cache is dropped.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.sdk_api import sdk_memories
from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext


class FakeResult:
    def __init__(self, row=None, scalar=0):
        self._row, self._scalar = row, scalar

    def fetchone(self):
        return self._row

    def scalar(self):
        return self._scalar


class FakeSession:
    def __init__(self, row):
        self.row = row
        self.sql = []
        self.commits = 0

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        self.sql.append(sql)
        if "SELECT content, memory_type" in sql:
            return FakeResult(self.row)
        return FakeResult(None, 0)

    async def commit(self):
        self.commits += 1


class FakeLifecycle:
    def __init__(self, fail=False, erased=None):
        self.release_calls, self.erase_calls = [], []
        self.fail, self.erased = fail, erased

    async def release(self, scope, memory_ids, *, mode, reason, superseded_by=None,
                      redact=False):
        self.release_calls.append(dict(ids=list(memory_ids), mode=mode, reason=reason,
                                       redact=redact))
        if self.fail:
            raise RuntimeError("neo4j unavailable")
        return SimpleNamespace(edges_pruned=0, edges_retired=0, edges_deleted=1,
                               nodes_pruned=0, nodes_retired=0, nodes_deleted=1)

    async def erase_record(self, scope, record_id):
        self.erase_calls.append(record_id)
        return self.erased


class FakeVector:
    async def delete_memory_vectors(self, memory_ids, scope):
        return None


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_LIFECYCLE", "true")
    state = SimpleNamespace(lifecycle=FakeLifecycle(), cache=[], legacy=[])

    class LegacyGraph:
        async def soft_delete_memory_from_edges(self, memory_id, scope, redact=False):
            state.legacy.append(memory_id)
            return 0

    from synap.cloud.application.shared import providers as providers_mod
    monkeypatch.setattr(providers_mod, "get_cached_providers",
                        lambda: {"vector": FakeVector(), "graph": LegacyGraph()})
    monkeypatch.setattr(sdk_memories, "_graph_lifecycle", lambda g: state.lifecycle)

    async def fake_invalidate(client_id):
        state.cache.append(client_id)
        return True

    monkeypatch.setattr(sdk_memories, "_invalidate_query_cache", fake_invalidate)

    async def no_propagation(*a, **k):
        return None

    from synap.cloud.shared import edition
    monkeypatch.setattr(edition, "acall", no_propagation)
    return state


def _auth():
    return SDKAuthContext(client_id=str(uuid.uuid4()), instance_id="inst_x",
                          credential_id=uuid.uuid4(), correlation_id="")


def _row():
    return SimpleNamespace(content="I work at Acme Corp", memory_type="fact",
                           customer_id=uuid.uuid4(), user_id=uuid.uuid4(),
                           source_document_id=None)


async def test_delete_erases_graph_support_and_reports_it(wired):
    mid = str(uuid.uuid4())
    session = FakeSession(_row())
    answer = await sdk_memories.delete_memory(mid, _auth(), session)
    assert wired.lifecycle.release_calls == [dict(ids=[mid], mode="erase", reason="deleted",
                                                  redact=False)]
    assert wired.legacy == []  # the copied-text helper no longer decides
    assert answer.kind == "memory"
    assert answer.fully_removed is True
    assert answer.graph_released.edges_deleted == 1
    assert answer.graph_released.nodes_deleted == 1
    # Postgres delete committed before the graph erase; cache dropped.
    assert session.commits >= 1
    assert len(wired.cache) == 1


async def test_graph_failure_is_reported_never_fully_removed(wired):
    wired.lifecycle.fail = True
    answer = await sdk_memories.delete_memory(str(uuid.uuid4()), _auth(), FakeSession(_row()))
    assert answer.still_present_in == ["graph"]
    assert answer.fully_removed is False
    assert answer.graph_released is None


async def test_delete_by_graph_edge_id_erases_record_and_reports_assertors(wired, monkeypatch):
    edge_id = str(uuid.uuid4())
    m_live, m_dead = str(uuid.uuid4()), str(uuid.uuid4())
    wired.lifecycle.erased = SimpleNamespace(kind="relationship", memory_ids=[m_live, m_dead],
                                             edges_deleted=1, nodes_deleted=0)
    from synap.cloud.shared import memory_liveness

    async def rows(ids, session=None):
        return {m_live: SimpleNamespace(alive=True)}

    monkeypatch.setattr(memory_liveness, "load_supporter_rows", rows)
    monkeypatch.setattr(memory_liveness, "is_alive", lambda r: r is not None)

    answer = await sdk_memories.delete_memory(edge_id, _auth(), FakeSession(None))
    assert wired.lifecycle.erase_calls == [edge_id]
    assert wired.lifecycle.release_calls == []  # nothing else released
    assert answer.kind == "relationship"
    assert answer.still_asserted_by == [m_live]
    assert answer.fully_removed is False
    assert answer.graph_released.edges_deleted == 1
    assert len(wired.cache) == 1


async def test_unknown_id_is_still_404(wired):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as err:
        await sdk_memories.delete_memory(str(uuid.uuid4()), _auth(), FakeSession(None))
    assert err.value.status_code == 404


async def test_update_retires_old_support_with_reason_updated(wired, monkeypatch):
    mid = str(uuid.uuid4())

    class UpdSession(FakeSession):
        async def execute(self, stmt, params=None):
            sql = str(stmt)
            if "SELECT content, customer_id" in sql:
                return FakeResult(SimpleNamespace(
                    content="old", customer_id=uuid.uuid4(), user_id=uuid.uuid4(),
                    memory_type="fact", confidence=0.9, category=None, subcategory=None))
            return FakeResult(None, 0)

    async def fake_get(memory_id, auth, session):
        return {"id": memory_id}

    monkeypatch.setattr(sdk_memories, "get_memory", fake_get)
    req = sdk_memories.UpdateMemoryRequest(document="I now work at Globex", merge_strategy="replace")
    await sdk_memories.update_memory(mid, req, _auth(), UpdSession(None))
    assert wired.lifecycle.release_calls == [dict(ids=[mid], mode="retire", reason="updated",
                                                  redact=False)]
    assert len(wired.cache) == 1
