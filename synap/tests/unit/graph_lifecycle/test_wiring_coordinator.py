"""Every KU path that retires a memory releases its graph support (W5).

The release waits until Stage 4's graph write (dossier 5.2, ordering note), so
the tests check both halves: the retirement queues it, `release_pending_graph`
applies it with the right mode, reason and successor, and the query cache is
dropped.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.agents.ingestion.storage import coordinator as coord_mod
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator


class FakeLifecycle:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def release(self, scope, memory_ids, *, mode, reason, superseded_by=None,
                      redact=False):
        self.calls.append(dict(ids=list(memory_ids), mode=mode, reason=reason,
                               superseded_by=superseded_by, redact=redact))
        if self.fail:
            raise RuntimeError("neo4j down")
        return SimpleNamespace(edges_pruned=1, edges_retired=1, edges_deleted=0,
                               nodes_pruned=0, nodes_retired=0, nodes_deleted=0)


class FakeRdbms:
    def __init__(self, accepts_enqueue=False):
        self.retired = []
        if accepts_enqueue:
            self.retire_memory_with_audit = self._retire_enqueue

    async def retire_memory_with_audit(self, *, memory_id, action, scope, previous_content,
                                       content=None, memory_type=None, superseded_by=None,
                                       replaced_by_memory_id=None, correlation_id=None,
                                       ingestion_id=None, reason=None,
                                       deprecate_with_link=False, link_kind="DEPRECATES",
                                       link_to_memory_id=None, require_retire=True):
        self.retired.append((memory_id, action, superseded_by, None))
        return True

    async def _retire_enqueue(self, *, memory_id, action, scope, previous_content,
                              superseded_by=None, enqueue_graph=False, **kw):
        self.retired.append((memory_id, action, superseded_by, enqueue_graph))
        return True


class FakeVector:
    def __init__(self):
        self.deleted = []

    async def delete_memory_vectors(self, memory_ids, scope):
        self.deleted.extend(memory_ids)


def _scope():
    return SimpleNamespace(client_id=uuid.uuid4(), instance_id="inst_x",
                           customer_id=uuid.uuid4(), user_id=uuid.uuid4())


def _coordinator(rdbms=None, lifecycle=None):
    c = StorageCoordinator(rdbms=rdbms or FakeRdbms(), graph=object(), vector=FakeVector(),
                           embedding_client=None)
    c._graph_lifecycle = lifecycle or FakeLifecycle()
    return c


@pytest.fixture
def cache_calls(monkeypatch):
    calls = []

    async def fake_invalidate(client_id):
        calls.append(client_id)

    monkeypatch.setattr(coord_mod, "_invalidate_query_cache", fake_invalidate)
    return calls


@pytest.fixture
def lifecycle_on(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_LIFECYCLE", "true")
    # KU guards read their own flags; keep them at their defaults (off).
    monkeypatch.delenv("SYNAP_KU_RETIRED_VALUE_GUARD", raising=False)
    monkeypatch.delenv("SYNAP_KU_VALUE_SIBLING_SWEEP", raising=False)


def _supersede(old, new):
    return SimpleNamespace(old_memory_id=old, new_memory_id=new,
                           old_memory_content="I work at Acme Corp",
                           new_memory_content="I now work at Globex", reason="job change")


async def test_supersession_queues_retire_and_applies_after_graph_write(lifecycle_on, cache_calls):
    lc = FakeLifecycle()
    c = _coordinator(lifecycle=lc)
    scope = _scope()
    n = await c.execute_supersessions([_supersede("m1", "m2")], scope, ingestion_id="ing1")
    assert n == 1
    # Not yet: the new memory's edges are written after KU (Stage 4 ordering).
    assert lc.calls == []
    assert [p.memory_id for p in c.pending_graph_releases("ing1")] == ["m1"]

    released = await c.release_pending_graph("ing1")
    assert released == 1
    assert lc.calls == [dict(ids=["m1"], mode="retire", reason="superseded",
                             superseded_by="m2", redact=False)]
    assert c.pending_graph_releases("ing1") == []
    # Supersession and the release both drop the client's cached bundles.
    assert str(scope.client_id) in cache_calls


async def test_supersession_writes_outbox_row_when_rdbms_supports_it(lifecycle_on, cache_calls):
    rdbms = FakeRdbms(accepts_enqueue=True)
    c = _coordinator(rdbms=rdbms)
    await c.execute_supersessions([_supersede("m1", "m2")], _scope(), ingestion_id="ing1")
    assert rdbms.retired == [("m1", "SUPERSEDE", "m2", True)]


async def test_flag_off_queues_nothing(monkeypatch, cache_calls):
    monkeypatch.delenv("SYNAP_GRAPH_LIFECYCLE", raising=False)
    monkeypatch.delenv("SYNAP_GRAPH_LIFECYCLE_INSTANCES", raising=False)
    lc = FakeLifecycle()
    c = _coordinator(lifecycle=lc)
    await c.execute_supersessions([_supersede("m1", "m2")], _scope(), ingestion_id="ing1")
    assert c.pending_graph_releases("ing1") == []
    assert await c.release_pending_graph("ing1") == 0
    assert lc.calls == []


async def test_ku_delete_retires_without_successor(lifecycle_on, cache_calls):
    lc = FakeLifecycle()
    c = _coordinator(lifecycle=lc)
    action = SimpleNamespace(old_memory_id="m9", old_memory_content="I am vegan",
                             reason="user retracted")
    assert await c.execute_deletions([action], _scope(), ingestion_id="ing2") == 1
    await c.release_pending_graph("ing2")
    assert lc.calls == [dict(ids=["m9"], mode="retire", reason="unsupported",
                             superseded_by=None, redact=False)]


async def test_every_retire_site_goes_through_the_release_wrapper():
    """No KU path may call retire_memory_with_audit directly (it would skip the
    graph release). Guards against a new path added later."""
    import inspect
    src = inspect.getsource(coord_mod.StorageCoordinator)
    direct = src.count("self._rdbms.retire_memory_with_audit(")
    # The single direct call is the one inside _retire_and_release itself.
    assert direct == 1
    assert src.count("await self._retire_and_release(") >= 7


async def test_release_failure_is_reported_not_raised(lifecycle_on, cache_calls):
    lc = FakeLifecycle(fail=True)
    c = _coordinator(lifecycle=lc)
    await c.execute_supersessions([_supersede("m1", "m2")], _scope(), ingestion_id="ing3")
    assert await c.release_pending_graph("ing3") == 0
    assert len(lc.calls) == 1


async def test_attach_filters_dead_memories(monkeypatch, lifecycle_on):
    """A DLQ replay must not put a since-superseded id back on an edge."""
    from synap.cloud.shared import memory_liveness

    async def fake_rows(ids, session=None):
        return {"alive": SimpleNamespace(superseded_by=None, deprecated_at=None,
                                         deleted_at=None)}

    monkeypatch.setattr(memory_liveness, "load_supporter_rows", fake_rows)
    monkeypatch.setattr(memory_liveness, "is_alive", lambda row: row is not None)
    c = _coordinator()
    mems = [SimpleNamespace(memory_id="alive"), SimpleNamespace(memory_id="dead")]
    kept = await c._alive_memories(mems)
    assert [m.memory_id for m in kept] == ["alive"]
