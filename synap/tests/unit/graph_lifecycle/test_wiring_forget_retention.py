"""Forget erases graph support with redact; retention's graph retry erases (W5)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.memory_forget import soft_delete


class FakeLifecycle:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    async def release(self, scope, memory_ids, *, mode, reason, superseded_by=None,
                      redact=False):
        self.calls.append(dict(ids=list(memory_ids), mode=mode, reason=reason, redact=redact))
        if self.fail:
            raise RuntimeError("graph down")
        return SimpleNamespace()


class FakeSession:
    def __init__(self, ids):
        self.ids = ids

    async def execute(self, stmt, params=None):
        rows = [(i, "text of " + i) for i in self.ids]
        return SimpleNamespace(fetchall=lambda: rows)

    async def commit(self):
        return None


def _scope():
    return SimpleNamespace(client_id=uuid.uuid4(), instance_id="inst_x",
                           customer_id=uuid.uuid4(), user_id=uuid.uuid4())


@pytest.fixture
def lc(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_LIFECYCLE", "true")
    fake = FakeLifecycle()
    monkeypatch.setattr(soft_delete, "lifecycle_for", lambda graph: fake)
    return fake


async def test_forget_erases_with_redact(lc):
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    res = await soft_delete.soft_delete_memories(
        FakeSession(ids), scope=_scope(), memory_ids=ids, reason="user asked",
        graph=object())
    assert lc.calls == [dict(ids=ids, mode="erase", reason="deleted", redact=True)]
    assert res.still_present_in == []


async def test_forget_graph_failure_reported(lc):
    lc.fail = True
    ids = [str(uuid.uuid4())]
    res = await soft_delete.soft_delete_memories(
        FakeSession(ids), scope=_scope(), memory_ids=ids, reason="user asked",
        graph=object())
    assert res.still_present_in == ["graph"]


async def test_retention_graph_retry_erases_and_drops_cache(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_LIFECYCLE", "true")
    from synap.cloud.application.retention import sweep
    from synap.cloud.application.sdk_api import sdk_memories
    from synap.cloud.application.shared import providers as providers_mod

    fake = FakeLifecycle()
    cache = []

    async def inval(cid):
        cache.append(cid)
        return True

    monkeypatch.setattr(sdk_memories, "_graph_lifecycle", lambda g: fake)
    monkeypatch.setattr(sdk_memories, "_invalidate_query_cache", inval)
    monkeypatch.setattr(providers_mod, "get_cached_providers",
                        lambda: {"graph": object(), "vector": object()})

    deps = sweep._real_deps()
    cand = SimpleNamespace(memory_id=str(uuid.uuid4()), client_id=str(uuid.uuid4()),
                           customer_id=str(uuid.uuid4()), user_id=str(uuid.uuid4()))
    item = SimpleNamespace(candidate=cand, instance_id="inst_x", stores=["graph"])
    left = await deps.retry_cleanup(item)
    assert left == []
    assert fake.calls == [dict(ids=[cand.memory_id], mode="erase", reason="deleted",
                               redact=False)]
    assert cache == [cand.client_id]
