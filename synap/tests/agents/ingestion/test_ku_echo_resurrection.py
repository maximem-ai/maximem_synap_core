"""Echo resurrection guard (ENABLE_ECHO_GUARD, WS2 4b).

The pair-level echo guard only fires on SUPERSEDE verdicts; most echoes are
classified UNRELATED (different attribute phrasing — "original email",
"billed to") and accumulate as current facts. This sweep runs after KU
actions in stage 4: an assistant-sourced memory whose concrete value token
is RETIRED for this user (present in superseded/deprecated rows, absent from
other current rows) is stored pre-deprecated — deprecate + vector delete +
NONE changelog row — so it never re-enters retrieval, while the row remains
for audit.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _mem(content: str, assertion_source=None) -> ExtractedMemory:
    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=MemoryType.FACTS,
        content=content,
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="doc-x",
        assertion_source=assertion_source,
    )


def _coordinator(value_retired: bool):
    rdbms = MagicMock()
    rdbms.value_retired_for_user = AsyncMock(return_value=value_retired)
    rdbms.set_deprecated_at = AsyncMock(return_value=True)
    rdbms.insert_memory_action = AsyncMock()
    vector = MagicMock()
    vector.delete_memory_vectors = AsyncMock()
    coord = StorageCoordinator.__new__(StorageCoordinator)
    coord._rdbms = rdbms
    attach_retire_shim(rdbms)
    coord._vector = vector
    return coord, rdbms, vector


ECHO = "The user's original email is casey.old@example.com"


@pytest.mark.asyncio
class TestResurrectionSweep:
    async def test_assistant_echo_of_retired_value_pre_deprecated(self, monkeypatch):
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        coord, rdbms, vector = _coordinator(value_retired=True)
        mem = _mem(ECHO, assertion_source="assistant")
        n = await coord.suppress_retired_value_echoes(
            memories=[mem], scope=_scope(),
            http_correlation_id="corr-1",
        )
        assert n == {str(mem.memory_id)}
        rdbms.set_deprecated_at.assert_awaited_once_with(memory_id=str(mem.memory_id))
        vector.delete_memory_vectors.assert_awaited_once()
        action_kwargs = rdbms.insert_memory_action.call_args.kwargs
        assert action_kwargs["action"] == "NONE"
        assert "echo-guard" in action_kwargs["reason"]
        assert "retired value" in action_kwargs["reason"]

    async def test_flag_off_is_noop(self, monkeypatch):
        monkeypatch.delenv("ENABLE_ECHO_GUARD", raising=False)
        coord, rdbms, _ = _coordinator(value_retired=True)
        n = await coord.suppress_retired_value_echoes(
            memories=[_mem(ECHO, assertion_source="assistant")], scope=_scope(),
        )
        assert not n
        rdbms.value_retired_for_user.assert_not_awaited()
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_user_sourced_memory_never_swept(self, monkeypatch):
        """The user re-stating an old value is a real assertion (maybe they
        changed back) — only assistant echoes are swept."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        coord, rdbms, _ = _coordinator(value_retired=True)
        n = await coord.suppress_retired_value_echoes(
            memories=[_mem(ECHO, assertion_source="user")], scope=_scope(),
        )
        assert not n
        rdbms.value_retired_for_user.assert_not_awaited()

    async def test_value_still_current_not_swept(self, monkeypatch):
        """Assistant restating a value that is still current elsewhere is a
        correct statement, not an echo of a retired one."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        coord, rdbms, _ = _coordinator(value_retired=False)
        n = await coord.suppress_retired_value_echoes(
            memories=[_mem(ECHO, assertion_source="assistant")], scope=_scope(),
        )
        assert not n
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_no_value_tokens_not_swept(self, monkeypatch):
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        coord, rdbms, _ = _coordinator(value_retired=True)
        n = await coord.suppress_retired_value_echoes(
            memories=[_mem("The user prefers email for updates", assertion_source="assistant")],
            scope=_scope(),
        )
        assert not n
        rdbms.value_retired_for_user.assert_not_awaited()

    async def test_already_skipped_by_ku_not_double_processed(self, monkeypatch):
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        coord, rdbms, _ = _coordinator(value_retired=True)
        mem = _mem(ECHO, assertion_source="assistant")
        n = await coord.suppress_retired_value_echoes(
            memories=[mem], scope=_scope(),
            already_skipped_ids={str(mem.memory_id)},
        )
        assert not n

    async def test_lookup_failure_leaves_memory_live(self, monkeypatch):
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        coord, rdbms, _ = _coordinator(value_retired=True)
        rdbms.value_retired_for_user = AsyncMock(side_effect=RuntimeError("db down"))
        n = await coord.suppress_retired_value_echoes(
            memories=[_mem(ECHO, assertion_source="assistant")], scope=_scope(),
        )
        assert not n
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_customer_scope_without_user_is_noop(self, monkeypatch):
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        coord, rdbms, _ = _coordinator(value_retired=True)
        scope = SimpleNamespace(
            client_id=uuid4(), instance_id=uuid4(),
            customer_id=uuid4(), user_id=None,
        )
        n = await coord.suppress_retired_value_echoes(
            memories=[_mem(ECHO, assertion_source="assistant")], scope=scope,
        )
        assert not n


class TestWiring:
    def test_stage4_calls_sweep_after_ku_actions(self):
        import inspect
        from synap.cloud.agents.ingestion.pipeline import stage4_organization
        src = inspect.getsource(stage4_organization)
        assert "suppress_retired_value_echoes" in src
        # after deletions, before merge synthesis
        assert src.index("execute_deletions") < src.index("suppress_retired_value_echoes")

    def test_stage4_excludes_suppressed_ids_from_vector_persist(self):
        """The vector upsert runs AFTER the sweep, so suppressed echoes must
        be excluded from downstream persistence via the KU skip set — the
        sweep's own vector delete is only a backstop (an ordering
        bug otherwise)."""
        import inspect
        from synap.cloud.agents.ingestion.pipeline import stage4_organization
        src = inspect.getsource(stage4_organization)
        assert "skipped_new_memory_ids |= set(suppressed_echo_ids" in src
        # the union must happen before the keep_mask filter is applied
        assert src.index("suppressed_echo_ids") < src.index("keep_mask")

    def test_sqlalchemy_lookup_checks_retired_and_live(self):
        import inspect
        from synap.cloud.agents.ingestion.storage import sqlalchemy_rdbms_wrapper
        src = inspect.getsource(sqlalchemy_rdbms_wrapper.SQLAlchemyRDBMSWrapper.value_retired_for_user)
        assert "superseded_by IS NOT NULL" in src
        assert "deprecated_at IS NULL" in src
        assert "ILIKE" in src
