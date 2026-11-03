"""Tests for the memory_actions changelog emitted by Stage 4 + SDK handlers.

These tests mock the RDBMS protocol and verify the StorageCoordinator
emits the right `insert_memory_action` calls on CREATE and SUPERSEDE paths.
Real DB I/O is covered by end-to-end verification, not here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.pipeline.knowledge_update import SupersessionAction
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim


pytestmark = pytest.mark.asyncio


def _make_scope() -> Scope:
    return Scope(
        client_id=uuid4(),
        instance_id=uuid4(),
        customer_id=uuid4(),
        user_id=uuid4(),
    )


def _make_memory(content: str = "user likes espresso") -> ExtractedMemory:
    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=MemoryType.FACTS,
        content=content,
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="doc-abc",
        entities_mentioned=[],
    )


def _make_coordinator(rdbms_mock: AsyncMock) -> StorageCoordinator:
    attach_retire_shim(rdbms_mock)
    return StorageCoordinator(
        rdbms=rdbms_mock,
        graph=None,
        vector=None,
        embedding_client=SimpleNamespace(dimension=768),
    )


async def test_persist_rdbms_emits_create_memory_actions_when_correlation_given():
    rdbms = AsyncMock()
    mem_id = uuid4()
    rdbms.insert_memory.return_value = mem_id

    coord = _make_coordinator(rdbms)
    scope = _make_scope()
    memory = _make_memory("new fact")

    await coord.persist_rdbms(
        memories=[memory],
        candidates=[],
        entities=[],
        scope=scope,
        document_id="doc-abc",
        http_correlation_id="req_abc_123",
        ingestion_id="ing-123",
    )

    rdbms.insert_memory_action.assert_awaited_once()
    kwargs = rdbms.insert_memory_action.await_args.kwargs
    assert kwargs["correlation_id"] == "req_abc_123"
    assert kwargs["ingestion_id"] == "ing-123"
    assert kwargs["action"] == "CREATE"
    assert kwargs["memory_id"] == str(mem_id)
    assert kwargs["content"] == "new fact"
    assert kwargs["previous_content"] is None
    assert kwargs["memory_type"] == MemoryType.FACTS.value


async def test_persist_rdbms_skips_changelog_when_no_correlation_id():
    rdbms = AsyncMock()
    rdbms.insert_memory.return_value = uuid4()

    coord = _make_coordinator(rdbms)
    await coord.persist_rdbms(
        memories=[_make_memory()],
        candidates=[],
        entities=[],
        scope=_make_scope(),
        document_id="doc-abc",
    )

    rdbms.insert_memory_action.assert_not_awaited()


async def test_execute_supersessions_emits_supersede_row_per_action():
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True

    coord = _make_coordinator(rdbms)
    scope = _make_scope()

    old_id_a, new_id_a = str(uuid4()), str(uuid4())
    old_id_b, new_id_b = str(uuid4()), str(uuid4())
    actions = [
        SupersessionAction(
            old_memory_id=old_id_a,
            old_memory_content="yoga 2x/week",
            new_memory_id=new_id_a,
            new_memory_content="yoga 3x/week",
            reason="frequency changed",
        ),
        SupersessionAction(
            old_memory_id=old_id_b,
            old_memory_content="lives in NYC",
            new_memory_id=new_id_b,
            new_memory_content="moved to SF",
            reason="location changed",
        ),
    ]

    superseded = await coord.execute_supersessions(
        actions=actions,
        scope=scope,
        http_correlation_id="req_super_xyz",
        ingestion_id="ing-xyz",
    )

    assert superseded == 2
    assert rdbms.insert_memory_action.await_count == 2
    first_call = rdbms.insert_memory_action.await_args_list[0].kwargs
    assert first_call["action"] == "SUPERSEDE"
    assert first_call["memory_id"] == old_id_a
    assert first_call["replaced_by_memory_id"] == new_id_a
    assert first_call["previous_content"] == "yoga 2x/week"
    assert first_call["content"] == "yoga 3x/week"
    assert first_call["correlation_id"] == "req_super_xyz"


async def test_execute_supersessions_skips_changelog_when_soft_delete_noop():
    """If the row was already superseded by an earlier action, we must NOT
    emit a memory_actions row — otherwise the UI shows a phantom deletion."""
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = False  # already superseded

    coord = _make_coordinator(rdbms)

    actions = [
        SupersessionAction(
            old_memory_id=str(uuid4()),
            old_memory_content="stale",
            new_memory_id=str(uuid4()),
            new_memory_content="fresh",
            reason="x",
        ),
    ]

    superseded = await coord.execute_supersessions(
        actions=actions,
        scope=_make_scope(),
        http_correlation_id="req_x",
        ingestion_id="ing-x",
    )

    assert superseded == 0
    rdbms.insert_memory_action.assert_not_awaited()


async def test_persist_rdbms_writes_memory_source_documents_for_create(monkeypatch):
    """KU v2: every CREATE writes a row to memory_source_documents
    so chunk-fetch fan-out works without a backfill gap."""
    rdbms = AsyncMock()
    mem_id = uuid4()
    rdbms.insert_memory.return_value = mem_id

    coord = _make_coordinator(rdbms)
    await coord.persist_rdbms(
        memories=[_make_memory()],
        candidates=[],
        entities=[],
        scope=_make_scope(),
        document_id="doc-abc",
        http_correlation_id="req_msd",
        ingestion_id="ing-msd",
    )

    rdbms.insert_memory_source_documents.assert_awaited_once()
    call = rdbms.insert_memory_source_documents.await_args
    assert call.kwargs["memory_id"] == str(mem_id)
    assert call.kwargs["source_document_ids"] == ["doc-abc"]


async def test_execute_supersessions_writes_memory_link_when_flag_on(monkeypatch):
    """Path A′ dual-write: with ENABLE_KU_DEPRECATE_WITH_LINK on, supersession
    sets deprecated_at and inserts a memory_links row alongside the legacy
    soft_delete. ChromaDB delete still happens (Path A′ retrieval-clean rule)."""
    monkeypatch.setenv("ENABLE_KU_DEPRECATE_WITH_LINK", "true")
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True
    rdbms.set_deprecated_at.return_value = True

    coord = _make_coordinator(rdbms)
    old_id, new_id = str(uuid4()), str(uuid4())
    actions = [
        SupersessionAction(
            old_memory_id=old_id, old_memory_content="old",
            new_memory_id=new_id, new_memory_content="new",
            reason="state revised",
        ),
    ]

    n = await coord.execute_supersessions(
        actions=actions, scope=_make_scope(),
        http_correlation_id="req_ku", ingestion_id="ing-ku",
    )

    assert n == 1
    # Both legacy and new RDBMS writes happened
    rdbms.soft_delete_memory.assert_awaited_once()
    rdbms.set_deprecated_at.assert_awaited_once_with(memory_id=old_id)
    rdbms.insert_memory_link.assert_awaited_once()
    link_call = rdbms.insert_memory_link.await_args.kwargs
    assert link_call["from_memory_id"] == old_id
    assert link_call["to_memory_id"] == new_id
    assert link_call["kind"] == "DEPRECATES"
    assert link_call["reason"] == "state revised"


async def test_execute_supersessions_skips_link_when_flag_off(monkeypatch):
    """Without the flag, supersession falls back to legacy soft_delete only —
    no deprecated_at, no memory_links. Backward-compat path until lifecycle-aware retrieval
    ships."""
    monkeypatch.delenv("ENABLE_KU_DEPRECATE_WITH_LINK", raising=False)
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True

    coord = _make_coordinator(rdbms)
    actions = [
        SupersessionAction(
            old_memory_id=str(uuid4()), old_memory_content="o",
            new_memory_id=str(uuid4()), new_memory_content="n",
            reason="r",
        ),
    ]

    await coord.execute_supersessions(
        actions=actions, scope=_make_scope(),
        http_correlation_id="req", ingestion_id="ing",
    )

    rdbms.set_deprecated_at.assert_not_awaited()
    rdbms.insert_memory_link.assert_not_awaited()
