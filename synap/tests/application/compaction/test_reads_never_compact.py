"""A read never starts a compaction (engine item E5).

From 4 Feb the read path compacted whenever a conversation had no summary or
one older than 15 minutes. The first read of a conversation therefore
compacted the user's first message alone, and that summary is where the
"verify the user's identity" asks came from (B1 in
docs/PLAYGROUND_AGENT_FIXES_ACTION_ITEMS.md). Compaction belongs to its own
triggers: thresholds, the idle sweep and session end.

Two holes, both pinned here:
  - ``RetrievalManager._get_compacted_context`` started a compaction;
  - ``CompactionManager.compact`` compacted anyway when the check said "not
    due" and nothing was stored yet, so no threshold stopped a first summary.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# Core compacts on the configured LLM; this gives it one (a no-op in Cloud). See tests/conftest.py.
pytestmark = pytest.mark.usefixtures("core_llm_env")

from synap.cloud.application.client_api import retrieval_manager as rm
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager

from .test_compaction_v3 import FIRST_OUTPUT, FIRST_TURNS, ScriptedLLM, manager_with, msgs


def _stale_summary():
    """A stored compaction two hours old: the read path used to recompact it."""
    return SimpleNamespace(metadata=SimpleNamespace(
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        compaction_id="cmp-1",
        source_message_range={},
    ))


async def _drain():
    """Let any fire-and-forget task run: the old trigger compacted in one."""
    for _ in range(5):
        await asyncio.sleep(0)


def _retrieval(stored):
    storage = MagicMock()
    storage.get_latest = AsyncMock(return_value=stored)
    storage.is_compaction_in_progress = AsyncMock(return_value=False)
    storage.load_conversation_messages = AsyncMock(return_value=msgs("hi"))
    manager = MagicMock()
    manager.get_or_create = AsyncMock()
    manager.compact = AsyncMock()
    return RetrievalManager(compaction_storage=storage, compaction_manager=manager), storage, manager


class TestTheReadPath:
    @pytest.mark.asyncio
    async def test_no_summary_starts_nothing(self):
        retrieval, storage, manager = _retrieval(None)
        assert await retrieval._get_compacted_context("conv-1", "inst-1", user_id="u1") is None
        await _drain()
        manager.get_or_create.assert_not_called()
        manager.compact.assert_not_called()
        storage.load_conversation_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_old_summary_is_served_and_starts_nothing(self):
        stale = _stale_summary()
        retrieval, storage, manager = _retrieval(stale)
        assert await retrieval._get_compacted_context("conv-1", "inst-1") is stale
        await _drain()
        manager.get_or_create.assert_not_called()
        manager.compact.assert_not_called()
        storage.load_conversation_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_storage_failure_serves_nothing(self):
        retrieval, storage, _ = _retrieval(None)
        storage.get_latest = AsyncMock(side_effect=RuntimeError("redis down"))
        assert await retrieval._get_compacted_context("conv-1", "inst-1") is None

    def test_the_trigger_is_gone(self):
        # Nothing on the read side can start one again by accident.
        assert not hasattr(RetrievalManager, "_trigger_async_compaction")
        assert not hasattr(RetrievalManager, "_is_compaction_stale")


class TestRecentTurns:
    """Without a read-path compaction, a long exchange of short messages can sit
    uncompacted past 20 turns. Its opening must not be cut off."""

    def test_a_short_tail_is_served_whole(self):
        tail = msgs(*["ok, thanks"] * 35)
        assert rm._uncompacted_tail(tail) == tail

    def test_a_long_tail_keeps_the_last_twenty(self):
        tail = msgs(*["word " * 300] * 35)  # ~300 tokens each, far past 2,000
        assert rm._uncompacted_tail(tail) == tail[-20:]

    def test_up_to_twenty_is_unchanged(self):
        tail = msgs(*["word " * 300] * 12)
        assert rm._uncompacted_tail(tail) == tail


class TestCompactWithNothingStored:
    @pytest.mark.asyncio
    async def test_not_due_and_nothing_stored_compacts_nothing(self, monkeypatch):
        llm = ScriptedLLM()  # any model call fails the test
        manager, storage = manager_with(llm, monkeypatch, existing=None, min_span=2_000)
        assert await manager.compact("c1", "inst", msgs(*FIRST_TURNS)) is None
        assert not llm.calls and not storage.save.await_count

    @pytest.mark.asyncio
    async def test_a_forced_first_compaction_still_runs(self, monkeypatch):
        manager, storage = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch, existing=None, min_span=2_000)
        result = await manager.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        assert result is not None and storage.save.await_count == 1

    @pytest.mark.asyncio
    async def test_a_due_first_compaction_still_runs(self, monkeypatch):
        # Over the message trigger (30) and the minimum span: due, so it runs.
        turns = [f"{t} " + "detail " * 80 for t in FIRST_TURNS] * 16  # 32 turns, ~2.7K tokens
        llm = ScriptedLLM(FIRST_OUTPUT, FIRST_OUTPUT, FIRST_OUTPUT, FIRST_OUTPUT)
        manager, storage = manager_with(llm, monkeypatch, existing=None, min_span=2_000)
        result = await manager.compact("c1", "inst", msgs(*turns))
        assert result is not None and llm.calls
