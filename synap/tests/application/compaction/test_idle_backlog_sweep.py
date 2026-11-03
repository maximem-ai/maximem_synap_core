"""After a restart, idle conversations of any age that still hold unflushed content are
flushed: in pages, in the background, never in the way of start-up."""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synap.cloud.application.compaction import storage as storage_mod
from synap.cloud.application.compaction.storage import CompactionStorage
from synap.cloud.application.compaction.worker import CompactionWorker

NOW = datetime.now(timezone.utc)


def _conv(n, age_hours=48):
    return {"conversation_id": f"c{n:03d}", "instance_id": "inst-1", "user_id": "u", "customer_id": "cu",
            "client_id": "cl", "last_message_at": NOW - timedelta(hours=age_hours, minutes=n)}


def _worker(pages, size=2):
    """A worker whose storage hands out the given pages one after another."""
    w = CompactionWorker(backlog_page_size=size, backlog_pause_seconds=0)
    w._running = True
    w.storage = MagicMock()
    w.storage.get_unflushed_idle_conversations = AsyncMock(side_effect=list(pages) + [[]])
    w.storage.set_compaction_in_progress = AsyncMock(return_value=True)
    w.storage.clear_compaction_in_progress = AsyncMock()
    w.storage.get_latest = AsyncMock(return_value=None)
    w._load_messages = AsyncMock(return_value=["m"])
    w._flush_promotion = AsyncMock()
    w._get_instance_thresholds = AsyncMock(return_value=(1, 1, 1, 5))
    w._compact_conversation = AsyncMock()
    return w


@pytest.mark.asyncio
async def test_old_idle_conversations_are_flushed_a_page_at_a_time():
    pages = [[_conv(1), _conv(2)], [_conv(3), _conv(4)], [_conv(5)]]
    w = _worker(pages)
    with patch("synap.cloud.application.compaction.instance_config.automatic_compaction_enabled",
               AsyncMock(return_value=True)):
        await w._sweep_backlog()
    flushed = [c.args[0]["conversation_id"] for c in w._flush_promotion.await_args_list]
    assert flushed == ["c001", "c002", "c003", "c004", "c005"]
    assert w._compact_conversation.await_count == 5
    calls = w.storage.get_unflushed_idle_conversations.await_args_list
    assert len(calls) == 3  # the short last page ends it
    assert all(c.args[2] == 2 for c in calls)  # never more than a page at once
    before = calls[0].args[0]
    assert before <= datetime.now(timezone.utc) - timedelta(minutes=29)
    assert calls[0].args[1] is None
    assert calls[1].args[1] == (pages[0][-1]["last_message_at"], "c002")
    assert calls[2].args[1] == (pages[1][-1]["last_message_at"], "c004")


@pytest.mark.asyncio
async def test_a_page_is_finished_before_the_next_is_read():
    order = []
    pages = [[_conv(1), _conv(2)], [_conv(3)]]
    w = _worker(pages)
    real = w.storage.get_unflushed_idle_conversations

    async def read(*a):
        order.append("read")
        return await real(*a)

    async def compact(**kw):
        await asyncio.sleep(0)
        order.append("compact")
    w.storage.get_unflushed_idle_conversations = read
    w._compact_conversation = compact
    with patch("synap.cloud.application.compaction.instance_config.automatic_compaction_enabled",
               AsyncMock(return_value=True)):
        await w._sweep_backlog()
    assert order == ["read", "compact", "compact", "read", "compact"]


@pytest.mark.asyncio
async def test_a_failing_read_ends_the_sweep_quietly():
    w = _worker([])
    w.storage.get_unflushed_idle_conversations = AsyncMock(side_effect=RuntimeError("db down"))
    await w._sweep_backlog()  # does not raise


@pytest.mark.asyncio
async def test_start_runs_the_backlog_in_the_background_and_stop_cancels_it():
    w = CompactionWorker(check_interval_seconds=0)
    gate = asyncio.Event()

    async def slow_backlog():
        await gate.wait()

    async def one_round():
        w._running = False

    w._sweep_backlog = slow_backlog
    w._check_and_compact = one_round
    with patch("synap.cloud.application.compaction.worker.get_compaction_manager", return_value=MagicMock()):
        await asyncio.wait_for(w.start(), timeout=2)  # start-up is not held by the backlog
    assert w._backlog_task is not None and not w._backlog_task.done()
    await w.stop()
    assert w._backlog_task is None


def _session_factory(seen):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt, params):
            seen.append((str(stmt), params))
            return MagicMock(fetchall=lambda: [])
    return lambda: Session()


@pytest.mark.asyncio
async def test_the_query_has_no_age_floor_and_asks_only_for_unflushed_content(monkeypatch):
    seen = []
    monkeypatch.setattr(storage_mod, "get_session_factory", lambda: _session_factory(seen))
    st = CompactionStorage.__new__(CompactionStorage)
    before = NOW - timedelta(minutes=30)
    await st.get_unflushed_idle_conversations(before, None, 10)
    await st.get_unflushed_idle_conversations(before, (NOW, "c9"), 10)
    sql, params = seen[0]
    assert "last_message_at < :before" in sql and "last_message_at >=" not in sql
    assert "promoted_message_count < c.message_count" in sql and "compacted_contexts" in sql
    assert "LIMIT :lim" in sql and params == {"before": before, "lim": 10}
    assert ":after_at" not in sql
    sql2, params2 = seen[1]
    assert ":after_at" in sql2 and params2["after_at"] == NOW and params2["after_id"] == "c9"
