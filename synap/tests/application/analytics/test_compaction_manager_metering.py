"""CompactionManager metering wiring.

Most compactions are not explicit API calls — they are auto-triggered from
``conversation.tracker`` when a conversation crosses its threshold, and that
path has no ``client_id`` to pass (the ``conversation`` table has no such
column). Metering keyed on the caller-supplied ``client_id`` alone would
therefore miss exactly the compactions the metric is meant to count.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synap.cloud.application.compaction.manager import CompactionManager


@pytest.fixture
def manager():
    mgr = CompactionManager.__new__(CompactionManager)   # skip pass construction
    mgr._usage_tracker = MagicMock()
    mgr._usage_tracker.track_compaction = MagicMock()
    mgr._client_id_cache = {}
    return mgr


class _FakeSession:
    def __init__(self, row, recorder=None):
        self._row = row
        self._recorder = recorder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, _stmt, params=None):
        if self._recorder is not None:
            self._recorder.append(params)
        return SimpleNamespace(fetchone=lambda: self._row)


def _patch_instance_row(row, recorder=None):
    return patch(
        "synap.cloud.database.connection.get_session_factory",
        return_value=lambda: _FakeSession(row, recorder),
    )


class TestResolveClientId:
    @pytest.mark.asyncio
    async def test_resolves_owner_from_instance(self, manager):
        with _patch_instance_row(SimpleNamespace(client_id="cli_owner")):
            assert await manager._resolve_client_id("inst_1") == "cli_owner"

    @pytest.mark.asyncio
    async def test_unknown_instance_returns_none(self, manager):
        with _patch_instance_row(None):
            assert await manager._resolve_client_id("inst_missing") is None

    @pytest.mark.asyncio
    async def test_db_failure_returns_none_instead_of_raising(self, manager):
        """Attribution must never be able to fail a completed compaction."""
        with patch(
            "synap.cloud.database.connection.get_session_factory",
            side_effect=RuntimeError("db down"),
        ):
            assert await manager._resolve_client_id("inst_1") is None

    @pytest.mark.asyncio
    async def test_blank_instance_short_circuits(self, manager):
        assert await manager._resolve_client_id("") is None

    @pytest.mark.asyncio
    async def test_lookup_is_memoised(self, manager):
        recorder = []
        with _patch_instance_row(SimpleNamespace(client_id="cli_owner"), recorder):
            await manager._resolve_client_id("inst_1")
            await manager._resolve_client_id("inst_1")
        assert len(recorder) == 1


class TestTrackCompactionEvent:
    def test_schedules_count_only_tracking(self, manager):
        """No cost, no credit burn — the sibling llm_* events own the cost."""
        loop = MagicMock()
        with patch("asyncio.get_event_loop", return_value=loop):
            manager._track_compaction_event(
                client_id="cli_abc",
                instance_id="inst_1",
                input_tokens=700,
                output_tokens=300,
                latency_ms=1200,
                conversation_id="conv_1",
            )

        loop.create_task.assert_called_once()
        kwargs = manager._usage_tracker.track_compaction.call_args.kwargs
        assert kwargs["client_id"] == "cli_abc"
        assert kwargs["conversation_id"] == "conv_1"
        assert kwargs["latency_ms"] == 1200
        # The defaults are the safety property; assert they are not overridden.
        assert "attribute_cost" not in kwargs or kwargs["attribute_cost"] is False
        assert "charge_credits" not in kwargs or kwargs["charge_credits"] is False

    def test_scheduling_failure_is_swallowed(self, manager):
        with patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
            manager._track_compaction_event(
                client_id="cli_abc",
                instance_id="inst_1",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )  # must not raise
