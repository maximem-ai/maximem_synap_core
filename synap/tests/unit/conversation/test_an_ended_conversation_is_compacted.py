"""A conversation that ends is compacted then, not at the idle sweep.

A turn reaches long-term memory when its conversation is compacted. Before
this, a conversation that ended under its thresholds waited for the idle
sweep, so a user who opened a new conversation straight away was asked to
repeat what they had just said.

Two things start the compaction: the client ending the session over gRPC, and
the same user opening a new conversation. Both go through
``ConversationTracker.compact_on_session_end``.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synap.cloud.application.compaction import CompactionTriggerType
from synap.cloud.application.conversation.tracker import ConversationTracker

CONV = "caadeafd-fdfc-4cc0-85e4-3f4de2aac07f"
PREVIOUS = "0b0b5f0e-6c57-4b7e-9d53-0a4a2f6f1c11"

# message_count, instance_id, user_id, customer_id, client_id, scope_path,
# transcript_hash
ROW = (6, "inst-1", "user-1", "cust-1", "cli-1", None, None)


def _factory(first):
    """A session factory whose every query returns *first* from .first()."""
    result = MagicMock()
    result.first.return_value = first
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm), session


def _manager(*, compacted_count=None, lock=True, messages=("m",) * 6):
    storage = MagicMock()
    if compacted_count is None:
        storage.get_latest = AsyncMock(return_value=None)
    else:
        existing = MagicMock()
        existing.metadata.source_message_range = {"message_count": compacted_count}
        storage.get_latest = AsyncMock(return_value=existing)
    storage.set_compaction_in_progress = AsyncMock(return_value=lock)
    storage.clear_compaction_in_progress = AsyncMock()
    storage.load_conversation_messages = AsyncMock(return_value=list(messages))
    manager = MagicMock()
    manager.storage = storage
    manager.compact = AsyncMock()
    return manager


async def _run(tracker, row=ROW, manager=None, flag=True, **kwargs):
    """Call compact_on_session_end and let the background compaction finish."""
    manager = manager or _manager()
    factory, _ = _factory(row)
    tracker._session_end_trigger_enabled = AsyncMock(return_value=flag)
    with patch(
        "synap.cloud.application.conversation.tracker.get_session_factory",
        return_value=factory,
    ), patch(
        "synap.cloud.application.compaction.get_compaction_manager",
        return_value=manager,
    ):
        started = await tracker.compact_on_session_end(CONV, **kwargs)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    return started, manager


@pytest.fixture
def tracker():
    return ConversationTracker()


class TestCompactOnSessionEnd:

    @pytest.mark.asyncio
    async def test_it_forces_a_compaction_with_the_conversations_own_ids(self, tracker):
        started, manager = await _run(tracker)

        assert started is True
        manager.compact.assert_awaited_once()
        kwargs = manager.compact.call_args.kwargs
        # Forced: a six-message conversation is under every threshold, and
        # without this the manager would hand back the existing compaction.
        assert kwargs["force"] is True
        assert kwargs["trigger_type"] is CompactionTriggerType.SESSION_END
        assert kwargs["conversation_id"] == CONV
        assert kwargs["instance_id"] == "inst-1"
        assert kwargs["user_id"] == "user-1"
        assert kwargs["customer_id"] == "cust-1"
        assert kwargs["client_id"] == "cli-1"
        manager.storage.clear_compaction_in_progress.assert_awaited_once_with(CONV)

    @pytest.mark.asyncio
    async def test_new_turns_after_an_earlier_compaction_are_compacted(self, tracker):
        started, manager = await _run(tracker, manager=_manager(compacted_count=4))
        assert started is True
        manager.compact.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_nothing_new_since_the_last_compaction_is_left_alone(self, tracker):
        """Forcing here would run the whole pipeline to promote no turns."""
        started, manager = await _run(tracker, manager=_manager(compacted_count=6))
        assert started is False
        manager.storage.set_compaction_in_progress.assert_not_called()
        manager.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_empty_conversation_is_left_alone(self, tracker):
        started, manager = await _run(tracker, row=(0,) + ROW[1:])
        assert started is False
        manager.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unknown_conversation_is_left_alone(self, tracker):
        started, manager = await _run(tracker, row=None)
        assert started is False
        manager.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_pushed_transcript_is_left_alone(self, tracker):
        """Its endpoint ingested it whole; promotion would skip it anyway."""
        started, manager = await _run(tracker, row=ROW[:6] + ("abc123",))
        assert started is False
        manager.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_another_instances_conversation_is_refused(self, tracker):
        """The gRPC caller names the conversation, so ownership is checked."""
        started, manager = await _run(tracker, instance_id="inst-other")
        assert started is False
        manager.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_owning_instance_is_accepted(self, tracker):
        started, _ = await _run(tracker, instance_id="inst-1")
        assert started is True

    @pytest.mark.asyncio
    async def test_the_agents_config_can_turn_it_off(self, tracker):
        started, manager = await _run(tracker, flag=False)
        assert started is False
        manager.compact.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["0", "false", "OFF", "no"])
    async def test_the_deployment_can_turn_it_off(self, tracker, monkeypatch, value):
        monkeypatch.setenv("SYNAP_SESSION_END_COMPACTION", value)
        started, manager = await _run(tracker)
        assert started is False
        manager.compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_compaction_already_running_is_not_doubled(self, tracker):
        started, manager = await _run(tracker, manager=_manager(lock=False))
        assert started is False
        manager.compact.assert_not_called()
        # The lock is someone else's to release.
        manager.storage.clear_compaction_in_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failure_never_reaches_the_caller(self, tracker):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            side_effect=RuntimeError("db down"),
        ):
            assert await tracker.compact_on_session_end(CONV) is False


class TestTheConfigFlag:

    async def _flag(self, tracker, content):
        config = MagicMock()
        config.config_content = content
        repo = MagicMock()
        repo.get_active_for_instance = AsyncMock(return_value=config)
        factory, _ = _factory(None)
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=factory,
        ), patch(
            "synap.cloud.database.repositories.maca_config_repo.MACAConfigRepository",
            return_value=repo,
        ):
            return await tracker._session_end_trigger_enabled("inst-1")

    @pytest.mark.asyncio
    async def test_on_when_the_config_says_so(self, tracker):
        content = {"memory_extraction": {"triggers": {"on_session_end": True}}}
        assert await self._flag(tracker, content) is True

    @pytest.mark.asyncio
    async def test_off_when_the_config_says_so(self, tracker):
        content = {"memory_extraction": {"triggers": {"on_session_end": False}}}
        assert await self._flag(tracker, content) is False

    @pytest.mark.asyncio
    async def test_on_when_the_config_does_not_say(self, tracker):
        """The schema default is true; a missing key has not chosen false."""
        assert await self._flag(tracker, {"compaction_config": {}}) is True


class TestANewConversationEndsThePreviousOne:

    async def _ensure(self, tracker, first):
        factory, session = _factory(first)
        tracker._compact_previous_conversation = AsyncMock()
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=factory,
        ):
            await tracker.ensure_conversation(
                conversation_id=CONV,
                instance_id="inst-1",
                user_id="user-1",
                customer_id="cust-1",
            )
            await asyncio.sleep(0)
        return session

    @pytest.mark.asyncio
    async def test_the_turn_that_creates_a_conversation_fires_it(self, tracker):
        session = await self._ensure(tracker, (True,))
        tracker._compact_previous_conversation.assert_awaited_once_with(
            conversation_id=CONV,
            instance_id="inst-1",
            user_id="user-1",
            customer_id="cust-1",
        )
        # How the insert is told apart from the conflict update.
        assert "RETURNING (xmax = 0)" in str(session.execute.call_args[0][0])

    @pytest.mark.asyncio
    async def test_a_later_turn_that_updated_the_row_does_not(self, tracker):
        await self._ensure(tracker, (False,))
        tracker._compact_previous_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_later_turn_that_changed_nothing_does_not(self, tracker):
        """The conflict WHERE matched no row, so RETURNING gave none back."""
        await self._ensure(tracker, None)
        tracker._compact_previous_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_it_compacts_the_users_most_recent_other_conversation(self, tracker):
        factory, session = _factory((PREVIOUS,))
        tracker.compact_on_session_end = AsyncMock(return_value=True)
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=factory,
        ):
            await tracker._compact_previous_conversation(
                conversation_id=CONV,
                instance_id="inst-1",
                user_id="user-1",
                customer_id="cust-1",
            )

        tracker.compact_on_session_end.assert_awaited_once_with(
            PREVIOUS, instance_id="inst-1", reason="next_conversation",
        )
        params = session.execute.call_args[0][1]
        # Same user under the same customer on the same instance, and never
        # the conversation that was just created.
        assert params == {
            "instance_id": "inst-1",
            "user_id": "user-1",
            "customer_id": "cust-1",
            "conversation_id": CONV,
        }

    @pytest.mark.asyncio
    async def test_a_first_ever_conversation_has_nothing_to_compact(self, tracker):
        factory, _ = _factory(None)
        tracker.compact_on_session_end = AsyncMock()
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=factory,
        ):
            await tracker._compact_previous_conversation(
                conversation_id=CONV,
                instance_id="inst-1",
                user_id="user-1",
                customer_id="cust-1",
            )
        tracker.compact_on_session_end.assert_not_called()


class TestThePlaygroundAgentsSweepAfterOneMinute:

    def test_every_playground_agent_goes_idle_after_one_minute(self):
        """Both numbers: the sweep picks a conversation up after
        ``idle_minutes``, but the manager then declines to redo a recent
        compaction until ``recompaction_idle_minutes`` has passed."""
        import json
        import pathlib

        maca = (pathlib.Path(__file__).resolve().parents[3]
                / "cloud/application/playground/v2/maca")
        configs = sorted(maca.glob("*.json"))
        assert configs
        for path in configs:
            thresholds = json.loads(path.read_text(encoding="utf-8"))[
                "compaction_config"]["adaptive_thresholds"]
            assert thresholds["idle_minutes"] == 1, path.name
            assert thresholds["recompaction_idle_minutes"] == 1, path.name
