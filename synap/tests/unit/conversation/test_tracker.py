"""Unit tests for ConversationTracker.

Tests use mocked database sessions to verify SQL generation, idempotency,
and error handling without requiring a live Postgres instance.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synap.cloud.application.conversation.tracker import ConversationTracker


@pytest.fixture
def tracker():
    return ConversationTracker()


# Valid UUID inputs so the tracker's _coerce_uuid() is an identity no-op.
# The product deterministically coerces any non-UUID id (e.g. "conv-1") into a
# UUIDv5, so tests that assert pass-through equality must supply real UUIDs.
CONV_UUID = "caadeafd-fdfc-4cc0-85e4-3f4de2aac07f"
SESS_UUID = "cfcd23db-c1c4-4523-ad59-f31906bef867"


@pytest.fixture
def mock_session():
    """Mock async SQLAlchemy session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    # A result is read synchronously. `ensure_conversation` asks whether its
    # upsert created the row; None is the "it did not" answer.
    session.execute.return_value.first = MagicMock(return_value=None)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    """Mock session factory that yields the mock session."""
    factory = MagicMock()

    # __call__ returns an async context manager yielding mock_session
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = cm

    return factory


@pytest.fixture
def mock_redis():
    """Mock async Redis client (rpush / expire are awaited)."""
    return AsyncMock()


@pytest.fixture
def tracker_with_redis(mock_redis):
    return ConversationTracker(redis_client=mock_redis)


# -----------------------------------------------------------------------
# ensure_conversation
# -----------------------------------------------------------------------

class TestEnsureConversation:

    @pytest.mark.asyncio
    async def test_inserts_conversation(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            result = await tracker.ensure_conversation(
                conversation_id=CONV_UUID,
                instance_id="inst-1",
                user_id="user-1",
                customer_id="cust-1",
            )

        assert result == CONV_UUID
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

        # The insert used to end `ON CONFLICT DO NOTHING`. It cannot any more:
        # a conversation row is created by the first turn, and the scope path
        # is not always known on that turn, so a later turn that DOES know it
        # has to be able to fill the column in. The upsert is deliberately
        # one-way, it only ever fills a NULL, so a second turn arriving with a
        # different path can never move a conversation out from under the
        # rung it was already filed at.
        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "ON CONFLICT" in sql_text
        assert "DO UPDATE" in sql_text
        assert "DO NOTHING" not in sql_text
        assert "conversation.scope_path IS NULL" in sql_text
        # The one-way guarantee moved out of the WHERE clause and into the SET
        # when customer_id gained the same self-heal: COALESCE fills only a
        # stored NULL, and COALESCE(NULL, NULL) is NULL, so a later turn that
        # omits the path still leaves it unset. Asserting the old
        # `EXCLUDED.scope_path IS NOT NULL` guard kept this red on main while
        # the property it protects was intact.
        assert "COALESCE(conversation.scope_path, EXCLUDED.scope_path)" in sql_text
        assert "SET scope_path = EXCLUDED.scope_path" not in sql_text

    @pytest.mark.asyncio
    async def test_passes_correct_params(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.ensure_conversation(
                conversation_id=CONV_UUID,
                instance_id="inst-xyz",
                user_id="user-42",
                customer_id=None,
            )

        call_args = mock_session.execute.call_args
        params = call_args[0][1]
        assert params["conversation_id"] == CONV_UUID
        assert params["instance_id"] == "inst-xyz"
        assert params["user_id"] == "user-42"
        assert params["customer_id"] is None
        # Present and None, not absent: the column exists on every insert and
        # a caller that does not know the rung writes NULL rather than
        # dropping the bind param and taking a missing-parameter error.
        assert "scope_path" in params
        assert params["scope_path"] is None


# -----------------------------------------------------------------------
# ensure_session
# -----------------------------------------------------------------------

class TestEnsureSession:

    @pytest.mark.asyncio
    async def test_auto_generates_deterministic_session_id(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            sid1 = await tracker.ensure_session("conv-1", "user-1")
            sid2 = await tracker.ensure_session("conv-1", "user-1")

        # Same conversation_id → same deterministic session_id
        assert sid1 == sid2
        # Must be a valid UUID
        uuid.UUID(sid1)

    @pytest.mark.asyncio
    async def test_uses_provided_session_id(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            sid = await tracker.ensure_session("conv-1", "user-1", session_id=SESS_UUID)

        # A valid UUID session_id is passed through unchanged (no coercion).
        assert sid == SESS_UUID

    @pytest.mark.asyncio
    async def test_different_conversations_get_different_sessions(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            sid1 = await tracker.ensure_session("conv-1", "user-1")
            sid2 = await tracker.ensure_session("conv-2", "user-1")

        assert sid1 != sid2


# -----------------------------------------------------------------------
# record_message
# -----------------------------------------------------------------------

class TestRecordMessage:

    @pytest.mark.asyncio
    async def test_inserts_message(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            mid = await tracker.record_message(
                conversation_id="conv-1",
                session_id="sess-1",
                role="user",
                content="Hello, how are you?",
                metadata={"source": "sdk"},
            )

        # Returns a valid UUID
        uuid.UUID(mid)
        # record_message INSERTs the message row, then runs a compaction-trigger
        # check that issues a second SELECT on the same session — so execute()
        # is called twice. The first call is the message INSERT.
        assert mock_session.execute.call_count >= 1
        insert_sql = str(mock_session.execute.call_args_list[0][0][0])
        assert "INSERT INTO message" in insert_sql
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_content_stored_as_json_string(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_message("conv-1", "sess-1", "user", "Hello world")

        # First execute() call is the message INSERT (a later SELECT runs for
        # the compaction-trigger check).
        params = mock_session.execute.call_args_list[0][0][1]
        # Content should be json.dumps("Hello world") → '"Hello world"'
        assert params["content"] == json.dumps("Hello world")

    @pytest.mark.asyncio
    async def test_role_mapping(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_message("conv-1", "sess-1", "human", "Hi")

        params = mock_session.execute.call_args_list[0][0][1]
        assert params["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_assistant_role(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_message("conv-1", "sess-1", "assistant", "I'm fine!")

        params = mock_session.execute.call_args_list[0][0][1]
        assert params["message_type"] == "assistant"

    @pytest.mark.asyncio
    async def test_metadata_serialized_as_json(self, tracker, mock_session_factory, mock_session):
        meta = {"word_count": 5, "has_question": True}
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_message("conv-1", "sess-1", "user", "Hello", metadata=meta)

        params = mock_session.execute.call_args_list[0][0][1]
        assert params["metadata"] == json.dumps(meta)

    @pytest.mark.asyncio
    async def test_null_metadata(self, tracker, mock_session_factory, mock_session):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_message("conv-1", "sess-1", "user", "Hello")

        params = mock_session.execute.call_args_list[0][0][1]
        assert params["metadata"] is None


# -----------------------------------------------------------------------
# record_conversation_event (high-level)
# -----------------------------------------------------------------------

class TestRecordConversationEvent:

    @pytest.mark.asyncio
    async def test_orchestrates_all_three_steps(self, tracker):
        tracker.ensure_conversation = AsyncMock(return_value="conv-1")
        tracker.ensure_session = AsyncMock(return_value="sess-1")
        tracker.record_message = AsyncMock(return_value="msg-123")

        result = await tracker.record_conversation_event(
            instance_id="inst-1",
            conversation_id="conv-1",
            user_id="user-1",
            role="user",
            content="Hello",
            customer_id=None,
        )

        assert result == "msg-123"
        tracker.ensure_conversation.assert_called_once_with(
            conversation_id="conv-1",
            instance_id="inst-1",
            user_id="user-1",
            customer_id=None,
            scope_path=None,
        )
        tracker.ensure_session.assert_called_once_with(
            conversation_id="conv-1",
            user_id="user-1",
            session_id=None,
        )
        tracker.record_message.assert_called_once_with(
            conversation_id="conv-1",
            session_id="sess-1",
            role="user",
            content="Hello",
            metadata=None,
            timestamp=None,
        )

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self, tracker):
        tracker.ensure_conversation = AsyncMock(side_effect=Exception("DB down"))

        result = await tracker.record_conversation_event(
            instance_id="inst-1",
            conversation_id="conv-1",
            user_id="user-1",
            role="user",
            content="Hello",
            customer_id=None,
        )

        assert result is None  # Non-fatal

    @pytest.mark.asyncio
    async def test_passes_optional_fields(self, tracker):
        tracker.ensure_conversation = AsyncMock(return_value="conv-1")
        tracker.ensure_session = AsyncMock(return_value="custom-sess")
        tracker.record_message = AsyncMock(return_value="msg-456")

        await tracker.record_conversation_event(
            instance_id="inst-1",
            conversation_id="conv-1",
            user_id="user-1",
            role="assistant",
            content="Sure!",
            customer_id="cust-99",
            session_id="custom-sess",
            metadata={"tool": "search"},
        )

        tracker.ensure_conversation.assert_called_once_with(
            conversation_id="conv-1",
            instance_id="inst-1",
            user_id="user-1",
            customer_id="cust-99",
            scope_path=None,
        )
        tracker.ensure_session.assert_called_once_with(
            conversation_id="conv-1",
            user_id="user-1",
            session_id="custom-sess",
        )
        tracker.record_message.assert_called_once_with(
            conversation_id="conv-1",
            session_id="custom-sess",
            role="assistant",
            content="Sure!",
            metadata={"tool": "search"},
            timestamp=None,
        )


# -----------------------------------------------------------------------
# the rung survives the whole call, not just the signature
# -----------------------------------------------------------------------

class TestTheRungIsThreadedNotJustAccepted:
    """⚠ Written because the guard that was supposed to cover this did not.

    `test_a_conversation_can_name_a_rung.py` asserted that
    `record_conversation_event` DECLARES a `scope_path` parameter. A parameter
    can be declared and thrown away. Replacing the forwarded
    `scope_path=scope_path` with `scope_path=None` left every test in both
    files green, which is precisely the failure that file's own docstring says
    it exists to prevent: a field the client fills in that nothing downstream
    reads. These two tests follow the value to the bind param instead of
    reading the source.
    """

    @pytest.mark.asyncio
    async def test_record_conversation_event_carries_the_path_to_the_insert(
        self, tracker, mock_session_factory, mock_session
    ):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_conversation_event(
                conversation_id="conv-1",
                instance_id="inst-1",
                user_id="user-1",
                customer_id="cust-1",
                session_id="sess-1",
                role="user",
                content="hello",
                scope_path={"tenant": "acme", "practice": "high_street"},
            )

        conv_params = [
            c[0][1]
            for c in mock_session.execute.call_args_list
            if "INSERT INTO conversation (" in str(c[0][0])
        ]
        assert conv_params, "no conversation insert was issued at all"
        # jsonb column, so the bind param is the encoded form. Decoded here
        # rather than compared as a string, because comparing the raw text
        # would pin key order that json.dumps is not obliged to keep.
        assert conv_params[0]["scope_path"] is not None, (
            "the servicer passes a rung and the tracker drops it before the row"
        )
        assert json.loads(conv_params[0]["scope_path"]) == {
            "tenant": "acme", "practice": "high_street"
        }

    @pytest.mark.asyncio
    async def test_record_transcript_carries_the_path_to_the_insert(
        self, tracker, mock_session_factory, mock_session
    ):
        transcript_kwargs = dict(
            conversation_id=CONV_UUID,
            instance_id="inst-1",
            user_id="+919812345678",
            customer_id="cust-1",
            transcript_hash="deadbeef" * 8,
            scope_path={"tenant": "acme", "practice": "high_street"},
        )
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=_turns(2), **transcript_kwargs)

        conv_params = [
            c[0][1]
            for c in mock_session.execute.call_args_list
            if "INSERT INTO conversation (" in str(c[0][0])
        ]
        assert conv_params, "no conversation insert was issued at all"
        assert conv_params[0]["scope_path"] is not None, (
            "record_transcript takes a rung and drops it before the row"
        )
        assert json.loads(conv_params[0]["scope_path"]) == {
            "tenant": "acme", "practice": "high_street"
        }


# -----------------------------------------------------------------------
# record_message — client-supplied timestamp (additive, migration-074 PR)
# -----------------------------------------------------------------------

class TestRecordMessageTimestamp:

    @pytest.mark.asyncio
    async def test_client_timestamp_used_for_insert(self, tracker, mock_session_factory, mock_session):
        client_ts = datetime(2026, 7, 1, 12, 30, 0, tzinfo=timezone.utc)
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_message(
                "conv-1", "sess-1", "user", "Hello", timestamp=client_ts
            )

        params = mock_session.execute.call_args_list[0][0][1]
        assert params["timestamp"] == client_ts

    @pytest.mark.asyncio
    async def test_client_timestamp_used_for_redis_buffer(
        self, tracker_with_redis, mock_redis, mock_session_factory
    ):
        client_ts = datetime(2026, 7, 1, 12, 30, 0, tzinfo=timezone.utc)
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker_with_redis.record_message(
                "conv-1", "sess-1", "user", "Hello", timestamp=client_ts
            )

        mock_redis.rpush.assert_called_once()
        entry = json.loads(mock_redis.rpush.call_args[0][1])
        assert entry["timestamp"] == client_ts.isoformat()
        mock_redis.expire.assert_called_once()
        assert mock_redis.expire.call_args[0][1] == 14400

    @pytest.mark.asyncio
    async def test_default_is_server_now_utc(self, tracker, mock_session_factory, mock_session):
        """Omitting timestamp keeps the previous behavior: server now(UTC)."""
        before = datetime.now(timezone.utc)
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_message("conv-1", "sess-1", "user", "Hello")
        after = datetime.now(timezone.utc)

        params = mock_session.execute.call_args_list[0][0][1]
        assert params["timestamp"].tzinfo is not None
        assert before <= params["timestamp"] <= after


# -----------------------------------------------------------------------
# record_transcript (bulk one-shot transcript storage, spec §4.4 step 3)
# -----------------------------------------------------------------------

def _turns(n, start_hour=10):
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"turn {i}",
            "timestamp": datetime(2026, 7, 1, start_hour, i, 0, tzinfo=timezone.utc),
        }
        for i in range(n)
    ]


class TestRecordTranscript:

    @pytest.fixture
    def transcript_kwargs(self):
        return dict(
            conversation_id=CONV_UUID,
            instance_id="inst-1",
            user_id="+919812345678",
            customer_id="cust-1",
            conversation_type="voice",
            analysis={"disposition": "qualified"},
            conversation_metadata={"source": "voice-agent"},
            transcript_hash="deadbeef" * 8,
            started_at=datetime(2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 7, 1, 10, 30, 0, tzinfo=timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_happy_path_inserts_and_finalizes_in_one_transaction(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            result = await tracker.record_transcript(
                turns=_turns(4), **transcript_kwargs
            )

        # 4 message INSERTs + 1 conversation UPDATE, single commit.
        assert mock_session.execute.call_count == 5
        for call in mock_session.execute.call_args_list[:4]:
            assert "INSERT INTO message" in str(call[0][0])
        update_sql = str(mock_session.execute.call_args_list[4][0][0])
        assert "UPDATE conversation" in update_sql
        assert "status = 'completed'" in update_sql
        assert "start_timestamp = :start_timestamp" in update_sql
        mock_session.commit.assert_called_once()

        # Finalize params carry the transcript-level fields.
        update_params = mock_session.execute.call_args_list[4][0][1]
        assert update_params["conversation_type"] == "voice"
        assert update_params["analysis"] == json.dumps({"disposition": "qualified"})
        assert update_params["conversation_metadata"] == json.dumps({"source": "voice-agent"})
        assert update_params["transcript_hash"] == "deadbeef" * 8
        assert update_params["start_timestamp"] == transcript_kwargs["started_at"]
        assert update_params["end_timestamp"] == transcript_kwargs["ended_at"]

        assert result == {
            "conversation_id": CONV_UUID,
            "session_id": SESS_UUID,
            "turns_recorded": 4,
        }

    @pytest.mark.asyncio
    async def test_client_turn_timestamps_used(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)
        turns = _turns(2)

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=turns, **transcript_kwargs)

        for i, call in enumerate(mock_session.execute.call_args_list[:2]):
            params = call[0][1]
            assert params["timestamp"] == turns[i]["timestamp"]
            assert params["content"] == json.dumps(f"turn {i}")

    @pytest.mark.asyncio
    async def test_speaker_and_turn_metadata_merge_into_message_metadata(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)
        turns = [
            {
                "role": "assistant",
                "content": "Hi, this is Priya from Acme.",
                "speaker": "Agent Priya",
                "metadata": {"stt_confidence": 0.98},
            },
            {"role": "user", "content": "Hello."},
        ]

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=turns, **transcript_kwargs)

        first_meta = json.loads(mock_session.execute.call_args_list[0][0][1]["metadata"])
        assert first_meta == {"stt_confidence": 0.98, "speaker": "Agent Priya"}
        # Turn without speaker/metadata stores NULL metadata (same as record_message).
        assert mock_session.execute.call_args_list[1][0][1]["metadata"] is None

    @pytest.mark.asyncio
    async def test_start_timestamp_untouched_when_started_at_absent(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)
        transcript_kwargs["started_at"] = None

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=_turns(1), **transcript_kwargs)

        update_sql = str(mock_session.execute.call_args_list[-1][0][0])
        assert "start_timestamp" not in update_sql
        assert "status = 'completed'" in update_sql

    @pytest.mark.asyncio
    async def test_redis_buffer_gets_last_20_turns(
        self, tracker_with_redis, mock_redis, mock_session_factory, transcript_kwargs
    ):
        tracker_with_redis.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker_with_redis.ensure_session = AsyncMock(return_value=SESS_UUID)

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker_with_redis.record_transcript(
                turns=_turns(25, start_hour=9), **transcript_kwargs
            )

        assert mock_redis.rpush.call_count == 20
        # Buffer starts at turn 5 (25 - 20) and preserves the entry shape.
        first_entry = json.loads(mock_redis.rpush.call_args_list[0][0][1])
        assert first_entry["content"] == "turn 5"
        assert set(first_entry) == {"message_id", "role", "content", "timestamp"}
        assert mock_redis.rpush.call_args_list[0][0][0] == f"conv_messages:{CONV_UUID}"
        mock_redis.expire.assert_called_once_with(f"conv_messages:{CONV_UUID}", 14400)

    @pytest.mark.asyncio
    async def test_does_not_trigger_auto_compaction(
        self, tracker, mock_session_factory, transcript_kwargs
    ):
        """Spec §4.4 step 3: the >=10-message auto-trigger must NOT fire
        mid-push — the ingest endpoint compacts explicitly afterwards."""
        tracker._maybe_trigger_compaction = AsyncMock()

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=_turns(15), **transcript_kwargs)

        tracker._maybe_trigger_compaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensures_conversation_and_session_idempotently(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        """Un-mocked ensure_* path: each upsert runs exactly once, and a
        non-UUID conversation_id coerces deterministically.

        The two inserts resolve their conflict DIFFERENTLY and the difference
        is asserted below rather than papered over. The session row is fixed
        once written, so it stays DO NOTHING. The conversation row carries
        `scope_path`, which a later turn may be the first to know, so it is a
        one-way DO UPDATE that only ever fills a NULL.
        """
        transcript_kwargs["conversation_id"] = "voice-call-42"

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            result = await tracker.record_transcript(
                turns=_turns(2), **transcript_kwargs
            )

        expected_cid = str(uuid.uuid5(uuid.NAMESPACE_URL, "voice-call-42"))
        assert result["conversation_id"] == expected_cid

        sqls = [str(c[0][0]) for c in mock_session.execute.call_args_list]
        conv_inserts = [s for s in sqls if "INSERT INTO conversation " in s or "INSERT INTO conversation (" in s]
        sess_inserts = [s for s in sqls if "INSERT INTO conversation_session" in s]
        assert len(conv_inserts) == 1
        assert len(sess_inserts) == 1
        assert all("ON CONFLICT" in s for s in conv_inserts + sess_inserts)
        assert all("DO NOTHING" in s for s in sess_inserts)
        assert all(
            "DO UPDATE" in s
            and "conversation.scope_path IS NULL" in s
            and "DO NOTHING" not in s
            for s in conv_inserts
        )
        # ensure_conversation + ensure_session + 2 message inserts + finalize
        assert mock_session.execute.call_count == 5

    @pytest.mark.asyncio
    async def test_turn_without_timestamp_defaults_to_server_now(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)

        before = datetime.now(timezone.utc)
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(
                turns=[{"role": "user", "content": "no ts"}], **transcript_kwargs
            )
        after = datetime.now(timezone.utc)

        params = mock_session.execute.call_args_list[0][0][1]
        assert before <= params["timestamp"] <= after


# -----------------------------------------------------------------------
# record_transcript — PR-B additions (external id round-trip, insert order)
# -----------------------------------------------------------------------

class TestRecordTranscriptEndpointAdditions:

    @pytest.fixture
    def transcript_kwargs(self):
        return dict(
            conversation_id=CONV_UUID,
            instance_id="inst-1",
            user_id="+919812345678",
            customer_id="cust-1",
            transcript_hash="deadbeef" * 8,
        )

    @pytest.mark.asyncio
    async def test_external_conversation_id_persisted(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(
                turns=_turns(2),
                external_conversation_id="voice-call-42",
                **transcript_kwargs,
            )

        update_sql = str(mock_session.execute.call_args_list[-1][0][0])
        assert "external_conversation_id = :external_conversation_id" in update_sql
        update_params = mock_session.execute.call_args_list[-1][0][1]
        assert update_params["external_conversation_id"] == "voice-call-42"

    @pytest.mark.asyncio
    async def test_external_conversation_id_defaults_null(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=_turns(1), **transcript_kwargs)

        update_params = mock_session.execute.call_args_list[-1][0][1]
        assert update_params["external_conversation_id"] is None

    @pytest.mark.asyncio
    async def test_turns_sorted_by_timestamp_asc(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        """011 stats trigger copies the LAST inserted row's timestamp into
        last_message_at (not the max) — out-of-order input must be sorted."""
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)
        turns = [
            {"role": "assistant", "content": "third",
             "timestamp": datetime(2026, 7, 1, 10, 30, 0, tzinfo=timezone.utc)},
            {"role": "user", "content": "first",
             "timestamp": datetime(2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc)},
            {"role": "assistant", "content": "second",
             "timestamp": datetime(2026, 7, 1, 10, 15, 0, tzinfo=timezone.utc)},
        ]

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=turns, **transcript_kwargs)

        inserted = [
            mock_session.execute.call_args_list[i][0][1] for i in range(3)
        ]
        assert [json.loads(p["content"]) for p in inserted] == [
            "first", "second", "third",
        ]
        timestamps = [p["timestamp"] for p in inserted]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_naive_timestamps_coerced_to_utc(
        self, tracker, mock_session_factory, mock_session, transcript_kwargs
    ):
        tracker.ensure_conversation = AsyncMock(return_value=CONV_UUID)
        tracker.ensure_session = AsyncMock(return_value=SESS_UUID)
        turns = [
            {"role": "user", "content": "naive",
             "timestamp": datetime(2026, 7, 1, 10, 0, 0)},  # no tzinfo
        ]

        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.record_transcript(turns=turns, **transcript_kwargs)

        params = mock_session.execute.call_args_list[0][0][1]
        assert params["timestamp"].tzinfo == timezone.utc
        assert params["timestamp"] == datetime(2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc)


# -----------------------------------------------------------------------
# set_last_ingestion_id
# -----------------------------------------------------------------------

class TestSetLastIngestionId:

    @pytest.mark.asyncio
    async def test_updates_conversation_row(
        self, tracker, mock_session_factory, mock_session
    ):
        ingestion_id = str(uuid.uuid4())
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.set_last_ingestion_id(CONV_UUID, ingestion_id)

        mock_session.execute.assert_called_once()
        sql = str(mock_session.execute.call_args[0][0])
        assert "UPDATE conversation" in sql
        assert "last_ingestion_id" in sql
        params = mock_session.execute.call_args[0][1]
        assert params["ingestion_id"] == ingestion_id
        assert params["cid"] == CONV_UUID
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_coerces_non_uuid_conversation_id(
        self, tracker, mock_session_factory, mock_session
    ):
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=mock_session_factory,
        ):
            await tracker.set_last_ingestion_id("voice-call-42", str(uuid.uuid4()))

        params = mock_session.execute.call_args[0][1]
        assert params["cid"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "voice-call-42"))


# -----------------------------------------------------------------------
# The automatic compaction must carry the client id
# -----------------------------------------------------------------------


class TestAutoCompactionCarriesTheClientId:
    """The post-compaction callback clears the conversation-context cache with a
    key built from client, customer, user and conversation. This call left the
    client out, so it cleared a key that did not exist and the agent was served the
    FIRST compaction's summary for the cache's whole 30 minutes (the
    cache held compaction 1 while the database held three newer ones)."""

    @pytest.mark.asyncio
    async def test_the_client_id_is_passed_to_the_compaction(self, tracker):
        session = AsyncMock()
        row = (5000, 40, "inst_x", "user_1", "cust_uuid", "client-uuid-1")
        result = MagicMock()
        result.first.return_value = row
        session.execute = AsyncMock(return_value=result)

        seen = {}

        class _Manager:
            storage = MagicMock()

            async def compact(self, **kwargs):
                seen.update(kwargs)

        _Manager.storage.set_compaction_in_progress = AsyncMock(return_value=True)
        _Manager.storage.clear_compaction_in_progress = AsyncMock()
        _Manager.storage.load_conversation_messages = AsyncMock(return_value=[object()])

        with patch(
            "synap.cloud.application.compaction.get_compaction_manager",
            return_value=_Manager(),
        ):
            await tracker._maybe_trigger_compaction("conv-1", session)
            import asyncio

            await asyncio.sleep(0.05)  # the compaction runs as a fire-and-forget task

        assert seen["client_id"] == "client-uuid-1"
        assert seen["user_id"] == "user_1"
        assert seen["customer_id"] == "cust_uuid"

    @pytest.mark.asyncio
    async def test_the_query_joins_the_instance_for_its_client(self, tracker):
        session = AsyncMock()
        result = MagicMock()
        result.first.return_value = None
        session.execute = AsyncMock(return_value=result)
        await tracker._maybe_trigger_compaction("conv-1", session)
        sql = str(session.execute.call_args[0][0]).lower()
        assert "join instance" in sql and "client_id" in sql
