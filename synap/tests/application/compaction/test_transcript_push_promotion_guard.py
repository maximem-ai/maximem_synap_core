"""Transcript-push double-extraction guard (spec §4.4) + CompactionEvent
trigger_type plumbing.

The one-shot transcript ingest already ingests the FULL transcript, so the
post-compaction long-term promotion callback must skip those conversations.
The skip is CONVERSATION-level (row transcript_hash IS NOT NULL), not just
trigger-level: a failed transcript-push compaction followed by a MANUAL_API
or BACKGROUND_WORKER *full* compaction would otherwise promote the whole
transcript a second time (adversarial review, backend finding 1).

The other three post-compaction callbacks (retrieval cache invalidation,
Redis buffer trim, gRPC push) keep firing for transcript-push events.
"""

import inspect
import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.compaction import CompactionTriggerType
from synap.cloud.application.compaction.models import CompactionEvent
import synap.cloud.application.compaction.manager as manager_mod
import synap.cloud.application.shared.providers as providers_mod
from synap.cloud.application.shared.providers import transcript_push_promotion_skip

CONV_ID = str(uuid.uuid4())


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _FakeSession:
    def __init__(self, row):
        self._row = row
        self.queries = []

    async def execute(self, stmt, params=None):
        self.queries.append((str(stmt), params))
        return _FakeResult(self._row)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _wire_row(monkeypatch, row):
    import synap.cloud.database.connection as conn_mod
    session = _FakeSession(row)
    monkeypatch.setattr(conn_mod, "get_session_factory", lambda: (lambda: session))
    return session


def _wire_db_down(monkeypatch):
    import synap.cloud.database.connection as conn_mod

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(conn_mod, "get_session_factory", _boom)


def _event(trigger_type=None):
    return SimpleNamespace(
        conversation_id=CONV_ID,
        instance_id="inst_test",
        trigger_type=trigger_type,
    )


# ---------------------------------------------------------------------------
# transcript_push_promotion_skip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTranscriptPushPromotionSkip:

    async def test_transcript_push_trigger_skips_without_touching_db(self, monkeypatch):
        _wire_db_down(monkeypatch)  # would raise if the guard queried
        assert await transcript_push_promotion_skip(
            _event(CompactionTriggerType.TRANSCRIPT_PUSH)
        ) is True

    async def test_raw_string_trigger_value_also_skips(self, monkeypatch):
        _wire_db_down(monkeypatch)
        assert await transcript_push_promotion_skip(_event("transcript_push")) is True

    async def test_manual_full_compaction_of_transcript_conversation_skips(
        self, monkeypatch
    ):
        """THE finding-1 case: transcript-push compaction failed, a later
        MANUAL_API full compaction fires — the row's transcript_hash must
        still block promotion."""
        _wire_row(monkeypatch, ("a1b2c3" * 10,))
        assert await transcript_push_promotion_skip(
            _event(CompactionTriggerType.MANUAL_API)
        ) is True

    async def test_normal_conversation_promotes(self, monkeypatch):
        _wire_row(monkeypatch, (None,))
        assert await transcript_push_promotion_skip(
            _event(CompactionTriggerType.MESSAGE_COUNT)
        ) is False

    async def test_missing_row_promotes(self, monkeypatch):
        _wire_row(monkeypatch, None)
        assert await transcript_push_promotion_skip(
            _event(CompactionTriggerType.TOKEN_THRESHOLD)
        ) is False

    async def test_db_error_falls_back_to_promote(self, monkeypatch):
        _wire_db_down(monkeypatch)
        assert await transcript_push_promotion_skip(
            _event(CompactionTriggerType.MANUAL_API)
        ) is False

    async def test_event_without_trigger_attr_still_checks_row(self, monkeypatch):
        session = _wire_row(monkeypatch, ("somehash",))
        event = SimpleNamespace(conversation_id=CONV_ID, instance_id="inst_test")
        assert await transcript_push_promotion_skip(event) is True
        assert len(session.queries) == 1

    async def test_row_lookup_coerces_conversation_id(self, monkeypatch):
        session = _wire_row(monkeypatch, (None,))
        event = _event(CompactionTriggerType.MANUAL_API)
        event.conversation_id = "voice-call-42"
        await transcript_push_promotion_skip(event)
        _, params = session.queries[0]
        assert params["cid"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "voice-call-42"))


# ---------------------------------------------------------------------------
# CompactionEvent / manager plumbing
# ---------------------------------------------------------------------------

class TestEventTriggerType:

    def test_enum_member_exists_and_parses(self):
        assert CompactionTriggerType.TRANSCRIPT_PUSH.value == "transcript_push"
        # CompactedContext.from_dict strict-parses the enum: a stored
        # transcript_push row must round-trip.
        assert CompactionTriggerType("transcript_push") is CompactionTriggerType.TRANSCRIPT_PUSH

    def test_event_carries_trigger_type(self):
        event = CompactionEvent(
            conversation_id=CONV_ID,
            instance_id="inst_test",
            compaction_id="compact_abc",
            compression_ratio=0.4,
            validation_score=0.9,
            is_incremental=False,
            message_count=10,
            trigger_type=CompactionTriggerType.TRANSCRIPT_PUSH,
        )
        assert event.trigger_type is CompactionTriggerType.TRANSCRIPT_PUSH

    def test_event_trigger_type_defaults_none(self):
        event = CompactionEvent(
            conversation_id=CONV_ID,
            instance_id="inst_test",
            compaction_id="compact_abc",
            compression_ratio=0.4,
            validation_score=0.9,
            is_incremental=False,
            message_count=10,
        )
        assert event.trigger_type is None

    def test_manager_wires_trigger_type_into_event(self):
        """compact() must pass its trigger_type into the CompactionEvent it
        emits (it already persists it on CompactionMetadata)."""
        src = inspect.getsource(manager_mod)
        start = src.index("event = CompactionEvent(")
        end = src.index(")", src.index("trigger_type", start))
        constructor = src[start:end]
        assert "trigger_type=trigger_type" in constructor


class TestPromotionCallbackWiring:

    def test_promotion_checks_the_guard_before_ingesting(self):
        """Promotion runs on its own schedule now (conversation/promotion.py);
        the guard still runs before anything is sent to ingestion."""
        import synap.cloud.application.conversation.promotion as promotion
        src = inspect.getsource(promotion._promote if hasattr(promotion, "_promote")
                                else promotion.LongTermPromoter._promote)
        assert src.index("transcript_push_promotion_skip(event)") < src.index("ingest_document")

    def test_no_compaction_callback_promotes(self):
        src = inspect.getsource(providers_mod)
        assert "_promote_to_longterm_on_compaction" not in src

    def test_other_callbacks_do_not_use_the_guard(self):
        """Cache invalidation, Redis trim and gRPC push keep firing for
        transcript-push compactions — only promotion skips."""
        src = inspect.getsource(providers_mod)
        for cb in (
            "_invalidate_retrieval_cache_on_compaction",
            "_trim_redis_buffer_on_compaction",
            "_push_compaction_to_grpc_stream",
        ):
            cb_start = src.index(f"async def {cb}")
            # Slice up to the callback's registration line.
            cb_end = src.index(f"on_compaction_complete({cb})", cb_start)
            assert "transcript_push_promotion_skip" not in src[cb_start:cb_end], cb
