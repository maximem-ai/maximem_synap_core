"""Long-term promotion on its own schedule (COMPACTION_V3_PLAN.md §4.6).

Raw turns go to ingestion every PROMOTE_EVERY turns, and on idle or session
end (flush), whatever compaction does. The cursor is the conversation row's
promoted_message_count; each batch's document id comes from its first and
last message id.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import uuid

import pytest

from synap.cloud.agents.ingestion.models.inputs import Scope

import synap.cloud.application.shared.providers as providers
from synap.cloud.application.compaction.models import ConversationMessage
from synap.cloud.application.conversation import promotion as promotion_mod
from synap.cloud.application.conversation.promotion import PROMOTE_EVERY, LongTermPromoter

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)


def _messages(n):
    return [ConversationMessage(f"id{i}", "user" if i % 2 == 0 else "assistant", f"turn {i}",
                                T0 + timedelta(minutes=i)) for i in range(n)]


def _promoter(monkeypatch, promoted, total, skip=False):
    storage = MagicMock()
    storage.load_conversation_messages = AsyncMock(return_value=_messages(total))
    ingestion = MagicMock()
    ingestion.ingest_document = AsyncMock(return_value=MagicMock(ingestion_id="ing_1"))
    promoter = LongTermPromoter(storage, ingestion, redis=None)
    promoter._counts = AsyncMock(return_value=(promoted, total))
    promoter._advance = AsyncMock(return_value=True)
    monkeypatch.setattr(providers, "transcript_push_promotion_skip", AsyncMock(return_value=skip))
    scope = Scope(client_id=uuid.uuid4(), instance_id=uuid.uuid4(), customer_id=uuid.uuid4())
    monkeypatch.setattr(providers, "resolve_compaction_promotion_scope", AsyncMock(return_value=scope))
    return promoter, ingestion


@pytest.mark.asyncio
async def test_waits_for_a_batch(monkeypatch):
    promoter, ingestion = _promoter(monkeypatch, promoted=10, total=10 + PROMOTE_EVERY - 1)
    assert await promoter.maybe_promote("c1", "inst") == 0
    ingestion.ingest_document.assert_not_called()


@pytest.mark.asyncio
async def test_sends_only_the_unsent_turns_and_advances_the_cursor(monkeypatch):
    promoter, ingestion = _promoter(monkeypatch, promoted=10, total=10 + PROMOTE_EVERY)
    assert await promoter.maybe_promote("c1", "inst") == PROMOTE_EVERY
    request = ingestion.ingest_document.call_args.args[0]
    assert request.content.splitlines()[0] == "user: turn 10"
    assert request.metadata["source_message_range"]["start_id"] == "id10"
    assert request.metadata["_trace"]["correlation_id"] == request.document_id
    promoter._advance.assert_awaited_once()
    assert promoter._advance.call_args.args[1:] == (10, 10 + PROMOTE_EVERY)


@pytest.mark.asyncio
async def test_the_same_batch_gets_the_same_document_id(monkeypatch):
    first, ing1 = _promoter(monkeypatch, promoted=0, total=PROMOTE_EVERY)
    second, ing2 = _promoter(monkeypatch, promoted=0, total=PROMOTE_EVERY)
    await first.maybe_promote("c1", "inst")
    await second.maybe_promote("c1", "inst")
    assert ing1.ingest_document.call_args.args[0].document_id == ing2.ingest_document.call_args.args[0].document_id


@pytest.mark.asyncio
async def test_flush_sends_a_partial_batch(monkeypatch):
    promoter, ingestion = _promoter(monkeypatch, promoted=4, total=6)
    assert await promoter.maybe_promote("c1", "inst", flush=True) == 2
    ingestion.ingest_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_pushed_transcript_is_skipped_and_the_cursor_moves_past_it(monkeypatch):
    promoter, ingestion = _promoter(monkeypatch, promoted=0, total=12, skip=True)
    assert await promoter.maybe_promote("c1", "inst", flush=True) == 0
    ingestion.ingest_document.assert_not_called()
    assert promoter._advance.call_args.args[1:] == (0, 12)


@pytest.mark.asyncio
async def test_a_failure_never_raises(monkeypatch):
    promoter, ingestion = _promoter(monkeypatch, promoted=0, total=PROMOTE_EVERY)
    ingestion.ingest_document = AsyncMock(side_effect=RuntimeError("queue down"))
    assert await promoter.maybe_promote("c1", "inst") == 0
    promoter._advance.assert_not_called()


def test_providers_installs_the_promoter_not_a_callback():
    import inspect
    src = inspect.getsource(providers)
    assert "set_promoter(LongTermPromoter(" in src
    assert promotion_mod.get_promoter is not None


@pytest.mark.asyncio
async def test_a_sent_batch_tells_the_forget_queue_where_it_starts(monkeypatch):
    """A forget that found nothing may be waiting for exactly this batch: its timers count from
    the moment the batch is sent (memory_forget/retry.py)."""
    from synap.cloud.application.memory_forget import retry as forget_retry

    seen = {}

    async def on_batch_sent(*, scope, conversation_id, from_position):
        seen.update(conversation_id=conversation_id, from_position=from_position,
                    user_id=scope.user_id)
        return 1

    monkeypatch.setattr(forget_retry, "on_batch_sent", on_batch_sent)
    promoter, ingestion = _promoter(monkeypatch, promoted=10, total=10 + PROMOTE_EVERY)
    assert await promoter.maybe_promote("c1", "inst") == PROMOTE_EVERY
    # the cursor says 10 messages were sent before, so this batch starts at message 11
    assert seen["from_position"] == 11
    assert seen["conversation_id"]                       # the normalised conversation id


@pytest.mark.asyncio
async def test_a_failing_forget_hook_never_stops_promotion(monkeypatch):
    from synap.cloud.application.memory_forget import retry as forget_retry

    async def boom(**kw):
        raise RuntimeError("queue down")

    monkeypatch.setattr(forget_retry, "on_batch_sent", boom)
    promoter, ingestion = _promoter(monkeypatch, promoted=10, total=10 + PROMOTE_EVERY)
    # maybe_promote swallows errors and returns 0; the batch must still have been sent and
    # the cursor advanced before the hook ran
    await promoter.maybe_promote("c1", "inst")
    ingestion.ingest_document.assert_awaited_once()
    promoter._advance.assert_awaited_once()
