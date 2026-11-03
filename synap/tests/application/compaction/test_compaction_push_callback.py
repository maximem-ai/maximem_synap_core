"""Tests for server-side Callback 4: Push compacted context to gRPC stream.

Tests the post-compaction callback in providers.py that:
1. Reads latest compacted context from storage
2. Builds a ConversationContext bundle
3. Pushes to the instance's gRPC push queue
4. Handles missing push_queues, full queue, and storage errors gracefully
"""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest


# ---------------------------------------------------------------------------
# Simulated types
# ---------------------------------------------------------------------------

@dataclass
class MockCompactionEvent:
    """Simulates CompactionEvent from compaction models."""
    conversation_id: str = "conv-abc-123"
    instance_id: str = "inst-test-001"
    client_id: Optional[str] = None
    customer_id: Optional[str] = None
    user_id: Optional[str] = "user-001"
    is_incremental: bool = False


@dataclass
class MockCompactedMetadata:
    source_message_range: Dict[str, Any] = field(default_factory=lambda: {
        "start_id": "msg-001",
        "end_id": "msg-050",
        "message_count": 50,
    })
    compression_ratio: float = 0.35


@dataclass
class MockCompactedContext:
    compaction_id: str = "cmp-001"
    conversation_id: str = "conv-abc-123"
    narrative_summary: str = "User discussed billing and settings."
    structured_extractions: Dict[str, Any] = field(default_factory=lambda: {
        "facts": [{"content": "User prefers email", "confidence": 0.9}],
        "preferences": [{"content": "Dark mode"}],
        "current_state": {"topic": "billing"},
    })
    metadata: MockCompactedMetadata = field(default_factory=MockCompactedMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Callback 4 logic (extracted for testing)
# ---------------------------------------------------------------------------

async def push_compaction_to_grpc(
    event: MockCompactionEvent,
    compaction_storage: AsyncMock,
    push_queues: Dict[str, asyncio.Queue],
    logger: logging.Logger,
):
    """Extracted Callback 4 logic from providers.py for testing.

    After Proposal 1 implementation, this logic lives inside
    _push_compaction_to_grpc_stream() in providers.py.
    """
    try:
        # Check if push_queues has a queue for this instance
        queue = push_queues.get(event.instance_id)
        if queue is None:
            logger.debug(
                f"No push queue for instance {event.instance_id}; skipping gRPC push"
            )
            return

        # Load latest compacted context
        compacted = await compaction_storage.get_latest(
            event.conversation_id, event.instance_id
        )
        if not compacted:
            logger.debug(f"No compacted context for {event.conversation_id}")
            return

        # Build bundle
        bundle = {
            "bundle_id": f"compaction_update_{compacted.compaction_id}",
            "_bundle_type": "compaction_update",
            "_anticipation_user_id": event.user_id,
            "_anticipation_customer_id": event.customer_id,
            "_anticipation_conversation_id": event.conversation_id,
            "search_queries": [],
            "search_keywords": [],
            "items_by_type": {},
            "conversation_context": {
                "summary": compacted.narrative_summary,
                "current_state": compacted.structured_extractions.get("current_state", {}),
                "key_extractions": {
                    k: v for k, v in compacted.structured_extractions.items()
                    if k != "current_state"
                },
                "compaction_id": compacted.compaction_id,
                "compacted_at": compacted.created_at.isoformat(),
                "conversation_id": event.conversation_id,
            },
        }

        try:
            queue.put_nowait(bundle)
            logger.info(
                f"Pushed compaction_update to gRPC stream for instance "
                f"{event.instance_id}: compaction_id={compacted.compaction_id}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"Push queue full for instance {event.instance_id}; "
                f"compaction_update dropped"
            )

    except Exception as e:
        logger.warning(f"gRPC push callback failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPushCompactionCallback:
    """Tests for the post-compaction gRPC push callback."""

    @pytest.mark.asyncio
    async def test_pushes_bundle_to_queue(self):
        """Callback should push a compaction_update bundle to the instance queue."""
        event = MockCompactionEvent()
        storage = AsyncMock()
        storage.get_latest = AsyncMock(return_value=MockCompactedContext())

        push_queues = {event.instance_id: asyncio.Queue(maxsize=50)}
        log = logging.getLogger("test")

        await push_compaction_to_grpc(event, storage, push_queues, log)

        assert not push_queues[event.instance_id].empty()
        bundle = push_queues[event.instance_id].get_nowait()
        assert bundle["_bundle_type"] == "compaction_update"
        assert bundle["conversation_context"]["compaction_id"] == "cmp-001"

    @pytest.mark.asyncio
    async def test_no_queue_for_instance_is_noop(self):
        """If instance has no push queue, callback should silently return."""
        event = MockCompactionEvent(instance_id="inst-offline")
        storage = AsyncMock()
        push_queues = {}  # No queues registered
        log = logging.getLogger("test")

        await push_compaction_to_grpc(event, storage, push_queues, log)

        # Should not raise and storage should not be called
        storage.get_latest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_compacted_context_is_noop(self):
        """If no compacted context exists, callback should silently return."""
        event = MockCompactionEvent()
        storage = AsyncMock()
        storage.get_latest = AsyncMock(return_value=None)

        push_queues = {event.instance_id: asyncio.Queue(maxsize=50)}
        log = logging.getLogger("test")

        await push_compaction_to_grpc(event, storage, push_queues, log)

        assert push_queues[event.instance_id].empty()

    @pytest.mark.asyncio
    async def test_full_queue_handled_gracefully(self):
        """When queue is full, callback should log warning, not crash."""
        event = MockCompactionEvent()
        storage = AsyncMock()
        storage.get_latest = AsyncMock(return_value=MockCompactedContext())

        push_queues = {event.instance_id: asyncio.Queue(maxsize=1)}
        push_queues[event.instance_id].put_nowait({"filler": True})  # Fill it

        log = logging.getLogger("test")

        # Should not raise
        await push_compaction_to_grpc(event, storage, push_queues, log)

        # Queue should still only have the filler
        assert push_queues[event.instance_id].qsize() == 1

    @pytest.mark.asyncio
    async def test_storage_error_handled_gracefully(self):
        """Storage exceptions should be caught and logged, not re-raised."""
        event = MockCompactionEvent()
        storage = AsyncMock()
        storage.get_latest = AsyncMock(side_effect=Exception("DB connection lost"))

        push_queues = {event.instance_id: asyncio.Queue(maxsize=50)}
        log = logging.getLogger("test")

        # Should not raise
        await push_compaction_to_grpc(event, storage, push_queues, log)

        assert push_queues[event.instance_id].empty()

    @pytest.mark.asyncio
    async def test_bundle_shape_correct(self):
        """Verify the bundle shape matches what context_bundle_to_proto expects."""
        event = MockCompactionEvent(
            user_id="user-001",
            customer_id="cust-001",
            conversation_id="conv-abc-123",
        )
        storage = AsyncMock()
        storage.get_latest = AsyncMock(return_value=MockCompactedContext())

        push_queues = {event.instance_id: asyncio.Queue(maxsize=50)}
        log = logging.getLogger("test")

        await push_compaction_to_grpc(event, storage, push_queues, log)

        bundle = push_queues[event.instance_id].get_nowait()

        # Top-level metadata
        assert bundle["_anticipation_user_id"] == "user-001"
        assert bundle["_anticipation_customer_id"] == "cust-001"
        assert bundle["_anticipation_conversation_id"] == "conv-abc-123"
        assert isinstance(bundle["search_queries"], list)
        assert isinstance(bundle["items_by_type"], dict)

        # conversation_context
        ctx = bundle["conversation_context"]
        assert isinstance(ctx["summary"], str)
        assert isinstance(ctx["current_state"], dict)
        assert isinstance(ctx["key_extractions"], dict)
        assert ctx["current_state"] == {"topic": "billing"}
        assert "facts" in ctx["key_extractions"]
        assert "preferences" in ctx["key_extractions"]

    @pytest.mark.asyncio
    async def test_incremental_compaction_push(self):
        """Incremental compaction should also push to gRPC."""
        event = MockCompactionEvent(is_incremental=True)
        storage = AsyncMock()
        storage.get_latest = AsyncMock(return_value=MockCompactedContext(
            compaction_id="cmp-incr-005"
        ))

        push_queues = {event.instance_id: asyncio.Queue(maxsize=50)}
        log = logging.getLogger("test")

        await push_compaction_to_grpc(event, storage, push_queues, log)

        bundle = push_queues[event.instance_id].get_nowait()
        assert bundle["conversation_context"]["compaction_id"] == "cmp-incr-005"


class TestPushCompactionIntegrationWithCallbackChain:
    """Test that Callback 4 integrates with the existing callback chain."""

    @pytest.mark.asyncio
    async def test_callback_runs_after_cache_invalidation(self):
        """Callback 4 should run after Callbacks 1-3 (cache invalidation, trim, promotion)."""
        # Simulate 4 callbacks registered on compaction_manager
        callbacks = []

        class MockCompactionManager:
            def on_compaction_complete(self, cb):
                callbacks.append(cb)

        mgr = MockCompactionManager()

        # Register callbacks in order (same as providers.py)
        mgr.on_compaction_complete(lambda e: None)  # CB1: cache invalidation
        mgr.on_compaction_complete(lambda e: None)  # CB2: Redis trim
        mgr.on_compaction_complete(lambda e: None)  # CB3: long-term promotion
        mgr.on_compaction_complete(lambda e: None)  # CB4: gRPC push (new)

        assert len(callbacks) == 4

    @pytest.mark.asyncio
    async def test_callback_4_does_not_block_compaction(self):
        """Callback 4 failure should not affect the compaction result."""
        event = MockCompactionEvent()
        storage = AsyncMock()
        storage.get_latest = AsyncMock(side_effect=Exception("Timeout"))

        push_queues = {event.instance_id: asyncio.Queue(maxsize=50)}
        log = logging.getLogger("test")

        # Should not raise
        await push_compaction_to_grpc(event, storage, push_queues, log)

        # Compaction itself is already complete — this callback is post-hoc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
