"""Tests for IngestionQueue."""

from datetime import datetime
from uuid import uuid4

import pytest
import fakeredis.aioredis

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    IngestPriority,
    Scope,
)
from synap.cloud.application.ingestion.queue import IngestionQueue


class TestIngestionQueue:
    """Tests for IngestionQueue."""

    @pytest.fixture
    async def redis(self):
        """Create fake Redis client.

        Production constructs the ingestion Redis client with
        ``decode_responses=True`` (see ingestion/startup.py), so the queue's
        per-client registry stores/reads str client_ids. Match that config
        here, otherwise SET members come back as bytes and the round-robin
        queue-name reconstruction (str(bytes) -> "b'...'") never matches the
        actual per-client list keys.
        """
        return fakeredis.aioredis.FakeRedis(decode_responses=True)

    @pytest.fixture
    def queue(self, redis):
        """Create ingestion queue."""
        return IngestionQueue(redis=redis)

    @pytest.fixture
    def scope(self):
        """Create test scope."""
        return Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )

    @pytest.fixture
    def ingest_request(self, scope):
        """Create test request."""
        return IngestDocumentRequest(
            content="Test content",
            scope=scope,
        )

    @pytest.mark.asyncio
    async def test_enqueue_adds_to_queue(self, queue, ingest_request):
        """Test enqueue adds item to queue."""
        ingestion_id = uuid4()
        
        await queue.enqueue(
            ingestion_id=ingestion_id,
            request=ingest_request,
            priority=IngestPriority.EXPLICIT,
        )
        
        lengths = await queue.queue_lengths()
        assert lengths["explicit"] == 1
        assert lengths["realtime"] == 0
        assert lengths["bootstrap"] == 0

    @pytest.mark.asyncio
    async def test_enqueue_to_different_priorities(self, queue, ingest_request):
        """Test enqueue to different priority queues."""
        await queue.enqueue(uuid4(), ingest_request, IngestPriority.REALTIME)
        await queue.enqueue(uuid4(), ingest_request, IngestPriority.EXPLICIT)
        await queue.enqueue(uuid4(), ingest_request, IngestPriority.BOOTSTRAP)
        
        lengths = await queue.queue_lengths()
        assert lengths["realtime"] == 1
        assert lengths["explicit"] == 1
        assert lengths["bootstrap"] == 1

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_when_empty(self, queue):
        """Test dequeue returns None when queues are empty."""
        result = await queue.dequeue(timeout=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_dequeue_returns_item(self, queue, ingest_request):
        """Test dequeue returns enqueued item."""
        ingestion_id = uuid4()
        
        await queue.enqueue(
            ingestion_id=ingestion_id,
            request=ingest_request,
            priority=IngestPriority.EXPLICIT,
        )
        
        result = await queue.dequeue(timeout=1)
        
        assert result is not None
        assert result["ingestion_id"] == str(ingestion_id)
        assert "request" in result
        assert "enqueued_at" in result

    @pytest.mark.asyncio
    async def test_dequeue_priority_order(self, queue, ingest_request):
        """Test dequeue respects priority order: REALTIME > EXPLICIT > BOOTSTRAP."""
        id_bootstrap = uuid4()
        id_explicit = uuid4()
        id_realtime = uuid4()
        
        # Enqueue in non-priority order
        await queue.enqueue(id_bootstrap, ingest_request, IngestPriority.BOOTSTRAP)
        await queue.enqueue(id_explicit, ingest_request, IngestPriority.EXPLICIT)
        await queue.enqueue(id_realtime, ingest_request, IngestPriority.REALTIME)
        
        # Dequeue should return REALTIME first
        result1 = await queue.dequeue(timeout=1)
        assert result1["ingestion_id"] == str(id_realtime)
        
        # Then EXPLICIT
        result2 = await queue.dequeue(timeout=1)
        assert result2["ingestion_id"] == str(id_explicit)
        
        # Then BOOTSTRAP
        result3 = await queue.dequeue(timeout=1)
        assert result3["ingestion_id"] == str(id_bootstrap)

    @pytest.mark.asyncio
    async def test_dequeue_fifo_within_priority(self, queue, ingest_request):
        """Test dequeue is FIFO within same priority."""
        id1 = uuid4()
        id2 = uuid4()
        id3 = uuid4()
        
        # Enqueue multiple items with same priority
        await queue.enqueue(id1, ingest_request, IngestPriority.EXPLICIT)
        await queue.enqueue(id2, ingest_request, IngestPriority.EXPLICIT)
        await queue.enqueue(id3, ingest_request, IngestPriority.EXPLICIT)
        
        # Dequeue should return in FIFO order
        result1 = await queue.dequeue(timeout=1)
        result2 = await queue.dequeue(timeout=1)
        result3 = await queue.dequeue(timeout=1)
        
        assert result1["ingestion_id"] == str(id1)
        assert result2["ingestion_id"] == str(id2)
        assert result3["ingestion_id"] == str(id3)

    @pytest.mark.asyncio
    async def test_queue_lengths_returns_all_queues(self, queue, ingest_request):
        """Test queue_lengths returns counts for all queues."""
        await queue.enqueue(uuid4(), ingest_request, IngestPriority.REALTIME)
        await queue.enqueue(uuid4(), ingest_request, IngestPriority.REALTIME)
        await queue.enqueue(uuid4(), ingest_request, IngestPriority.EXPLICIT)
        
        lengths = await queue.queue_lengths()
        
        assert lengths["realtime"] == 2
        assert lengths["explicit"] == 1
        assert lengths["bootstrap"] == 0

    @pytest.mark.asyncio
    async def test_request_serialization_roundtrip(self, queue, ingest_request):
        """Test request can be serialized and deserialized."""
        ingestion_id = uuid4()
        
        await queue.enqueue(
            ingestion_id=ingestion_id,
            request=ingest_request,
            priority=IngestPriority.EXPLICIT,
        )
        
        result = await queue.dequeue(timeout=1)
        
        assert result is not None
        assert result["request"]["content"] == ingest_request.content
        assert result["request"]["scope"]["client_id"] == str(ingest_request.scope.client_id)
