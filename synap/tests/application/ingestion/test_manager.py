"""Tests for IngestionManager."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import fakeredis.aioredis

from synap.cloud.agents.ingestion.models import (
    ConvertSessionRequest,
    DocumentSource,
    DocumentType,
    IngestBatchRequest,
    IngestDocumentRequest,
    IngestFromListeningAgentRequest,
    IngestPriority,
    IngestStatus,
    Scope,
)
from synap.cloud.application.ingestion import (
    IdempotencyChecker,
    IngestionManager,
    IngestionQueue,
    RateLimiter,
    RequestValidator,
    StatusTracker,
)
from synap.cloud.application.models.errors import ValidationError, RateLimitExceeded


class MockStorageProvider:
    """Mock storage provider for testing."""
    
    def __init__(self):
        self.persist_original = AsyncMock()
        self.persist_short_term = AsyncMock()
        self.get_short_term_by_session = AsyncMock(return_value=[])


class TestIngestionManager:
    """Tests for IngestionManager."""

    @pytest.fixture
    async def redis(self):
        """Create fake Redis client."""
        return fakeredis.aioredis.FakeRedis()

    @pytest.fixture
    def storage(self):
        """Create mock storage provider."""
        return MockStorageProvider()

    @pytest.fixture
    def manager(self, redis, storage):
        """Create ingestion manager with all dependencies."""
        queue = IngestionQueue(redis)
        idempotency = IdempotencyChecker(redis)
        rate_limiter = RateLimiter(redis, requests_per_minute=10, requests_per_day=100)
        status_tracker = StatusTracker(redis)
        validator = RequestValidator(max_content_size_mb=1)
        
        return IngestionManager(
            queue=queue,
            idempotency=idempotency,
            rate_limiter=rate_limiter,
            status_tracker=status_tracker,
            validator=validator,
            storage_provider=storage,
        )

    @pytest.fixture
    def scope(self):
        """Create test scope."""
        return Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )

    @pytest.fixture
    def doc_request(self, scope):
        """Create test document request."""
        return IngestDocumentRequest(
            content="Test content",
            scope=scope,
        )

    @pytest.mark.asyncio
    async def test_ingest_document_success(self, manager, doc_request):
        """Test successful document ingestion."""
        response = await manager.ingest_document(doc_request)

        assert response.status == IngestStatus.QUEUED
        assert response.request_id == doc_request.request_id
        assert response.ingestion_id is not None
        assert response.document_id is not None

    @pytest.mark.asyncio
    async def test_ingest_document_validates_request(self, manager, scope):
        """Test ingestion validates request."""
        # Empty content should fail validation
        invalid_request = IngestDocumentRequest(
            content="",
            scope=scope,
        )
        
        with pytest.raises(ValidationError):
            await manager.ingest_document(invalid_request)

    @pytest.mark.asyncio
    async def test_ingest_document_checks_rate_limit(self, manager, doc_request):
        """Test ingestion checks rate limits."""
        # Use up rate limit
        for _ in range(10):
            await manager.ingest_document(doc_request)

        # Next request should be rate limited
        with pytest.raises(RateLimitExceeded) as exc_info:
            await manager.ingest_document(doc_request)

        assert exc_info.value.retry_after is not None

    @pytest.mark.asyncio
    async def test_ingest_document_checks_idempotency(self, manager, doc_request):
        """Test ingestion returns cached response for duplicate request."""
        # First ingestion
        response1 = await manager.ingest_document(doc_request)

        # Same request should return cached response
        response2 = await manager.ingest_document(doc_request)

        assert response1.ingestion_id == response2.ingestion_id
        assert response1.request_id == response2.request_id

    @pytest.mark.asyncio
    async def test_ingest_document_persists_to_storage(self, manager, doc_request, storage):
        """Test ingestion persists original to S3."""
        await manager.ingest_document(doc_request)
        
        # Give async task time to execute
        import asyncio
        await asyncio.sleep(0.1)
        
        # Storage provider should have been called
        assert storage.persist_original.called

    @pytest.mark.asyncio
    async def test_ingest_document_tracks_status(self, manager, doc_request):
        """Test ingestion tracks status."""
        response = await manager.ingest_document(doc_request)
        
        # Status should be retrievable
        status = await manager.get_status(response.ingestion_id)
        
        assert status is not None
        assert status.ingestion_id == response.ingestion_id
        assert status.status == IngestStatus.QUEUED

    @pytest.mark.asyncio
    async def test_ingest_batch_success(self, manager, scope):
        """Test successful batch ingestion."""
        doc1 = IngestDocumentRequest(content="Doc 1", scope=scope)
        doc2 = IngestDocumentRequest(content="Doc 2", scope=scope)
        
        batch_request = IngestBatchRequest(
            documents=[doc1, doc2],
            scope=scope,
        )
        
        response = await manager.ingest_batch(batch_request)
        
        assert response.status == IngestStatus.QUEUED
        assert response.total_documents == 2
        assert response.succeeded == 2
        assert response.failed == 0
        assert len(response.document_results) == 2

    @pytest.mark.asyncio
    async def test_ingest_batch_partial_failure(self, manager, scope):
        """Test batch ingestion with partial failure."""
        doc1 = IngestDocumentRequest(content="Valid doc", scope=scope)
        doc2 = IngestDocumentRequest(content="", scope=scope)  # Invalid
        
        batch_request = IngestBatchRequest(
            documents=[doc1, doc2],
            scope=scope,
            fail_fast=False,
        )
        
        response = await manager.ingest_batch(batch_request)
        
        assert response.status == IngestStatus.PARTIAL_SUCCESS
        assert response.succeeded == 1
        assert response.failed == 1

    @pytest.mark.asyncio
    async def test_ingest_batch_fail_fast(self, manager, scope):
        """Test batch ingestion with fail_fast."""
        doc1 = IngestDocumentRequest(content="", scope=scope)  # Invalid
        doc2 = IngestDocumentRequest(content="Valid doc", scope=scope)
        
        batch_request = IngestBatchRequest(
            documents=[doc1, doc2],
            scope=scope,
            fail_fast=True,
        )
        
        with pytest.raises(ValidationError):
            await manager.ingest_batch(batch_request)

    @pytest.mark.asyncio
    async def test_ingest_from_listening_agent_short_term_direct(
        self, manager, scope, storage
    ):
        """Test listening agent short-term direct storage path."""
        request = IngestFromListeningAgentRequest(
            session_id="session-123",
            conversation_id="conv-456",
            content="Important fact",
            content_type="compacted_context",
            scope=scope,
            destination="short_term",
            pre_qualified=True,
            qualification_result={"nature": "substantive"},
        )
        
        response = await manager.ingest_from_listening_agent(request)
        
        assert response.status == IngestStatus.COMPLETED
        assert response.memories_created == 1
        assert storage.persist_short_term.called

    @pytest.mark.asyncio
    async def test_ingest_from_listening_agent_full_pipeline(
        self, manager, scope, storage
    ):
        """Test listening agent full pipeline path."""
        request = IngestFromListeningAgentRequest(
            session_id="session-123",
            conversation_id="conv-456",
            content="Need processing",
            content_type="turn",
            scope=scope,
            destination="long_term",
            pre_qualified=False,
        )
        
        response = await manager.ingest_from_listening_agent(request)
        
        assert response.status == IngestStatus.QUEUED
        assert not storage.persist_short_term.called

    @pytest.mark.asyncio
    async def test_convert_session_to_longterm_empty_session(self, manager, scope):
        """Test session conversion with no short-term memories."""
        request = ConvertSessionRequest(
            session_id="session-789",
            scope=scope,
        )
        
        response = await manager.convert_session_to_longterm(request)
        
        assert response.status == IngestStatus.COMPLETED
        assert response.memories_created == 0

    @pytest.mark.asyncio
    async def test_convert_session_to_longterm_with_memories(
        self, manager, scope, storage
    ):
        """Test session conversion with short-term memories."""
        # Mock short-term memories
        storage.get_short_term_by_session.return_value = [
            {"content": "Memory 1"},
            {"content": "Memory 2"},
        ]
        
        request = ConvertSessionRequest(
            session_id="session-789",
            scope=scope,
        )
        
        response = await manager.convert_session_to_longterm(request)
        
        assert response.status == IngestStatus.QUEUED
        assert storage.get_short_term_by_session.called

    @pytest.mark.asyncio
    async def test_get_status_returns_none_for_unknown_id(self, manager):
        """Test get_status returns None for unknown ingestion ID."""
        status = await manager.get_status(uuid4())
        assert status is None

    @pytest.mark.asyncio
    async def test_get_status_returns_current_status(self, manager, doc_request):
        """Test get_status returns current status."""
        response = await manager.ingest_document(doc_request)
        
        status = await manager.get_status(response.ingestion_id)
        
        assert status is not None
        assert status.ingestion_id == response.ingestion_id

    @pytest.mark.asyncio
    async def test_content_hash_deduplication(self, manager, scope):
        """Test content hash prevents duplicate ingestion."""
        # Two requests with same content but different request IDs
        request1 = IngestDocumentRequest(
            request_id="req-1",
            content="Same content",
            scope=scope,
        )
        request2 = IngestDocumentRequest(
            request_id="req-2",
            content="Same content",
            scope=scope,
        )
        
        response1 = await manager.ingest_document(request1)
        response2 = await manager.ingest_document(request2)
        
        # Should return same ingestion_id due to content hash match
        assert response1.ingestion_id == response2.ingestion_id

    @pytest.mark.asyncio
    async def test_whitespace_normalization_in_hash(self, manager, scope):
        """Test content hash normalizes whitespace."""
        request1 = IngestDocumentRequest(
            request_id="req-1",
            content="Hello    world",
            scope=scope,
        )
        request2 = IngestDocumentRequest(
            request_id="req-2",
            content="Hello world",
            scope=scope,
        )
        
        response1 = await manager.ingest_document(request1)
        response2 = await manager.ingest_document(request2)
        
        # Should match due to whitespace normalization
        assert response1.ingestion_id == response2.ingestion_id

    @pytest.mark.asyncio
    async def test_trigger_post_batch_consolidation_passes_client_id(
        self, manager, scope
    ):
        """Retired (D-023): the hook must REFUSE rather than start the legacy
        hard-deleting ConsolidationManager. Consolidation now runs in its own
        worker (synap/cloud/application/consolidation)."""
        import pytest
        with pytest.raises(RuntimeError, match="retired"):
            await manager.trigger_post_batch_consolidation(
                batch_id=uuid4(),
                instance_id="inst_def456",
                client_id="cli_abc123",
            )

    @pytest.mark.asyncio
    async def test_trigger_post_batch_consolidation_returns_result(
        self, manager, scope
    ):
        """Retired (D-023): the hook refuses even with full scope args; it must
        never start the legacy hard-deleting engine."""
        import pytest
        with pytest.raises(RuntimeError, match="retired"):
            await manager.trigger_post_batch_consolidation(
                batch_id=uuid4(),
                instance_id="inst_abc",
                client_id="cli_xyz",
                user_id="user_1",
                customer_id="cust_2",
            )
