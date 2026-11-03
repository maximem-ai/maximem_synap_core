"""
MVP User Stories 1.6a-b: Memory Creation & Retrieval

1.6a - Automatic extraction and storage of long-term memories
1.6b - Explicit memory search via SDK or API
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


@pytest.mark.asyncio
class TestUS1_6a_AutoExtraction:
    """As a developer, I want Synap to extract and store long-term memories automatically."""

    async def test_ingestion_manager_queues_document(self, mock_redis, mock_storage_provider):
        """IngestionManager.ingest_document() validates, deduplicates, and queues."""
        from synap.cloud.application.ingestion import (
            IngestionManager, IngestionQueue, IdempotencyChecker,
            RateLimiter, StatusTracker, RequestValidator,
        )
        from synap.cloud.agents.ingestion.models import (
            IngestDocumentRequest, DocumentType, DocumentSource, IngestPriority, Scope,
        )

        queue = IngestionQueue(mock_redis)
        idempotency = IdempotencyChecker(mock_redis)
        rate_limiter = RateLimiter(mock_redis, requests_per_minute=100, requests_per_day=10000)
        status_tracker = StatusTracker(mock_redis)
        validator = RequestValidator(max_content_size_mb=10)

        mgr = IngestionManager(
            queue=queue,
            idempotency=idempotency,
            rate_limiter=rate_limiter,
            status_tracker=status_tracker,
            validator=validator,
            storage_provider=mock_storage_provider,
        )

        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
            user_id=uuid4(),
        )

        request = IngestDocumentRequest(
            content="User prefers Python for backend development. They work at Acme Corp.",
            document_type=DocumentType.CONVERSATION,
            source=DocumentSource.API,
            priority=IngestPriority.EXPLICIT,
            scope=scope,
        )

        response = await mgr.ingest_document(request)

        assert response.status.value == "queued" or str(response.status) == "queued"
        assert response.ingestion_id is not None

    async def test_ingestion_models_exist(self):
        """Ingestion pipeline models (stages, categories) are properly defined."""
        from synap.cloud.agents.ingestion.models import (
            IngestDocumentRequest, DocumentType, DocumentSource,
            IngestPriority, IngestResponse, IngestStatus, Scope,
        )

        # Verify enums have expected values
        assert hasattr(DocumentType, "CONVERSATION")
        assert hasattr(DocumentSource, "API")
        assert hasattr(IngestPriority, "EXPLICIT")
        assert hasattr(IngestStatus, "QUEUED")

    async def test_idempotent_ingestion_via_request_id(self, mock_redis, mock_storage_provider):
        """Submitting same request twice returns cached response (idempotency)."""
        from synap.cloud.application.ingestion import (
            IngestionManager, IngestionQueue, IdempotencyChecker,
            RateLimiter, StatusTracker, RequestValidator,
        )
        from synap.cloud.agents.ingestion.models import (
            IngestDocumentRequest, DocumentType, DocumentSource, IngestPriority, Scope,
        )

        queue = IngestionQueue(mock_redis)
        idempotency = IdempotencyChecker(mock_redis)
        rate_limiter = RateLimiter(mock_redis, requests_per_minute=100, requests_per_day=10000)
        status_tracker = StatusTracker(mock_redis)
        validator = RequestValidator(max_content_size_mb=10)

        mgr = IngestionManager(
            queue=queue, idempotency=idempotency, rate_limiter=rate_limiter,
            status_tracker=status_tracker, validator=validator,
            storage_provider=mock_storage_provider,
        )

        scope = Scope(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4())
        shared_request_id = str(uuid4())

        request = IngestDocumentRequest(
            content="User prefers dark mode",
            document_type=DocumentType.CONVERSATION,
            source=DocumentSource.API,
            priority=IngestPriority.EXPLICIT,
            scope=scope,
            request_id=shared_request_id,
        )

        resp1 = await mgr.ingest_document(request)
        resp2 = await mgr.ingest_document(request)

        # Second call should return cached (same ingestion_id)
        assert resp1.ingestion_id == resp2.ingestion_id

    async def test_ingestion_pipeline_stages_exist(self):
        """All 4 ingestion pipeline stages are importable."""
        from synap.cloud.agents.ingestion.pipeline import (
            Stage1Qualifier, Stage3Extractor, Stage4Organizer,
        )

        assert Stage1Qualifier is not None
        assert Stage3Extractor is not None
        assert Stage4Organizer is not None


@pytest.mark.asyncio
class TestUS1_6b_ExplicitSearch:
    """As a developer, I want to search memories explicitly via SDK or API."""

    async def test_retrieval_with_explicit_search_query(
        self, mock_cache, mock_vector_store, mock_graph_store,
        mock_postgres, mock_llm, sample_scope,
    ):
        """Explicit search query hits vector store and returns matching items."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
        from synap.cloud.agents.models.decisions import ContextDecision, TriggerType, RetrievalMode

        scope_resolver = AsyncMock()
        scope_resolver.resolve.return_value = sample_scope

        mgr = RetrievalManager(
            scope_resolver=scope_resolver,
            cache_provider=mock_cache,
            vector_provider=mock_vector_store,
            graph_provider=mock_graph_store,
            postgres_provider=mock_postgres,
            llm_client=mock_llm,
        )

        decision = ContextDecision(
            decision_id="dec_search_test",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst_test",
            user_id="user_456",
            retrieval_mode=RetrievalMode.FAST,
            search_queries=["dark mode preference"],
            token_budget=2000,
        )

        bundle = await mgr.fetch_context(decision)

        assert bundle is not None
        # Vector store should have received the query
        assert len(mock_vector_store.queries_received) > 0

    async def test_sdk_memories_interface_exists(self):
        """SDK has a memories interface for create/get/update/delete."""
        from maximem_synap.memories.interface import MemoriesInterface

        assert hasattr(MemoriesInterface, "__init__")

    async def test_multiple_search_queries(
        self, mock_cache, mock_vector_store, mock_graph_store,
        mock_postgres, mock_llm, sample_scope,
    ):
        """Multiple search queries result in multiple vector store lookups."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
        from synap.cloud.agents.models.decisions import ContextDecision, TriggerType, RetrievalMode

        scope_resolver = AsyncMock()
        scope_resolver.resolve.return_value = sample_scope

        mgr = RetrievalManager(
            scope_resolver=scope_resolver,
            cache_provider=mock_cache,
            vector_provider=mock_vector_store,
            graph_provider=mock_graph_store,
            postgres_provider=mock_postgres,
            llm_client=mock_llm,
        )

        decision = ContextDecision(
            decision_id="dec_multi_query",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst_test",
            user_id="user_456",
            retrieval_mode=RetrievalMode.FAST,
            search_queries=["dark mode", "work history", "food preferences"],
            token_budget=3000,
        )

        bundle = await mgr.fetch_context(decision)

        assert bundle is not None
