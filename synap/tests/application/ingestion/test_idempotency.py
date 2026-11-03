"""Tests for IdempotencyChecker."""

from datetime import datetime
from uuid import uuid4

import pytest
import fakeredis.aioredis

from synap.cloud.agents.ingestion.models import (
    IngestResponse,
    IngestStatus,
    Scope,
)
from synap.cloud.application.ingestion.idempotency import IdempotencyChecker


class TestIdempotencyChecker:
    """Tests for IdempotencyChecker."""

    @pytest.fixture
    async def redis(self):
        """Create fake Redis client."""
        return fakeredis.aioredis.FakeRedis()

    @pytest.fixture
    def idempotency(self, redis):
        """Create idempotency checker."""
        return IdempotencyChecker(redis=redis)

    @pytest.fixture
    def scope(self):
        """Create test scope."""
        return Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )

    @pytest.fixture
    def response(self):
        """Create test response."""
        return IngestResponse(
            request_id="req-123",
            ingestion_id=uuid4(),
            status=IngestStatus.QUEUED,
            document_id="doc-456",
            queued_at=datetime.utcnow(),
        )

    @pytest.mark.asyncio
    async def test_check_returns_none_for_new_request(self, idempotency, scope):
        """Test check returns None for new request."""
        result = await idempotency.check(
            request_id="new-request",
            content_hash="hash123",
            scope=scope,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_check_returns_cached_for_duplicate_request_id(
        self, idempotency, scope, response
    ):
        """Test check returns cached response for duplicate request ID."""
        # Store response
        await idempotency.store(
            request_id=response.request_id,
            response=response,
        )
        
        # Check should return cached response
        result = await idempotency.check(
            request_id=response.request_id,
            content_hash="any-hash",
            scope=scope,
        )
        
        assert result is not None
        assert result.request_id == response.request_id
        assert result.ingestion_id == response.ingestion_id

    @pytest.mark.asyncio
    async def test_check_returns_cached_for_duplicate_content_hash(
        self, idempotency, scope, response
    ):
        """Test check returns cached response for duplicate content hash."""
        content_hash = "duplicate-content"
        
        # Store response with content hash
        await idempotency.store(
            request_id=response.request_id,
            response=response,
            content_hash=content_hash,
            scope=scope,
        )
        
        # Check with same content hash but different request ID
        result = await idempotency.check(
            request_id="different-request-id",
            content_hash=content_hash,
            scope=scope,
        )
        
        assert result is not None
        assert result.request_id == response.request_id

    @pytest.mark.asyncio
    async def test_check_scopes_content_hash_by_client_and_customer(
        self, idempotency, response
    ):
        """Test content hash is scoped by client and customer."""
        content_hash = "same-content"
        
        scope1 = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        scope2 = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
        )
        
        # Store for scope1
        await idempotency.store(
            request_id=response.request_id,
            response=response,
            content_hash=content_hash,
            scope=scope1,
        )
        
        # Check with scope2 should return None (different scope)
        result = await idempotency.check(
            request_id="different-request",
            content_hash=content_hash,
            scope=scope2,
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_store_without_content_hash(self, idempotency, response):
        """Test store works without content hash."""
        await idempotency.store(
            request_id=response.request_id,
            response=response,
        )
        
        # Should be retrievable by request ID
        result = await idempotency.check(
            request_id=response.request_id,
            content_hash="any",
            scope=Scope(
                client_id=uuid4(),
                instance_id=uuid4(),
                customer_id=uuid4(),
            ),
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_response_serialization_roundtrip(
        self, idempotency, scope, response
    ):
        """Test response can be serialized and deserialized."""
        await idempotency.store(
            request_id=response.request_id,
            response=response,
            content_hash="hash",
            scope=scope,
        )
        
        result = await idempotency.check(
            request_id=response.request_id,
            content_hash="hash",
            scope=scope,
        )
        
        assert result.status == response.status
        assert result.document_id == response.document_id
        assert result.memories_created == response.memories_created
