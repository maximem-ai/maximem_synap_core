"""Tests for StatusTracker."""

from datetime import datetime
from uuid import uuid4

import pytest
import fakeredis.aioredis

from synap.cloud.agents.ingestion.models import IngestResponse, IngestStatus
from synap.cloud.application.ingestion.status import StatusTracker


class TestStatusTracker:
    """Tests for StatusTracker."""

    @pytest.fixture
    async def redis(self):
        """Create fake Redis client."""
        return fakeredis.aioredis.FakeRedis()

    @pytest.fixture
    def tracker(self, redis):
        """Create status tracker."""
        return StatusTracker(redis=redis)

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
    async def test_create_stores_status(self, tracker, response):
        """Test create stores status."""
        await tracker.create(
            ingestion_id=response.ingestion_id,
            response=response,
        )
        
        result = await tracker.get(response.ingestion_id)
        
        assert result is not None
        assert result.request_id == response.request_id
        assert result.status == response.status

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_id(self, tracker):
        """Test get returns None for missing ingestion ID."""
        result = await tracker.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_stored_response(self, tracker, response):
        """Test get returns stored response."""
        await tracker.create(response.ingestion_id, response)
        
        result = await tracker.get(response.ingestion_id)
        
        assert result is not None
        assert result.ingestion_id == response.ingestion_id
        assert result.document_id == response.document_id

    @pytest.mark.asyncio
    async def test_update_changes_status(self, tracker, response):
        """Test update changes status."""
        await tracker.create(response.ingestion_id, response)
        
        # Update status
        await tracker.update(
            ingestion_id=response.ingestion_id,
            status=IngestStatus.PROCESSING,
            started_at=datetime.utcnow(),
        )
        
        result = await tracker.get(response.ingestion_id)
        
        assert result is not None
        assert result.status == IngestStatus.PROCESSING
        assert result.started_at is not None

    @pytest.mark.asyncio
    async def test_update_changes_multiple_fields(self, tracker, response):
        """Test update can change multiple fields."""
        await tracker.create(response.ingestion_id, response)
        
        completed_time = datetime.utcnow()
        await tracker.update(
            ingestion_id=response.ingestion_id,
            status=IngestStatus.COMPLETED,
            completed_at=completed_time,
            memories_created=5,
            entities_created=3,
        )
        
        result = await tracker.get(response.ingestion_id)
        
        assert result.status == IngestStatus.COMPLETED
        assert result.completed_at == completed_time
        assert result.memories_created == 5
        assert result.entities_created == 3

    @pytest.mark.asyncio
    async def test_update_ignores_invalid_fields(self, tracker, response):
        """Test update ignores invalid field names."""
        await tracker.create(response.ingestion_id, response)
        
        # Update with invalid field should not raise error
        await tracker.update(
            ingestion_id=response.ingestion_id,
            status=IngestStatus.PROCESSING,
            invalid_field="should be ignored",
        )
        
        result = await tracker.get(response.ingestion_id)
        assert result.status == IngestStatus.PROCESSING
        assert not hasattr(result, "invalid_field")

    @pytest.mark.asyncio
    async def test_update_noop_for_missing_id(self, tracker):
        """Test update does nothing for missing ingestion ID."""
        # Should not raise error
        await tracker.update(
            ingestion_id=uuid4(),
            status=IngestStatus.COMPLETED,
        )

    @pytest.mark.asyncio
    async def test_status_serialization_roundtrip(self, tracker, response):
        """Test response can be serialized and deserialized."""
        response.memories_created = 10
        response.candidates_created = 3
        
        await tracker.create(response.ingestion_id, response)
        
        result = await tracker.get(response.ingestion_id)
        
        assert result.memories_created == 10
        assert result.candidates_created == 3
        assert result.status == response.status
