"""Shared test fixtures for ingestion workers."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
import fakeredis.aioredis

from synap.cloud.agents.ingestion.models import (
    Scope,
    IngestDocumentRequest,
    IngestResponse,
    IngestStatus,
    IngestPriority,
    DocumentType,
    DocumentSource,
)


@pytest_asyncio.fixture
async def redis():
    """Create fake Redis client.

    Mirror the production ingestion Redis client, which is created with
    decode_responses=True (see cloud/application/ingestion/startup.py). The
    router's round-robin reads client IDs from a registry set via smembers and
    interpolates them into queue names; with bytes responses those names would
    include the b'...' repr and never match the real per-client queue keys.
    """
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_status_tracker():
    """Mock status tracker."""
    tracker = AsyncMock()
    tracker.update = AsyncMock()
    tracker.get = AsyncMock()
    tracker.create = AsyncMock()
    return tracker


@pytest.fixture
def mock_reporting():
    """Mock reporting interface."""
    reporting = MagicMock()
    reporting.write_metric = MagicMock()
    return reporting


@pytest.fixture
def scope():
    """Create test scope."""
    return Scope(
        client_id=uuid4(),
        instance_id=uuid4(),
        customer_id=uuid4(),
        user_id=uuid4(),
    )


@pytest.fixture
def ingestion_id():
    """Create test ingestion ID."""
    return uuid4()


@pytest.fixture
def sample_request(scope):
    """Create sample ingestion request."""
    return IngestDocumentRequest(
        content="This is a test document about project planning.",
        document_type=DocumentType.DOCUMENT,
        source=DocumentSource.API,
        scope=scope,
        priority=IngestPriority.EXPLICIT,
    )


@pytest.fixture
def sample_payload(ingestion_id, sample_request):
    """Create sample queue payload."""
    return {
        "ingestion_id": str(ingestion_id),
        "request": sample_request.model_dump(mode="json"),
        "enqueued_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def mock_config_loader():
    """Mock MACA config loader."""
    loader = AsyncMock()
    
    # Mock config with metadata
    mock_config = MagicMock()
    mock_config.metadata.config_version = "v1.0.0"
    
    loader.get_config = AsyncMock(return_value=mock_config)
    return loader


@pytest.fixture
def mock_qualifier():
    """Mock Stage1 qualifier."""
    return AsyncMock()


@pytest.fixture
def mock_extractor():
    """Mock Stage3 extractor."""
    return AsyncMock()


@pytest.fixture
def mock_organizer():
    """Mock Stage4 organizer."""
    return AsyncMock()
