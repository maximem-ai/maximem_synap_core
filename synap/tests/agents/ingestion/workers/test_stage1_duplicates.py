"""Stage 1 drops a duplicate only when another ingestion holds the same text in the same scope."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    DocumentSource,
    DocumentType,
    IngestDocumentRequest,
    IngestPriority,
    IngestResponse,
    IngestStatus,
    Scope,
)
from synap.cloud.agents.ingestion.pipeline.stage1_qualification import Stage1Qualifier
from synap.cloud.agents.ingestion.workers.stage1_worker import Stage1Worker
from synap.cloud.application.ingestion.idempotency import IdempotencyChecker, normalized_content_hash

TEXT = "The quarterly planning meeting covers hiring, budget and the product roadmap for the next two quarters."
LOW = UUID("00000000-0000-4000-8000-000000000001")
HIGH = UUID("ffffffff-0000-4000-8000-000000000001")


def _worker(redis):
    config = MagicMock()
    config.metadata.config_version = "v1"
    loader = AsyncMock()
    loader.get_config = AsyncMock(return_value=config)
    return Stage1Worker(
        qualifier=Stage1Qualifier(config=MagicMock()),
        config_loader=loader,
        redis=redis,
        status_tracker=AsyncMock(),
        reporting=MagicMock(),
        worker_id="stage1-test",
    )


def _payload(scope, ingestion_id):
    request = IngestDocumentRequest(
        content=TEXT,
        document_type=DocumentType.DOCUMENT,
        source=DocumentSource.API,
        scope=scope,
        priority=IngestPriority.EXPLICIT,
    )
    return {"ingestion_id": str(ingestion_id), "request": request.model_dump(mode="json")}


async def _record(redis, scope, ingestion_id):
    response = IngestResponse(
        request_id="r", ingestion_id=ingestion_id, status=IngestStatus.QUEUED,
        document_id="d", queued_at=datetime.utcnow(),
    )
    await IdempotencyChecker(redis).store(
        "r", response, content_hash=normalized_content_hash(TEXT), scope=scope)


@pytest.fixture
def scope():
    return Scope(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(), user_id=uuid4())


@pytest.mark.asyncio
async def test_larger_id_yields_to_the_recorded_ingestion(redis, scope):
    await _record(redis, scope, LOW)
    worker = _worker(redis)
    assert await worker.process(_payload(scope, HIGH)) is None
    assert "already stored" in worker._status.update.await_args.kwargs["error_message"]


@pytest.mark.asyncio
async def test_the_recorded_ingestion_itself_is_processed(redis, scope):
    await _record(redis, scope, LOW)
    assert await _worker(redis).process(_payload(scope, LOW)) is not None


@pytest.mark.asyncio
async def test_smaller_id_is_processed_so_a_pair_is_never_both_dropped(redis, scope):
    await _record(redis, scope, HIGH)
    assert await _worker(redis).process(_payload(scope, LOW)) is not None


@pytest.mark.asyncio
async def test_same_text_for_another_user_is_not_a_duplicate(redis, scope):
    await _record(redis, scope, LOW)
    other = Scope(
        client_id=scope.client_id, instance_id=scope.instance_id,
        customer_id=scope.customer_id, user_id=uuid4(),
    )
    assert await _worker(redis).process(_payload(other, HIGH)) is not None


@pytest.mark.asyncio
async def test_nothing_recorded_is_processed(redis, scope):
    assert await _worker(redis).process(_payload(scope, HIGH)) is not None
