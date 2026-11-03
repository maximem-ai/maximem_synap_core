"""A document Stage 0 splits is counted once against the per-minute limit, not skipped and not per chunk."""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import fakeredis.aioredis
import pytest

from synap.cloud.agents.ingestion.models import IngestDocumentRequest, Scope
from synap.cloud.agents.ingestion.pipeline.stage0_preprocessor import Stage0Preprocessor
from synap.cloud.application.ingestion import (
    IdempotencyChecker,
    IngestionManager,
    IngestionQueue,
    RateLimiter,
    RequestValidator,
    StatusTracker,
)
from synap.cloud.application.models.errors import RateLimitExceeded


def _big(tag: str) -> str:
    return "\n\n".join(
        f"{tag} paragraph {i}. " + "The customer prefers email over phone. " * 20 for i in range(12))


class _Storage:
    persist_original = AsyncMock()
    persist_short_term = AsyncMock()
    get_short_term_by_session = AsyncMock(return_value=[])


@pytest.fixture
async def redis():
    return fakeredis.aioredis.FakeRedis()


def _manager(redis, per_minute):
    m = IngestionManager(
        queue=IngestionQueue(redis),
        idempotency=IdempotencyChecker(redis),
        rate_limiter=RateLimiter(redis, requests_per_minute=per_minute, requests_per_day=1000),
        status_tracker=StatusTracker(redis),
        validator=RequestValidator(max_content_size_mb=1),
        storage_provider=_Storage(),
    )
    m._preprocessor = Stage0Preprocessor(max_tokens=200)
    return m


@pytest.fixture
def scope():
    return Scope(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(), user_id=uuid4())


@pytest.mark.asyncio
async def test_a_split_document_is_one_request_against_the_limit(redis, scope):
    manager = _manager(redis, per_minute=1)
    first = await manager.ingest_document(IngestDocumentRequest(content=_big("alpha"), scope=scope))
    assert first.ingestion_id is not None
    with pytest.raises(RateLimitExceeded):
        await manager.ingest_document(IngestDocumentRequest(content=_big("beta"), scope=scope))


@pytest.mark.asyncio
async def test_a_refused_split_document_queues_no_chunk(redis, scope):
    manager = _manager(redis, per_minute=1)
    await manager.ingest_document(IngestDocumentRequest(content="Prefers email, not phone, for all of the quarterly updates and notices.", scope=scope))
    with pytest.raises(RateLimitExceeded):
        await manager.ingest_document(IngestDocumentRequest(content=_big("gamma"), scope=scope))
    assert len(await redis.keys("ingestion:status:*")) == 1


@pytest.mark.asyncio
async def test_the_skip_flag_still_skips_the_check(redis, scope):
    manager = _manager(redis, per_minute=1)
    for tag in ("one", "two", "three"):
        await manager.ingest_document(
            IngestDocumentRequest(content=_big(tag), scope=scope), skip_per_minute_check=True)
