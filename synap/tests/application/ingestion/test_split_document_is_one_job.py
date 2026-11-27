"""A document Stage 0 splits into chunks is still one job to the caller.

`ingest_document` used to return the first chunk's response, so for any
document over the chunk budget `memories.status(ingestion_id)` reported one
chunk: "completed" while the rest were still queued. With no `document_id`
supplied, every chunk also minted its own.
"""
from __future__ import annotations

import json
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    IngestStatus,
    Scope,
    ScopeRung,
)
from synap.cloud.agents.ingestion.pipeline.stage0_preprocessor import Stage0Preprocessor
from synap.cloud.application.ingestion import (
    IdempotencyChecker,
    IngestionManager,
    IngestionQueue,
    RateLimiter,
    RequestValidator,
    StatusTracker,
)
from unittest.mock import AsyncMock

BIG = "\n\n".join(f"Paragraph {i}. " + "The customer prefers email over phone. " * 20 for i in range(12))


class _Storage:
    persist_original = AsyncMock()
    persist_short_term = AsyncMock()
    get_short_term_by_session = AsyncMock(return_value=[])


@pytest.fixture
async def redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def manager(redis):
    m = IngestionManager(
        queue=IngestionQueue(redis),
        idempotency=IdempotencyChecker(redis),
        rate_limiter=RateLimiter(redis, requests_per_minute=100, requests_per_day=1000),
        status_tracker=StatusTracker(redis),
        validator=RequestValidator(max_content_size_mb=1),
        storage_provider=_Storage(),
    )
    m._preprocessor = Stage0Preprocessor(max_tokens=200)
    return m


@pytest.fixture
def scope():
    return Scope(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(), user_id=uuid4())


async def _children(redis, ingestion_id) -> list[UUID]:
    raw = await redis.get(f"ingestion:children:{ingestion_id}")
    return [UUID(c) for c in json.loads(raw)] if raw else []


@pytest.mark.asyncio
async def test_the_returned_id_reports_every_chunk(manager, redis, scope):
    resp = await manager.ingest_document(IngestDocumentRequest(content=BIG, scope=scope))
    children = await _children(redis, resp.ingestion_id)
    assert len(children) > 1, "control: the document must actually be split"
    assert resp.ingestion_id not in children

    assert (await manager.get_status(resp.ingestion_id)).status == IngestStatus.QUEUED

    await manager._status.update(children[0], IngestStatus.COMPLETED, memories_created=2)
    status = await manager.get_status(resp.ingestion_id)
    assert status.status == IngestStatus.PROCESSING, (
        "one finished chunk must not report the whole document as completed")

    for c in children[1:]:
        await manager._status.update(c, IngestStatus.COMPLETED, memories_created=1)
    status = await manager.get_status(resp.ingestion_id)
    assert status.status == IngestStatus.COMPLETED
    assert status.memories_created == 2 + len(children) - 1


@pytest.mark.asyncio
async def test_a_failed_chunk_makes_it_partial_success(manager, redis, scope):
    resp = await manager.ingest_document(IngestDocumentRequest(content=BIG, scope=scope))
    children = await _children(redis, resp.ingestion_id)
    await manager._status.update(children[0], IngestStatus.FAILED, error_message="boom")
    for c in children[1:]:
        await manager._status.update(c, IngestStatus.COMPLETED)
    status = await manager.get_status(resp.ingestion_id)
    assert status.status == IngestStatus.PARTIAL_SUCCESS
    assert status.error_message == "boom"


@pytest.mark.parametrize("supplied", [None, "my-doc-1"])
@pytest.mark.asyncio
async def test_every_chunk_carries_one_document_id(manager, redis, scope, supplied):
    resp = await manager.ingest_document(
        IngestDocumentRequest(content=BIG, scope=scope, document_id=supplied))
    if supplied:
        assert resp.document_id == supplied
    ids = {(await manager._status.get(c)).document_id for c in await _children(redis, resp.ingestion_id)}
    assert ids == {resp.document_id}


@pytest.mark.asyncio
async def test_sending_it_again_returns_the_first_ingestion(manager, scope):
    first = await manager.ingest_document(IngestDocumentRequest(content=BIG, scope=scope))
    again = await manager.ingest_document(IngestDocumentRequest(content=BIG, scope=scope))
    assert again.ingestion_id == first.ingestion_id
    assert again.document_id == first.document_id


@pytest.mark.asyncio
async def test_a_small_document_is_unchanged(manager, redis, scope):
    resp = await manager.ingest_document(IngestDocumentRequest(content="Prefers email.", scope=scope))
    assert await _children(redis, resp.ingestion_id) == []
    assert (await manager.get_status(resp.ingestion_id)).status == IngestStatus.QUEUED


def _at(scope: Scope, node) -> Scope:
    return scope.model_copy(update={"user_id": None, "chain": [
        ScopeRung(node_id=scope.customer_id, level_key="customer"),
        *([ScopeRung(node_id=node, level_key="team")] if node else []),
    ]})


@pytest.mark.asyncio
async def test_the_same_text_at_two_rungs_is_two_writes(manager, scope):
    """Dedup used to key on customer and user only, so a team-level write had
    the same key as a customer-level one and as every sibling team's."""
    payments, billing = uuid4(), uuid4()
    a = await manager.ingest_document(IngestDocumentRequest(content="Uses SEPA.", scope=_at(scope, payments)))
    b = await manager.ingest_document(IngestDocumentRequest(content="Uses SEPA.", scope=_at(scope, billing)))
    c = await manager.ingest_document(IngestDocumentRequest(content="Uses SEPA.", scope=_at(scope, None)))
    assert len({a.ingestion_id, b.ingestion_id, c.ingestion_id}) == 3


@pytest.mark.asyncio
async def test_the_same_text_at_one_rung_is_still_deduplicated(manager, scope):
    payments = uuid4()
    a = await manager.ingest_document(IngestDocumentRequest(content="Uses SEPA.", scope=_at(scope, payments)))
    b = await manager.ingest_document(IngestDocumentRequest(content="Uses SEPA.", scope=_at(scope, payments)))
    assert a.ingestion_id == b.ingestion_id


class _RecordingStorage(_Storage):
    def __init__(self):
        self.saved = []

    async def persist_original(self, ingestion_id, content, scope):
        self.saved.append((ingestion_id, content))


@pytest.mark.asyncio
async def test_a_split_document_saves_its_text_under_the_returned_id(redis, scope):
    """The uploaded file's original is saved under the returned id, so the text copy
    has to be under that id too, once, as the whole document."""
    import asyncio

    storage = _RecordingStorage()
    m = IngestionManager(
        queue=IngestionQueue(redis),
        idempotency=IdempotencyChecker(redis),
        rate_limiter=RateLimiter(redis, requests_per_minute=100, requests_per_day=1000),
        status_tracker=StatusTracker(redis),
        validator=RequestValidator(max_content_size_mb=1),
        storage_provider=storage,
    )
    m._preprocessor = Stage0Preprocessor(max_tokens=200)

    resp = await m.ingest_document(IngestDocumentRequest(content=BIG, scope=scope))
    assert len(await _children(redis, resp.ingestion_id)) > 1, "control: the document must be split"
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert storage.saved == [(resp.ingestion_id, BIG)]


@pytest.mark.asyncio
async def test_a_small_document_still_saves_its_text_under_its_own_id(redis, scope):
    import asyncio

    storage = _RecordingStorage()
    m = IngestionManager(
        queue=IngestionQueue(redis),
        idempotency=IdempotencyChecker(redis),
        rate_limiter=RateLimiter(redis, requests_per_minute=100, requests_per_day=1000),
        status_tracker=StatusTracker(redis),
        validator=RequestValidator(max_content_size_mb=1),
        storage_provider=storage,
    )
    resp = await m.ingest_document(IngestDocumentRequest(content="Prefers email.", scope=scope))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert storage.saved == [(resp.ingestion_id, "Prefers email.")]


def test_single_over_budget_line_is_split_into_chunks():
    """A document containing a single line exceeding token budget must be split into chunks (#33)."""
    preprocessor = Stage0Preprocessor(max_tokens=30)
    single_long_line = "The quick brown fox jumps over the lazy dog. " * 10

    chunks = preprocessor._force_split(single_long_line, max_tokens=30)
    assert len(chunks) > 1, "Must split single line into multiple chunks"
    for chunk in chunks:
        assert preprocessor._counter.count(chunk) <= 30, f"Chunk exceeds max_tokens: {chunk}"


def test_process_splits_document_with_single_over_budget_line(scope):
    """Stage0Preprocessor.preprocess must produce multiple sub-requests for a single over-budget line (#33)."""
    preprocessor = Stage0Preprocessor(max_tokens=30)
    single_long_line = "Client prefers remote communication via asynchronous updates. " * 8
    req = IngestDocumentRequest(content=single_long_line, scope=scope)

    results = preprocessor.preprocess(req)
    assert len(results) > 1, "Must split single over-budget line into sub-requests"
    for sub in results:
        assert "_chunk" in sub.metadata
        assert sub.metadata["_chunk"]["total_chunks"] == len(results)
        assert preprocessor._counter.count(sub.content) <= 30


def test_process_splits_unbroken_string_over_budget(scope):
    """A single unbroken token sequence exceeding max_tokens must also be chunked safely (#33)."""
    preprocessor = Stage0Preprocessor(max_tokens=10)
    unbroken_string = "a" * 300
    req = IngestDocumentRequest(content=unbroken_string, scope=scope)

    results = preprocessor.preprocess(req)
    assert len(results) > 1, "Must split unbroken string into sub-requests"
    for sub in results:
        assert preprocessor._counter.count(sub.content) <= 10


def test_process_splits_numeric_space_separated_line_without_exceeding_budget(scope):
    """Joined word chunks must not exceed token budget when spaces add tokens (#33)."""
    preprocessor = Stage0Preprocessor(max_tokens=50)
    # A single line with many space-separated numbers
    numeric_line = " ".join(["2"] * 500)
    req = IngestDocumentRequest(content=numeric_line, scope=scope)

    results = preprocessor.preprocess(req)
    assert len(results) > 1, "Must split numeric line into multiple sub-requests"
    for sub in results:
        token_count = preprocessor._counter.count(sub.content)
        assert token_count <= 50, f"Sub-request exceeds budget: {token_count} tokens"


def test_conversation_preamble_fits_with_first_turn():
    """When leading preamble fits with first turn within max_tokens, it bundles into the first chunk (#32)."""
    preprocessor = Stage0Preprocessor(max_tokens=60)
    # Preamble without a recognized role prefix (e.g. not User:, Assistant:, System:, etc.)
    preamble = "Session metadata: Client is Alice from Berlin and prefers asynchronous updates."
    turn1 = "user: What is the weather in Berlin today?"
    turn2 = "assistant: It is currently 15 degrees and cloudy in Berlin."
    turn3 = "user: Thanks! Also remember that I like cycling."
    turn4 = "assistant: Noted, I will keep your interest in cycling in mind."

    full_conversation = f"{preamble}\n{turn1}\n{turn2}\n{turn3}\n{turn4}"

    chunks = preprocessor._chunk_conversation_by_turns(full_conversation, max_tokens=35)
    assert len(chunks) > 1, "Must split into multiple chunks"
    # Preamble should be preserved in the first chunk together with turn1
    assert "Session metadata: Client is Alice" in chunks[0]
    assert "What is the weather" in chunks[0]
    assert any("cycling" in c for c in chunks)


def test_conversation_preamble_needs_separate_chunk():
    """When leading preamble exceeds budget when paired with first turn, it is emitted as its own chunk (#32)."""
    preprocessor = Stage0Preprocessor(max_tokens=30)
    # Long preamble that does not fit with the first turn under max_tokens=25
    preamble = "Session context and historical summary: Client is Alice from Berlin, works as an architect, and has multiple ongoing projects."
    turn1 = "user: What is the weather in Berlin today?"
    turn2 = "assistant: It is currently 15 degrees and cloudy in Berlin."

    full_conversation = f"{preamble}\n{turn1}\n{turn2}"

    chunks = preprocessor._chunk_conversation_by_turns(full_conversation, max_tokens=25)
    assert len(chunks) > 1, "Must split into multiple chunks"
    # First chunk must contain the preamble
    assert "Session context and historical summary" in chunks[0]
    # Subsequent chunk must contain the user query
    assert any("What is the weather" in c for c in chunks[1:])
