"""guardrails.rate_limits.max_memories_per_user_per_day, enforced at the entry.

Every test drives the real IngestionManager over fakeredis. The config loader
is replaced by a fake that records its calls, so "the config was not read" is
something the tests can assert.
"""

from datetime import datetime, timezone
from uuid import uuid4

import fakeredis.aioredis
import pytest

from synap.cloud.agents.ingestion.models import IngestDocumentRequest, Scope
from synap.cloud.application.ingestion import (
    IdempotencyChecker,
    IngestionManager,
    IngestionQueue,
    RateLimiter,
    RequestValidator,
    StatusTracker,
)
from synap.cloud.application.models.errors import RateLimitExceeded
from synap.cloud.memory_config.config_schema import ConfigMetadata, MemoryArchitectureConfig
from unittest.mock import AsyncMock

FLAG = "SYNAP_GUARDRAIL_INGEST_CAP"


def _config(**rate_limits) -> MemoryArchitectureConfig:
    raw = {"metadata": ConfigMetadata(
        client_id="c", instance_id="i", config_version="1.0.0",
        generated_at=datetime.now(timezone.utc), generated_by="test")}
    if rate_limits:
        raw["guardrails"] = {"rate_limits": rate_limits}
    return MemoryArchitectureConfig(**raw)


class FakeLoader:
    def __init__(self, config):
        self.config, self.calls = config, 0

    async def get_config(self, instance_id=None):
        self.calls += 1
        if isinstance(self.config, Exception):
            raise self.config
        return self.config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(f"{FLAG}_INSTANCES", raising=False)


@pytest.fixture
def loader(monkeypatch):
    fake = FakeLoader(_config(max_memories_per_user_per_day=2))
    monkeypatch.setattr("synap.cloud.memory_config.loader.get_config_loader", lambda: fake)
    return fake


@pytest.fixture
def limiter():
    return RateLimiter(fakeredis.aioredis.FakeRedis(), requests_per_minute=1000, requests_per_day=1000)


@pytest.fixture
def manager(limiter):
    redis = fakeredis.aioredis.FakeRedis()
    storage = type("S", (), {"persist_original": AsyncMock(), "persist_short_term": AsyncMock(),
                             "get_short_term_by_session": AsyncMock(return_value=[])})()
    return IngestionManager(
        queue=IngestionQueue(redis), idempotency=IdempotencyChecker(redis),
        rate_limiter=limiter, status_tracker=StatusTracker(redis),
        validator=RequestValidator(max_content_size_mb=1), storage_provider=storage)


INSTANCE = uuid4()
CLIENT = uuid4()


def _request(user, n=0):
    return IngestDocumentRequest(
        content=f"note number {n} {uuid4()}",
        scope=Scope(client_id=CLIENT, instance_id=INSTANCE, customer_id=uuid4(), user_id=user))


async def _send(manager, user, times):
    for i in range(times):
        await manager.ingest_document(_request(user, i))


@pytest.mark.asyncio
async def test_switch_off_nothing_is_counted_and_the_config_is_not_read(manager, loader):
    user = uuid4()
    await _send(manager, user, 5)
    assert loader.calls == 0


@pytest.mark.asyncio
async def test_cap_reached_refuses_with_the_per_client_limiters_error(manager, loader, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    user = uuid4()
    await _send(manager, user, 2)
    with pytest.raises(RateLimitExceeded) as exc:
        await manager.ingest_document(_request(user))
    assert exc.value.retry_after > 0
    assert exc.value.code == "RATE_LIMIT_EXCEEDED"


@pytest.mark.asyncio
async def test_below_the_cap_allows(manager, loader, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    await _send(manager, uuid4(), 2)


@pytest.mark.asyncio
async def test_the_switch_can_be_named_per_instance(manager, loader, monkeypatch):
    monkeypatch.setenv(f"{FLAG}_INSTANCES", str(INSTANCE))
    user = uuid4()
    await _send(manager, user, 2)
    with pytest.raises(RateLimitExceeded):
        await manager.ingest_document(_request(user))


@pytest.mark.asyncio
async def test_each_user_has_their_own_count(manager, loader, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    a, b = uuid4(), uuid4()
    await _send(manager, a, 2)
    with pytest.raises(RateLimitExceeded):
        await manager.ingest_document(_request(a))
    await _send(manager, b, 2)


@pytest.mark.asyncio
async def test_a_new_utc_day_starts_from_zero(manager, loader, limiter, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    user = uuid4()
    monkeypatch.setattr(limiter, "_current_day", lambda: "20260101")
    await _send(manager, user, 2)
    with pytest.raises(RateLimitExceeded):
        await manager.ingest_document(_request(user))
    monkeypatch.setattr(limiter, "_current_day", lambda: "20260102")
    await _send(manager, user, 2)


@pytest.mark.asyncio
async def test_the_counter_expires_after_about_two_days(manager, loader, limiter, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    user = uuid4()
    await _send(manager, user, 1)
    keys = [k async for k in limiter._redis.scan_iter("ratelimit:user_day:*")]
    assert len(keys) == 1
    ttl = await limiter._redis.ttl(keys[0])
    assert 86400 < ttl <= 2 * 86400


@pytest.mark.asyncio
async def test_no_user_id_means_no_per_user_cap(manager, loader, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    await _send(manager, None, 5)
    assert loader.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", [
    {},                                          # absent: the schema default of 100 is not a choice
    {"max_memories_per_user_per_day": 0},
    {"max_memories_per_user_per_day": -3},
])
async def test_no_positive_setting_means_no_cap(manager, loader, monkeypatch, limits):
    monkeypatch.setenv(FLAG, "true")
    loader.config = _config(**limits)
    await _send(manager, uuid4(), 5)


@pytest.mark.asyncio
async def test_a_setting_of_none_means_no_cap(manager, loader, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    config = _config(max_memories_per_user_per_day=2)
    object.__setattr__(config.guardrails.rate_limits, "max_memories_per_user_per_day", None)
    loader.config = config
    await _send(manager, uuid4(), 5)


@pytest.mark.asyncio
async def test_redis_down_allows_and_does_not_raise(manager, loader, limiter, monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "true")

    async def boom(*a, **k):
        raise ConnectionError("redis is down")
    monkeypatch.setattr(limiter, "check_user_day", boom)
    with caplog.at_level("WARNING"):
        await _send(manager, uuid4(), 5)
    assert "ingestion_user_cap_unchecked" in caplog.text


@pytest.mark.asyncio
async def test_a_config_that_cannot_be_read_allows(manager, loader, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    loader.config = RuntimeError("no config")
    await _send(manager, uuid4(), 3)


@pytest.mark.asyncio
async def test_a_chunk_is_not_counted_on_its_own(manager, loader, monkeypatch):
    """The recursion per chunk passes _chunk_index; only the request is counted."""
    monkeypatch.setenv(FLAG, "true")
    user = uuid4()
    for i in range(5):
        await manager.ingest_document(_request(user, i), skip_per_minute_check=True, _chunk_index=i)
    assert loader.calls == 0
