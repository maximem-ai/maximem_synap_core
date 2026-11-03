"""Tests for RateLimiter."""

from datetime import datetime
from uuid import uuid4

import pytest
import fakeredis.aioredis

from synap.cloud.application.ingestion.rate_limiter import RateLimiter, RateLimitResult


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    async def redis(self):
        """Create fake Redis client."""
        return fakeredis.aioredis.FakeRedis()

    @pytest.fixture
    def rate_limiter(self, redis):
        """Create rate limiter with test limits."""
        return RateLimiter(
            redis=redis,
            requests_per_minute=10,
            requests_per_day=100,
        )

    @pytest.mark.asyncio
    async def test_check_allows_first_request(self, rate_limiter):
        """Test first request is allowed."""
        client_id = uuid4()
        instance_id = uuid4()
        
        result = await rate_limiter.check(client_id, instance_id)
        
        assert result.allowed
        assert result.remaining_minute == 9
        assert result.remaining_day == 99
        assert result.retry_after is None

    @pytest.mark.asyncio
    async def test_check_increments_counters(self, rate_limiter):
        """Test multiple requests increment counters."""
        client_id = uuid4()
        instance_id = uuid4()
        
        # Make 3 requests
        for i in range(3):
            result = await rate_limiter.check(client_id, instance_id)
            assert result.allowed
            assert result.remaining_minute == 10 - i - 1
            assert result.remaining_day == 100 - i - 1

    @pytest.mark.asyncio
    async def test_check_blocks_after_minute_limit(self, rate_limiter):
        """Test requests blocked after minute limit reached."""
        client_id = uuid4()
        instance_id = uuid4()
        
        # Use up minute limit
        for _ in range(10):
            result = await rate_limiter.check(client_id, instance_id)
            assert result.allowed
        
        # Next request should be blocked
        result = await rate_limiter.check(client_id, instance_id)
        assert not result.allowed
        assert result.remaining_minute == 0
        assert result.retry_after is not None
        assert result.retry_after <= 60

    @pytest.mark.asyncio
    async def test_check_blocks_after_day_limit(self, rate_limiter, redis):
        """Test requests blocked after day limit reached."""
        client_id = uuid4()
        instance_id = uuid4()
        
        # Manually set day count to limit
        day_key = f"ratelimit:{client_id}:day:{datetime.utcnow().strftime('%Y%m%d')}"
        await redis.set(day_key, "100")
        
        result = await rate_limiter.check(client_id, instance_id)
        assert not result.allowed
        assert result.remaining_day == 0
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_check_batch_allows_within_limit(self, rate_limiter):
        """Test batch check allows when within limit."""
        client_id = uuid4()
        
        result = await rate_limiter.check_batch(client_id, count=50)
        assert result.allowed
        assert result.remaining_day == 50

    @pytest.mark.asyncio
    async def test_check_batch_blocks_exceeding_limit(self, rate_limiter):
        """Test batch check blocks when exceeding limit."""
        client_id = uuid4()
        
        result = await rate_limiter.check_batch(client_id, count=150)
        assert not result.allowed
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_rate_limit_per_client(self, rate_limiter):
        """Test rate limits are per-client."""
        client1 = uuid4()
        client2 = uuid4()
        instance_id = uuid4()
        
        # Use up client1's minute limit
        for _ in range(10):
            result = await rate_limiter.check(client1, instance_id)
            assert result.allowed
        
        # client1 should be blocked
        result = await rate_limiter.check(client1, instance_id)
        assert not result.allowed
        
        # client2 should still be allowed
        result = await rate_limiter.check(client2, instance_id)
        assert result.allowed

    @pytest.mark.asyncio
    async def test_rate_limit_result_structure(self, rate_limiter):
        """Test RateLimitResult structure."""
        result = await rate_limiter.check(uuid4(), uuid4())
        
        assert isinstance(result, RateLimitResult)
        assert isinstance(result.allowed, bool)
        assert isinstance(result.remaining_minute, int)
        assert isinstance(result.remaining_day, int)
