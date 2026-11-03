"""Rate limiting for ingestion requests."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from redis.asyncio import Redis


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining_minute: int
    remaining_day: int
    retry_after: Optional[int] = None  # Seconds


#: Named so callers can reason about the ceiling instead of rediscovering it
#: from a 429. A migration needs them up front: it is the one workload that
#: arrives as a whole account at once, so it must be paced to these and must
#: be able to tell the customer how long that will take.
DEFAULT_REQUESTS_PER_MINUTE = 100
DEFAULT_REQUESTS_PER_DAY = 10000


class RateLimiter:
    """
    Rate limiter using Redis for distributed rate limiting.
    
    Implements two limits per client:
    - Per-minute (burst protection)
    - Per-day (quota)
    """

    def __init__(
        self,
        redis: Redis,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        requests_per_day: int = DEFAULT_REQUESTS_PER_DAY,
    ):
        """Initialize rate limiter.
        
        Args:
            redis: Async Redis client
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day
        """
        self._redis = redis
        self._per_minute = requests_per_minute
        self._per_day = requests_per_day

    async def check(
        self,
        client_id: UUID,
        instance_id: UUID,
    ) -> RateLimitResult:
        """Check if request is allowed.
        
        Args:
            client_id: Client UUID
            instance_id: Instance UUID
            
        Returns:
            RateLimitResult with allowed flag and retry_after if denied
        """
        minute_key = f"ratelimit:{client_id}:minute:{self._current_minute()}"
        day_key = f"ratelimit:{client_id}:day:{self._current_day()}"

        # Get current counts
        minute_count = int(await self._redis.get(minute_key) or 0)
        day_count = int(await self._redis.get(day_key) or 0)

        # Check minute limit
        if minute_count >= self._per_minute:
            return RateLimitResult(
                allowed=False,
                remaining_minute=0,
                remaining_day=self._per_day - day_count,
                retry_after=60 - datetime.now(timezone.utc).second,
            )

        # Check day limit
        if day_count >= self._per_day:
            return RateLimitResult(
                allowed=False,
                remaining_minute=self._per_minute - minute_count,
                remaining_day=0,
                retry_after=self._seconds_until_midnight(),
            )

        # Increment counters
        pipe = self._redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incr(day_key)
        pipe.expire(day_key, 86400)
        await pipe.execute()

        return RateLimitResult(
            allowed=True,
            remaining_minute=self._per_minute - minute_count - 1,
            remaining_day=self._per_day - day_count - 1,
        )

    async def check_batch(
        self,
        client_id: UUID,
        count: int,
    ) -> RateLimitResult:
        """Check if batch is allowed.
        
        Simplified batch check - only checks daily quota.
        
        Args:
            client_id: Client UUID
            count: Number of documents in batch
            
        Returns:
            RateLimitResult
        """
        day_key = f"ratelimit:{client_id}:day:{self._current_day()}"
        day_count = int(await self._redis.get(day_key) or 0)

        if day_count + count > self._per_day:
            return RateLimitResult(
                allowed=False,
                remaining_minute=self._per_minute,
                remaining_day=self._per_day - day_count,
                retry_after=self._seconds_until_midnight(),
            )

        return RateLimitResult(
            allowed=True,
            remaining_minute=self._per_minute,
            remaining_day=self._per_day - day_count - count,
        )

    async def check_user_day(
        self,
        instance_id: UUID,
        user_id: UUID,
        cap: int,
    ) -> RateLimitResult:
        """Count one request against a user's daily cap and say if it is within it.

        The unit is a request that reaches the ingestion entry, not an extracted
        memory: the number of memories is not known until extraction, long after
        the caller must be answered. The day is the UTC day. The key outlives it
        by a day so a clock skew between hosts cannot reopen yesterday's count.
        """
        key = f"ratelimit:user_day:{instance_id}:{user_id}:{self._current_day()}"
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 2 * 86400)
        count = int((await pipe.execute())[0])
        if count > cap:
            return RateLimitResult(
                allowed=False,
                remaining_minute=self._per_minute,
                remaining_day=0,
                retry_after=self._seconds_until_midnight(),
            )
        return RateLimitResult(
            allowed=True,
            remaining_minute=self._per_minute,
            remaining_day=cap - count,
        )

    def _current_minute(self) -> str:
        """Get current minute as YYYYMMDDHHMM string."""
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

    def _current_day(self) -> str:
        """Get current day as YYYYMMDD string."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _seconds_until_midnight(self) -> int:
        """Calculate seconds until next UTC midnight."""
        now = datetime.now(timezone.utc)
        midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int((midnight - now).total_seconds())
