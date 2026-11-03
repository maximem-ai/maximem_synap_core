"""
LLM Usage Tracker - Tracks API usage for rate limiting and load balancing.
"""

import time
from typing import Optional

import redis.asyncio as redis

from .models import LLMProvider, ModelAvailability


class LLMUsageTracker:
    """
    Tracks LLM API usage using Redis for distributed rate limiting.

    Uses sliding window counters with TTL for rate limit tracking.
    Key format: llm_usage:{provider}:{model}:{metric}
    """

    # Redis key prefixes
    KEY_PREFIX = "llm_usage"
    RPM_SUFFIX = "rpm"
    TPM_SUFFIX = "tpm"
    ERROR_SUFFIX = "errors"

    # TTL values
    RPM_WINDOW_SECONDS = 60
    TPM_WINDOW_SECONDS = 60
    ERROR_PENALTY_SECONDS = 300  # 5 minutes

    # Availability thresholds
    HEAVY_LOAD_THRESHOLD = 0.8
    MODERATE_LOAD_THRESHOLD = 0.5

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize tracker with Redis client.

        Args:
            redis_client: Async Redis client instance
        """
        self._redis = redis_client

    def _get_key(self, provider: LLMProvider, model: str, metric: str) -> str:
        """Generate Redis key for a metric."""
        return f"{self.KEY_PREFIX}:{provider.value}:{model}:{metric}"

    async def record_request(
        self,
        provider: LLMProvider,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """
        Record a successful API request.

        Args:
            provider: The LLM provider
            model: The model name
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
        """
        rpm_key = self._get_key(provider, model, self.RPM_SUFFIX)
        tpm_key = self._get_key(provider, model, self.TPM_SUFFIX)
        total_tokens = input_tokens + output_tokens

        pipe = self._redis.pipeline()

        # Increment RPM counter
        pipe.incr(rpm_key)
        pipe.expire(rpm_key, self.RPM_WINDOW_SECONDS)

        # Increment TPM counter
        pipe.incrby(tpm_key, total_tokens)
        pipe.expire(tpm_key, self.TPM_WINDOW_SECONDS)

        await pipe.execute()

    async def record_error(
        self,
        provider: LLMProvider,
        model: str,
        is_rate_limit: bool = False
    ) -> None:
        """
        Record an API error.

        Args:
            provider: The LLM provider
            model: The model name
            is_rate_limit: Whether this was a rate limit error (429)
        """
        error_key = self._get_key(provider, model, self.ERROR_SUFFIX)

        # Increment error counter with longer TTL for rate limit errors
        ttl = self.ERROR_PENALTY_SECONDS * 2 if is_rate_limit else self.ERROR_PENALTY_SECONDS

        pipe = self._redis.pipeline()
        pipe.incr(error_key)
        pipe.expire(error_key, ttl)
        await pipe.execute()

    async def get_availability(
        self,
        provider: LLMProvider,
        model: str,
        rpm_limit: int,
        tpm_limit: int
    ) -> ModelAvailability:
        """
        Calculate availability status for a model.

        Returns availability factor:
        - 1.0: Fully available (< 50% usage)
        - 0.9: Moderate load (50-80% usage)
        - 0.5: Heavy load (80-99% usage)
        - 0.0: Exhausted (100% usage or recent errors)

        Args:
            provider: The LLM provider
            model: The model name
            rpm_limit: The RPM limit for this model
            tpm_limit: The TPM limit for this model

        Returns:
            ModelAvailability with status and factor
        """
        rpm_key = self._get_key(provider, model, self.RPM_SUFFIX)
        tpm_key = self._get_key(provider, model, self.TPM_SUFFIX)
        error_key = self._get_key(provider, model, self.ERROR_SUFFIX)

        pipe = self._redis.pipeline()
        pipe.get(rpm_key)
        pipe.get(tpm_key)
        pipe.get(error_key)

        results = await pipe.execute()

        rpm_current = int(results[0] or 0)
        tpm_current = int(results[1] or 0)
        error_count = int(results[2] or 0)

        # Calculate usage percentages
        rpm_usage = rpm_current / rpm_limit if rpm_limit > 0 else 0
        tpm_usage = tpm_current / tpm_limit if tpm_limit > 0 else 0
        max_usage = max(rpm_usage, tpm_usage)

        # Check for recent errors
        if error_count >= 3:
            return ModelAvailability(
                provider=provider,
                model=model,
                is_available=False,
                rpm_usage=rpm_usage,
                tpm_usage=tpm_usage,
                reason="Too many recent errors"
            )

        # Calculate availability factor
        if max_usage >= 1.0:
            return ModelAvailability(
                provider=provider,
                model=model,
                is_available=False,
                rpm_usage=rpm_usage,
                tpm_usage=tpm_usage,
                reason="Rate limit exhausted"
            )
        elif max_usage > self.HEAVY_LOAD_THRESHOLD:
            return ModelAvailability(
                provider=provider,
                model=model,
                is_available=True,
                rpm_usage=rpm_usage,
                tpm_usage=tpm_usage,
                reason="Heavy load"
            )
        elif max_usage > self.MODERATE_LOAD_THRESHOLD:
            return ModelAvailability(
                provider=provider,
                model=model,
                is_available=True,
                rpm_usage=rpm_usage,
                tpm_usage=tpm_usage,
                reason="Moderate load"
            )
        else:
            return ModelAvailability(
                provider=provider,
                model=model,
                is_available=True,
                rpm_usage=rpm_usage,
                tpm_usage=tpm_usage,
                reason=None
            )

    async def mark_unavailable(
        self,
        provider: LLMProvider,
        model: str,
        duration_seconds: int = 60
    ) -> None:
        """
        Temporarily mark a model as unavailable (e.g., after 429 error).

        Args:
            provider: The LLM provider
            model: The model name
            duration_seconds: How long to mark as unavailable
        """
        error_key = self._get_key(provider, model, self.ERROR_SUFFIX)

        # Set error count high enough to trigger unavailability
        await self._redis.setex(error_key, duration_seconds, 10)

    async def clear_usage(self, provider: LLMProvider, model: str) -> None:
        """Clear all usage data for a model (for testing)."""
        rpm_key = self._get_key(provider, model, self.RPM_SUFFIX)
        tpm_key = self._get_key(provider, model, self.TPM_SUFFIX)
        error_key = self._get_key(provider, model, self.ERROR_SUFFIX)

        await self._redis.delete(rpm_key, tpm_key, error_key)
