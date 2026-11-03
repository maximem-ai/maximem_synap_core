"""
LLM API Key Rotator
====================

Enables higher throughput by rotating between multiple API keys per provider.

Features:
- Round-robin key rotation for load distribution
- Per-key health tracking and automatic failover
- Automatic recovery after cooldown period
- Thread-safe for concurrent access

Adapted from maximem_memoryinfra/app/ingestion/llm_key_rotator.py
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIKeyStats:
    """Statistics for a single API key."""

    key_id: str  # Masked key for safe logging
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time: float = 0.0
    last_used: float = 0.0  # monotonic timestamp
    last_error: Optional[str] = None
    error_count_last_hour: int = 0
    is_healthy: bool = True
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[float] = None  # monotonic timestamp
    marked_unhealthy_at: float = 0.0  # for auto-recovery


# How long an unhealthy key stays unhealthy before auto-recovery (seconds)
HEALTH_RECOVERY_SECONDS = 60.0


class LLMKeyRotator:
    """
    Manages multiple API keys for a single LLM provider.

    Uses round-robin rotation with health checks. When a key hits a
    rate limit (429), it's marked unhealthy and skipped until recovery.
    """

    def __init__(
        self,
        api_keys: List[str],
        calls_per_key: int = 20,
    ):
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.api_keys = api_keys
        self.calls_per_key = calls_per_key
        self.total_capacity = len(api_keys) * calls_per_key

        # Thread-safe rotation
        self._current_index = 0
        self._lock = threading.Lock()

        # Per-key stats
        self._stats: Dict[str, APIKeyStats] = {}
        for key in api_keys:
            self._stats[key] = APIKeyStats(key_id=self._mask(key))

        logger.info(
            "LLM Key Rotator initialized: %d keys, %d calls/key, %d total capacity",
            len(api_keys),
            calls_per_key,
            self.total_capacity,
        )

    @staticmethod
    def _mask(key: str) -> str:
        if len(key) < 8:
            return "***"
        return f"{key[:6]}...{key[-4:]}"

    def get_next_key(self) -> str:
        """Get next healthy API key via round-robin."""
        with self._lock:
            now = time.monotonic()

            # Auto-recover keys whose cooldown has expired
            for stats in self._stats.values():
                if (
                    not stats.is_healthy
                    and stats.marked_unhealthy_at > 0
                    and (now - stats.marked_unhealthy_at) >= HEALTH_RECOVERY_SECONDS
                ):
                    stats.is_healthy = True
                    stats.error_count_last_hour = 0
                    logger.info("Key %s auto-recovered after cooldown", stats.key_id)

            # Try to find a healthy key
            for _ in range(len(self.api_keys)):
                key = self.api_keys[self._current_index]
                self._current_index = (self._current_index + 1) % len(self.api_keys)
                stats = self._stats[key]

                if stats.is_healthy:
                    stats.last_used = now
                    return key

            # All unhealthy — use next one anyway (fail-open)
            key = self.api_keys[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.api_keys)
            logger.warning("All %d API keys unhealthy, using %s anyway",
                           len(self.api_keys), self._stats[key].key_id)
            return key

    def record_success(
        self,
        api_key: str,
        processing_time: float,
        rate_limit_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        stats = self._stats.get(api_key)
        if not stats:
            return
        stats.total_calls += 1
        stats.successful_calls += 1
        stats.total_time += processing_time
        stats.is_healthy = True

        if rate_limit_info:
            stats.rate_limit_remaining = rate_limit_info.get("remaining")

    def record_failure(
        self,
        api_key: str,
        error: str,
        is_rate_limit: bool = False,
    ) -> None:
        stats = self._stats.get(api_key)
        if not stats:
            return
        stats.total_calls += 1
        stats.failed_calls += 1
        stats.last_error = error
        stats.error_count_last_hour += 1

        if is_rate_limit:
            stats.is_healthy = False
            stats.marked_unhealthy_at = time.monotonic()
            logger.warning("Key %s rate-limited, marked unhealthy", stats.key_id)
        elif stats.error_count_last_hour >= 10:
            stats.is_healthy = False
            stats.marked_unhealthy_at = time.monotonic()
            logger.warning(
                "Key %s has %d errors, marked unhealthy",
                stats.key_id, stats.error_count_last_hour,
            )

    def get_healthy_key_count(self) -> int:
        return sum(1 for s in self._stats.values() if s.is_healthy)

    def get_statistics(self) -> Dict[str, Any]:
        per_key = []
        for key, stats in self._stats.items():
            per_key.append({
                "key_id": stats.key_id,
                "total_calls": stats.total_calls,
                "successful_calls": stats.successful_calls,
                "failed_calls": stats.failed_calls,
                "is_healthy": stats.is_healthy,
                "last_error": stats.last_error,
            })
        return {
            "total_keys": len(self.api_keys),
            "healthy_keys": self.get_healthy_key_count(),
            "total_capacity": self.total_capacity,
            "per_key_stats": per_key,
        }

    def reset_health_checks(self) -> None:
        """Reset all keys to healthy (e.g., after cooldown)."""
        for stats in self._stats.values():
            stats.is_healthy = True
            stats.error_count_last_hour = 0
        logger.info("Reset health checks for all %d API keys", len(self.api_keys))
