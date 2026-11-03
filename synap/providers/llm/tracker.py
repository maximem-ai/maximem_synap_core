"""LLM usage tracking for rate limit management."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class UsageTracker(ABC):
    """Tracks LLM rate limits and usage.

    Monitors requests per minute (RPM) and tokens per minute (TPM) usage
    for each provider/model combination. Production implementations should
    use Redis or similar distributed cache for multi-instance coordination.
    """

    @abstractmethod
    def get_usage(self, provider: str, model: str) -> Dict[str, float]:
        """Returns current usage ratios for a provider/model.

        Args:
            provider: Provider name (openai, anthropic, groq)
            model: Model identifier

        Returns:
            Dict with 'rpm_usage' and 'tpm_usage' as floats 0.0-1.0
            representing current usage vs. limits
        """
        ...

    @abstractmethod
    def record_usage(self, provider: str, model: str, tokens: int) -> None:
        """Record usage for a completion request.

        Args:
            provider: Provider name
            model: Model identifier
            tokens: Total tokens used (prompt + completion)
        """
        ...


WINDOW_SECONDS = 60


class InMemoryTracker(UsageTracker):
    """In-memory usage tracker for development/testing.

    Not suitable for production - does not persist across restarts
    and does not coordinate across multiple instances.

    Uses a 60-second sliding window to track RPM and TPM.
    """

    def __init__(self, rate_limits: Dict[str, Dict[str, int]] = None):
        """Initialize in-memory tracker.

        Args:
            rate_limits: Dict mapping "provider:model" keys to
                         {'rpm': int, 'tpm': int} limits.
                         If None, get_usage returns 0.0 for all ratios.
        """
        self._rate_limits = rate_limits or {}
        self._records: Dict[str, List[Tuple[float, int]]] = {}

    def _key(self, provider: str, model: str) -> str:
        return f"{provider}:{model}"

    def _prune(self, key: str) -> None:
        """Remove records older than the sliding window."""
        if key not in self._records:
            return
        cutoff = time.monotonic() - WINDOW_SECONDS
        self._records[key] = [
            (ts, tok) for ts, tok in self._records[key] if ts > cutoff
        ]

    def get_usage(self, provider: str, model: str) -> Dict[str, float]:
        """Get current usage ratios.

        Args:
            provider: Provider name
            model: Model identifier

        Returns:
            Dict with 'rpm_usage' and 'tpm_usage' as 0.0-1.0
        """
        key = self._key(provider, model)
        self._prune(key)

        records = self._records.get(key, [])
        request_count = len(records)
        token_sum = sum(tok for _, tok in records)

        limits = self._rate_limits.get(key, {})
        rpm_limit = limits.get("rpm", 0)
        tpm_limit = limits.get("tpm", 0)

        rpm_usage = (request_count / rpm_limit) if rpm_limit > 0 else 0.0
        tpm_usage = (token_sum / tpm_limit) if tpm_limit > 0 else 0.0

        return {
            "rpm_usage": min(rpm_usage, 1.0),
            "tpm_usage": min(tpm_usage, 1.0),
        }

    def record_usage(self, provider: str, model: str, tokens: int) -> None:
        """Record usage.

        Args:
            provider: Provider name
            model: Model identifier
            tokens: Total tokens used
        """
        key = self._key(provider, model)
        if key not in self._records:
            self._records[key] = []
        self._records[key].append((time.monotonic(), tokens))
        self._prune(key)
