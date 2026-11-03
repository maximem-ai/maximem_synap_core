"""AI-1: FAST-mode LLM relevance-filter gate (opt-in, OFF by default).

The gate lets FAST mode skip the Groq re-rank when the candidate set already
fits the caller's max_results (nothing to *select*, only relevance-drop). It is
config-gated because skipping is a precision/latency tradeoff, not a no-op — so
the default must be a pure no-behavior-change.
"""

from unittest.mock import AsyncMock

import pytest

from synap.cloud.application.client_api.retrieval_manager import RetrievalManager


@pytest.fixture
def manager():
    return RetrievalManager(cache_provider=AsyncMock())


def test_gate_is_off_by_default(manager):
    """Default config never skips — no behavior change out of the box."""
    assert manager.config.fast_skip_filter_when_scarce is False
    assert manager._should_skip_fast_filter(3, 10) is False
    assert manager._should_skip_fast_filter(0, 10) is False
    assert manager._should_skip_fast_filter(50, 10) is False


def test_gate_skips_only_when_enabled_and_scarce(manager):
    manager.config.fast_skip_filter_when_scarce = True
    # Candidates already fit max_results → nothing to select → skip.
    assert manager._should_skip_fast_filter(3, 10) is True
    assert manager._should_skip_fast_filter(10, 10) is True
    # More candidates than max_results → filter still needed to select.
    assert manager._should_skip_fast_filter(11, 10) is False


def test_gate_requires_a_positive_max_results(manager):
    """When the caller didn't cap results, don't skip (falsy max_results)."""
    manager.config.fast_skip_filter_when_scarce = True
    assert manager._should_skip_fast_filter(3, 0) is False
