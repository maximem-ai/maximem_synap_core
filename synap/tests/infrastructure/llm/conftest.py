"""
Pytest fixtures for LLM infrastructure tests.
"""

import pytest
import fakeredis.aioredis
from pathlib import Path

from synap.cloud.infrastructure.llm import (
    LLMRegistry,
    LLMUsageTracker,
    LLMModelSelector,
)


@pytest.fixture
async def redis_client():
    """Create a fake Redis client for testing."""
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def test_config_path():
    """Get path to test YAML config."""
    # Use the actual config file in the repo.
    # conftest.py lives at synap/tests/infrastructure/llm/; four .parent hops
    # reach the synap/ package root, where providers/llm/llm_registry.yaml lives.
    return str(Path(__file__).parent.parent.parent.parent / "providers" / "llm" / "llm_registry.yaml")


@pytest.fixture
def registry(test_config_path):
    """Create a registry instance with test config."""
    return LLMRegistry(test_config_path)


@pytest.fixture
async def tracker(redis_client):
    """Create a tracker instance with fake Redis."""
    return LLMUsageTracker(redis_client)


@pytest.fixture
async def selector(registry, tracker):
    """Create a selector instance."""
    return LLMModelSelector(registry, tracker)
