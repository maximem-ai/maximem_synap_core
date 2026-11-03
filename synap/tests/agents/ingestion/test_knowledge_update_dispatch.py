"""Tests for KU v2 type-aware dispatch.

Verifies that EPISODES and EMOTIONS new memories never reach the LLM
classifier when ENABLE_KU_TYPE_AWARE_DISPATCH is on, and that
FACTS/PREFERENCES/TEMPORAL_EVENTS proceed as before.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    KnowledgeUpdateDetector,
    SIMILARITY_THRESHOLD,
)


pytestmark = pytest.mark.asyncio


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _mem(content: str, mtype: MemoryType) -> ExtractedMemory:
    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=mtype,
        content=content,
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="doc-x",
        entities_mentioned=[],
    )


def _vector_with_hit(existing_id: str, existing_content: str):
    """Vector wrapper mock returning one above-threshold candidate per query."""
    vec = MagicMock()
    vec.batch_search = AsyncMock(return_value=[[
        {
            "id": existing_id,
            "content": existing_content,
            "score": SIMILARITY_THRESHOLD + 0.05,
        },
    ]])
    return vec


def _llm_classifier(returns_classification: str = "UNRELATED"):
    """LLM client mock that returns a fixed classification when called."""
    llm = MagicMock()
    import json
    llm.complete = AsyncMock(return_value=json.dumps({
        "classification": returns_classification,
        "reason": "test stub",
    }))
    return llm


async def test_episodes_skipped_when_dispatch_flag_on(monkeypatch):
    """An EPISODES new memory must NOT reach the LLM classifier under type-aware dispatch."""
    monkeypatch.setenv("ENABLE_KU_TYPE_AWARE_DISPATCH", "true")

    vec = _vector_with_hit(str(uuid4()), "Caroline attended a pride parade in 2024")
    llm = _llm_classifier()
    detector = KnowledgeUpdateDetector(vector=vec, llm_client=llm)

    new_mem = _mem("Caroline attended a pride parade in 2025", MemoryType.EPISODES)
    result = await detector.detect_updates(
        memories=[new_mem],
        memory_embeddings=[[0.1] * 768],
        scope=_scope(),
    )

    # No actions emitted because the EPISODES memory was filtered before classify.
    assert result.supersessions == []
    assert result.skips == []
    assert result.enrichments == []
    # And critically: the LLM was never called.
    llm.complete.assert_not_called()


async def test_emotions_skipped_when_dispatch_flag_on(monkeypatch):
    """An EMOTIONS new memory must NOT reach the LLM classifier under type-aware dispatch."""
    monkeypatch.setenv("ENABLE_KU_TYPE_AWARE_DISPATCH", "true")

    vec = _vector_with_hit(str(uuid4()), "Caroline felt joyful at the parade")
    llm = _llm_classifier()
    detector = KnowledgeUpdateDetector(vector=vec, llm_client=llm)

    new_mem = _mem("Caroline felt thrilled at the parade", MemoryType.EMOTIONS)
    result = await detector.detect_updates(
        memories=[new_mem],
        memory_embeddings=[[0.1] * 768],
        scope=_scope(),
    )

    assert result.supersessions == []
    assert result.skips == []
    assert result.enrichments == []
    llm.complete.assert_not_called()


async def test_facts_still_classified_when_dispatch_flag_on(monkeypatch):
    """FACTS memories continue to go through the classifier under type-aware dispatch."""
    monkeypatch.setenv("ENABLE_KU_TYPE_AWARE_DISPATCH", "true")

    vec = _vector_with_hit(str(uuid4()), "User's manager is Person A")
    llm = _llm_classifier(returns_classification="UNRELATED")
    detector = KnowledgeUpdateDetector(vector=vec, llm_client=llm)

    new_mem = _mem("User's manager is Person B", MemoryType.FACTS)
    await detector.detect_updates(
        memories=[new_mem],
        memory_embeddings=[[0.1] * 768],
        scope=_scope(),
    )

    # Classifier WAS called for FACTS.
    llm.complete.assert_called_once()


async def test_episodes_still_classified_when_dispatch_flag_off(monkeypatch):
    """Without the flag, EPISODES go through the classifier — backward compat."""
    monkeypatch.delenv("ENABLE_KU_TYPE_AWARE_DISPATCH", raising=False)

    vec = _vector_with_hit(str(uuid4()), "Caroline attended a pride parade in 2024")
    llm = _llm_classifier()
    detector = KnowledgeUpdateDetector(vector=vec, llm_client=llm)

    new_mem = _mem("Caroline attended a pride parade in 2025", MemoryType.EPISODES)
    await detector.detect_updates(
        memories=[new_mem],
        memory_embeddings=[[0.1] * 768],
        scope=_scope(),
    )

    # Classifier WAS called — type-aware dispatch is gated off.
    llm.complete.assert_called_once()
