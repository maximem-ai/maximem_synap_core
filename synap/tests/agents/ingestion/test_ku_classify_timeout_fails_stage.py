"""A knowledge-update classification that times out must fail the stage.

Before: _classify_pair caught every exception and returned None, and Stage 4
also wrapped the whole knowledge-update block as non-fatal, so a timed-out
classification saved the new memory with no supersession check. Now an
LLMTimeoutError propagates (the worker retries the stage, then dead-letters);
every other classification failure keeps the old non-fatal behaviour.
"""
import os

# This repository is Synap Core: the manager, Stage 4 and the start-up check
# load Core's prompt pack and hooks only when the edition says so.
os.environ.setdefault("SYNAP_EDITION", "core")

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import ExtractedMemory, MemoryDestination, MemoryType, Scope
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    KnowledgeUpdateDetector,
    SIMILARITY_THRESHOLD,
)
from synap.cloud.infrastructure.llm.models import LLMError, LLMTimeoutError

pytestmark = pytest.mark.asyncio


def _scope() -> Scope:
    return Scope(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(), user_id=uuid4())


def _mem(content: str) -> ExtractedMemory:
    return ExtractedMemory(
        memory_id=uuid4(), memory_type=MemoryType.FACTS, content=content, confidence=0.9,
        destination=MemoryDestination.MEMORIES, source_document_id="doc-x", entities_mentioned=[],
    )


def _detector(llm_exc):
    vec = MagicMock()
    vec.batch_search = AsyncMock(return_value=[[
        {"id": str(uuid4()), "content": "User's manager is Person A", "score": SIMILARITY_THRESHOLD + 0.05},
    ]])
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=llm_exc)
    return KnowledgeUpdateDetector(vector=vec, llm_client=llm), llm


async def test_timeout_fails_detection(monkeypatch):
    # Per-pair path. The batch path re-raises the same way, but its prompt is
    # not in Core (CorePromptMissing), so it cannot be driven here.
    monkeypatch.setenv("ENABLE_KU_BATCH_CLASSIFY", "false")
    detector, llm = _detector(LLMTimeoutError("All models failed. gpt-4o: Request timeout (ReadTimeout)"))
    with pytest.raises(LLMTimeoutError):
        await detector.detect_updates(
            memories=[_mem("User's manager is Person B")],
            memory_embeddings=[[0.1] * 768], scope=_scope(),
        )
    # One call: the timeout stops detection at the first pair.
    assert llm.complete.await_count == 1


async def test_other_llm_failure_stays_non_fatal():
    detector, _ = _detector(LLMError("All models failed. gpt-4o: 400 bad request"))
    result = await detector.detect_updates(
        memories=[_mem("User's manager is Person B")],
        memory_embeddings=[[0.1] * 768], scope=_scope(),
    )
    assert result.supersessions == []


async def test_bad_json_stays_non_fatal():
    detector, _ = _detector(None)
    detector._llm.complete = AsyncMock(return_value="not json")
    result = await detector.detect_updates(
        memories=[_mem("User's manager is Person B")],
        memory_embeddings=[[0.1] * 768], scope=_scope(),
    )
    assert result.supersessions == []
