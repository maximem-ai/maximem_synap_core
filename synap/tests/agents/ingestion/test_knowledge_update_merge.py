"""Tests for KU v2 ENRICH merge synthesis.

Focus is on the deterministic clustering logic and the LLM-call wrapping;
the merge prompt content is covered by manual eval, not unit tests.
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    EnrichmentAction,
    KnowledgeUpdateDetector,
    MergeAction,
)


pytestmark = pytest.mark.asyncio


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _enrichment(existing_id: str, existing_content: str, new_id: str, new_content: str):
    return EnrichmentAction(
        existing_memory_id=existing_id,
        existing_memory_content=existing_content,
        new_memory_id=new_id,
        new_memory_content=new_content,
        reason="test stub",
    )


def _llm_returning(merged_text: str, rationale: str = "merged"):
    """LLM mock that returns a successful merge response."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=json.dumps({
        "merged_content": merged_text,
        "rationale": rationale,
    }))
    return llm


async def test_synthesize_merges_returns_empty_when_no_enrichments():
    detector = KnowledgeUpdateDetector(vector=MagicMock(), llm_client=MagicMock())
    result = await detector.synthesize_merges(enrichments=[], scope=_scope())
    assert result == []


async def test_synthesize_merges_two_pair_cluster():
    """Two ENRICH pairs (A-M1) and (B-M1) form a 3-memory cluster transitively
    via M1. One LLM call, one MergeAction with 3 contributors."""
    a, b, m1 = str(uuid4()), str(uuid4()), str(uuid4())
    enrichments = [
        _enrichment(a, "A: keen on counseling", m1, "M1: exploring counseling, focus on trans people"),
        _enrichment(b, "B: passionate about counseling", m1, "M1: exploring counseling, focus on trans people"),
    ]
    long_merged = "M1: exploring counseling, focus on trans people, also keen on it and passionate about helping others"
    llm = _llm_returning(long_merged, rationale="combined three Caroline-counseling memories")
    detector = KnowledgeUpdateDetector(vector=MagicMock(), llm_client=llm)

    merges = await detector.synthesize_merges(enrichments=enrichments, scope=_scope())

    assert len(merges) == 1
    merge = merges[0]
    assert set(merge.contributor_memory_ids) == {a, b, m1}
    assert merge.merged_content == long_merged
    assert merge.rationale.startswith("combined")
    llm.complete.assert_called_once()


async def test_synthesize_merges_independent_pairs_become_separate_clusters():
    """Two unrelated ENRICH pairs form two separate clusters → two LLM calls,
    two MergeActions."""
    a, b, c, d = (str(uuid4()) for _ in range(4))
    enrichments = [
        _enrichment(a, "A is a long memory about X with detail", b, "B is a long memory about X with detail and more stuff"),
        _enrichment(c, "C is a long memory about Y with detail", d, "D is a long memory about Y with detail and more stuff"),
    ]
    llm = _llm_returning("merged longer than any contributor with all the details preserved here")
    detector = KnowledgeUpdateDetector(vector=MagicMock(), llm_client=llm)

    merges = await detector.synthesize_merges(enrichments=enrichments, scope=_scope())

    assert len(merges) == 2
    assert llm.complete.await_count == 2


async def test_synthesize_merges_caps_cluster_size():
    """When a cluster grows beyond max_cluster_size, only the first N members
    are passed to the merge LLM."""
    # Build 7 memories in one transitively-connected cluster: M0-M1, M0-M2, ..., M0-M6
    m0 = str(uuid4())
    others = [str(uuid4()) for _ in range(6)]
    enrichments = [
        _enrichment(o, f"contributor {i} is a long enough memory", m0, "central long enough memory")
        for i, o in enumerate(others)
    ]
    llm = _llm_returning("merged content longer than any contributor preserved fully here for sure yes")
    detector = KnowledgeUpdateDetector(vector=MagicMock(), llm_client=llm)

    merges = await detector.synthesize_merges(
        enrichments=enrichments, scope=_scope(), max_cluster_size=5,
    )

    assert len(merges) == 1
    assert len(merges[0].contributor_memory_ids) == 5  # capped


async def test_synthesize_merges_rejects_short_merge():
    """If merged_content is shorter than the longest contributor, guardrail
    rejects the merge — no MergeAction is emitted."""
    a, b = str(uuid4()), str(uuid4())
    enrichments = [
        _enrichment(a, "long contributor content with a lot of important detail", b, "another very long contributor with even more important detail and named entities"),
    ]
    short_merged = "too short"
    llm = _llm_returning(short_merged)
    detector = KnowledgeUpdateDetector(vector=MagicMock(), llm_client=llm)

    merges = await detector.synthesize_merges(enrichments=enrichments, scope=_scope())

    assert merges == []  # rejected by guardrail


async def test_synthesize_merges_handles_llm_failure_gracefully():
    """If the merge LLM raises, the cluster is skipped (no merge), not crashed."""
    a, b = str(uuid4()), str(uuid4())
    enrichments = [_enrichment(a, "x", b, "y")]
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    detector = KnowledgeUpdateDetector(vector=MagicMock(), llm_client=llm)

    merges = await detector.synthesize_merges(enrichments=enrichments, scope=_scope())

    assert merges == []  # graceful skip
