"""The two retrieval gates, and the switch on each.

Measured on the LongMemEval lme2_ run, over the exact contexts
served. Handing the reader back only what the 0.45 near-duplicate
rule removed moved the same reader and judge from 81.70% to 85.74% on 470
questions, against a +/-1.3 point noise floor. So the near-duplicate rule now
defaults to OFF and both gates are operator-switchable.
"""

import logging

import pytest

from synap.cloud.agents.models.bundles import ContextItem
from synap.cloud.application.client_api.retrieval_manager import (
    _DEFAULT_DEDUP_JACCARD,
    _DEFAULT_MIN_SIMILARITY,
    _deduplicate_jaccard,
    dedup_jaccard_threshold,
    min_similarity,
    record_retrieval_fault,
    _RETRIEVAL_FAULTS,
)


def _item(item_id: str, content: str, score: float = 0.5) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        content=content,
        context_type="fact",
        source="vector",
        confidence=score,
        similarity_score=score,
        relevance_score=score,
    )


# ── the near-duplicate rule ─────────────────────────────────────────────────

def test_the_near_duplicate_rule_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SYNAP_RETRIEVAL_DEDUP_JACCARD", raising=False)
    assert dedup_jaccard_threshold() == _DEFAULT_DEDUP_JACCARD == 0.0


def test_it_can_be_restored_without_a_deploy(monkeypatch):
    monkeypatch.setenv("SYNAP_RETRIEVAL_DEDUP_JACCARD", "0.45")
    assert dedup_jaccard_threshold() == pytest.approx(0.45)


@pytest.mark.parametrize("word", ["off", "OFF", "Off"])
def test_off_is_spelled_off(monkeypatch, word):
    monkeypatch.setenv("SYNAP_RETRIEVAL_DEDUP_JACCARD", word)
    assert dedup_jaccard_threshold() == 0.0


@pytest.mark.parametrize("bad", ["abc", "-0.5", "1.5", "None", "none"])
def test_a_bad_value_falls_back_and_never_guesses(monkeypatch, bad):
    """'None' is the one that matters: a stringified Python None must not be
    read as a deliberate instruction. The same rule guards the floor, where
    mis-reading it would silently widen every client's retrieval."""
    monkeypatch.setenv("SYNAP_RETRIEVAL_DEDUP_JACCARD", bad)
    assert dedup_jaccard_threshold() == _DEFAULT_DEDUP_JACCARD


@pytest.mark.parametrize("bad", ["None", "none"])
def test_the_floor_is_not_opened_by_a_stringified_none(monkeypatch, bad):
    monkeypatch.setenv("SYNAP_RETRIEVAL_MIN_SIMILARITY", bad)
    assert min_similarity() == _DEFAULT_MIN_SIMILARITY


def test_the_floor_takes_the_word_off_too(monkeypatch):
    monkeypatch.setenv("SYNAP_RETRIEVAL_MIN_SIMILARITY", "off")
    assert min_similarity() == 0.0


# ── what the rule actually did to real memory pairs ─────────────────────────

REAL_PAIRS = [
    ("The user has watched 12 films in the last 3 months, including 5 MCU films.",
     "The user has watched 4 MCU films in the last 3 months."),
    ("The user sees Dr. Smith every week",
     "The user has a therapy session with Dr. Smith every two weeks"),
    ("Transcriptionist is a work-from-home job idea for seniors",
     "Remote IT support specialist is a work-from-home job idea for seniors"),
    ("The user started reading 'The Nightingale' by Kristin Hannah on 2022-01-01",
     "The user finished reading 'The Nightingale' by Kristin Hannah on 2022-01-15"),
]


@pytest.mark.parametrize("dropped,kept", REAL_PAIRS)
def test_at_0_45_these_real_pairs_collapse(dropped, kept):
    """Every pair here comes from a question the run got wrong. None of them are
    duplicates: they differ in a number, a frequency, a list item or a date. The
    test exists so that anyone restoring 0.45 sees what it costs."""
    items = [_item("a", kept, 0.9), _item("b", dropped, 0.5)]
    assert len(_deduplicate_jaccard(items, threshold=0.45)) == 1


@pytest.mark.parametrize("dropped,kept", REAL_PAIRS)
def test_with_the_rule_off_both_survive(dropped, kept):
    items = [_item("a", kept, 0.9), _item("b", dropped, 0.5)]
    assert len(_deduplicate_jaccard(items, threshold=dedup_jaccard_threshold())) == 2


def test_off_is_an_early_return_not_a_threshold_of_zero():
    """``similarity > 0`` collapses every pair that shares one token, so a
    threshold of 0 without the early return would delete almost everything."""
    items = [_item("a", "The user likes tea", 0.9),
             _item("b", "The user drives a van", 0.5),
             _item("c", "Completely unrelated sentence", 0.4)]
    assert len(_deduplicate_jaccard(items, threshold=0.0)) == 3


def test_exact_duplicates_are_still_handled_elsewhere():
    """Turning the near-duplicate rule off is not the same as shipping
    duplicates: identical content still collapses at 0.45 if it is restored, and
    the caller's exact-dedup pass runs before this one either way."""
    items = [_item("a", "The user likes tea", 0.9), _item("b", "The user likes tea", 0.5)]
    assert len(_deduplicate_jaccard(items, threshold=0.45)) == 1


# ── a failed leg has to reach the caller ────────────────────────────────────

def test_a_fault_outside_a_fetch_is_a_no_op():
    """No collector means not inside a fetch. Recording must not raise there, or
    a logging concern becomes a read-path outage."""
    assert _RETRIEVAL_FAULTS.get() is None
    record_retrieval_fault("vector search (scope rung 0)", ValueError("boom"))


def test_a_fault_inside_a_fetch_is_collected_once():
    faults = []
    token = _RETRIEVAL_FAULTS.set(faults)
    try:
        record_retrieval_fault("vector search (scope rung 0)", ValueError("Error finding id"))
        record_retrieval_fault("vector search (scope rung 0)", ValueError("Error finding id"))
        record_retrieval_fault("question search (scope rung 1)", "Nothing found on disk")
    finally:
        _RETRIEVAL_FAULTS.reset(token)
    assert len(faults) == 2, faults
    assert "vector search (scope rung 0) failed: ValueError: Error finding id" in faults
    assert "question search (scope rung 1) failed: Nothing found on disk" in faults


# ── the wrapper that carries a fault out to the caller ──────────────────────

@pytest.mark.asyncio
async def test_fetch_context_folds_a_failed_leg_into_the_bundle():
    """The fix for the failure mode that cost us the benchmark: a store leg fails,
    the request still returns 200 with fewer items, and before this nothing in
    the bundle or the response said so."""
    from synap.cloud.agents.models.bundles import ContextBundle, DegradationLevel
    from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
    from synap.cloud.agents.models.decisions import ContextDecision, TriggerType, RetrievalMode

    mgr = RetrievalManager.__new__(RetrievalManager)

    async def _inner(decision, scope=None):
        record_retrieval_fault(
            "vector search (scope rung 0)",
            RuntimeError("Error creating hnsw segment reader: Nothing found on disk"),
        )
        return ContextBundle(bundle_id="b", decision_id="d", retrieval_mode="accurate",
                             token_budget=100)

    mgr._fetch_context_inner = _inner
    decision = ContextDecision(
        decision_id="d", trigger_type=TriggerType.EXPLICIT_REQUEST,
        retrieval_mode=RetrievalMode.ACCURATE, search_queries=["q"],
    )
    bundle = await mgr.fetch_context(decision)

    assert bundle.degradation_level == DegradationLevel.PARTIAL
    assert len(bundle.warnings) == 1
    assert "Nothing found on disk" in bundle.warnings[0]


@pytest.mark.asyncio
async def test_a_clean_fetch_says_nothing():
    """The field has to stay empty on a healthy fetch, or a caller learns to
    ignore it."""
    from synap.cloud.agents.models.bundles import ContextBundle, DegradationLevel
    from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
    from synap.cloud.agents.models.decisions import ContextDecision, TriggerType, RetrievalMode

    mgr = RetrievalManager.__new__(RetrievalManager)

    async def _inner(decision, scope=None):
        return ContextBundle(bundle_id="b", decision_id="d", retrieval_mode="accurate",
                             token_budget=100)

    mgr._fetch_context_inner = _inner
    decision = ContextDecision(
        decision_id="d", trigger_type=TriggerType.EXPLICIT_REQUEST,
        retrieval_mode=RetrievalMode.ACCURATE, search_queries=["q"],
    )
    bundle = await mgr.fetch_context(decision)
    assert bundle.warnings == []
    assert bundle.degradation_level in (None, DegradationLevel.NONE)


@pytest.mark.asyncio
async def test_the_collector_does_not_leak_between_fetches():
    from synap.cloud.agents.models.bundles import ContextBundle
    from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
    from synap.cloud.agents.models.decisions import ContextDecision, TriggerType, RetrievalMode

    mgr = RetrievalManager.__new__(RetrievalManager)
    calls = {"n": 0}

    async def _inner(decision, scope=None):
        calls["n"] += 1
        if calls["n"] == 1:
            record_retrieval_fault("vector search (scope rung 0)", RuntimeError("boom"))
        return ContextBundle(bundle_id="b", decision_id="d", retrieval_mode="accurate",
                             token_budget=100)

    mgr._fetch_context_inner = _inner
    decision = ContextDecision(
        decision_id="d", trigger_type=TriggerType.EXPLICIT_REQUEST,
        retrieval_mode=RetrievalMode.ACCURATE, search_queries=["q"],
    )
    first = await mgr.fetch_context(decision)
    second = await mgr.fetch_context(decision)
    assert len(first.warnings) == 1
    assert second.warnings == []
    assert _RETRIEVAL_FAULTS.get() is None
