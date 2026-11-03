"""The raw excerpt we attach to a fact should be an excerpt about that fact.

`score_chunks` ranks a candidate excerpt on query coverage AND on novelty,
where novelty is the share of its words that are NOT in the memory. So 40% of
the ranking prefers an excerpt that says something other than the fact it is
attached to. Measured over 6,688 enriched items of one 500-question run: the
attached excerpt shares a median 50% of its own fact's content words, 17% share
under a quarter, 5% share nothing. The reader then answers from the excerpt
instead of from the fact.

The support floor is the targeted correction, and the case that matters is
`test_floor_re_picks_the_supporting_excerpt`: when a document holds a
supporting passage the floor must yield THAT passage. Dropping the excerpt is
the fallback for a document with no supporting passage, not the mechanism. The
measured arm could only drop, and still scored 441.3 against 436.7 over three
replicates of 500 questions, so re-picking is a floor on the gain, not the gain.

The fixtures are verbatim from question 92a0aa75 of the benchmark, where the
career fact was served with an excerpt about a colleague on the sales team and
the reader answered "2 years and 4 months" where the gold is "1 year and 5
months".
"""
from __future__ import annotations

import os

import pytest

from synap.cloud.agents.retrieval.sentence_scorer import (
    evidence_min_support, extract_evidence, support_pct)

FACT = ("The user started as a Marketing Coordinator and worked their way up "
        "to Senior Marketing Specialist over a period of 2 years and 4 months.")
SUPPORTING = ("User: I've been in marketing for a while now, started as a "
              "Marketing Coordinator and worked my way up to Senior Marketing "
              "Specialist after 2 years and 4 months.")
UNRELATED = ("Assistant: By presenting complex data in a clear and concise "
             "manner you can help your sales team make better decisions. "
             "User: I've been working closely with Tom, our point of contact "
             "on the sales team, to align on sales strategies and goals.")
QUERY = "How long have I been working in my current role?"


@pytest.fixture(autouse=True)
def _no_inherited_floor(monkeypatch):
    monkeypatch.delenv("SYNAP_EVIDENCE_MIN_SUPPORT", raising=False)


def test_support_separates_the_two_real_excerpts():
    assert support_pct(SUPPORTING, FACT) > 50
    assert support_pct(UNRELATED, FACT) < 25


def test_off_by_default_so_nothing_changes_for_a_caller():
    assert evidence_min_support() == 0.0
    got = extract_evidence(chunk_texts=[UNRELATED, SUPPORTING], query=QUERY,
                           memory_content=FACT, max_chunks=1)
    assert got is not None and len(got) == 1


def test_floor_re_picks_the_supporting_excerpt():
    got = extract_evidence(chunk_texts=[UNRELATED, SUPPORTING], query=QUERY,
                           memory_content=FACT, max_chunks=1, min_support=25)
    assert got == [SUPPORTING]


def test_floor_withholds_only_when_nothing_supports_the_fact():
    got = extract_evidence(chunk_texts=[UNRELATED], query=QUERY,
                           memory_content=FACT, max_chunks=1, min_support=25)
    assert got is None


@pytest.mark.parametrize("raw,want", [("25", 25.0), ("0", 0.0),
                                      ("nonsense", 0.0), ("-5", 0.0),
                                      ("500", 100.0)])
def test_the_env_var_is_read_and_clamped(monkeypatch, raw, want):
    monkeypatch.setenv("SYNAP_EVIDENCE_MIN_SUPPORT", raw)
    assert evidence_min_support() == want


def test_a_fact_with_no_content_words_is_left_alone():
    got = extract_evidence(chunk_texts=[UNRELATED], query=QUERY,
                           memory_content="it is a the", max_chunks=1,
                           min_support=25)
    assert got == [UNRELATED]
