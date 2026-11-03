"""Jaccard dedupe must not depend on the order items arrive in.

`_deduplicate_jaccard` keeps the FIRST of a duplicate pair. `all_items` in
ACCURATE mode is assembled from concurrently gathered sources, so the same
request can present the same items in a different sequence between runs, and the
answer changed with it.

Measured after pinning the retrieval LLM temperatures:
two cold runs of the same build still disagreed on 2 requests, and in both the
same client-scope fact, identical content and identical score 0.398602, appeared
in one run and not the other, for two different users. The score was not moving.
The decision was.

The same code also picked the wrong winner: the tiebreak only replaced an
existing item when the new one had a score and the existing one did not, so
between two scored duplicates the first to arrive won even when the other scored
higher.
"""
from __future__ import annotations

import random

import pytest

from synap.cloud.application.client_api.retrieval_manager import _deduplicate_jaccard


class _Item:
    """Minimal stand-in: the function only reads these four attributes."""

    def __init__(self, item_id, content, similarity=0.0, relevance=0.0):
        self.item_id = item_id
        self.content = content
        self.similarity_score = similarity
        self.relevance_score = relevance

    def __repr__(self):
        return f"<{self.item_id} {self.similarity_score}>"


# Near-identical text, so Jaccard puts them well over the 0.45 threshold.
_DUP_A = "Dietary preferences vegetarian vegan Jain non-veg and seating notes"
_DUP_B = "Dietary preferences vegetarian vegan Jain non-veg and seating notes today"
_OTHER = "Completely unrelated content about quarterly revenue forecasting models"


def test_the_same_items_in_a_different_order_give_the_same_survivors():
    """The regression this was written for."""
    base = [
        _Item("a", _DUP_A, similarity=0.398602),
        _Item("b", _DUP_B, similarity=0.512000),
        _Item("c", _OTHER, similarity=0.300000),
    ]
    expected = {i.item_id for i in _deduplicate_jaccard(list(base))}

    rng = random.Random(20260824)
    for _ in range(25):
        shuffled = list(base)
        rng.shuffle(shuffled)
        got = {i.item_id for i in _deduplicate_jaccard(shuffled)}
        assert got == expected, (
            f"dedupe depends on arrival order: {got} != {expected}. "
            f"Order was {[i.item_id for i in shuffled]}"
        )


def test_the_higher_scoring_duplicate_is_the_one_kept():
    """Not just stable, correct. Previously whichever arrived first won."""
    low = _Item("low", _DUP_A, similarity=0.398602)
    high = _Item("high", _DUP_B, similarity=0.512000)

    for order in ([low, high], [high, low]):
        kept = _deduplicate_jaccard(list(order))
        assert [i.item_id for i in kept] == ["high"], (
            f"expected the higher-scoring duplicate to survive, got "
            f"{[i.item_id for i in kept]} for input order "
            f"{[i.item_id for i in order]}"
        )


def test_equal_scores_still_resolve_the_same_way_every_time():
    """A tie must not reintroduce arrival-order dependence."""
    a = _Item("aaa", _DUP_A, similarity=0.4)
    b = _Item("bbb", _DUP_B, similarity=0.4)
    first = [i.item_id for i in _deduplicate_jaccard([a, b])]
    second = [i.item_id for i in _deduplicate_jaccard([b, a])]
    assert first == second, f"tie resolved differently: {first} vs {second}"


def test_the_returned_order_is_still_the_callers_order():
    """Selection is score-ordered; the sequence handed back is not. Downstream
    ranking is a separate step and must not silently change shape."""
    items = [
        _Item("x", _OTHER, similarity=0.10),
        _Item("y", "Totally different text about shipping logistics", similarity=0.90),
        _Item("z", "A third distinct subject entirely, about payroll", similarity=0.50),
    ]
    kept = _deduplicate_jaccard(list(items))
    assert [i.item_id for i in kept] == ["x", "y", "z"]


def test_an_unscored_duplicate_still_loses_to_a_scored_one():
    """The original intent of the tiebreak, preserved."""
    unscored = _Item("unscored", _DUP_A)
    scored = _Item("scored", _DUP_B, similarity=0.2)
    for order in ([unscored, scored], [scored, unscored]):
        kept = _deduplicate_jaccard(list(order))
        assert [i.item_id for i in kept] == ["scored"], [i.item_id for i in kept]


def test_empty_and_single_item_inputs_are_unchanged():
    assert _deduplicate_jaccard([]) == []
    one = _Item("solo", _OTHER, similarity=0.3)
    assert [i.item_id for i in _deduplicate_jaccard([one])] == ["solo"]
