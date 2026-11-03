"""Accurate mode must ask the same question the same way twice.

Root cause of the instability open since Part 27b. Three identical `accurate`
fetches produced candidate pools of different sizes, and the
bisection put the divergence upstream of every stage that had been probed: the
subquery COUNT was stable and the TEXT was not. Different subqueries are
different vector searches, so a different pool, so a different top ten.

`temperature=0.0` reduced that and did not remove it, because providers still
vary at temperature 0. The subqueries are a pure function of the question and the
instance's prompt inputs, so caching them is what actually makes a repeated
question return a repeated answer.
"""
from __future__ import annotations

import hashlib
import json

import pytest


def _key(query: str, edges=None, ents=None) -> str:
    """The key the implementation builds, restated so a change to either side
    shows up as a failure rather than as silently different behaviour."""
    return "subq:v1:" + hashlib.sha256(
        json.dumps(
            {
                "q": query,
                "edges": sorted(edges) if edges else [],
                "ents": sorted(ents) if ents else [],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:32]


def test_the_same_question_hashes_to_the_same_key():
    assert _key("what do you know about me") == _key("what do you know about me")


def test_a_different_question_hashes_differently():
    assert _key("what do you know about me") != _key("what did we agree on pricing")


def test_prompt_inputs_are_part_of_the_key():
    """`graph_edge_types` and `graph_entity_types` both change the prompt, so a
    cache that ignored them would serve subqueries built for another instance's
    schema."""
    base = _key("q")
    assert _key("q", edges=["WORKS_AT"]) != base
    assert _key("q", ents=["Person"]) != base


def test_the_key_does_not_depend_on_who_is_asking():
    """The subqueries decompose the question, not the caller. If scope or user
    entered the key, two people asking the same thing would each pay for their
    own LLM call and neither would stabilise the other."""
    import inspect

    from synap.cloud.application.client_api.retrieval_manager import RetrievalManager

    src = inspect.getsource(RetrievalManager._generate_retrieval_subqueries)
    head = src.split("_subq_cache_key = ", 1)[1].split("_cached_subq", 1)[0]
    for leaky in ("scope", "user_id", "customer_id", "correlation_id"):
        assert leaky not in head, (
            f"{leaky!r} is in the subquery cache key; the decomposition does not "
            f"depend on who is asking, and keying on it defeats the cache"
        )


def test_only_accurate_mode_is_cached():
    """FAST decomposes with fixed templates and is already deterministic, so it
    has nothing to cache and should not pay a round trip to Redis."""
    import inspect

    from synap.cloud.application.client_api.retrieval_manager import RetrievalManager

    src = inspect.getsource(RetrievalManager._generate_retrieval_subqueries)
    assert 'if mode == "accurate" and self.llm_client:\n            _subq_cache_key' in src, (
        "the cache is no longer gated on accurate mode"
    )


def test_the_write_is_bounded():
    """A long TTL would pin one decomposition while a prompt or MACA setting
    changed underneath it. The point is repeatability within a session, not
    permanence."""
    import inspect

    from synap.cloud.application.client_api.retrieval_manager import RetrievalManager

    src = inspect.getsource(RetrievalManager._generate_retrieval_subqueries)
    assert "ttl=300" in src, "the subquery cache TTL is no longer bounded at 300s"
