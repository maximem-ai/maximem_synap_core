"""R-10: is the requester's own material in the pool at all?

The test that matters is the last one. It builds the situation the rule exists
for, an ancestor level holding far more matching material than the leaf, and
asserts the leaf's records come back. Everything above it is the machinery that
makes that possible.
"""
from __future__ import annotations

import uuid as _uuid

import pytest

from synap.cloud.shared.scope_filter import (
    NothingReadable,
    fields_referenced,
    to_chroma_where,
    to_python_predicate,
)
from synap.cloud.shared.scope_pool import (
    DEFAULT_LEAF_RESERVE,
    leaf_reserve,
    merge_pool,
    plan_pool,
)

CLIENT = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "cli"))
REGION = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "emea"))
CUST = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "infosys"))
TEAM = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "payments"))
USER = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "d-42"))
CHAIN = [CLIENT, REGION, CUST, TEAM, USER]


# ── the plan ────────────────────────────────────────────────────────────────

def test_a_deep_request_gets_a_second_narrow_fetch():
    plan = plan_pool(CHAIN, 25, include_legacy=False)
    assert plan.fetches == 2
    assert plan.main_top_k == 25
    assert plan.leaf is not None
    assert to_chroma_where(plan.leaf) == {"node_id": USER}
    assert plan.leaf_top_k == DEFAULT_LEAF_RESERVE


def test_the_reserve_never_becomes_most_of_the_answer():
    """At that point the main fetch is the thing being starved, which is this
    failure inverted."""
    assert leaf_reserve(2) <= 1
    assert leaf_reserve(4) <= 2
    assert leaf_reserve(25) == DEFAULT_LEAF_RESERVE


def test_a_root_level_requester_gets_no_second_fetch():
    """The requester IS the root, so the main fetch already reads only their
    node and a second identical query would be pure cost."""
    plan = plan_pool([CLIENT], 25, include_legacy=False)
    assert plan.fetches == 1
    assert plan.leaf is None


def test_a_b2c_chain_that_collapses_gets_no_second_fetch():
    plan = plan_pool([CUST, CUST], 25, include_legacy=False)
    assert plan.fetches == 1


def test_the_leaf_fetch_is_built_from_the_same_predicate_tree():
    """So it compiles to whatever store is in use and cannot drift from the
    main filter's semantics."""
    plan = plan_pool(CHAIN, 25, include_legacy=False)
    assert fields_referenced(plan.leaf) == {"node_id"}
    assert fields_referenced(plan.main) == {"node_id"}


def test_the_leaf_fetch_reads_only_the_requester_own_node():
    plan = plan_pool(CHAIN, 25, include_legacy=False)
    predicate = to_python_predicate(plan.leaf)
    assert predicate({"node_id": USER})
    for ancestor in (CLIENT, REGION, CUST, TEAM):
        assert not predicate({"node_id": ancestor})


# ── R-17: the reserve is subject to everything the main fetch is ────────────
#
# These are leak tests, not tidiness tests. The reserve narrows the SCOPE arm
# and nothing else, so anything the main fetch is filtered by must filter it
# too. It was built from `nested_scope_predicate` for a while, which is scope
# only, and that version would have returned another instance's records on
# every request from a requester more than one rung deep.

def test_the_reserve_carries_the_cross_instance_visibility_clause():
    """The leak. Without this the reserve reads instances the viewer may not."""
    plan = plan_pool(
        CHAIN, 25, include_legacy=False, visibility_allowed=["inst_a", "inst_b"]
    )
    assert "instance_id" in fields_referenced(plan.leaf)

    predicate = to_python_predicate(plan.leaf)
    assert predicate({"node_id": USER, "instance_id": "inst_a"})
    assert not predicate({"node_id": USER, "instance_id": "inst_elsewhere"})


def test_the_reserve_carries_the_memory_type_clause():
    plan = plan_pool(
        CHAIN,
        25,
        include_legacy=False,
        memory_types=["facts"],
        memory_type_universe=frozenset({"facts", "preferences", "events"}),
    )
    assert "memory_type" in fields_referenced(plan.leaf)

    predicate = to_python_predicate(plan.leaf)
    assert predicate({"node_id": USER, "memory_type": "facts"})
    assert not predicate({"node_id": USER, "memory_type": "preferences"})


def test_every_non_scope_field_the_main_fetch_filters_on_the_reserve_does_too():
    """The general form, so a clause added to the builder later cannot be
    silently absent from the reserve. `node_id` is excluded on purpose: the
    reserve narrowing it is the entire point of the reserve."""
    plan = plan_pool(
        CHAIN,
        25,
        include_legacy=False,
        visibility_allowed=["inst_a"],
        memory_types=["facts"],
        memory_type_universe=frozenset({"facts", "preferences"}),
    )
    main_fields = fields_referenced(plan.main) - {"node_id"}
    leaf_fields = fields_referenced(plan.leaf) - {"node_id"}
    assert main_fields <= leaf_fields, (
        f"main fetch filters on {sorted(main_fields - leaf_fields)} and the "
        "reserve does not, so the reserve is wider than the query it tops up"
    )


def test_the_reserve_drops_the_legacy_arm_even_when_the_main_fetch_keeps_it():
    """Deliberate, not an oversight. Before the backfill the legacy arm matches
    exactly the requester's own records and the main fetch already carries it,
    so a reserve carrying it too is a duplicate query. After the backfill the
    node id arm is the only one needed."""
    plan = plan_pool(
        CHAIN,
        25,
        include_legacy=True,
        tenant_node_id=CUST,
        identity_node_id=USER,
    )
    assert "customer_id" in fields_referenced(plan.main)
    assert fields_referenced(plan.leaf) == {"node_id"}


def test_a_reserve_that_matches_nothing_before_the_backfill_is_not_an_error():
    """During the transition no record carries `node_id`, so the reserve finds
    nothing. That is the correct behaviour and must not raise: the main fetch's
    legacy arm is still returning the requester's own material."""
    plan = plan_pool(CHAIN, 25, include_legacy=False)
    predicate = to_python_predicate(plan.leaf)
    assert not predicate({"customer_id": CUST, "user_id": USER})


def test_an_empty_chain_still_raises_from_the_filter_builder():
    """Not second-guessed here. One place decides what an unresolvable request
    means."""
    with pytest.raises(NothingReadable):
        plan_pool([], 25)


def test_a_reserve_of_zero_disables_the_second_fetch():
    assert plan_pool(CHAIN, 25, reserve=0, include_legacy=False).fetches == 1


# ── the merge ───────────────────────────────────────────────────────────────

def test_merge_unions_without_ranking():
    """R-09. Scores drawn from separately ranked lists are not comparable, so
    any ordering imposed here would be a lie the ranker would have to undo."""
    main = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]
    leaf = [{"id": "c", "score": 0.2}]
    out = merge_pool(main, leaf)
    assert [r["id"] for r in out] == ["a", "b", "c"]


def test_a_record_in_both_fetches_appears_once_at_its_better_score():
    """The same record can come back from both with different scores when a
    store reports distance for one query shape and similarity for another.
    Keeping the higher is the safe direction: the ranker can demote it, but it
    cannot recover one that was dropped for looking worse than it is."""
    out = merge_pool(
        [{"id": "a", "score": 0.4}],
        [{"id": "a", "score": 0.8}],
    )
    assert len(out) == 1
    assert out[0]["score"] == 0.8

    # And the other way round. With only the first ordering, an implementation
    # that simply lets the last one win passes: the better score happens to
    # arrive second. Both orderings is what makes it a test of the rule.
    out = merge_pool(
        [{"id": "a", "score": 0.8}],
        [{"id": "a", "score": 0.4}],
    )
    assert len(out) == 1
    assert out[0]["score"] == 0.8


def test_records_with_no_usable_id_are_dropped_not_merged():
    out = merge_pool([{"score": 0.9}, {"id": "", "score": 0.8}, None, {"id": "x", "score": 0.1}])
    assert [r["id"] for r in out] == ["x"]


def test_a_missing_or_broken_score_does_not_break_the_merge():
    out = merge_pool([{"id": "a"}], [{"id": "a", "score": "not a number"}])
    assert len(out) == 1


# ── the rule itself ─────────────────────────────────────────────────────────

def test_a_lopsided_pool_still_returns_the_leaf_records():
    """R-10's own verification, and the reason any of this exists.

    The ancestor level holds a hundred records, every one of them a fraction
    more similar to the query than anything the leaf has. One similarity
    ordered fetch of 25 therefore comes back with no leaf material at all. The
    result is well ordered and wrong, and nothing about it looks broken.
    """
    top_k = 25
    ancestors = [{"id": f"anc-{i}", "score": 0.90 - i * 0.001, "node_id": CUST}
                 for i in range(100)]
    leaf_records = [{"id": f"leaf-{i}", "score": 0.60 - i * 0.001, "node_id": USER}
                    for i in range(10)]

    # what one fetch would return: the store ranks by similarity and stops
    main_fetch = sorted(ancestors + leaf_records, key=lambda r: -r["score"])[:top_k]
    assert not [r for r in main_fetch if r["node_id"] == USER], (
        "fixture is wrong: the leaf should be crowded out of a single fetch"
    )

    plan = plan_pool(CHAIN, top_k, include_legacy=False)
    leaf_fetch = sorted(leaf_records, key=lambda r: -r["score"])[: plan.leaf_top_k]
    pool = merge_pool(main_fetch, leaf_fetch)

    survivors = [r for r in pool if r["node_id"] == USER]
    assert survivors, "the requester's own material never reached the pool"
    assert len(survivors) == DEFAULT_LEAF_RESERVE
    assert survivors[0]["id"] == "leaf-0", "the leaf's best record should be the one kept"


def test_adding_the_reserve_never_displaces_a_record_that_would_have_won():
    """The safety argument, asserted. Ranking still runs once over the union
    and still returns top_k, so every record that would have won before is
    still present and still scores the same."""
    top_k = 5
    main_fetch = [{"id": f"m-{i}", "score": 0.9 - i * 0.01, "node_id": CUST} for i in range(top_k)]
    leaf_fetch = [{"id": "leaf-0", "score": 0.2, "node_id": USER}]
    pool = merge_pool(main_fetch, leaf_fetch)
    for record in main_fetch:
        assert record["id"] in {r["id"] for r in pool}
        assert next(r for r in pool if r["id"] == record["id"])["score"] == record["score"]
