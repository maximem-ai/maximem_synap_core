"""Closeness ranking, and the boundary it must not cross.

Two things are being checked. The ordinary one is that nearer material outranks
identical material from further up, which is R-02 and is what stops a large
generic pool drowning a specific leaf memory.

The other one is R-07, and it is the reason this module is separate at all: no
ranking term may reference permission, and no permission decision may reference
distance. Once "far away" and "forbidden" differ only by degree, a high enough
similarity score defeats isolation. That is asserted structurally here, by
reading the module's own signatures, because it is exactly the kind of rule
that survives review and then dies to one convenient extra argument.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from synap.cloud.shared import scope_closeness as sc
from synap.cloud.shared.scope_closeness import (
    DEFAULT_PER_LEVEL_COST,
    apply_closeness,
    closeness_multiplier,
    distance_from_leaf,
    per_level_cost_for,
)

CHAIN = ["client-node", "region-node", "customer-node", "team-node", "user-node"]


# ── distance ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("node,expected", [
    ("user-node", 0),
    ("team-node", 1),
    ("customer-node", 2),
    ("region-node", 3),
    ("client-node", 4),
])
def test_distance_counts_rungs_up_from_the_requester(node, expected):
    assert distance_from_leaf(CHAIN, node) == expected


def test_a_record_off_the_chain_has_no_distance():
    """Not a large distance. A record off the chain is one the requester cannot
    read, and turning that into a number would let a high enough similarity
    score outrank the fact that they are not allowed to see it."""
    assert distance_from_leaf(CHAIN, "someone-elses-node") is None
    assert distance_from_leaf(CHAIN, None) is None
    assert distance_from_leaf(CHAIN, "") is None


def test_a_b2c_chain_with_a_repeated_node_measures_from_the_leaf():
    """B2C collapses identity onto the tenant, so the same id appears twice.
    The distance has to be the nearer of the two or shared material would be
    penalised for something that is an artefact of the shape."""
    chain = ["client-node", "cust-node", "cust-node"]
    assert distance_from_leaf(chain, "cust-node") == 0


# ── R-02: nearer wins at equal similarity ───────────────────────────────────

def test_at_equal_similarity_the_nearer_record_ranks_higher():
    leaf = apply_closeness(0.80, CHAIN, "user-node")
    ancestor = apply_closeness(0.80, CHAIN, "client-node")
    assert leaf > ancestor


def test_the_requester_own_node_is_never_penalised():
    assert closeness_multiplier(0) == 1.0
    assert apply_closeness(0.62, CHAIN, "user-node") == pytest.approx(0.62)


def test_a_much_better_ancestor_still_beats_a_poor_leaf():
    """The point of a gentle default. Shared knowledge is often the most
    valuable material a client has, and closeness is a tiebreaker, not a
    filter."""
    strong_ancestor = apply_closeness(0.95, CHAIN, "customer-node")
    weak_leaf = apply_closeness(0.40, CHAIN, "user-node")
    assert strong_ancestor > weak_leaf


def test_ordering_within_one_level_is_untouched():
    """Every record at the same distance is scaled by the same factor, so the
    original ranking inside a level survives unchanged."""
    a = apply_closeness(0.9, CHAIN, "customer-node")
    b = apply_closeness(0.8, CHAIN, "customer-node")
    c = apply_closeness(0.7, CHAIN, "customer-node")
    assert a > b > c


def test_scores_never_go_negative_or_invert():
    """Multiplicative and not subtractive: a subtractive penalty on a small
    score goes below zero and can flip an ordering that was already right."""
    for score in (0.0, 0.01, 0.5, 1.0, 100.0):
        for node in CHAIN:
            assert apply_closeness(score, CHAIN, node) >= 0.0


# ── R-03: tunable per client ────────────────────────────────────────────────

def test_the_default_is_a_named_constant():
    assert closeness_multiplier(1) == pytest.approx(1.0 - DEFAULT_PER_LEVEL_COST)


def test_two_clients_with_different_costs_can_order_the_same_fixture_differently():
    """R-03's own verification: the knob has to actually change an outcome, or
    it is a setting that does nothing."""
    leaf, ancestor = 0.70, 0.80
    gentle = 0.05
    steep = 0.60
    assert apply_closeness(ancestor, CHAIN, "customer-node", gentle) > apply_closeness(
        leaf, CHAIN, "user-node", gentle
    )
    assert apply_closeness(ancestor, CHAIN, "customer-node", steep) < apply_closeness(
        leaf, CHAIN, "user-node", steep
    )


def test_a_cost_of_zero_disables_closeness_entirely():
    """A legitimate thing for a client to ask for, and it must be exactly a
    no-op rather than nearly one."""
    for node in CHAIN:
        assert apply_closeness(0.5, CHAIN, node, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [1.0, 5.0, -3.0, None, "steep", float("nan")])
def test_a_nonsense_cost_falls_back_or_clamps_rather_than_breaking_ranking(bad):
    """This value comes from client configuration. A cost of 1.0 would zero out
    everything above the requester, which is not a tuning choice but a
    different feature, and a scorer is the wrong place to discover that."""
    out = apply_closeness(0.5, CHAIN, "customer-node", bad)
    assert 0.0 < out <= 0.5


def test_the_configured_value_is_read_from_a_plain_mapping():
    assert per_level_cost_for(None) == DEFAULT_PER_LEVEL_COST
    assert per_level_cost_for({}) == DEFAULT_PER_LEVEL_COST
    assert per_level_cost_for({"scope_closeness_per_level_cost": 0.3}) == pytest.approx(0.3)
    assert per_level_cost_for({"other": 1}) == DEFAULT_PER_LEVEL_COST


# ── R-07: the boundary ──────────────────────────────────────────────────────

_FORBIDDEN = (
    "permission", "grant", "policy", "allowed", "visibility", "acl",
    "authorized", "authorised", "can_read", "predicate", "filter",
)


# A scorer may take a FLAG or a WEIGHT about permission-shaped things. It may
# not take a permission SET, mapping or callable.
#
# The distinction is the rule, not a loophole for it. `from_grant: bool` says
# which fetch a record arrived on, decided by the caller that issued the
# fetches; `grant_penalty: float` is a tuning weight. Neither lets the scorer
# decide whether anything is readable. A `granted_node_ids: set` would: the
# scorer would then be evaluating membership, and the next person holding both
# a score and a set is one line away from including instead of penalising,
# which is exactly the blend R-07 forbids.
# Both forms, because `from __future__ import annotations` makes every
# annotation a string at runtime. A guard that only knew the real types would
# pass on any module using postponed evaluation, which is all of them here.
_SAFE_ANNOTATIONS = (bool, int, float, "bool", "int", "float")


@pytest.mark.parametrize("fn", [distance_from_leaf, closeness_multiplier, apply_closeness])
def test_no_scoring_function_can_be_handed_a_permission_set(fn):
    """R-07 asserted structurally, not by review. A scorer that cannot be
    handed a permission cannot blend one into a score, however tempting the
    shortcut looks to whoever is holding both at the call site."""
    for name, param in inspect.signature(fn).parameters.items():
        if not any(word in name.lower() for word in _FORBIDDEN):
            continue
        assert param.annotation in _SAFE_ANNOTATIONS, (
            f"{fn.__name__} takes {name!r} as {param.annotation!r}. A flag or a weight "
            "is fine; anything that can hold a set of permissions is not, because the "
            "scorer would then be deciding membership."
        )


def test_the_module_imports_nothing_that_carries_a_permission():
    """The other half of the same rule. A function whose parameters are clean
    can still reach for a policy through an import."""
    tree = ast.parse(Path(sc.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for mod in imported:
        assert not any(word in mod.lower() for word in ("privacy", "policy", "auth", "grant")), (
            f"scope_closeness imports {mod!r}"
        )


def test_an_off_chain_record_scores_zero_rather_than_ranking_last():
    """R-08 says a record the requester may not see never enters the pool, so
    this is the belt to that braces. Scoring it zero rather than assigning it a
    large distance keeps the two ideas from blending: it is refused, not
    ranked."""
    assert closeness_multiplier(None) == 0.0
    assert apply_closeness(0.99, CHAIN, "another-tenants-node") == 0.0


def test_a_negative_distance_is_a_bug_and_says_so():
    with pytest.raises(ValueError):
        closeness_multiplier(-1)


def test_a_nan_cost_falls_back_rather_than_clamping_to_the_extreme():
    """Every comparison against NaN is false, so a naive clamp lands on the
    steepest allowed cost and a typo in a config file quietly becomes the most
    aggressive setting available."""
    assert per_level_cost_for({"scope_closeness_per_level_cost": float("nan")}) == pytest.approx(
        DEFAULT_PER_LEVEL_COST
    )
    assert apply_closeness(0.5, CHAIN, "customer-node", float("nan")) == pytest.approx(
        apply_closeness(0.5, CHAIN, "customer-node")
    )


# ── G-09: borrowed context ranks below your own ─────────────────────────────

def test_a_granted_record_ranks_below_an_identical_one_of_your_own():
    """Borrowed context is less about the requester than their own is, and a
    client's own material should not be pushed down the page by something
    another team happened to write."""
    from synap.cloud.shared.scope_closeness import DEFAULT_GRANT_PENALTY

    own = apply_closeness(0.8, CHAIN, "user-node")
    borrowed = apply_closeness(0.8, CHAIN, "another-teams-node", from_grant=True)
    assert borrowed < own
    assert borrowed == pytest.approx(0.8 * DEFAULT_GRANT_PENALTY)


def test_a_granted_record_is_not_also_penalised_for_being_off_the_chain():
    """It IS off the chain, so the closeness term alone scores it zero. The
    penalty replaces that rather than compounding with it, or every granted
    record would be invisible and G-09 would be G-nothing."""
    borrowed = apply_closeness(0.9, CHAIN, "another-teams-node", from_grant=True)
    assert borrowed > 0.0


def test_a_strong_granted_record_can_still_beat_a_weak_own_one():
    """A penalty, not a filter. Grants exist to surface material, so a much
    better borrowed record should still win."""
    borrowed = apply_closeness(0.95, CHAIN, "another-teams-node", from_grant=True)
    weak_own = apply_closeness(0.20, CHAIN, "user-node")
    assert borrowed > weak_own


@pytest.mark.parametrize("bad", [0.0, -1.0, 5.0, None, "half", float("nan")])
def test_a_nonsense_penalty_clamps_or_falls_back(bad):
    """0 would make granted material invisible, which is a different feature
    reached by a setting that says "penalty"."""
    out = apply_closeness(0.5, CHAIN, "another-teams-node", from_grant=True, grant_penalty=bad)
    assert 0.0 < out <= 0.5


def test_the_grant_flag_is_provenance_and_not_a_permission():
    """R-07 again. The scorer is told which fetch a record arrived on, by the
    caller that issued the fetches. It is never handed a grant list, a policy
    or a predicate, and it never decides whether anything is readable."""
    params = inspect.signature(apply_closeness).parameters
    assert params["from_grant"].annotation in (bool, "bool")
    assert params["grant_penalty"].annotation in (float, "float")
    for name, param in params.items():
        if any(word in name.lower() for word in _FORBIDDEN):
            assert param.annotation in _SAFE_ANNOTATIONS


def test_default_cost_is_the_measured_value_not_a_drifted_one():
    """R-03 / SHIP-07. The default is 0.30 because 1,440 judged bundles said so.

    This asserts the literal, which every other test in this file deliberately
    avoids doing. The others take the constant as given because they are testing
    the curve, and pinning the number in them would make an intentional change
    look like fifteen failures. This one exists to make that change land in
    exactly one place, with a reason attached.

    If you are changing this number: the instrument is
    `scripts/measure_closeness_answer_quality.py`, it needs a fresh run, and the
    trigger recorded for re-measuring is more than 25% of records sitting in
    multi-rung pools. It was 8.2% on 2026-08-23, and for the other 91.6% this
    constant changes nothing at all because every record takes the same
    multiplier.
    """
    assert DEFAULT_PER_LEVEL_COST == 0.30
    # And the shape it implies, so a change that keeps the number but breaks the
    # curve is caught too: two rungs up keeps roughly half.
    assert closeness_multiplier(2) == pytest.approx(0.49)
