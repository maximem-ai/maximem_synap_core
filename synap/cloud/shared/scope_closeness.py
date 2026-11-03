"""How much nearer material should outrank material from further up.

The problem this solves is dilution. A requester deep in a ladder can read
everything above them, and everything above them is usually a much larger pool
than their own node holds. Without a closeness term, a generic fact true of the
whole client outranks the specific fact about this person, because both are
similar to the query and there are a thousand times more of the former. That
failure mode is on record from the LoCoMo and LongMemEval work; it is not
hypothetical.

R-07 IS WHY THIS FILE IS SMALL AND SEPARATE
-------------------------------------------
No ranking term may reference permission, and no permission decision may
reference distance. Once "far away" and "forbidden" differ only by degree, a
high enough similarity score defeats isolation, and the whole thing stops being
a boundary and becomes a preference.

So nothing here takes a permission, a grant, a policy or a scope predicate.
Distance in, multiplier out. What a requester may read is settled earlier and
elsewhere, by the pool builder (R-08), and a record they may not see never
reaches a scorer at all. The signature is the enforcement, and
`test_scope_closeness.py` asserts it structurally rather than trusting review.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

# R-03. The cost of one level of distance, as a fraction of score lost per rung.
#
# MEASURED, and moved from 0.15 to 0.30 on the result. See SHIP-07.
#
# The measurement judges the ANSWER, not the ordering: 1,440 bundles assembled
# in the order served, each handed to a model with a question whose true
# value differs by rung, scored against ground truth known by construction.
# Both directions are measured, because the term cannot tell them apart. When
# the answer is the requester's own, raising the cost from 0.15 to 0.30 takes
# correct answers from 56% to 82% and confident wrong answers from 11% to 2%.
# When the answer is the organisation's, it goes the other way, 9% correct
# falling to 4% and 45% wrong rising to 55%.
#
# Which direction dominates is a fact about real data, not a preference. Most stored
# memories sit at a collapsed single rung where
# customer_id equals user_id, so every record takes the same multiplier and this
# term cannot reorder anything for them. Among the minority in pools where it
# can fire, nearly all records sit at the requester's own rung. At that mix 0.30 wins
# on both counts at once, and 0.15 is beaten at every mix by one end or the
# other: it is not a safe middle, it is dominated.
#
# Two limits, stated rather than buried. The 91.6% is a fact about the
# pre-nested product, and real ladders are what will change it, so the trigger
# to re-measure is numeric: re-run the sweep once more than 25% of records sit
# in multi-rung pools. And for those 91.6% this term still changes nothing at
# all, so its value is not what decides whether they get good answers.
#
# R-03 keeps this per-client regardless. A client whose ladder is shallow and
# whose shared material carries the weight wants a gentler value than one whose
# leaves hold everything, and no single constant serves both.
DEFAULT_PER_LEVEL_COST = 0.30

# G-09. What a record borrowed through a grant keeps, relative to one the
# requester owns. Borrowed context is less about the requester than their own
# is, and a client's own material should not be pushed down the page by
# something another team happened to write.
#
# ALSO NOT MEASURED, and it shares the closeness cost's problem rather than the
# pool reserve's: it changes ordering, so a wrong value produces a worse answer
# rather than a slower one. 0.5 is a deliberate halving, chosen to be visible
# in a benchmark rather than to be right.
DEFAULT_GRANT_PENALTY = 0.5

# A cost of 1.0 would zero out everything above the requester's own node, which
# is not a tuning choice but a different feature (no upward reads at all). A
# cost of 0 disables closeness entirely, which is a legitimate thing to ask for.
_MIN_COST = 0.0
_MAX_COST = 0.9


def distance_from_leaf(chain_node_ids: Sequence[str], record_node_id: Optional[str]) -> Optional[int]:
    """How many rungs above the requester a record sits. 0 is their own node.

    None when the record is not on the chain at all, which the caller must
    treat as "do not score", never as "very far". A record off the chain is one
    the requester cannot read, and turning that into a large distance is
    precisely the blend R-07 forbids: it would let a high enough similarity
    score outrank the fact that they are not allowed to see it.
    """
    if not record_node_id:
        return None
    for distance, node in enumerate(reversed(list(chain_node_ids))):
        if node == record_node_id:
            return distance
    return None


def closeness_multiplier(distance: Optional[int], per_level_cost: float = DEFAULT_PER_LEVEL_COST) -> float:
    """The factor to apply to a record's relevance score.

    Multiplicative and not subtractive, for two reasons. It is scale free, so
    it behaves the same whether scores run 0 to 1 or 0 to 100 and does not need
    to know which. And it cannot push a score below zero or invert an ordering
    that was already correct: every record at the same distance is scaled by the
    same amount, so within a level the original ranking is untouched.

    Returns 0.0 for a record that is not on the chain. That is a scorer refusing
    to rank something the pool builder should never have handed it, not a
    permission decision: by R-08 such a record is absent from the pool, so this
    is the belt to that braces.
    """
    if distance is None:
        return 0.0
    if distance < 0:
        raise ValueError(f"distance cannot be negative, got {distance}")
    cost = _clamp_cost(per_level_cost)
    return (1.0 - cost) ** distance


def _clamp_cost(cost: float) -> float:
    """Keep a client's configured value inside a range that still means something."""
    try:
        cost = float(cost)
    except (TypeError, ValueError):
        return DEFAULT_PER_LEVEL_COST
    if cost != cost:
        # NaN. Clamping it lands on the steepest allowed cost, because every
        # comparison against NaN is false, so a typo in a config file would
        # quietly become the most aggressive setting available. Fall back
        # instead: an unreadable value should mean "use the default", not
        # "use the extreme".
        return DEFAULT_PER_LEVEL_COST
    return max(_MIN_COST, min(_MAX_COST, cost))


def per_level_cost_for(config: Optional[Mapping] = None) -> float:
    """R-03. The client's configured cost, or the system default.

    Reads a plain mapping rather than a MACA object so the scorer depends on
    nothing that could carry a permission into it, which is R-07 again.
    """
    if not config:
        return DEFAULT_PER_LEVEL_COST
    raw = config.get("scope_closeness_per_level_cost")
    return DEFAULT_PER_LEVEL_COST if raw is None else _clamp_cost(raw)


def apply_closeness(
    score: float,
    chain_node_ids: Sequence[str],
    record_node_id: Optional[str],
    per_level_cost: float = DEFAULT_PER_LEVEL_COST,
    *,
    from_grant: bool = False,
    grant_penalty: float = DEFAULT_GRANT_PENALTY,
) -> float:
    """One record's score, adjusted for how far up it sits and where it came from.

    R-09: this is a term inside the single relevance pass, not a second pass.
    Scores drawn from separately ranked per-level lists are not comparable, so
    a merge of them cannot produce a correct global order however carefully it
    is done.

    `from_grant` is provenance, not permission, and the difference is what
    keeps R-07 intact. The scorer is told which fetch a record arrived on, by
    the caller that issued the fetches. It is never handed a grant list, a
    policy, or a predicate, and it never decides whether anything is readable:
    by R-08 an unreadable record is absent from the pool before any of this
    runs. A record that came through a grant is off the chain, so its distance
    is None and the closeness term alone would score it zero; the penalty
    replaces that rather than compounding with it.
    """
    if from_grant:
        return score * _clamp_penalty(grant_penalty)
    return score * closeness_multiplier(
        distance_from_leaf(chain_node_ids, record_node_id), per_level_cost
    )


def _clamp_penalty(penalty: float) -> float:
    """Keep a configured penalty inside a range that still means something.

    0 would make granted material invisible, which is a different feature
    (no grants) reached by a setting that says "penalty". 1 would make it
    indistinguishable from the requester's own, which is G-09 not happening.
    """
    try:
        penalty = float(penalty)
    except (TypeError, ValueError):
        return DEFAULT_GRANT_PENALTY
    if penalty != penalty:
        return DEFAULT_GRANT_PENALTY
    return max(0.05, min(1.0, penalty))
