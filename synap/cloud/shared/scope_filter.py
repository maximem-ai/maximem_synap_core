"""Compile a resolved scope chain into a filter, once, for any store.

WHY THIS EXISTS AT ALL
----------------------
Today the same scope predicate is written twice: `_build_customer_scope_filter`
produces a Chroma `where` clause and `_scope_filter_predicate` produces the
equivalent Python callable for the post-filter path. The docstring on the second
one says any change to the first must be mirrored here. S-08 asks that they stay
in lockstep, and the only way to actually get that is to stop writing them
twice.

So the predicate is built once as a small tree of Eq / In / And / Or, and each
backend compiles that tree. Chroma and Python cannot disagree about semantics
because neither one decides the semantics. Adding Qdrant is then a third
compiler over the same tree, which is what S-09 needs: the store decision was
deliberately left open, and it stays open only while swapping stores is a
compile target rather than a rewrite.

WHAT THE PREDICATE IS
---------------------
S-07: an upward read is the requester's node and its ancestors, which is exactly
the chain resolution already produced, so the nested form is one `IN` over node
ids. It does not change shape when the ladder gets deeper, and it is immune to a
mid-ladder insert because nothing in it refers to a level.

R-11 says this predicate goes to the store rather than to a Python post-filter.
Post-filtering scope fails by pool starvation, not by ranking: measured at
overfetch 3, realised recall equals pool fill exactly, 0.621 at 20% readable
falling to 0.021 at 1%, and raising the multiplier does not rescue it. The
Python compiler here exists for tests and for filtering material that is already
in hand, never as the scope gate on a nested read.

THE LEGACY ARM
--------------
Records written before this feature carry `customer_id` and `user_id` and no
`node_id`, so the nested form alone would match nothing that already exists.
The one case that needs care is customer-shared material, which stores the
sentinel in `user_id` rather than a uuid (S-05), so it is matched through
`customer_id` instead.

⚠ This used to say "a legacy record's `user_id` *is* its node id", and the
caller acted on it by taking `tenant_node_id` and `identity_node_id` off the
resolved chain. That equality is real only because `resolve_legacy_chain` passes
the data-plane uuid as a node id hint (N-14). It is an accident of one code
path, not a property of the model: N-04 makes a node id a SURROGATE, and a node
that a `scope` path created first was minted with `gen_random_uuid()`.

Where it was false, both arms missed and a person's whole history stopped being
readable behind an HTTP 200. See `_legacy_arm_ids` in the retrieval manager for
the measurement and B-15 for the report. The values now come from the request's
own resolved data-plane uuids, so this arm no longer depends on what the
registry happened to record.

The legacy arm is therefore an `$or` branch and it is deliberately temporary.
R-12 says the scope predicate stays narrow, and this widens it, so it is a flag
that can be measured and then turned off once every record carries `node_id`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL

# Metadata field names. Named here rather than inline so the write side and the
# read side cannot disagree about spelling, which is a defect this repo has had
# more than once.
FIELD_NODE_ID = "node_id"
FIELD_CUSTOMER_ID = "customer_id"
FIELD_USER_ID = "user_id"
FIELD_INSTANCE_ID = "instance_id"
# A record written by several instances carries them as a LIST. Graph nodes and
# relationships do (shared/graph_boundary), because one node is shared by every
# instance of a customer and a scalar would be overwritten by the last writer.
FIELD_INSTANCE_IDS = "instance_ids"
FIELD_MEMORY_TYPE = "memory_type"

ANCESTOR_PREFIX = "anc_"


# ---------------------------------------------------------------------------
# The predicate tree. Deliberately tiny: four node types is enough for every
# scope predicate we have, and a bigger language would be a second query
# planner nobody asked for.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Eq:
    field: str
    value: str


@dataclass(frozen=True)
class In:
    field: str
    values: Tuple[str, ...]


@dataclass(frozen=True)
class AnyOf:
    """True when the record's LIST under `field` shares a value with `values`.

    For fields that hold several values (a graph record's `instance_ids`). A
    record holding a scalar there is treated as a one-element list.
    """
    field: str
    values: Tuple[str, ...]


@dataclass(frozen=True)
class And:
    clauses: Tuple["Predicate", ...]


@dataclass(frozen=True)
class Or:
    clauses: Tuple["Predicate", ...]


Predicate = Union[Eq, In, AnyOf, And, Or]


class NothingReadable(Exception):
    """The request can read nothing at all, so there is no query to compile.

    Raised rather than returned as a predicate because "matches nothing" has no
    portable spelling: an empty `$in` is rejected by some Chroma versions and
    accepted by others, and a Qdrant `should` with no clauses is not a filter.
    Guessing a spelling that quietly matches *everything* is the failure mode
    this exists to prevent, and it would be a cross-tenant one.

    The caller returns an empty result without touching the store, which is also
    the cheapest thing it could do.
    """


def _flatten(kind, clauses: Iterable[Predicate]) -> Predicate:
    """Collapse a one-clause And/Or to the clause itself.

    Not cosmetic: Chroma rejects `$and` with a single operand, and a compiled
    query that a human cannot read is a compiled query nobody checks.
    """
    kept = [c for c in clauses if c is not None]
    if not kept:
        raise ValueError("empty predicate")
    if len(kept) == 1:
        return kept[0]
    return kind(tuple(kept))


def _one_of(field: str, values: Sequence[str]) -> Predicate:
    uniq = list(dict.fromkeys(str(v) for v in values))
    if not uniq:
        raise ValueError(f"no values for {field}")
    return Eq(field, uniq[0]) if len(uniq) == 1 else In(field, tuple(uniq))


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def nested_scope_predicate(chain_node_ids: Sequence[str]) -> Predicate:
    """S-07. The requester's node and its ancestors, and nothing else."""
    if not chain_node_ids:
        # Resolution always yields at least the root, so an empty chain means
        # the caller skipped resolution. Compiling it away would produce a
        # predicate with no scope arm at all.
        raise NothingReadable("empty scope chain: nothing was resolved for this request")
    return _one_of(FIELD_NODE_ID, chain_node_ids)


def legacy_scope_predicate(
    *,
    tenant_node_id: Optional[str],
    identity_node_id: Optional[str],
    roll_up_to_customer: bool = True,
    cross_user: bool = False,
    cross_customer: bool = False,
) -> Optional[Predicate]:
    """Today's filter, expressed in the same tree.

    This is `_build_customer_scope_filter` with the names changed: the customer
    node is the tenant_of_record and the user node is the identity, which is the
    whole of what those two ids ever meant. Keeping it in the same language is
    what makes the two forms comparable, and comparing them is how we will know
    the legacy arm is safe to drop.
    """
    clauses: list[Predicate] = []
    if tenant_node_id and not cross_customer:
        clauses.append(Eq(FIELD_CUSTOMER_ID, str(tenant_node_id)))

    if not cross_user:
        if identity_node_id:
            if roll_up_to_customer:
                clauses.append(_one_of(FIELD_USER_ID, [str(identity_node_id), CUSTOMER_SCOPE_SENTINEL]))
            else:
                clauses.append(Eq(FIELD_USER_ID, str(identity_node_id)))
        elif tenant_node_id:
            # A caller who named no identity reads shared material only, never
            # some individual's personal memories.
            clauses.append(Eq(FIELD_USER_ID, CUSTOMER_SCOPE_SENTINEL))

    if not clauses:
        return None
    return _flatten(And, clauses)


def instance_visibility_predicate(allowed: Sequence[str]) -> Predicate:
    """The cross-instance policy: only records written by these instances.

    The single place this clause is spelled. Every read that applies a
    visibility set (the nested builder, the legacy builder, short-term turns,
    the starvation repair) goes through here, so a change to it cannot reach
    some reads and miss others.
    """
    values = list(dict.fromkeys(str(v) for v in allowed))
    if not values:
        # An isolated viewer holding no readable source. An empty `$in` is not
        # a portable "matches nothing", so there is no query to compile.
        raise NothingReadable("no source instance is visible to this requester")
    # Always an In, even for one instance: the store has been sent
    # `{"$in": [...]}` for this clause since the feature shipped, and the golden
    # read-path tests pin it. Eq and a one-element In select the same records.
    return In(FIELD_INSTANCE_ID, tuple(values))


def graph_instance_predicate(allowed: Sequence[str]) -> Predicate:
    """The cross-instance policy for records that carry a LIST of instances.

    The sibling of `instance_visibility_predicate`, for graph nodes and
    relationships. Same input, same refusal when nothing is readable; only the
    spelling differs, because the field is a list.
    """
    values = list(dict.fromkeys(str(v) for v in allowed))
    if not values:
        raise NothingReadable("no source instance is visible to this requester")
    return AnyOf(FIELD_INSTANCE_IDS, tuple(values))


def memory_type_predicate(memory_types: Sequence[str]) -> In:
    """The type condition: only records of these memory types.

    Not a scope rule, but it is the same kind of rule: the store clause and the
    Python post-filter must agree on it, so it is spelled once here and both are
    compiled from it. An empty list stays an empty `In`, exactly what the
    callers have always sent; they decide whether to ask the store at all.
    """
    return In(FIELD_MEMORY_TYPE, tuple(dict.fromkeys(str(t) for t in memory_types)))


def legacy_read_predicate(
    *,
    customer_id: Optional[str],
    user_id: Optional[str],
    roll_up_to_customer: bool = True,
    cross_user: bool = False,
    cross_customer: bool = False,
    visibility_allowed: Optional[Sequence[str]] = None,
) -> Optional[Predicate]:
    """The whole read predicate for a request with NO resolved chain.

    This is what the long-term collections have always been read with: the
    cross-instance policy ANDed with the customer and user rule. It is the
    legacy arm of `build_scope_filter` plus the visibility clause, in one
    flat conjunction, so the Chroma where-clause and the Python post-filter
    are two compilations of one value rather than two functions that must be
    kept in step by hand.

    None means "no restriction at all" (a client-scope caller, with the
    escape hatches on and no visibility policy).
    """
    clauses: list[Predicate] = []
    if visibility_allowed is not None:
        clauses.append(instance_visibility_predicate(visibility_allowed))
    legacy = legacy_scope_predicate(
        tenant_node_id=customer_id,
        identity_node_id=user_id,
        roll_up_to_customer=roll_up_to_customer,
        cross_user=cross_user,
        cross_customer=cross_customer,
    )
    if legacy is not None:
        clauses.extend(legacy.clauses if isinstance(legacy, And) else [legacy])
    if not clauses:
        return None
    return _flatten(And, clauses)


def build_scope_filter(
    chain_node_ids: Sequence[str],
    *,
    tenant_node_id: Optional[str] = None,
    identity_node_id: Optional[str] = None,
    roll_up_to_customer: bool = True,
    cross_user: bool = False,
    cross_customer: bool = False,
    include_legacy: bool = True,
    visibility_allowed: Optional[Sequence[str]] = None,
    memory_types: Optional[Sequence[str]] = None,
    memory_type_universe: Optional[frozenset] = None,
) -> Predicate:
    """The whole read predicate for one request.

    `visibility_allowed` is the cross-instance policy and stays outside the
    scope arm: it is an `AND` over both forms, because a legacy record is no
    less subject to it than a new one.

    `memory_types` is dropped when it covers the whole alphabet. That is R-12
    and it is not a micro-optimisation: measured on Chroma, the legacy filter
    shape whose `memory_type $in` matches every document costs 858ms against
    33ms for a selective one on the same collection. A condition that excludes
    nothing still pays for itself on every visited candidate.
    """
    arms: list[Predicate] = [nested_scope_predicate(chain_node_ids)]

    if include_legacy:
        legacy = legacy_scope_predicate(
            tenant_node_id=tenant_node_id,
            identity_node_id=identity_node_id,
            roll_up_to_customer=roll_up_to_customer,
            cross_user=cross_user,
            cross_customer=cross_customer,
        )
        if legacy is not None:
            arms.append(legacy)

    clauses: list[Predicate] = [_flatten(Or, arms)]

    if visibility_allowed is not None:
        # An empty set raises NothingReadable: an isolated viewer holding no
        # grants has an empty read set, which is neither an error nor
        # "unrestricted".
        clauses.append(instance_visibility_predicate(visibility_allowed))

    if memory_types:
        covers_everything = bool(
            memory_type_universe and memory_type_universe.issubset(frozenset(memory_types))
        )
        if not covers_everything:
            clauses.append(_one_of(FIELD_MEMORY_TYPE, list(memory_types)))

    return _flatten(And, clauses)


def descendant_predicate(level_key: str, ancestor_node_id: str) -> Predicate:
    """R-05 downward reads, which are flag-gated and off by default.

    Cheap only because S-03 puts a level-keyed ancestor on every record: asking
    for everything beneath a node is one equality on `anc_<level_key>`, with no
    tree walk and no second query.
    """
    return Eq(f"{ANCESTOR_PREFIX}{level_key}", str(ancestor_node_id))


# ---------------------------------------------------------------------------
# Compilers. One per backend, all over the same tree.
# ---------------------------------------------------------------------------

def to_chroma_where(pred: Predicate) -> Dict[str, Any]:
    """Chroma's `where` document."""
    if isinstance(pred, Eq):
        return {pred.field: pred.value}
    if isinstance(pred, In):
        return {pred.field: {"$in": list(pred.values)}}
    if isinstance(pred, AnyOf):
        # Chroma has no "list overlaps list"; one `$contains` per value, ORed.
        # Verified against a Chroma server (list metadata, $contains, $or).
        parts = [{pred.field: {"$contains": v}} for v in pred.values]
        return parts[0] if len(parts) == 1 else {"$or": parts}
    if isinstance(pred, And):
        return {"$and": [to_chroma_where(c) for c in pred.clauses]}
    if isinstance(pred, Or):
        return {"$or": [to_chroma_where(c) for c in pred.clauses]}
    raise TypeError(f"not a predicate: {pred!r}")


def to_flat_where(pred: Optional[Predicate]) -> Dict[str, Any]:
    """Chroma's `where` as ONE flat dict, for a conjunction of leaf clauses.

    The legacy read has always sent `{"customer_id": ..., "user_id": {...}}`
    rather than `{"$and": [...]}`, and the golden read-path tests pin that shape
    byte for byte. It is also what the hot path's callers merge further keys
    into. Only a conjunction of distinct-field Eq/In clauses can be written this
    way; anything with an Or, or two clauses on one field, is refused rather
    than silently collapsed, because collapsing is how a scope clause gets lost.

    None compiles to {}, which means "no restriction" and is only ever produced
    by `legacy_read_predicate` when the request genuinely has none.
    """
    if pred is None:
        return {}
    leaves = pred.clauses if isinstance(pred, And) else (pred,)
    out: Dict[str, Any] = {}
    for leaf in leaves:
        if not isinstance(leaf, (Eq, In)):
            raise TypeError(f"not a flat conjunction: {leaf!r}")
        if leaf.field in out:
            raise ValueError(f"two clauses on {leaf.field}: a flat where would drop one")
        out.update(to_chroma_where(leaf))
    return out


def to_qdrant_filter(pred: Predicate) -> Dict[str, Any]:
    """Qdrant's filter document.

    `must` is AND and `should` is OR. Written as plain dicts rather than
    qdrant_client models so this module imports no vendor package, which is the
    point of S-09.

    ⚠ It used to emit `min_should: 1` alongside `should`, on the reasoning that
    "a `should` without minimum_should_match ranks rather than excludes". That
    is Elasticsearch's behaviour and Elasticsearch's spelling. In Qdrant a bare
    `should` already excludes, and `min_should` takes a `{"conditions": [...],
    "min_count": n}` object rather than an integer, so the document this
    produced was **rejected outright** by the client's own validation. The
    transitional `$or`, which is every request during the transition, could not
    be expressed at all.

    Validated against the installed `qdrant_client`: `min_should=1` raises a
    ValidationError; the same filter without it is accepted. Removing the key
    both fixes the document and preserves the semantics. Found by audit
    2026-08-23.
    """
    if isinstance(pred, Eq):
        return {"key": pred.field, "match": {"value": pred.value}}
    if isinstance(pred, (In, AnyOf)):
        # Qdrant's `match.any` already means "any of these values", and on an
        # array payload it matches when any element does.
        return {"key": pred.field, "match": {"any": list(pred.values)}}
    if isinstance(pred, And):
        return {"must": [to_qdrant_filter(c) for c in pred.clauses]}
    if isinstance(pred, Or):
        return {"should": [to_qdrant_filter(c) for c in pred.clauses]}
    raise TypeError(f"not a predicate: {pred!r}")


def to_python_predicate(pred: Predicate) -> Callable[[Mapping[str, Any]], bool]:
    """The same predicate over a metadata dict.

    For tests and for filtering material already in hand. Never the scope gate
    on a nested read: see R-11 and the module docstring.
    """
    if isinstance(pred, Eq):
        field, value = pred.field, pred.value
        return lambda m: m.get(field) == value
    if isinstance(pred, In):
        field, values = pred.field, frozenset(pred.values)
        return lambda m: m.get(field) in values
    if isinstance(pred, AnyOf):
        field, values = pred.field, frozenset(pred.values)

        def _overlaps(m, field=field, values=values):
            held = m.get(field)
            if held is None:
                return False
            if isinstance(held, (list, tuple, set, frozenset)):
                return not values.isdisjoint(str(h) for h in held)
            return str(held) in values
        return _overlaps
    if isinstance(pred, And):
        parts = [to_python_predicate(c) for c in pred.clauses]
        return lambda m: all(p(m) for p in parts)
    if isinstance(pred, Or):
        parts = [to_python_predicate(c) for c in pred.clauses]
        return lambda m: any(p(m) for p in parts)
    raise TypeError(f"not a predicate: {pred!r}")


def fields_referenced(pred: Predicate) -> set[str]:
    """Every metadata field the predicate touches.

    Used by the R-12 guard test: a five-level request must compile to chain
    node ids and nothing else once the legacy arm is off.
    """
    if isinstance(pred, (Eq, In, AnyOf)):
        return {pred.field}
    out: set[str] = set()
    for c in pred.clauses:
        out |= fields_referenced(c)
    return out
