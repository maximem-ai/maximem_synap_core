"""Unit tests for cross-instance visibility filtering.

Guards the two non-negotiable invariants of the read-side visibility feature:

1. Default-open no-op: with VISIBILITY_ALL (the default — no policy configured),
   the produced Chroma where-clause and Python predicate are byte-identical to
   pre-feature behaviour (no ``instance_id`` clause at all).
2. Mirror consistency: the Chroma where-clause ``_build_customer_scope_filter``
   and the Python predicate ``_scope_filter_predicate`` accept exactly the same
   documents — including the new ``instance_id`` clause under isolation.
"""

import pytest

from synap.cloud.application.client_api.retrieval_manager import (
    _build_customer_scope_filter,
    _scope_filter_predicate,
    _resolve_allowed_from_bundle,
    _VisibilityBundle,
    VISIBILITY_ALL,
    MACARetrievalSettings,
)
from synap.cloud.shared.scope import (
    ScopeHierarchy,
    CUSTOMER_SCOPE_SENTINEL,
    parse_or_deterministic_uuid,
)


def _uuid(raw):
    return str(parse_or_deterministic_uuid(raw))


def _scope(customer_id="cust-uuid", user_id="user-uuid", instance_id="inst-uuid",
           client_id="client-uuid"):
    return ScopeHierarchy(
        instance_id=instance_id,
        client_id=client_id,
        customer_id=customer_id,
        user_id=user_id,
    )


def _where_clause_accepts(where: dict, meta: dict) -> bool:
    """Evaluate a (small) Chroma-style where-clause against a metadata dict.

    Supports the only two shapes this code produces: ``{key: value}`` (equality)
    and ``{key: {"$in": [...]}}`` (membership).
    """
    for key, cond in where.items():
        if isinstance(cond, dict) and "$in" in cond:
            if meta.get(key) not in cond["$in"]:
                return False
        else:
            if meta.get(key) != cond:
                return False
    return True


# --- Invariant 1: default-open no-op -----------------------------------------

def test_default_open_is_identical_to_pre_feature():
    """VISIBILITY_ALL must produce the exact same where-clause as before, with
    no instance_id key anywhere."""
    scope = _scope()
    maca = MACARetrievalSettings()

    baseline = _build_customer_scope_filter(scope, maca)               # no vis arg
    with_all = _build_customer_scope_filter(scope, maca, VISIBILITY_ALL)

    assert baseline == with_all
    assert "instance_id" not in with_all


def test_default_open_predicate_accepts_any_instance():
    scope = _scope()
    maca = MACARetrievalSettings()
    pred = _scope_filter_predicate(scope, maca, VISIBILITY_ALL)

    for inst in ("inst-uuid", "some-other-instance", "anything"):
        assert pred({
            "customer_id": "cust-uuid",
            "user_id": "user-uuid",
            "instance_id": inst,
        }) is True


# --- Invariant 2: isolation adds the instance clause -------------------------

def test_isolation_adds_instance_in_clause():
    scope = _scope()
    maca = MACARetrievalSettings()
    allowed = frozenset({"inst-uuid", "child-uuid"})

    where = _build_customer_scope_filter(scope, maca, allowed)
    assert where["instance_id"] == {"$in": list(allowed)}


def test_isolation_predicate_rejects_disallowed_instance():
    scope = _scope()
    maca = MACARetrievalSettings()
    allowed = frozenset({"inst-uuid", "child-uuid"})
    pred = _scope_filter_predicate(scope, maca, allowed)

    base = {"customer_id": "cust-uuid", "user_id": "user-uuid"}
    assert pred({**base, "instance_id": "inst-uuid"}) is True       # self
    assert pred({**base, "instance_id": "child-uuid"}) is True      # granted/child
    assert pred({**base, "instance_id": "stranger-uuid"}) is False  # not allowed


# --- Invariant 2 (continued): Chroma where-clause and predicate agree --------

@pytest.mark.parametrize("visibility", [
    VISIBILITY_ALL,
    frozenset({"inst-uuid"}),
    frozenset({"inst-uuid", "child-uuid"}),
])
def test_where_clause_and_predicate_agree(visibility):
    """For a synthetic metadata corpus, the Chroma where-clause and the Python
    predicate must accept exactly the same documents."""
    scope = _scope()
    maca = MACARetrievalSettings()

    where = _build_customer_scope_filter(scope, maca, visibility)
    pred = _scope_filter_predicate(scope, maca, visibility)

    corpus = []
    for cust in ("cust-uuid", "other-cust"):
        for uid in ("user-uuid", CUSTOMER_SCOPE_SENTINEL, "other-user"):
            for inst in ("inst-uuid", "child-uuid", "stranger-uuid"):
                corpus.append({"customer_id": cust, "user_id": uid, "instance_id": inst})

    for meta in corpus:
        assert _where_clause_accepts(where, meta) == pred(meta), meta


# --- Resolution algorithm (isolated mode) ------------------------------------

def _bundle(grants=None, children=None, shareable=None):
    by_viewer = {}
    for v, s in (grants or []):
        by_viewer.setdefault(v, set()).add(s)
    return _VisibilityBundle(
        grants_by_viewer={v: frozenset(s) for v, s in by_viewer.items()},
        children_by_parent={p: tuple(c) for p, c in (children or {}).items()},
        shareable=dict(shareable or {}),
    )


def test_resolve_self_only_when_no_grants():
    bundle = _bundle()
    assert _resolve_allowed_from_bundle(bundle, "inst_a") == frozenset({_uuid("inst_a")})


def test_resolve_explicit_grant():
    bundle = _bundle(grants=[("inst_a", "inst_b")])
    assert _resolve_allowed_from_bundle(bundle, "inst_a") == frozenset(
        {_uuid("inst_a"), _uuid("inst_b")}
    )
    # The grant is directional: inst_b does not get inst_a.
    assert _resolve_allowed_from_bundle(bundle, "inst_b") == frozenset({_uuid("inst_b")})


def test_resolve_derived_parent_sees_children():
    # inst_child's parent is inst_parent -> parent sees child, child isolated.
    bundle = _bundle(children={"inst_parent": ["inst_child"]})
    assert _resolve_allowed_from_bundle(bundle, "inst_parent") == frozenset(
        {_uuid("inst_parent"), _uuid("inst_child")}
    )
    assert _resolve_allowed_from_bundle(bundle, "inst_child") == frozenset(
        {_uuid("inst_child")}
    )


def test_resolve_shareable_false_vetoes_grant():
    # inst_a is granted inst_b, but inst_b opted out of sharing -> not visible.
    bundle = _bundle(grants=[("inst_a", "inst_b")], shareable={"inst_b": False})
    assert _resolve_allowed_from_bundle(bundle, "inst_a") == frozenset({_uuid("inst_a")})


def test_resolve_shareable_false_vetoes_derived_child():
    bundle = _bundle(children={"inst_parent": ["inst_child"]}, shareable={"inst_child": False})
    assert _resolve_allowed_from_bundle(bundle, "inst_parent") == frozenset(
        {_uuid("inst_parent")}
    )
