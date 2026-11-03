"""The scope predicate, and the thing S-08 actually asks for.

S-08 says the store's where-clause and its Python equivalent must stay in
lockstep, and today they are two hand-written functions with a comment asking
whoever edits one to remember the other. These tests are written against a
design where that is not a thing anyone has to remember: one tree, three
compilers. The property test at the bottom is the one that would catch a drift
if the design ever regressed to two implementations.
"""
from __future__ import annotations

import itertools
import uuid as _uuid

import pytest

from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL
from synap.cloud.shared.scope_filter import (
    And,
    Eq,
    FIELD_NODE_ID,
    In,
    Or,
    build_scope_filter,
    descendant_predicate,
    fields_referenced,
    nested_scope_predicate,
    to_chroma_where,
    to_python_predicate,
    to_qdrant_filter,
)

ALL_TYPES = frozenset(("facts", "preferences", "episodes", "emotions", "temporal_events"))

CLIENT = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "cli_test"))
REGION = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "emea"))
CUST = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "infosys"))
TEAM = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "payments"))
USER = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "d-42"))
OTHER_USER = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "d-99"))
OTHER_CUST = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "tcs"))

CHAIN = [CLIENT, REGION, CUST, TEAM, USER]


# ── shape ───────────────────────────────────────────────────────────────────

def test_nested_predicate_is_one_in_over_the_chain() -> None:
    """S-07. Not a path prefix, not a per-level clause: the chain is already
    known from resolution, so this is today's two-element user filter
    generalised and nothing more."""
    p = nested_scope_predicate(CHAIN)
    assert isinstance(p, In)
    assert p.field == FIELD_NODE_ID
    assert list(p.values) == CHAIN


def test_depth_does_not_change_the_shape() -> None:
    """R-06 in miniature. Three levels and five levels compile to the same kind
    of clause, so a deeper ladder cannot cost a different query plan."""
    three = to_chroma_where(nested_scope_predicate(CHAIN[:3]))
    five = to_chroma_where(nested_scope_predicate(CHAIN))
    assert set(three) == set(five) == {FIELD_NODE_ID}
    assert len(five[FIELD_NODE_ID]["$in"]) == 5


def test_single_value_collapses_to_equality() -> None:
    """A one-element `$in` is a `$eq` that reads like a bug, and Chroma rejects
    a one-operand `$and` outright."""
    assert nested_scope_predicate([CLIENT]) == Eq(FIELD_NODE_ID, CLIENT)
    assert to_chroma_where(nested_scope_predicate([CLIENT])) == {FIELD_NODE_ID: CLIENT}


def test_duplicate_chain_entries_are_collapsed() -> None:
    """B2C puts the same uuid on the chain twice, at the tenant rung and at the
    identity rung."""
    p = nested_scope_predicate([CLIENT, CUST, CUST])
    assert isinstance(p, In) and list(p.values) == [CLIENT, CUST]


# ── R-12: nothing non-selective in the scope predicate ──────────────────────

def test_full_memory_type_alphabet_is_dropped() -> None:
    """Measured on Chroma: the legacy shape whose memory_type matches every
    document costs 858ms where a selective filter costs 33ms on the same
    collection. A clause that excludes nothing is not free."""
    p = build_scope_filter(
        CHAIN, include_legacy=False,
        memory_types=sorted(ALL_TYPES), memory_type_universe=ALL_TYPES,
    )
    assert fields_referenced(p) == {FIELD_NODE_ID}


def test_a_selective_memory_type_subset_is_kept() -> None:
    p = build_scope_filter(
        CHAIN, include_legacy=False,
        memory_types=["facts"], memory_type_universe=ALL_TYPES,
    )
    assert fields_referenced(p) == {FIELD_NODE_ID, "memory_type"}


def test_scope_predicate_alone_names_only_node_id() -> None:
    """R-12's assertion, verbatim: the compiled predicate for a five-level
    request contains only chain node ids."""
    p = build_scope_filter(CHAIN, include_legacy=False)
    assert fields_referenced(p) == {FIELD_NODE_ID}
    assert to_chroma_where(p) == {FIELD_NODE_ID: {"$in": CHAIN}}


# ── S-05: pre-migration records still resolve ───────────────────────────────

def _legacy_user_doc(cust: str, user: str) -> dict:
    return {"customer_id": cust, "user_id": user, "instance_id": "i1"}


def _legacy_shared_doc(cust: str) -> dict:
    """Customer-shared material. The sentinel is a one-bit materialised path
    and predates all of this, so it has to keep working untouched."""
    return {"customer_id": cust, "user_id": CUSTOMER_SCOPE_SENTINEL, "instance_id": "i1"}


def test_legacy_records_are_reachable_without_a_node_id() -> None:
    p = to_python_predicate(
        build_scope_filter(CHAIN, tenant_node_id=CUST, identity_node_id=USER)
    )
    assert p(_legacy_user_doc(CUST, USER))
    assert p(_legacy_shared_doc(CUST))


def test_the_sentinel_is_still_matched_at_customer_level(  ) -> None:
    """S-05, stated as its own test because the sentinel is the one legacy
    value that is not a uuid and so cannot be caught by the node_id arm."""
    p = to_python_predicate(
        build_scope_filter([CLIENT, CUST], tenant_node_id=CUST, identity_node_id=None)
    )
    assert p(_legacy_shared_doc(CUST))
    assert not p(_legacy_user_doc(CUST, USER))


def test_new_records_are_reachable_by_node_id_alone() -> None:
    p = to_python_predicate(
        build_scope_filter(CHAIN, tenant_node_id=CUST, identity_node_id=USER)
    )
    assert p({"node_id": TEAM})       # a team-level record, no legacy fields at all
    assert p({"node_id": REGION})


# ── R-01: the invariant ─────────────────────────────────────────────────────

@pytest.mark.parametrize("doc", [
    {"node_id": OTHER_USER},
    {"node_id": OTHER_CUST},
    _legacy_user_doc(CUST, OTHER_USER),
    _legacy_user_doc(OTHER_CUST, USER),
    _legacy_shared_doc(OTHER_CUST),
    {"customer_id": CUST},                      # user_id missing entirely
    {},
])
def test_nothing_outside_the_chain_passes(doc) -> None:
    """R-01 is the security invariant of the whole feature, and the `$or` that
    keeps legacy data reachable is exactly where a hole would open up. Each of
    these is a record that shares one field with the requester and must still
    be refused."""
    p = to_python_predicate(
        build_scope_filter(CHAIN, tenant_node_id=CUST, identity_node_id=USER)
    )
    assert not p(doc)


def test_a_sibling_teams_records_are_not_readable() -> None:
    """The originating case, from the read side. Infosys/payments must not see
    Infosys/risk even though they share every ancestor above the team."""
    risk = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "risk"))
    p = to_python_predicate(build_scope_filter(CHAIN, include_legacy=False))
    assert not p({"node_id": risk})
    assert p({"node_id": TEAM})


def test_cross_user_widening_is_explicit_and_still_bounded() -> None:
    """allow_cross_user_retrieval is a deliberate privacy widening, but it must
    not also widen across customers."""
    p = to_python_predicate(
        build_scope_filter(CHAIN, tenant_node_id=CUST, identity_node_id=USER, cross_user=True)
    )
    assert p(_legacy_user_doc(CUST, OTHER_USER))
    assert not p(_legacy_user_doc(OTHER_CUST, OTHER_USER))


def test_roll_up_off_hides_customer_shared_material() -> None:
    p = to_python_predicate(
        build_scope_filter(
            CHAIN, tenant_node_id=CUST, identity_node_id=USER, roll_up_to_customer=False
        )
    )
    assert p(_legacy_user_doc(CUST, USER))
    assert not p(_legacy_shared_doc(CUST))


def test_visibility_applies_to_both_arms() -> None:
    """A legacy record is no less subject to the cross-instance policy than a
    new one, so the instance clause is an AND outside the `$or`, not a third
    branch inside it."""
    p = to_python_predicate(
        build_scope_filter(
            CHAIN, tenant_node_id=CUST, identity_node_id=USER, visibility_allowed=["i1"]
        )
    )
    assert p(_legacy_user_doc(CUST, USER))
    assert not p({**_legacy_user_doc(CUST, USER), "instance_id": "i2"})
    assert not p({"node_id": TEAM, "instance_id": "i2"})
    assert p({"node_id": TEAM, "instance_id": "i1"})


# ── R-05 ────────────────────────────────────────────────────────────────────

def test_descendants_are_one_equality_on_a_level_keyed_ancestor() -> None:
    """S-03 is what makes a downward read cheap: no tree walk, no second query,
    one equality. That is why the flag can exist at all."""
    p = descendant_predicate("customer", CUST)
    assert to_chroma_where(p) == {"anc_customer": CUST}
    assert to_python_predicate(p)({"anc_customer": CUST, "node_id": USER})
    assert not to_python_predicate(p)({"anc_customer": OTHER_CUST})


# ── compilers ───────────────────────────────────────────────────────────────

def test_qdrant_compiles_from_the_same_tree() -> None:
    """S-09. The store decision stays reversible only while a new backend is a
    compiler over this tree rather than a second implementation of it."""
    p = build_scope_filter(CHAIN, include_legacy=False, visibility_allowed=["i1", "i2"])
    q = to_qdrant_filter(p)
    assert q["must"][0] == {"key": "node_id", "match": {"any": CHAIN}}
    assert q["must"][1] == {"key": "instance_id", "match": {"any": ["i1", "i2"]}}


def test_qdrant_or_is_a_bare_should() -> None:
    """⚠ This test used to assert `min_should == 1`, justified by "a `should`
    without minimum_should_match ranks rather than excludes". That is
    Elasticsearch's behaviour and its spelling. In Qdrant a bare `should`
    already excludes, and `min_should` takes an object rather than an integer,
    so the document the compiler produced was rejected outright by the client's
    own validation: the transitional `$or`, which is every request during the
    transition, could not be expressed at all. The test locked the defect in.
    Found by audit 2026-08-23."""
    q = to_qdrant_filter(build_scope_filter(CHAIN, tenant_node_id=CUST, identity_node_id=USER))
    top = q["must"][0] if "must" in q else q
    assert "min_should" not in top, (
        "min_should takes {'conditions': [...], 'min_count': n}, not an int; "
        "qdrant_client rejects the integer form outright"
    )
    assert len(top["should"]) == 2


def test_the_qdrant_document_is_one_qdrant_actually_accepts() -> None:
    """The reason the bug survived: nothing ever handed the output to Qdrant.
    S-09's verify line says the suite passes against both providers, and there
    is no Qdrant adapter, so "both providers" was never exercised. Validating
    against the client's own models is the cheapest honest substitute."""
    qm = pytest.importorskip(
        "qdrant_client.models",
        reason="qdrant_client not installed; the compiler is unvalidated here",
    )
    for pred in (
        build_scope_filter(CHAIN, tenant_node_id=CUST, identity_node_id=USER),
        build_scope_filter(CHAIN, include_legacy=False),
        build_scope_filter(CHAIN, include_legacy=False, visibility_allowed=["i1", "i2"]),
    ):
        doc = to_qdrant_filter(pred)
        # A bare condition is not a Filter; wrap it the way a caller would.
        if "key" in doc:
            doc = {"must": [doc]}
        qm.Filter(**doc)      # raises if the document is not one Qdrant takes


def test_no_vendor_package_is_imported() -> None:
    """S-09 again, structurally. `retrieval_manager` reaches Chroma semantics
    directly in several places today, which is precisely why swapping stores
    looked like a rewrite.

    Read as an AST, not as text. A substring search over the source also matches
    the word in a comment explaining why the import is absent, which is a test
    that fails on its own documentation and passes on a file that imports the
    package inside a function.
    """
    import ast

    import synap.cloud.shared.scope_filter as mod

    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"chromadb", "chroma", "qdrant_client", "qdrant", "pymilvus", "weaviate"}


# ── S-08, the property test ─────────────────────────────────────────────────

def _metadata_space():
    """Every combination of the fields a scope predicate can look at, including
    the absent case, which is how legacy records actually differ from new
    ones."""
    node = [None, USER, TEAM, OTHER_USER, CLIENT]
    cust = [None, CUST, OTHER_CUST]
    user = [None, USER, OTHER_USER, CUSTOMER_SCOPE_SENTINEL]
    inst = [None, "i1", "i2"]
    for n, c, u, i in itertools.product(node, cust, user, inst):
        m = {}
        if n: m["node_id"] = n
        if c: m["customer_id"] = c
        if u: m["user_id"] = u
        if i: m["instance_id"] = i
        yield m


def _chroma_eval(where, meta) -> bool:
    """A minimal interpreter for the subset of Chroma's `where` grammar this
    module emits. Small enough to read, which is the point: it is a second
    opinion, not a second copy of the builder."""
    if "$and" in where:
        return all(_chroma_eval(c, meta) for c in where["$and"])
    if "$or" in where:
        return any(_chroma_eval(c, meta) for c in where["$or"])
    (field, cond), = where.items()
    if isinstance(cond, dict):
        return meta.get(field) in cond["$in"]
    return meta.get(field) == cond


@pytest.mark.parametrize("legacy,rollup,cross_user,vis", [
    (True, True, False, None),
    (True, False, False, None),
    (True, True, True, None),
    (False, True, False, None),
    (True, True, False, ["i1"]),
])
def test_store_clause_and_python_predicate_agree_everywhere(legacy, rollup, cross_user, vis) -> None:
    """S-08. Run both over the same generated metadata and assert identical
    results, across every option combination that changes the tree."""
    pred = build_scope_filter(
        CHAIN, tenant_node_id=CUST, identity_node_id=USER,
        include_legacy=legacy, roll_up_to_customer=rollup,
        cross_user=cross_user, visibility_allowed=vis,
    )
    where = to_chroma_where(pred)
    py = to_python_predicate(pred)
    disagreements = [m for m in _metadata_space() if _chroma_eval(where, m) != py(m)]
    assert not disagreements, f"{len(disagreements)} disagreements, first: {disagreements[0]}"


# ── the empty read set ──────────────────────────────────────────────────────

def test_an_empty_visibility_set_is_an_empty_result_not_an_error() -> None:
    """An isolated client whose viewer holds no grants can read nothing. That
    has to be distinguishable from "unrestricted", because the two differ by
    exactly one dropped clause and getting it backwards is a cross-tenant
    leak."""
    from synap.cloud.shared.scope_filter import NothingReadable

    with pytest.raises(NothingReadable):
        build_scope_filter(CHAIN, visibility_allowed=[])

    # and None still means unrestricted, which is the case it must not become
    p = build_scope_filter(CHAIN, include_legacy=False, visibility_allowed=None)
    assert fields_referenced(p) == {FIELD_NODE_ID}


def test_an_empty_chain_is_refused_rather_than_compiled_away() -> None:
    """Resolution always yields at least the root, so an empty chain means the
    caller never resolved. Compiling it away would leave a query with no scope
    arm, which is the whole collection."""
    from synap.cloud.shared.scope_filter import NothingReadable

    with pytest.raises(NothingReadable):
        build_scope_filter([])
    with pytest.raises(NothingReadable):
        nested_scope_predicate([])
