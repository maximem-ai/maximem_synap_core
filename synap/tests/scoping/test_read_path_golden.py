"""What the three vector searches actually send to the store, today.

This is a golden test written *before* the read path is changed, not after. The
nested filter cannot be merged into the flat where-clause dicts these call sites
build (a `$or` does not survive `{**a, **b}`), so wiring S-07 in means touching
three functions that run on every client's every request.

The safety net for that is knowing exactly what they emit now. Each case below
records the `filters` and `top_k` handed to the vector provider and pins them.
If a refactor changes one byte for a caller that has not opted in, this fails
and names the case.

Nothing here asserts that today's behaviour is *correct*. It asserts that it is
what it is, which is the only thing a golden test can honestly claim.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from synap.cloud.application.client_api.retrieval_manager import (
    MACARetrievalSettings,
    RetrievalManager,
)
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL, ScopeHierarchy

pytestmark = pytest.mark.asyncio

CLIENT = str(UUID("11111111-1111-5111-8111-111111111111"))
CUST = str(UUID("22222222-2222-5222-8222-222222222222"))
USER = str(UUID("33333333-3333-5333-8333-333333333333"))
ALL_TYPES = ["facts", "preferences", "episodes", "emotions", "temporal_events"]


def _scope(user_id: str | None = USER) -> ScopeHierarchy:
    return ScopeHierarchy(
        instance_id=str(uuid4()), client_id=CLIENT, customer_id=CUST, user_id=user_id,
    )


class _Recorder:
    """Captures every call the retrieval path makes to the vector store."""

    def __init__(self):
        self.calls: list[dict] = []

    def _record(self, method):
        async def _fn(**kwargs):
            self.calls.append({
                "method": method,
                "filters": kwargs.get("filters"),
                "top_k": kwargs.get("top_k"),
                "scope_level": kwargs.get("scope_level"),
            })
            return [[] for _ in kwargs.get("embeddings", [])]
        return _fn

    def as_provider(self) -> AsyncMock:
        v = AsyncMock()
        for m in ("batch_search", "batch_search_questions", "batch_search_sentences"):
            setattr(v, m, AsyncMock(side_effect=self._record(m)))
        return v


def _manager(rec: _Recorder, maca: MACARetrievalSettings | None = None) -> RetrievalManager:
    m = RetrievalManager(vector_provider=rec.as_provider(), postgres_provider=AsyncMock())
    m._current_maca = maca or MACARetrievalSettings()
    return m


def _vec():
    return [np.zeros(8, dtype=np.float32)]


def _filters(rec: _Recorder, method: str, scope_level: str | None = None):
    for c in rec.calls:
        if c["method"] == method and (scope_level is None or c["scope_level"] == scope_level):
            return c["filters"], c["top_k"]
    raise AssertionError(f"no {method} call recorded, got {[c['method'] for c in rec.calls]}")


# ── memories, the main path ─────────────────────────────────────────────────

async def test_memories_full_alphabet_sends_no_filter_and_overfetches():
    """The shipped default. Every memory_type is asked for, so the type filter
    has pass-rate 1.0 and the whole predicate moves to Python with a 3x
    overfetch. This is the behaviour R-11 says is wrong for nested ladders and
    correct for today's, so it has to keep working for callers who have not
    opted in."""
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    f, k = _filters(rec, "batch_search", "customer")
    assert f is None
    assert k == 75
    f_client, k_client = _filters(rec, "batch_search", "client")
    assert f_client is None
    assert k_client == 24


async def test_memories_selective_type_keeps_the_type_at_the_store():
    """A subset of types is genuinely selective, so dropping it would lose
    high-similarity items past the overfetch window. Type stays at Chroma,
    scope goes to Python, and there is no overfetch."""
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _scope(), ["q"], _vec(), ["facts"], top_k=25
    )
    f, k = _filters(rec, "batch_search", "customer")
    assert f == {"memory_type": {"$in": ["facts"]}}
    assert k == 25


async def test_memories_customer_scope_caller():
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _scope(user_id=None), ["q"], _vec(), ["facts"], top_k=10
    )
    f, _ = _filters(rec, "batch_search", "customer")
    assert f == {"memory_type": {"$in": ["facts"]}}


async def test_memories_visibility_pins_instance_at_the_store():
    """Cross-instance isolation is exact at Chroma even when everything else is
    post-filtered, because a selective allowed-set starves the overfetch
    window."""
    rec = _Recorder()
    m = _manager(rec)
    m._current_visibility = frozenset({"inst-a"})
    await m._direct_similarity_search(_scope(), ["q"], _vec(), ALL_TYPES, top_k=25)
    f, _ = _filters(rec, "batch_search", "customer")
    assert f == {"instance_id": {"$in": ["inst-a"]}}
    f_client, _ = _filters(rec, "batch_search", "client")
    assert f_client is None, "client-scope shared knowledge is never instance-filtered"


# ── questions ───────────────────────────────────────────────────────────────

async def test_questions_post_filters_and_overfetches():
    rec = _Recorder()
    await _manager(rec)._search_questions(_scope(), ["q"], _vec(), top_k=10)
    f, k = _filters(rec, "batch_search_questions", "customer")
    assert f is None
    assert k == 30


async def test_questions_visibility_pins_instance():
    rec = _Recorder()
    m = _manager(rec)
    m._current_visibility = frozenset({"inst-a"})
    await m._search_questions(_scope(), ["q"], _vec(), top_k=10)
    f, _ = _filters(rec, "batch_search_questions", "customer")
    assert f == {"instance_id": {"$in": ["inst-a"]}}


# ── sentences ───────────────────────────────────────────────────────────────

async def test_sentences_post_filters_and_overfetches():
    rec = _Recorder()
    await _manager(rec)._search_sentences(_scope(), ["q"], _vec(), top_k=30)
    f, k = _filters(rec, "batch_search_sentences")
    assert f is None
    assert k == 90


async def test_sentences_visibility_pins_instance():
    rec = _Recorder()
    m = _manager(rec)
    m._current_visibility = frozenset({"inst-a", "inst-b"})
    await m._search_sentences(_scope(), ["q"], _vec(), top_k=30)
    f, _ = _filters(rec, "batch_search_sentences")
    assert set(f) == {"instance_id"}
    assert sorted(f["instance_id"]["$in"]) == ["inst-a", "inst-b"]


# ── the post-filter escape hatch ────────────────────────────────────────────

async def test_with_post_filter_disabled_everything_goes_to_the_store(monkeypatch):
    """SYNAP_VECTOR_POST_FILTER=false is the documented escape hatch, and it is
    also the shape the nested path will use: the full scope predicate compiled
    to the store. Pinning it here means the refactor has a same-shaped case to
    match before any nested chain exists."""
    import synap.cloud.application.client_api.retrieval_manager as rm

    monkeypatch.setattr(rm, "_USE_PYTHON_POST_FILTER", False)
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    f, k = _filters(rec, "batch_search", "customer")
    assert f == {
        "memory_type": {"$in": ALL_TYPES},
        "customer_id": CUST,
        "user_id": {"$in": [USER, CUSTOMER_SCOPE_SENTINEL]},
    }
    assert k == 25


# ── the nested path ─────────────────────────────────────────────────────────

TEAM = str(UUID("44444444-4444-5444-8444-444444444444"))
REGION = str(UUID("55555555-5555-5555-8555-555555555555"))


def _nested_scope(user_id: str | None = USER) -> ScopeHierarchy:
    """A five-level chain, so nothing here can pass by accident on a depth-3
    shape that happens to match today's product."""
    s = _scope(user_id)
    s.chain_rungs = [
        {"node_id": UUID(CLIENT), "level_key": "client", "role": "root"},
        {"node_id": UUID(REGION), "level_key": "region", "role": None},
        {"node_id": UUID(CUST), "level_key": "customer", "role": "tenant_of_record"},
        {"node_id": UUID(TEAM), "level_key": "team", "role": None},
    ] + ([{"node_id": UUID(user_id), "level_key": "user", "role": "identity"}] if user_id else [])
    return s


async def test_nested_read_sends_the_chain_to_the_store():
    """S-07 and R-11: the predicate is the chain and it goes to the store.

    ⚠ This used to assert `k == 25`, justified by "overfetch is for
    post-filtering, and nothing is post-filtered here". True of the mechanism
    and false of the effect, and the test pinned the effect: the shipped default
    asks for `top_k * 3` and post-filters at a pass rate of about 1.0, so
    roughly 75 candidates reach the ranker, and switching the feature on would
    have handed it 25. The closeness pass reorders that pool, so a smaller pool
    is strictly less to reorder. Found by audit 2026-08-23.
    """
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _nested_scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    f, k = _filters(rec, "batch_search", "customer")
    assert k == 75, (
        "the nested path must hand the ranker the same number of candidates the "
        "legacy path does, or turning the feature on shrinks recall"
    )
    assert "$or" in f
    nested_arm = f["$or"][0]
    assert nested_arm == {"node_id": {"$in": [CLIENT, REGION, CUST, TEAM, USER]}}


async def test_nested_read_drops_the_full_type_alphabet():
    """R-12. A clause that excludes nothing still costs on every visited
    candidate; measured, the legacy shape whose memory_type matches everything
    costs 858ms against 33ms for a selective one."""
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _nested_scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    f, _ = _filters(rec, "batch_search", "customer")
    assert "memory_type" not in str(f)


async def test_nested_read_keeps_a_selective_type():
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _nested_scope(), ["q"], _vec(), ["facts"], top_k=25
    )
    f, _ = _filters(rec, "batch_search", "customer")
    assert {"memory_type": "facts"} in f["$and"]


async def test_nested_read_carries_visibility_inside_the_clause():
    """Merging it afterwards would put a second key beside a `$or` at the same
    level, which is not a where-clause Chroma accepts."""
    rec = _Recorder()
    m = _manager(rec)
    m._current_visibility = frozenset({"inst-a"})
    await m._direct_similarity_search(_nested_scope(), ["q"], _vec(), ALL_TYPES, top_k=25)
    f, _ = _filters(rec, "batch_search", "customer")
    assert set(f) == {"$and"}, f"expected a single $and, got keys {set(f)}"
    # The instance clause has one spelling, shared with the legacy read
    # (`scope_filter.instance_visibility_predicate`): a one-element `$in`. It
    # used to be a bare equality here, two spellings of one clause.
    assert {"instance_id": {"$in": ["inst-a"]}} in f["$and"]
    assert any("$or" in c for c in f["$and"])


async def test_nested_read_leaves_the_client_collection_alone():
    """Client-level records carry no customer or user id, so the chain
    predicate would match none of them. The collection name is already the
    scope."""
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _nested_scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    f_client, _ = _filters(rec, "batch_search", "client")
    assert f_client is None


async def test_nested_questions_and_sentences_also_go_to_the_store():
    rec = _Recorder()
    m = _manager(rec)
    await m._search_questions(_nested_scope(), ["q"], _vec(), top_k=10)
    f, k = _filters(rec, "batch_search_questions", "customer")
    assert k == 10 and "$or" in f

    rec2 = _Recorder()
    await _manager(rec2)._search_sentences(_nested_scope(), ["q"], _vec(), top_k=30)
    f2, k2 = _filters(rec2, "batch_search_sentences")
    assert k2 == 30 and "$or" in f2


async def test_a_requester_who_can_read_nothing_never_touches_the_store():
    """An isolated client whose viewer holds no grants. Returning early is both
    correct and the cheapest thing available; compiling an empty `$in` instead
    risks a where-clause that matches everything."""
    rec = _Recorder()
    m = _manager(rec)
    m._current_visibility = frozenset()
    out = await m._direct_similarity_search(_nested_scope(), ["q"], _vec(), ALL_TYPES, top_k=25)
    assert out == []
    assert rec.calls == [], "the store was queried for a requester who can read nothing"


async def test_a_customer_scope_nested_caller_has_no_identity_arm():
    rec = _Recorder()
    await _manager(rec)._direct_similarity_search(
        _nested_scope(user_id=None), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    f, _ = _filters(rec, "batch_search", "customer")
    legacy_arm = f["$or"][1]
    assert legacy_arm == {"$and": [{"customer_id": CUST}, {"user_id": CUSTOMER_SCOPE_SENTINEL}]}


async def test_a_broken_chain_falls_back_instead_of_widening():
    """If the nested predicate cannot be compiled, the legacy path is narrower
    than it for every record that exists today, so the failure mode is a missing
    new-style record and never a leak."""
    rec = _Recorder()
    s = _nested_scope()
    s.chain_rungs = [{"level_key": "client"}]        # no node_id at all
    await _manager(rec)._direct_similarity_search(s, ["q"], _vec(), ALL_TYPES, top_k=25)
    f, k = _filters(rec, "batch_search", "customer")
    assert f is None and k == 75, "expected the legacy post-filter path"


async def test_a_team_level_record_survives_the_post_filter_stage():
    """The load-bearing reason the nested path must not post-filter.

    `_scope_filter_predicate` only knows customer_id and user_id. A record
    filed at a team, which carries node_id and neither of those, fails it. The
    store has already applied the chain predicate exactly, so running the legacy
    predicate over the result would silently drop exactly the records nested
    scoping exists to return, and it would look like nothing was ever written.
    """
    rec = _Recorder()

    async def _one_team_record(**kwargs):
        if kwargs.get("scope_level") == "client":
            return [[] for _ in kwargs.get("embeddings", [])]
        return [[{
            "id": "mem-team-1",
            "score": 0.9,
            "document": "the payments team deploys on Thursdays",
            "metadata": {"node_id": TEAM, "anc_customer": CUST, "memory_type": "facts"},
        }] for _ in kwargs.get("embeddings", [])]

    provider = rec.as_provider()
    provider.batch_search = AsyncMock(side_effect=_one_team_record)
    m = RetrievalManager(vector_provider=provider, postgres_provider=AsyncMock())
    m._current_maca = MACARetrievalSettings()

    items = await m._direct_similarity_search(
        _nested_scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["mem-team-1"], (
        "a team-level record was dropped after the store had already approved it"
    )


async def test_switching_the_feature_on_does_not_shrink_the_candidate_pool():
    """The property directly, rather than as two numbers in two tests.

    Whatever the two paths do differently, the ranker must not be given less to
    work with when nested scoping is switched on."""
    legacy, nested = _Recorder(), _Recorder()
    await _manager(legacy)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    await _manager(nested)._direct_similarity_search(
        _nested_scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    _, k_legacy = _filters(legacy, "batch_search", "customer")
    _, k_nested = _filters(nested, "batch_search", "customer")
    assert k_nested >= k_legacy, (
        f"the nested path asks for {k_nested} candidates and the legacy path "
        f"asks for {k_legacy}; turning the feature on would shrink the pool"
    )

    _, c_legacy = _filters(legacy, "batch_search", "client")
    _, c_nested = _filters(nested, "batch_search", "client")
    assert c_nested >= c_legacy, (
        f"the client-scope fetch shrinks from {c_legacy} to {c_nested}"
    )
