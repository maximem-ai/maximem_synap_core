"""Gate 1: the ranking half, wired into the read path.

The filter decides WHAT a request may read and these decide what order it comes
back in. Only one of those is visible when it is missing. A request with the
filter and without this returns exactly the right records ranked by raw
similarity, which is the dilution R-02 exists to prevent, and nothing about the
answer looks broken.

Two of these tests guard against a wiring that would have been destructive
rather than merely wrong. `test_a_record_with_no_node_id_keeps_its_score` is the
data-loss guard: `distance_from_leaf` returns None for a missing id and
`closeness_multiplier` turns None into 0.0, so the naive reading drops every
record written before the metadata backfill. `test_client_level_knowledge_is_not_deleted`
is the same shape for shared knowledge.
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
from synap.cloud.shared.scope import ScopeHierarchy

pytestmark = pytest.mark.asyncio

CLIENT = str(UUID("11111111-1111-5111-8111-111111111111"))
REGION = str(UUID("55555555-5555-5555-8555-555555555555"))
CUST = str(UUID("22222222-2222-5222-8222-222222222222"))
TEAM = str(UUID("44444444-4444-5444-8444-444444444444"))
USER = str(UUID("33333333-3333-5333-8333-333333333333"))
STRANGER = str(UUID("99999999-9999-5999-8999-999999999999"))
ALL_TYPES = ["facts", "preferences", "episodes", "emotions", "temporal_events"]


def _scope(with_chain: bool = True, depth: int = 5) -> ScopeHierarchy:
    s = ScopeHierarchy(
        instance_id=str(uuid4()), client_id=CLIENT, customer_id=CUST, user_id=USER,
    )
    if not with_chain:
        return s
    full = [
        {"node_id": UUID(CLIENT), "level_key": "client", "role": "root"},
        {"node_id": UUID(REGION), "level_key": "region", "role": None},
        {"node_id": UUID(CUST), "level_key": "customer", "role": "tenant_of_record"},
        {"node_id": UUID(TEAM), "level_key": "team", "role": None},
        {"node_id": UUID(USER), "level_key": "user", "role": "identity"},
    ]
    s.chain_rungs = full[:depth] if depth < 5 else full
    return s


def _rec(rid: str, score: float, node_id: str | None, legacy_ids: bool = False):
    """`legacy_ids` puts the flat customer/user ids on the record.

    Needed only for the no-chain cases. On the legacy path the scope predicate
    runs in Python over `customer_id` and `user_id`, so a record carrying
    neither is dropped before it is ever scored, which is correct behaviour and
    nothing to do with closeness.
    """
    meta = {"memory_type": "facts"}
    if node_id is not None:
        meta["node_id"] = node_id
    if legacy_ids:
        meta["customer_id"] = CUST
        meta["user_id"] = USER
    return {"id": rid, "score": score, "content": rid, "metadata": meta}


class _Store:
    """Returns whatever each scope_level was told to return, and counts calls."""

    def __init__(self, customer=(), client=(), reserve=None):
        self.customer = list(customer)
        self.client = list(client)
        # None means "the reserve sees the same collection as the main fetch",
        # which is the truth: it is the same collection, narrower predicate.
        self.reserve = list(reserve) if reserve is not None else None
        self.calls: list[dict] = []

    def provider(self) -> AsyncMock:
        async def _search(**kw):
            level = kw.get("scope_level")
            self.calls.append({"scope_level": level, "top_k": kw.get("top_k"),
                               "filters": kw.get("filters")})
            n = len(kw.get("embeddings", []))
            if level == "client":
                rows = self.client
            elif self._is_reserve(kw):
                rows = self.reserve if self.reserve is not None else []
            else:
                rows = self.customer
            return [[dict(r) for r in rows] for _ in range(n)]

        v = AsyncMock()
        v.batch_search = AsyncMock(side_effect=_search)
        for m in ("batch_search_questions", "batch_search_sentences"):
            setattr(v, m, AsyncMock(
                side_effect=lambda **kw: [[] for _ in kw.get("embeddings", [])]
            ))
        return v

    def _is_reserve(self, kw) -> bool:
        """The reserve is the second customer-level call."""
        seen = [c for c in self.calls if c["scope_level"] == "customer"]
        return len(seen) > 1

    @property
    def customer_calls(self) -> int:
        return len([c for c in self.calls if c["scope_level"] == "customer"])


def _manager(store: _Store, cost: float | None = None) -> RetrievalManager:
    m = RetrievalManager(vector_provider=store.provider(), postgres_provider=AsyncMock())
    maca = MACARetrievalSettings()
    if cost is not None:
        maca.scope_closeness_per_level_cost = cost
    m._current_maca = maca
    return m


def _vec():
    return [np.zeros(8, dtype=np.float32)]


def _by_id(items):
    return {i.item_id: i for i in items}


# ── SHIP-01: closeness is a term in the single relevance pass ───────────────

async def test_a_nearer_record_can_beat_a_more_similar_distant_one():
    """The whole point of R-02. A generic fact true of the whole client is
    similar to almost any query and there is far more of it than there is of
    the specific fact about this person."""
    store = _Store(customer=[
        _rec("far", 0.90, CLIENT),   # root, four rungs up
        _rec("near", 0.80, USER),    # the requester's own node
    ])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["near"].relevance_score > got["far"].relevance_score, (
        "the ancestor record outranked the requester's own despite being four "
        "rungs away, so closeness is not reaching the score the sort reads"
    )
    # The raw number is untouched, because the store's similarity is still the
    # store's similarity and other things read it.
    assert got["far"].similarity_score == 0.90


async def test_a_cost_of_zero_returns_the_raw_order_unchanged():
    """R-03 says a client can turn this off, and off has to mean off."""
    store = _Store(customer=[
        _rec("far", 0.90, CLIENT),
        _rec("near", 0.80, USER),
    ])
    items = await _manager(store, cost=0.0)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["far"].relevance_score == pytest.approx(0.90)
    assert got["near"].relevance_score == pytest.approx(0.80)


async def test_records_on_the_same_rung_keep_their_relative_order():
    """Multiplicative and not subtractive: everything at one distance is scaled
    by the same amount, so within a level the original ranking is untouched."""
    store = _Store(customer=[
        _rec("a", 0.90, TEAM),
        _rec("b", 0.60, TEAM),
    ])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["a"].relevance_score > got["b"].relevance_score


# ── SHIP-20 / R-14: the data-loss guard ────────────────────────────────────

async def test_a_record_with_no_node_id_keeps_its_score():
    """Every record written before the metadata backfill has no `node_id`, and
    the only reason it is in this result set is the transitional `$or`'s legacy
    arm, which is exactly the requester's own scope. Scoring it by absence would
    give it distance None, a multiplier of 0, and delete it. That presents as
    data loss, not as a reordering."""
    store = _Store(customer=[_rec("legacy", 0.77, None)])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["legacy"], (
        "a record written before the backfill was dropped entirely"
    )
    assert items[0].relevance_score == pytest.approx(0.77), (
        "an un-stamped record was penalised for distance it does not have"
    )


async def test_un_stamped_and_own_node_records_are_treated_identically():
    """Because they are the same thing: the legacy arm that admitted the first
    is the requester's own customer and user scope."""
    store = _Store(customer=[
        _rec("stamped", 0.70, USER),
        _rec("unstamped", 0.70, None),
    ])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["stamped"].relevance_score == pytest.approx(
        got["unstamped"].relevance_score
    )


# ── SHIP-03 / R-07, R-08: off the chain is dropped, not demoted ─────────────

async def test_a_record_off_the_chain_is_dropped():
    """By R-08 the pool builder made this impossible, so this is the belt to
    that braces. It must not be collapsed into the missing-id case above: one is
    a record that predates the backfill and one is a record belonging to
    somebody else."""
    store = _Store(customer=[
        _rec("mine", 0.50, USER),
        _rec("theirs", 0.99, STRANGER),
    ])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["mine"], (
        "a record whose node is not on the chain was returned"
    )


async def test_off_chain_and_missing_id_are_not_the_same_case():
    store = _Store(customer=[
        _rec("nothing", 0.60, None),
        _rec("stranger", 0.60, STRANGER),
    ])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["nothing"]


# ── SHIP-21 / R-15: client-level shared knowledge ──────────────────────────

async def test_client_level_knowledge_is_not_deleted():
    """Client-level records carry no customer id, no user id and no node id,
    because the collection name is their entire scope. Reading that absence as
    "not on the chain" deletes every piece of shared knowledge a client has."""
    store = _Store(client=[_rec("shared", 0.80, None)])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["shared"]


async def test_client_level_knowledge_ranks_at_the_root_rung():
    """Not at distance 0. Scoring it as the requester's own would let generic
    client-wide material compete level with the specific fact about them, which
    is the dilution this whole mechanism exists to prevent."""
    store = _Store(
        customer=[_rec("mine", 0.80, USER)],
        client=[_rec("shared", 0.80, None)],
    )
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["mine"].relevance_score > got["shared"].relevance_score

    # It is the ROOT rung specifically, so it should score exactly as a record
    # filed at the root node does.
    store2 = _Store(
        customer=[_rec("at_root", 0.80, CLIENT)],
        client=[_rec("shared", 0.80, None)],
    )
    items2 = await _manager(store2)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got2 = _by_id(items2)
    assert got2["shared"].relevance_score == pytest.approx(
        got2["at_root"].relevance_score
    )


async def test_on_a_chain_of_one_shared_knowledge_ranks_level_with_own():
    """A chain of one means the requester IS the root, so there is no distance
    between them and client-level material."""
    store = _Store(
        customer=[_rec("mine", 0.80, CLIENT)],
        client=[_rec("shared", 0.80, None)],
    )
    items = await _manager(store)._direct_similarity_search(
        _scope(depth=1), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["mine"].relevance_score == pytest.approx(got["shared"].relevance_score)


# ── SHIP-02 / R-10: the reserve fetch ──────────────────────────────────────

async def test_a_deep_requester_gets_a_second_narrow_fetch():
    store = _Store(customer=[_rec("x", 0.5, USER)])
    await _manager(store)._direct_similarity_search(
        _scope(depth=5), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert store.customer_calls == 2, (
        f"expected a main fetch and a reserve, got {store.customer_calls} "
        "customer-level calls"
    )


async def test_a_root_requester_gets_no_reserve():
    """The main fetch already reads only their node, so a second identical
    query is pure cost."""
    store = _Store(customer=[_rec("x", 0.5, CLIENT)])
    await _manager(store)._direct_similarity_search(
        _scope(depth=1), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert store.customer_calls == 1


async def test_a_request_with_no_chain_gets_no_reserve():
    store = _Store(customer=[_rec("x", 0.5, None)])
    await _manager(store)._direct_similarity_search(
        _scope(with_chain=False), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert store.customer_calls == 1


async def test_a_leaf_record_the_main_fetch_missed_still_arrives():
    """The failure R-10 exists for. The main fetch comes back full of ancestor
    material and none of the requester's own, not because theirs is irrelevant
    but because there is far more of everything above them."""
    store = _Store(
        customer=[_rec(f"anc{i}", 0.9, CUST) for i in range(5)],
        reserve=[_rec("mine", 0.55, USER)],
    )
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert "mine" in _by_id(items), (
        "the requester's own record was never fetched, and no amount of "
        "re-ranking can recover a record that was never fetched"
    )


async def test_a_record_both_fetches_return_appears_once():
    """The reserve reads the same collection with a narrower predicate, so
    overlap is the normal case, not the exception. Two copies of one record is
    not a second candidate: it spends the token budget twice."""
    store = _Store(
        customer=[_rec("both", 0.8, USER)],
        reserve=[_rec("both", 0.8, USER)],
    )
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["both"]


async def test_the_reserve_is_subject_to_the_same_type_filter():
    """R-17, at the call site rather than in the builder. The reserve narrows
    the scope arm and nothing else."""
    store = _Store(customer=[_rec("x", 0.5, USER)])
    await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ["facts"], top_k=25
    )
    reserve = [c for c in store.calls if c["scope_level"] == "customer"][1]
    assert "memory_type" in str(reserve["filters"]), (
        f"the reserve was sent {reserve['filters']}, which is wider than the "
        "query it is topping up"
    )


# ── SHIP-04: inert for everyone who has not opted in ───────────────────────

async def test_without_a_chain_scores_are_untouched():
    """The feature is off for every client today, so this path has to stay
    byte for byte what it was."""
    store = _Store(customer=[
        _rec("a", 0.91, None, legacy_ids=True),
        _rec("b", 0.42, None, legacy_ids=True),
    ])
    items = await _manager(store)._direct_similarity_search(
        _scope(with_chain=False), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["a"].relevance_score == pytest.approx(0.91)
    assert got["b"].relevance_score == pytest.approx(0.42)


async def test_without_a_chain_a_stranger_record_is_not_dropped():
    """Off-chain dropping is a nested-path rule. On the legacy path the store's
    own filter is exact and there is no chain to be off."""
    store = _Store(customer=[_rec("whoever", 0.7, STRANGER, legacy_ids=True)])
    items = await _manager(store)._direct_similarity_search(
        _scope(with_chain=False), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["whoever"]


async def test_a_malformed_chain_falls_back_rather_than_failing():
    """A rung with no node id is a chain that never resolved. Raising would take
    down a request the legacy path serves perfectly well."""
    s = _scope()
    s.chain_rungs = [{"level_key": "client"}]
    store = _Store(customer=[_rec("x", 0.7, None, legacy_ids=True)])
    items = await _manager(store)._direct_similarity_search(
        s, ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["x"]
    assert store.customer_calls == 1, "a malformed chain should buy no reserve"


# ── SHIP-14: the same wiring at questions and sentences ────────────────────
#
# The nested FILTER went in at three sites. Ranking going in at one is the
# asymmetry that produces a "works for memories only" bug nobody looks for,
# because every other symptom of it is identical to the feature working.


class _QStore(_Store):
    """The questions collection: same two-task shape, different method."""

    def provider(self) -> AsyncMock:
        v = super().provider()

        async def _q(**kw):
            level = kw.get("scope_level")
            self.calls.append({"scope_level": level, "top_k": kw.get("top_k"),
                               "filters": kw.get("filters")})
            n = len(kw.get("embeddings", []))
            if level == "client":
                rows = self.client
            elif self._is_reserve(kw):
                rows = self.reserve if self.reserve is not None else []
            else:
                rows = self.customer
            return [[dict(r) for r in rows] for _ in range(n)]

        v.batch_search_questions = AsyncMock(side_effect=_q)
        return v


def _qrec(rid: str, score: float, node_id: str | None, memory_id: str | None = None):
    meta = {"memory_id": memory_id or f"mem-{rid}", "memory_type": "facts"}
    if node_id is not None:
        meta["node_id"] = node_id
    return {"id": rid, "score": score, "content": f"question {rid}", "metadata": meta}


async def test_questions_apply_closeness_too():
    store = _QStore(customer=[
        _qrec("far", 0.90, CLIENT),
        _qrec("near", 0.80, USER),
    ])
    items = await _manager(store)._search_questions(_scope(), ["q"], _vec(), top_k=10)
    got = {i.item_id: i for i in items}
    assert got["mem-near"].relevance_score > got["mem-far"].relevance_score


async def test_questions_do_not_delete_pre_backfill_records():
    """R-14 at the second site. Same failure, same size."""
    store = _QStore(customer=[_qrec("legacy", 0.77, None)])
    items = await _manager(store)._search_questions(_scope(), ["q"], _vec(), top_k=10)
    assert [i.item_id for i in items] == ["mem-legacy"]
    assert items[0].relevance_score == pytest.approx(0.77)


async def test_questions_do_not_delete_client_level_knowledge():
    """R-15 at the second site."""
    store = _QStore(client=[_qrec("shared", 0.80, None)])
    items = await _manager(store)._search_questions(_scope(), ["q"], _vec(), top_k=10)
    assert [i.item_id for i in items] == ["mem-shared"]


async def test_questions_get_a_reserve_fetch_when_the_chain_is_deep():
    store = _QStore(customer=[_qrec("x", 0.7, USER)])
    await _manager(store)._search_questions(_scope(), ["q"], _vec(), top_k=10)
    assert store.customer_calls == 2


async def test_questions_two_questions_for_one_memory_both_survive():
    """The dedup here keys on the QUESTION id, not the memory id, because two
    distinct questions legitimately point at one memory and today's path
    returns both."""
    store = _QStore(customer=[
        _qrec("q1", 0.80, USER, memory_id="mem-shared-target"),
        _qrec("q2", 0.75, USER, memory_id="mem-shared-target"),
    ])
    items = await _manager(store)._search_questions(_scope(), ["q"], _vec(), top_k=10)
    assert len(items) == 2, "a legitimate second question for the same memory was dropped"


async def test_questions_the_reserve_returning_a_seen_question_is_collapsed():
    store = _QStore(
        customer=[_qrec("dupe", 0.80, USER)],
        reserve=[_qrec("dupe", 0.80, USER)],
    )
    items = await _manager(store)._search_questions(_scope(), ["q"], _vec(), top_k=10)
    assert [i.item_id for i in items] == ["mem-dupe"]


class _SStore(_Store):
    """Sentences read ONE collection, so there is no client-scope task here."""

    def provider(self) -> AsyncMock:
        v = super().provider()

        async def _s(**kw):
            self.calls.append({"scope_level": "customer", "top_k": kw.get("top_k"),
                               "filters": kw.get("filters")})
            n = len(kw.get("embeddings", []))
            rows = (
                (self.reserve if self.reserve is not None else [])
                if self._is_reserve(kw)
                else self.customer
            )
            return [[dict(r) for r in rows] for _ in range(n)]

        v.batch_search_sentences = AsyncMock(side_effect=_s)
        return v


# Genuinely unrelated sentences. The sentences path drops any candidate whose
# jaccard overlap with an already-selected one is above 0.5, so near-identical
# fixture text silently deletes half the fixture and looks like a ranking bug.
_SENTENCE_TEXT = {
    "far": "quarterly revenue rose after the pricing change in March",
    "near": "deployment freezes every Thursday because of the payments release",
    "good": "the office moved to a different building last winter",
    "weak": "coffee machine on floor two is broken again",
    "legacy": "onboarding documents were rewritten by the support team",
    "x": "customers in the northern region prefer email over phone",
    "a": "annual leave carries over only until the end of March",
    "b": "the security review happens before any production deploy",
}


def _srec(rid: str, score: float, node_id: str | None, content: str | None = None,
          legacy_ids: bool = False):
    meta = {}
    if node_id is not None:
        meta["node_id"] = node_id
    if legacy_ids:
        meta["customer_id"] = CUST
        meta["user_id"] = USER
    return {"id": rid, "score": score,
            "content": content or _SENTENCE_TEXT.get(rid, f"a sentence concerning {rid}"),
            "metadata": meta}


async def test_sentences_apply_closeness_to_the_order():
    """This path sorts by score and then breaks out of the sorted list, so the
    thresholds have to keep gating on the raw score while the order follows
    distance. Getting that wrong silently truncates the results."""
    store = _SStore(customer=[
        _srec("far", 0.90, CLIENT),
        _srec("near", 0.80, USER),
    ])
    items = await _manager(store)._search_sentences(_scope(), ["q"], _vec(), top_k=10)
    got = {i.item_id: i for i in items}
    assert set(got) == {"far", "near"}, f"a sentence was lost: {sorted(got)}"
    assert got["near"].relevance_score > got["far"].relevance_score
    assert [i.item_id for i in items][0] == "near", "closeness did not change the order"


async def test_sentences_eligibility_still_uses_the_raw_score():
    """A sentence below the floor stays out, however near it is. The floor is a
    relevance gate on what the store returned, not a statement about depth."""
    store = _SStore(customer=[
        _srec("good", 0.90, CLIENT),
        _srec("weak", 0.30, USER),     # under SENTENCE_SCORE_THRESHOLD
    ])
    items = await _manager(store)._search_sentences(_scope(), ["q"], _vec(), top_k=10)
    assert [i.item_id for i in items] == ["good"]


async def test_sentences_do_not_delete_pre_backfill_records():
    store = _SStore(customer=[_srec("legacy", 0.90, None)])
    items = await _manager(store)._search_sentences(_scope(), ["q"], _vec(), top_k=10)
    assert [i.item_id for i in items] == ["legacy"]
    assert items[0].relevance_score == pytest.approx(0.90)


async def test_sentences_get_a_reserve_fetch_when_the_chain_is_deep():
    store = _SStore(customer=[_srec("x", 0.90, USER)])
    await _manager(store)._search_sentences(_scope(), ["q"], _vec(), top_k=10)
    assert store.customer_calls == 2


async def test_sentences_without_a_chain_keep_the_raw_order():
    store = _SStore(customer=[
        _srec("a", 0.90, None, legacy_ids=True),
        _srec("b", 0.80, None, legacy_ids=True),
    ])
    items = await _manager(store)._search_sentences(
        _scope(with_chain=False), ["q"], _vec(), top_k=10
    )
    assert [i.item_id for i in items] == ["a", "b"]
    assert items[0].relevance_score == pytest.approx(0.90)


# ── the cross-user escape hatch ───────────────────────────────────────────
#
# Found by audit 2026-08-23. With allow_cross_user_retrieval on, the legacy arm
# drops its user clause and the store deliberately returns other users' records.
# Those are off the requester's chain by definition, so SHIP-03's drop rule
# deleted them. But it could only delete the ones carrying a node_id: unstamped
# records have no node_id and fall to R-14's own-node rule, so they survived.
#
# The result set became "whichever of the other users' records happen not to be
# stamped yet", and it changed as the backfill progressed. Two records from the
# same user, same query, different answers. And it fired
# synap.off_chain_dropped, whose comment tells an operator that a non-zero count
# means something upstream is wrong, so they would chase a phantom.


def _widened_manager(store):
    m = RetrievalManager(vector_provider=store.provider(), postgres_provider=AsyncMock())
    maca = MACARetrievalSettings(allow_cross_user_retrieval=True)
    m._current_maca = maca
    return m


async def test_a_widened_request_keeps_another_users_stamped_record():
    """The defect. This record was dropped while its unstamped twin survived."""
    store = _Store(customer=[
        _rec("mine", 0.60, USER),
        _rec("theirs_stamped", 0.90, STRANGER),
    ])
    items = await _widened_manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert "theirs_stamped" in _by_id(items), (
        "allow_cross_user_retrieval was on and the store returned this record, "
        "then closeness deleted it for being off the chain"
    )


async def test_a_widened_request_treats_stamped_and_unstamped_alike():
    """The property that matters more than either answer: the result must not
    depend on how far the metadata backfill has got."""
    store = _Store(customer=[
        _rec("theirs_stamped", 0.80, STRANGER),
        _rec("theirs_unstamped", 0.80, None, legacy_ids=True),
    ])
    items = await _widened_manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert {"theirs_stamped", "theirs_unstamped"} <= set(got), (
        f"only {sorted(got)} survived; the answer changes as records get stamped"
    )


async def test_a_widened_request_still_ranks_the_requesters_own_material_first():
    """Widening is not levelling. Another user's record is legitimate and it is
    still further away than anything on the requester's own chain."""
    store = _Store(customer=[
        _rec("mine", 0.70, USER),
        _rec("theirs", 0.70, STRANGER),
    ])
    items = await _widened_manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    got = _by_id(items)
    assert got["mine"].relevance_score > got["theirs"].relevance_score


async def test_without_widening_an_off_chain_record_is_still_dropped():
    """SHIP-03 must not be softened for everyone by the fix for the widened case."""
    store = _Store(customer=[
        _rec("mine", 0.50, USER),
        _rec("theirs", 0.99, STRANGER),
    ])
    items = await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["mine"]


# ── three smaller read-path findings from the same audit ──────────────────

async def test_a_rung_with_a_null_node_id_falls_back_rather_than_sending_None():
    """`str(None)` is "None", which is truthy, so the malformed-chain guard let
    the literal string through and it was sent to the store as a node id. The
    same defect D-05 records in the SDK cache key, in a function written to be
    defensive about exactly this."""
    s = _scope()
    s.chain_rungs = [
        {"node_id": CLIENT, "level_key": "client", "role": "root"},
        {"node_id": None, "level_key": "customer", "role": "tenant_of_record"},
    ]
    store = _Store(customer=[_rec("x", 0.7, None, legacy_ids=True)])
    items = await _manager(store)._direct_similarity_search(
        s, ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert [i.item_id for i in items] == ["x"], "should have fallen back to the legacy path"

    # Checked on the values, not on the repr: a filter that is Python None
    # stringifies to "None" and would make a repr check pass for the wrong
    # reason, which is the kind of assertion this audit kept finding.
    def _strings(obj):
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from _strings(k)
                yield from _strings(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _strings(v)

    for call in store.calls:
        leaked = [v for v in _strings(call["filters"]) if v == "None"]
        assert not leaked, (
            f"the literal string 'None' was sent as a filter value: {call['filters']}"
        )


async def test_a_b2c_request_buys_no_reserve_fetch():
    """The reserve only helps when it excludes a rung whose records live in the
    same collection. When the leaf IS the tenant, which is every B2C request,
    the only other rung is the root and its material is in the client-level
    collection, read by its own task. Measured at up to +19.9% p95 for no
    candidates at all."""
    s = _scope(depth=3)               # client, region, customer
    s.chain_rungs = [
        {"node_id": CLIENT, "level_key": "client", "role": "root", "roles": ["root"]},
        {"node_id": REGION, "level_key": "region", "role": None, "roles": []},
        {"node_id": CUST, "level_key": "customer", "role": "tenant_of_record",
         "roles": ["tenant_of_record", "identity"]},
    ]
    store = _Store(customer=[_rec("x", 0.7, CUST)])
    await _manager(store)._direct_similarity_search(
        s, ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert store.customer_calls == 1, (
        "a reserve fetch was issued that cannot return anything the main fetch "
        "did not already"
    )


async def test_a_deep_requester_still_gets_the_reserve():
    """The guard above must not switch the reserve off for the case it exists
    for: a leaf strictly below the tenant."""
    store = _Store(customer=[_rec("x", 0.7, USER)])
    await _manager(store)._direct_similarity_search(
        _scope(), ["q"], _vec(), ALL_TYPES, top_k=25
    )
    assert store.customer_calls == 2
