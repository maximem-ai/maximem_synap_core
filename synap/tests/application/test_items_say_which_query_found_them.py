"""Every returned item names the caller's search queries that found it.

A fetch can carry several queries and returns one set for all of them. Until
this, nothing on an item said which question it answered, and the server lost
the link itself as soon as the search arms merged.

The tag is the caller's own query. A result of a subquery or template the
server wrote carries the caller query it was written from.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from synap.cloud.agents.models.bundles import ContextBundle, ContextItem
from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
from synap.cloud.shared.scope import ScopeHierarchy


@pytest.fixture(autouse=True)
def _stub_query_decomposition():
    """spaCy does not import here; see test_retrieval_manager.py."""
    mod_name = "synap.cloud.application.client_api.query_decomposition"
    saved = sys.modules.get(mod_name)
    stub = types.ModuleType(mod_name)
    stub.decompose_queries = lambda queries: [
        part for q in queries for part in q.split(" or ")
    ]
    sys.modules[mod_name] = stub
    try:
        yield
    finally:
        if saved is not None:
            sys.modules[mod_name] = saved
        else:
            sys.modules.pop(mod_name, None)


def _scope() -> ScopeHierarchy:
    return ScopeHierarchy(
        instance_id="inst-1", client_id="cli-1", user_id="user-1",
        internal_user_id="iuser-1",
    )


def _hit(memory_id: str, text: str) -> dict:
    return {
        "id": memory_id,
        "content": text,
        "score": 0.9,
        "metadata": {
            "memory_type": "facts", "confidence": 0.8, "scope": "user",
            "user_id": "user-1",
        },
    }


class _Stores:
    """An embedder and a vector store that agree on which text is which.

    Each distinct query string gets its own vector, and the store answers a
    vector with whatever ``answers`` holds for that string. So a test states
    "this query finds this memory" and reads the tags that come back.
    """

    def __init__(self, answers: dict):
        self.answers = answers
        self._texts: list = []
        self.searched: list = []

        self.embedding = AsyncMock()
        self.embedding.embed_batch = AsyncMock(side_effect=self._embed)
        self.vector = AsyncMock()
        self.vector.batch_search = AsyncMock(side_effect=self._search)
        self.vector.batch_search_questions = AsyncMock(
            side_effect=lambda embeddings, **k: [[] for _ in embeddings]
        )

    async def _embed(self, queries, is_query=False, **kwargs):
        out = []
        for q in queries:
            if q not in self._texts:
                self._texts.append(q)
            # One-hot, so the vector still names its text after normalisation.
            vec = [0.0] * 16
            vec[self._texts.index(q)] = 1.0
            out.append(vec)
        return out

    async def _search(self, embeddings, scope=None, top_k=None, filters=None, scope_level=None, **kwargs):
        if scope_level == "client":
            return [[] for _ in embeddings]
        results = []
        for emb in embeddings:
            text = self._texts[int(np.argmax(emb))]
            self.searched.append(text)
            results.append([dict(h) for h in self.answers.get(text, [])])
        return results


def _manager(stores: _Stores) -> RetrievalManager:
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=_scope())
    return RetrievalManager(
        scope_resolver=resolver,
        vector_provider=stores.vector,
        embedding_client=stores.embedding,
    )


def _tags(items) -> dict:
    return {i.item_id: list(i.matched_queries) for i in items}


class TestTheVectorSearch:
    async def test_each_item_names_the_query_that_found_it(self):
        stores = _Stores({
            "diet": [_hit("m-diet", "The user is vegetarian")],
            "restaurants": [_hit("m-rest", "The user likes the corner bistro")],
        })
        items = await _manager(stores).fetch_long_term_context(
            _scope(), queries=["diet", "restaurants"], filters={"types": ["facts"]},
            mode="fast", skip_subquery_gen=True,
        )

        assert _tags(items) == {"m-diet": ["diet"], "m-rest": ["restaurants"]}

    async def test_a_memory_two_queries_find_names_both(self):
        shared = _hit("m-both", "The user eats vegetarian at the corner bistro")
        stores = _Stores({"diet": [shared], "restaurants": [shared]})
        items = await _manager(stores).fetch_long_term_context(
            _scope(), queries=["diet", "restaurants"], filters={"types": ["facts"]},
            mode="fast", skip_subquery_gen=True,
        )

        assert len(items) == 1
        assert sorted(items[0].matched_queries) == ["diet", "restaurants"]

    async def test_a_template_result_names_the_query_it_was_written_from(self):
        """FAST writes fixed templates around the first query and searches them too."""
        template = "What are the user's preferences related to diet?"
        stores = _Stores({
            template: [_hit("m-pref", "The user prefers oat milk")],
            "restaurants": [_hit("m-rest", "The user likes the corner bistro")],
        })
        items = await _manager(stores).fetch_long_term_context(
            _scope(), queries=["diet", "restaurants"], filters={"types": ["facts"]},
            mode="fast",
        )

        assert template in stores.searched
        assert _tags(items) == {"m-pref": ["diet"], "m-rest": ["restaurants"]}

    async def test_owners_passed_in_are_the_ones_used(self):
        """The ACCURATE subquery search: every string belongs to the first query."""
        stores = _Stores({
            "sub one": [_hit("m-1", "The user moved to Lisbon")],
            "sub two": [_hit("m-2", "The user works remotely")],
        })
        items = await _manager(stores).fetch_long_term_context(
            _scope(), queries=["sub one", "sub two"], filters={"types": ["facts"]},
            mode="fast", skip_subquery_gen=True,
            query_owners=["where do I live", "where do I live"],
        )

        assert _tags(items) == {"m-1": ["where do I live"], "m-2": ["where do I live"]}

    async def test_owners_of_the_wrong_length_are_ignored(self):
        stores = _Stores({"diet": [_hit("m-diet", "The user is vegetarian")]})
        items = await _manager(stores).fetch_long_term_context(
            _scope(), queries=["diet"], filters={"types": ["facts"]},
            mode="fast", skip_subquery_gen=True, query_owners=["a", "b"],
        )

        assert _tags(items) == {"m-diet": ["diet"]}


class TestTheCollapseOfTwoCopiesKeepsBothQueries:
    def _copy(self, score, queries, source="vector"):
        return ContextItem(
            item_id="m-1", content="The user is vegetarian", context_type="facts",
            source=source, similarity_score=score, matched_queries=list(queries),
        )

    def test_the_higher_scoring_copy_wins_and_inherits(self):
        m = RetrievalManager(scope_resolver=MagicMock())
        out = m._deduplicate_items([
            self._copy(0.5, ["diet"]), self._copy(0.9, ["restaurants"], source="discovery"),
        ])

        assert len(out) == 1
        assert out[0].source == "discovery"
        assert sorted(out[0].matched_queries) == ["diet", "restaurants"]

    def test_the_first_copy_wins_a_tie_and_inherits(self):
        m = RetrievalManager(scope_resolver=MagicMock())
        out = m._deduplicate_items([
            self._copy(0.9, ["diet"]), self._copy(0.9, ["restaurants"]),
        ])

        assert out[0].matched_queries == ["diet", "restaurants"]

    def test_graph_and_vector_ids_for_one_memory_still_collapse(self):
        m = RetrievalManager(scope_resolver=MagicMock())
        graph_copy = self._copy(0.4, ["diet"], source="graph")
        graph_copy.item_id = "memory:m-1"
        out = m._deduplicate_items([graph_copy, self._copy(0.9, ["restaurants"])])

        assert len(out) == 1
        assert sorted(out[0].matched_queries) == ["diet", "restaurants"]


class TestTheCacheRoundTrip:
    def test_the_tag_survives_to_dict_and_back(self):
        item = ContextItem(
            item_id="m-1", content="c", context_type="facts", source="vector",
            matched_queries=["diet", "restaurants"],
        )
        assert ContextItem.from_dict(item.to_dict()).matched_queries == ["diet", "restaurants"]

    def test_an_entry_cached_before_the_field_existed_reads_as_untagged(self):
        legacy = ContextItem(
            item_id="m-1", content="c", context_type="facts", source="vector",
        ).to_dict()
        del legacy["matched_queries"]

        assert ContextItem.from_dict(legacy).matched_queries == []

    def test_two_items_do_not_share_one_list(self):
        a = ContextItem(item_id="a", content="c", context_type="facts", source="vector")
        b = ContextItem(item_id="b", content="c", context_type="facts", source="vector")
        a.matched_queries.append("diet")

        assert b.matched_queries == []


def _decision(queries, mode=RetrievalMode.FAST) -> ContextDecision:
    return ContextDecision(
        decision_id="d-1", trigger_type=TriggerType.EXPLICIT_REQUEST,
        instance_id="inst-1", user_id="user-1", conversation_id="conv-1",
        retrieval_mode=mode, context_types=["facts"],
        search_queries=list(queries), token_budget=2000,
    )


def _untagged(item_id="m-graph") -> ContextItem:
    return ContextItem(
        item_id=item_id,
        content="The user mentioned a long and specific thing about the Lisbon office move",
        context_type="facts", source="graph", similarity_score=0.8, confidence=0.8,
    )


class TestAnItemNoArmAttributed:
    def _assemble(self, queries, items):
        m = RetrievalManager(scope_resolver=MagicMock())
        decision = _decision(queries)
        bundle = ContextBundle(bundle_id="b-1", decision_id=decision.decision_id)
        out = m.assemble_context_bundle(items, decision, bundle)
        return [i for group in out.items_by_type.values() for i in group]

    def test_one_query_owns_everything_the_fetch_returns(self):
        served = self._assemble(["diet"], [_untagged()])
        assert served[0].matched_queries == ["diet"]

    def test_the_same_query_sent_twice_is_still_one_query(self):
        served = self._assemble(["diet", "diet"], [_untagged()])
        assert served[0].matched_queries == ["diet"]

    def test_with_several_queries_it_stays_untagged(self):
        served = self._assemble(["diet", "restaurants"], [_untagged()])
        assert served[0].matched_queries == []

    def test_with_no_query_it_stays_untagged(self):
        served = self._assemble([], [_untagged()])
        assert served[0].matched_queries == []

    def test_a_tag_an_arm_set_is_not_overwritten(self):
        item = _untagged()
        item.matched_queries = ["restaurants"]
        served = self._assemble(["diet", "restaurants"], [item])
        assert served[0].matched_queries == ["restaurants"]


class TestTheModesHandOwnershipDown:
    def _wired(self):
        m = RetrievalManager(scope_resolver=MagicMock(), cache_provider=AsyncMock())
        m._get_maca_settings = AsyncMock(return_value=MagicMock())
        m._batch_encode_queries = AsyncMock(
            side_effect=lambda qs: [np.full(4, float(i + 1)) for i, _ in enumerate(qs)]
        )
        m.fetch_short_term_context = AsyncMock(return_value=([], None))
        m.fetch_long_term_context = AsyncMock(return_value=[])
        m._apply_temporal_window_filter = AsyncMock(side_effect=lambda items, *a, **k: items)
        m._apply_jev_rerank = AsyncMock(side_effect=lambda items, *a, **k: items)
        return m

    async def test_accurate_subqueries_belong_to_the_first_query(self, monkeypatch):
        monkeypatch.delenv("SYNAP_RETRIEVAL_ACCURATE_ALL_QUERIES", raising=False)
        m = self._wired()
        m._generate_retrieval_subqueries = AsyncMock(return_value={
            "queries": ["sub one", "sub two"], "relationship_types": [],
        })
        decision = _decision(["first", "second"], RetrievalMode.ACCURATE)
        await m._fetch_accurate(
            decision, _scope(), ContextBundle(bundle_id="b", decision_id="d-1"),
        )

        main_call, subquery_call = m.fetch_long_term_context.call_args_list
        assert "query_owners" not in main_call.kwargs      # each owns itself
        assert subquery_call.kwargs["query_owners"] == ["first", "first"]
        assert m.fetch_short_term_context.call_args.kwargs["query_owners"] == ["first", "second"]

    async def test_fast_gives_each_split_piece_to_the_query_it_came_from(self):
        m = self._wired()
        decision = _decision(["pick up or return", "refund policy"])
        await m._fetch_fast(
            decision, _scope(), ContextBundle(bundle_id="b", decision_id="d-1"),
        )

        kwargs = m.fetch_long_term_context.call_args.kwargs
        assert kwargs["queries"] == ["pick up", "return", "refund policy"]
        assert kwargs["query_owners"] == [
            "pick up or return", "pick up or return", "refund policy",
        ]

    async def test_fast_with_nothing_split_lets_each_query_own_itself(self):
        m = self._wired()
        decision = _decision(["diet", "restaurants"])
        await m._fetch_fast(
            decision, _scope(), ContextBundle(bundle_id="b", decision_id="d-1"),
        )

        kwargs = m.fetch_long_term_context.call_args.kwargs
        assert kwargs["query_owners"] == ["diet", "restaurants"]
