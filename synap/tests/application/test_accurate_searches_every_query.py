"""ACCURATE runs its long-term search on every query the caller sent.

From 2026-02-21 (498babb8) it searched only the first. ``_fetch_accurate`` took
``search_queries[0]`` as the query to write LLM subqueries from and passed that
one string to the long-term search, so a caller's second and third queries
reached the short-term search and nothing else. FAST never had the gap.

These tests call ``_fetch_accurate`` and read what it hands the long-term
search, because that argument is the whole bug.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import synap.cloud.application.client_api.retrieval_manager as RM
from synap.cloud.agents.models.bundles import ContextBundle, ContextItem
from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
from synap.cloud.shared.scope import ScopeHierarchy

_FLAG = "SYNAP_RETRIEVAL_ACCURATE_ALL_QUERIES"


def _scope() -> ScopeHierarchy:
    return ScopeHierarchy(
        instance_id="inst-1", client_id="cli-1", user_id="user-1",
        internal_user_id="iuser-1",
    )


def _decision(queries) -> ContextDecision:
    return ContextDecision(
        decision_id="d-accurate",
        trigger_type=TriggerType.EXPLICIT_REQUEST,
        instance_id="inst-1",
        user_id="user-1",
        conversation_id="conv-1",
        retrieval_mode=RetrievalMode.ACCURATE,
        context_types=["facts"],
        search_queries=list(queries),
        token_budget=2000,
    )


def _item(item_id: str) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        content=f"The user said something specific and useful about {item_id}",
        context_type="facts",
        source="vector",
        similarity_score=0.8,
        confidence=0.8,
    )


@pytest.fixture
def manager():
    """A manager whose stores are replaced by recorders.

    ``_fetch_accurate`` is left real. Everything it calls outward is a mock, so
    a test reads the arguments rather than a store's reaction to them.
    """
    m = RetrievalManager(scope_resolver=MagicMock(), cache_provider=AsyncMock())
    m._get_maca_settings = AsyncMock(return_value=MagicMock())
    m._batch_encode_queries = AsyncMock(
        side_effect=lambda qs: [np.full(4, float(i + 1)) for i, _ in enumerate(qs)]
    )
    m._generate_retrieval_subqueries = AsyncMock(
        return_value={"queries": [], "relationship_types": []}
    )
    m.fetch_short_term_context = AsyncMock(return_value=([], None))
    m.fetch_long_term_context = AsyncMock(return_value=[_item("m1")])
    m._apply_temporal_window_filter = AsyncMock(side_effect=lambda items, *a, **k: items)
    m._apply_jev_rerank = AsyncMock(side_effect=lambda items, *a, **k: items)
    return m


async def _run(manager, queries):
    decision = _decision(queries)
    bundle = ContextBundle(bundle_id="b-1", decision_id=decision.decision_id)
    await manager._fetch_accurate(decision, _scope(), bundle)
    return manager.fetch_long_term_context.call_args_list


class TestTheMainSearchGetsEveryQuery:
    async def test_three_queries_are_all_searched(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        calls = await _run(manager, ["dietary restrictions", "favorite restaurants", "dinner plans"])

        assert len(calls) == 1
        assert calls[0].kwargs["queries"] == [
            "dietary restrictions", "favorite restaurants", "dinner plans",
        ]

    async def test_the_vectors_already_computed_are_the_ones_used(self, manager, monkeypatch):
        """All three were always embedded; two of them were then thrown away."""
        monkeypatch.delenv(_FLAG, raising=False)
        calls = await _run(manager, ["a", "b", "c"])

        assert len(calls[0].kwargs["precomputed_query_vectors"]) == 3
        assert manager._batch_encode_queries.await_count == 1

    async def test_subqueries_are_still_written_from_the_first_query_only(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        await _run(manager, ["first", "second"])

        assert manager._generate_retrieval_subqueries.await_count == 1
        assert manager._generate_retrieval_subqueries.call_args.args[0] == "first"


class TestOneQueryIsUnchanged:
    async def test_the_search_receives_the_same_single_query(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        calls = await _run(manager, ["only question"])

        assert len(calls) == 1
        assert calls[0].kwargs["queries"] == ["only question"]
        assert len(calls[0].kwargs["precomputed_query_vectors"]) == 1

    async def test_no_query_at_all_still_searches_the_empty_string(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        calls = await _run(manager, [])

        assert calls[0].kwargs["queries"] == [""]


class TestTheSwitchRestoresTheOldSearch:
    @pytest.mark.parametrize("off", ["off", "0", "false", "no", "OFF"])
    async def test_off_searches_the_first_query_only(self, manager, monkeypatch, off):
        monkeypatch.setenv(_FLAG, off)
        calls = await _run(manager, ["first", "second", "third"])

        assert calls[0].kwargs["queries"] == ["first"]
        assert calls[0].kwargs["anchor_on_all_queries"] is False

    @pytest.mark.parametrize("on", ["", "on", "1", "true", "anything-else"])
    def test_on_unless_explicitly_switched_off(self, monkeypatch, on):
        monkeypatch.setenv(_FLAG, on)
        assert RM.accurate_searches_all_queries() is True


class TestAGeneratedSubqueryIsNotSearchedTwice:
    async def test_a_subquery_the_caller_already_sent_is_dropped(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        manager._generate_retrieval_subqueries = AsyncMock(return_value={
            "queries": ["first", "second", "something new"],
            "relationship_types": [],
        })
        calls = await _run(manager, ["first", "second"])

        assert len(calls) == 2
        assert calls[1].kwargs["queries"] == ["something new"]

    async def test_nothing_new_means_no_second_search(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        manager._generate_retrieval_subqueries = AsyncMock(return_value={
            "queries": ["first", "second"], "relationship_types": [],
        })
        calls = await _run(manager, ["first", "second"])

        assert len(calls) == 1


class TestTheReRankIsToldEveryQuery:
    """An item found by the second query used to be judged against the first."""

    async def test_window_filter_and_rerank_read_all_of_them(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        manager._generate_retrieval_subqueries = AsyncMock(return_value={
            "queries": ["generated"], "relationship_types": [],
        })
        await _run(manager, ["first", "second"])

        for step in (manager._apply_temporal_window_filter, manager._apply_jev_rerank):
            text = step.call_args.args[1]
            assert text == "first second generated"

    async def test_one_query_reads_as_it_did(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        manager._generate_retrieval_subqueries = AsyncMock(return_value={
            "queries": ["only", "generated"], "relationship_types": [],
        })
        await _run(manager, ["only"])

        assert manager._apply_jev_rerank.call_args.args[1] == "only generated"


class TestGraphAnchorsComeFromEveryQuery:
    async def test_the_caller_query_search_asks_for_all_anchors(self, manager, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        manager._generate_retrieval_subqueries = AsyncMock(return_value={
            "queries": ["generated"], "relationship_types": [],
        })
        calls = await _run(manager, ["first", "second"])

        assert calls[0].kwargs["anchor_on_all_queries"] is True
        # The subquery search is left as it was, so a one-query request is
        # anchored exactly as before.
        assert "anchor_on_all_queries" not in calls[1].kwargs

    async def test_the_lookup_is_one_store_call_and_keeps_queries_apart(self):
        vector = AsyncMock()
        # A real store always returns the record's metadata, and the anchor
        # search now keeps only records the caller may read, so the fixture
        # carries the caller's own tags.
        _mine = {"user_id": "user-1"}
        vector.batch_search = AsyncMock(return_value=[
            [{"content": "Asha booked the Lisbon trip", "metadata": _mine}],
            [{"content": "Ravi prefers the corner table", "metadata": _mine},
             {"content": "", "metadata": _mine}],
        ])
        m = RetrievalManager(scope_resolver=MagicMock(), vector_provider=vector)

        texts = await m._quick_memory_search(
            _scope(), [np.ones(4), np.zeros(4)], top_k=15,
        )

        assert vector.batch_search.await_count == 1
        assert len(vector.batch_search.call_args.kwargs["embeddings"]) == 2
        assert texts == [["Asha booked the Lisbon trip"], ["Ravi prefers the corner table"]]

    def test_each_query_gets_an_anchor_before_any_gets_a_second(self):
        m = RetrievalManager(scope_resolver=MagicMock())
        entities = ["Asha", "Lisbon", "Porto", "Ravi"]
        texts = [
            ["Asha went to Lisbon. Asha liked Lisbon. Asha also saw Porto."],
            ["Ravi called."],
        ]

        anchors = m._anchors_across_queries(texts, entities, max_anchors=2)

        # Pooled, "Ravi" (one mention) loses both places to Asha and Lisbon.
        assert anchors == ["Asha", "Ravi"]

    def test_one_query_is_ranked_exactly_as_before(self):
        m = RetrievalManager(scope_resolver=MagicMock())
        entities = ["Asha", "Lisbon", "Porto"]
        texts = ["Asha went to Lisbon. Asha liked Lisbon. Asha also saw Porto."]

        assert m._anchors_across_queries([texts], entities) == \
            m._match_entities_in_memories(texts, entities)

    def test_an_entity_two_queries_share_is_one_anchor(self):
        m = RetrievalManager(scope_resolver=MagicMock())
        anchors = m._anchors_across_queries(
            [["Asha and Lisbon"], ["Asha and Porto"]], ["Asha", "Lisbon", "Porto"],
        )

        assert anchors.count("Asha") == 1
        assert set(anchors) == {"Asha", "Lisbon", "Porto"}

    def test_no_memories_means_no_anchors(self):
        m = RetrievalManager(scope_resolver=MagicMock())
        assert m._anchors_across_queries([], ["Asha"]) == []
        assert m._anchors_across_queries([[], []], ["Asha"]) == []
