"""``max_results_per_query``: every search query gets a share of the response.

One cut by overall rank lets a query with many strong matches take every
place. The share gives each query a turn. ``max_results`` stays the ceiling on
the whole response whatever the share says.
"""

from unittest.mock import MagicMock

import pytest

from synap.cloud.agents.models.bundles import ContextBundle, ContextItem
from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType
from synap.cloud.application.client_api.retrieval_manager import (
    RetrievalManager,
    _reserve_per_query,
)
from synap.cloud.shared.scope import ScopeHierarchy


def _item(item_id, queries, score=0.9):
    return ContextItem(
        item_id=item_id,
        content=f"The user said a specific and distinct thing numbered {item_id} about their life",
        context_type="facts", source="vector",
        similarity_score=score, relevance_score=score, confidence=0.8,
        matched_queries=list(queries),
    )


def _ids(items):
    return [i.item_id for i in items]


def _count(items, query):
    return sum(1 for i in items if query in i.matched_queries)


class TestEveryQueryGetsATurn:
    def test_a_lone_weak_answer_is_no_longer_crowded_out(self):
        """Eight strong restaurant memories, one weaker diet memory, five places."""
        ranked = [_item(f"r{i}", ["restaurants"]) for i in range(8)] + [_item("d0", ["diet"])]

        assert "d0" not in _ids(ranked[:5])            # the single cut by rank
        served = _reserve_per_query(ranked, 5, 3, ["restaurants", "diet"])

        assert "d0" in _ids(served)
        assert _count(served, "restaurants") == 3

    def test_no_query_gets_more_than_its_share_of_its_own_items(self):
        ranked = [_item(f"a{i}", ["a"]) for i in range(10)]
        served = _reserve_per_query(ranked, 10, 4, ["a", "b"])

        assert len(served) == 4

    def test_the_turns_go_in_the_order_the_caller_sent(self):
        ranked = [_item("b0", ["b"]), _item("a0", ["a"])]
        served = _reserve_per_query(ranked, 1, 5, ["a", "b"])

        assert _ids(served) == ["a0"]


class TestMaxResultsIsTheCeiling:
    def test_shares_adding_up_to_more_than_it_stop_early(self):
        """Three queries at five each under ten places end 4, 3 and 3."""
        ranked = []
        for i in range(6):
            ranked += [_item(f"a{i}", ["a"]), _item(f"b{i}", ["b"]), _item(f"c{i}", ["c"])]
        served = _reserve_per_query(ranked, 10, 5, ["a", "b", "c"])

        assert len(served) == 10
        assert [_count(served, q) for q in ("a", "b", "c")] == [4, 3, 3]

    @pytest.mark.parametrize("budget,per", [(1, 50), (3, 50), (7, 2), (10, 10)])
    def test_it_is_never_exceeded(self, budget, per):
        ranked = [_item(f"{q}{i}", [q]) for i in range(20) for q in ("a", "b", "c")]
        assert len(_reserve_per_query(ranked, budget, per, ["a", "b", "c"])) <= budget

    def test_no_room_means_nothing(self):
        assert _reserve_per_query([_item("a0", ["a"])], 0, 5, ["a"]) == []


class TestAnItemSeveralQueriesFound:
    def test_it_takes_one_place_and_counts_against_each(self):
        ranked = [_item("both", ["a", "b"]), _item("a1", ["a"]), _item("b1", ["b"])]
        served = _reserve_per_query(ranked, 10, 1, ["a", "b"])

        # "both" fills a's share and b's share at once. Neither takes another.
        assert _ids(served) == ["both"]

    def test_a_full_query_does_not_stop_another_from_taking_a_shared_item(self):
        ranked = [_item("a0", ["a"]), _item("both", ["a", "b"])]
        served = _reserve_per_query(ranked, 10, 1, ["a", "b"])

        assert _ids(served) == ["a0", "both"]


class TestAnItemNoQueryAccountsFor:
    def test_it_fills_room_left_after_the_turns(self):
        ranked = [_item("a0", ["a"]), _item("x0", []), _item("b0", ["b"])]
        served = _reserve_per_query(ranked, 3, 1, ["a", "b"])

        assert _ids(served) == ["a0", "x0", "b0"]

    def test_it_does_not_take_a_place_a_query_wanted(self):
        ranked = [_item("x0", []), _item("x1", []), _item("a0", ["a"]), _item("b0", ["b"])]
        served = _reserve_per_query(ranked, 2, 1, ["a", "b"])

        assert _ids(served) == ["a0", "b0"]


def test_the_result_keeps_the_order_the_items_came_in():
    ranked = [_item("b0", ["b"]), _item("a0", ["a"]), _item("b1", ["b"]), _item("a1", ["a"])]
    served = _reserve_per_query(ranked, 4, 2, ["a", "b"])

    assert _ids(served) == ["b0", "a0", "b1", "a1"]


# --- through bundle assembly -------------------------------------------------

def _decision(queries, max_results=5, per_query=None) -> ContextDecision:
    return ContextDecision(
        decision_id="d-1", trigger_type=TriggerType.EXPLICIT_REQUEST,
        instance_id="inst-1", user_id="user-1", conversation_id="conv-1",
        retrieval_mode=RetrievalMode.FAST, context_types=["facts"],
        search_queries=list(queries), token_budget=20000,
        max_results=max_results, max_results_per_query=per_query,
    )


def _assemble(decision, items):
    m = RetrievalManager(scope_resolver=MagicMock())
    bundle = ContextBundle(bundle_id="b-1", decision_id=decision.decision_id)
    out = m.assemble_context_bundle(items, decision, bundle)
    return [i for group in out.items_by_type.values() for i in group]


def _pool():
    """Eight strong memories for one query, one weaker memory for the other."""
    strong = [_item(f"r{i}", ["restaurants"], score=0.95 - i * 0.01) for i in range(8)]
    return strong + [_item("d0", ["diet"], score=0.55)]


class TestThroughAssembly:
    def test_the_share_brings_the_weaker_querys_answer_back(self):
        served = _assemble(_decision(["restaurants", "diet"], per_query=3), _pool())

        assert "d0" in _ids(served)
        assert len(served) == 4

    def test_without_it_the_response_is_the_best_overall_as_before(self):
        served = _assemble(_decision(["restaurants", "diet"]), _pool())

        assert sorted(_ids(served)) == ["r0", "r1", "r2", "r3", "r4"]

    def test_one_query_ignores_it(self):
        pool = [_item(f"r{i}", ["restaurants"], score=0.95 - i * 0.01) for i in range(8)]
        with_share = _assemble(_decision(["restaurants"], per_query=2), list(pool))
        pool = [_item(f"r{i}", ["restaurants"], score=0.95 - i * 0.01) for i in range(8)]
        without = _assemble(_decision(["restaurants"]), list(pool))

        assert len(with_share) == 5
        assert sorted(_ids(with_share)) == sorted(_ids(without))

    def test_a_share_above_max_results_cannot_grow_the_response(self):
        served = _assemble(
            _decision(["restaurants", "diet"], max_results=5, per_query=50), _pool(),
        )
        assert len(served) == 5


class TestTheCacheKey:
    def _key(self, per_query):
        m = RetrievalManager(scope_resolver=MagicMock())
        scope = ScopeHierarchy(
            instance_id="inst-1", client_id="cli-1", user_id="user-1",
            internal_user_id="iuser-1",
        )
        return m._build_cache_key(_decision(["a", "b"], per_query=per_query), scope)

    def test_a_fetch_that_names_a_share_does_not_reuse_one_that_does_not(self):
        assert self._key(3) != self._key(None)

    def test_two_different_shares_are_two_entries(self):
        assert self._key(3) != self._key(4)

    def test_the_same_share_is_the_same_entry(self):
        assert self._key(3) == self._key(3)


class TestTheDecisionRoundTrip:
    def test_the_share_survives_to_dict_and_back(self):
        decision = _decision(["a", "b"], per_query=3)
        assert ContextDecision.from_dict(decision.to_dict()).max_results_per_query == 3

    def test_a_decision_stored_before_the_field_existed_has_none(self):
        data = _decision(["a"]).to_dict()
        del data["max_results_per_query"]
        assert ContextDecision.from_dict(data).max_results_per_query is None
