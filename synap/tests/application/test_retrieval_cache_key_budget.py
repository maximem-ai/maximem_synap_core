"""The L1 cache key and `max_results`.

Found while running the benchmark gates for PII Protection, and it briefly looked
like a hard server-side cap of ten items. It was the cache.

`max_results` is the assembly budget: `budget = decision.max_results` in
`_assemble_bundle` decides how many items the bundle carries. It was absent from
the cache key, so two fetches differing only in `max_results` shared an entry for
the 300 second TTL. Confirmed in both directions on a fresh query:
asking for 20 after asking for 5 returned 5, and asking for 5 after asking for 20
returned 20.

Nothing raises. The caller receives a bundle that looks complete and is the wrong
size. That is the same shape as the SDK cache defect where the key omitted
`instance_id` and two instances of one account shared a local database, and it is
the reason a cache key deserves a test per dimension rather than a review.
"""

import pytest

from synap.cloud.agents.models.decisions import (
    ContextDecision,
    RetrievalMode,
    TriggerType,
)
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager


def _decision(max_results=10, trigger_payload=None, **overrides):
    kwargs = dict(
        decision_id="test-decision",
        trigger_type=TriggerType.EXPLICIT_REQUEST,
        client_id="cli_test",
        instance_id="inst_test",
        user_id="user-1",
        retrieval_mode=RetrievalMode.FAST,
        trigger_payload=trigger_payload,
        search_queries=["what changed"],
        context_types=["facts"],
        max_results=max_results,
    )
    kwargs.update(overrides)
    return ContextDecision(**kwargs)


@pytest.fixture
def manager():
    return RetrievalManager.__new__(RetrievalManager)


class TestTheBudgetIsPartOfTheKey:
    def test_two_budgets_do_not_share_an_entry(self, manager):
        """The bug, stated as the one line that would have caught it."""
        assert manager._build_cache_key(_decision(5)) != manager._build_cache_key(
            _decision(20)
        )

    def test_every_budget_in_the_plausible_range_is_distinct(self, manager):
        """A single pair passing could be luck in a hash. The property is that
        the budget is carried, not that two particular values differ."""
        keys = {n: manager._build_cache_key(_decision(n)) for n in (1, 5, 10, 12, 20, 50, 100)}
        assert len(set(keys.values())) == len(keys)

    def test_the_same_budget_is_stable(self, manager):
        """Otherwise nothing is ever cached and the fix is a performance bug."""
        d = _decision(20)
        assert manager._build_cache_key(d) == manager._build_cache_key(d)

    def test_a_missing_budget_does_not_raise(self, manager):
        """`max_results` is optional on the decision and the key builder runs on
        every fetch. A crash here takes out retrieval entirely."""
        assert manager._build_cache_key(_decision(None))
        assert manager._build_cache_key(_decision(0))

    def test_none_and_zero_are_the_same_entry(self, manager):
        """Both mean "no budget stated". Splitting them would double the cache
        for no behavioural difference."""
        assert manager._build_cache_key(_decision(None)) == manager._build_cache_key(
            _decision(0)
        )


class TestItDoesNotDisturbTheOtherDimensions:
    def test_the_skip_flags_still_separate(self, manager):
        """The budget suffix is appended to the same component the skip flags
        use, so the risk is that one swallows the other."""
        plain = manager._build_cache_key(_decision(10))
        skipped = manager._build_cache_key(_decision(10, {"skip_llm_filter": True}))
        assert plain != skipped

    def test_a_flag_and_a_budget_are_independent(self, manager):
        """Four distinct combinations, or one dimension is masking the other."""
        keys = {
            manager._build_cache_key(_decision(10)),
            manager._build_cache_key(_decision(20)),
            manager._build_cache_key(_decision(10, {"skip_llm_filter": True})),
            manager._build_cache_key(_decision(20, {"skip_llm_filter": True})),
        }
        assert len(keys) == 4

    def test_precision_level_still_separates_at_one_budget(self, manager):
        high = manager._build_cache_key(_decision(10, {"precision_level": "high"}))
        medium = manager._build_cache_key(_decision(10, {"precision_level": "medium"}))
        assert high != medium

    def test_the_query_still_separates(self, manager):
        a = manager._build_cache_key(_decision(10, search_queries=["one"]))
        b = manager._build_cache_key(_decision(10, search_queries=["two"]))
        assert a != b

    def test_a_query_cannot_be_confused_with_a_budget(self, manager):
        """The suffix is joined into the query component, so a query that
        happens to look like the suffix must not collide with a real budget."""
        crafted = manager._build_cache_key(_decision(10, search_queries=["what changed|n20"]))
        real = manager._build_cache_key(_decision(20, search_queries=["what changed"]))
        assert crafted != real


class TestTheDeployCost:
    def test_the_default_budget_is_in_the_key_too(self, manager):
        """Unlike the skip flags, which were suffixed only when set so existing
        keys stayed byte-identical, the budget is included unconditionally.

        Entries already in the cache were written without it, and any of them may
        be the wrong size for the next caller, so none of them may be served. One
        cold TTL after deploy is the cost, and it was a choice.

        The way to assert that without pinning the hash is to show the default
        budget is load-bearing: if 10 were treated as "no suffix", it would share
        a key with a decision carrying no budget at all.
        """
        default_budget = manager._build_cache_key(_decision(10))
        no_budget = manager._build_cache_key(_decision(None))
        assert default_budget != no_budget

    def test_adjacent_budgets_do_not_collide(self, manager):
        assert manager._build_cache_key(_decision(10)) != manager._build_cache_key(
            _decision(11)
        )
