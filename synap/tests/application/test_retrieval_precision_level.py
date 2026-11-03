"""Tests for the precision_level retrieval knob (and the skip_llm_filter guard it shares).

precision_level="medium" is the public, validated form of the internal
skip_llm_filter opt-out (PRs #850/#851). These tests cover the shared
guard helper and the L1 cache-key salting that keeps medium/high fetches
from ever sharing a cache entry.
"""

import pytest

from synap.cloud.agents.models.decisions import (
    ContextDecision,
    RetrievalMode,
    TriggerType,
)
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager


def _decision(trigger_payload=None, **overrides):
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
    )
    kwargs.update(overrides)
    return ContextDecision(**kwargs)


# ---------------------------------------------------------------------------
# Guard helper: _llm_filter_opted_out
# ---------------------------------------------------------------------------

class TestLlmFilterOptedOut:
    def test_none_payload_does_not_opt_out(self):
        assert RetrievalManager._llm_filter_opted_out(_decision(None)) is False

    def test_empty_payload_does_not_opt_out(self):
        assert RetrievalManager._llm_filter_opted_out(_decision({})) is False

    def test_precision_high_does_not_opt_out(self):
        d = _decision({"precision_level": "high"})
        assert RetrievalManager._llm_filter_opted_out(d) is False

    def test_precision_medium_opts_out(self):
        d = _decision({"precision_level": "medium"})
        assert RetrievalManager._llm_filter_opted_out(d) is True

    def test_skip_llm_filter_opts_out(self):
        d = _decision({"skip_llm_filter": True})
        assert RetrievalManager._llm_filter_opted_out(d) is True

    def test_skip_llm_filter_false_does_not_opt_out(self):
        d = _decision({"skip_llm_filter": False, "precision_level": "high"})
        assert RetrievalManager._llm_filter_opted_out(d) is False

    def test_both_vehicles_opt_out(self):
        d = _decision({"skip_llm_filter": True, "precision_level": "medium"})
        assert RetrievalManager._llm_filter_opted_out(d) is True


# ---------------------------------------------------------------------------
# L1 cache-key salting
# ---------------------------------------------------------------------------

class TestCacheKeySalting:
    @pytest.fixture()
    def manager(self):
        return RetrievalManager()

    def test_high_key_is_byte_identical_to_legacy_key(self, manager):
        """Requests that pass precision_level="high" (the default the API
        always sends now) must map to the same cache entry as pre-feature
        requests whose payload had no precision_level at all — otherwise the
        deploy would cold-start the whole L1 cache."""
        legacy = manager._build_cache_key(_decision({"skip_subquery_gen": False}))
        high = manager._build_cache_key(
            _decision({"skip_subquery_gen": False, "precision_level": "high"})
        )
        assert high == legacy

    def test_medium_key_differs_from_high_key(self, manager):
        high = manager._build_cache_key(_decision({"precision_level": "high"}))
        medium = manager._build_cache_key(_decision({"precision_level": "medium"}))
        assert medium != high

    def test_medium_and_skip_flag_keys_are_distinct(self, manager):
        """medium and skip_llm_filter produce the same retrieval behavior but
        arrive via different vehicles; their salts are independent so a
        combined payload still differs from either alone."""
        medium = manager._build_cache_key(_decision({"precision_level": "medium"}))
        skip = manager._build_cache_key(_decision({"skip_llm_filter": True}))
        both = manager._build_cache_key(
            _decision({"skip_llm_filter": True, "precision_level": "medium"})
        )
        assert len({medium, skip, both}) == 3

    def test_medium_key_is_stable(self, manager):
        d = _decision({"precision_level": "medium"})
        assert manager._build_cache_key(d) == manager._build_cache_key(d)
