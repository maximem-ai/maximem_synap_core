"""Retrieval on the Core path: no LLM call, no HQE, no stored scope path.

Everything outward is a fake. Not run here: a real database, a real vector
store, a real model.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import synap.cloud.application.client_api.retrieval_manager as RM
from synap.cloud.agents.models.bundles import ContextBundle, ContextItem
from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
from synap.cloud.shared import edition
from synap.cloud.shared.scope import ScopeHierarchy

COMPOUND = "Do I prefer tea or coffee in the morning?"


class _CountingLLM:
    """An LLM client that records every call. The Core path must never reach it."""

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    async def complete(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return {"content": '{"subqueries": ["from the model"]}'}


def _scope():
    return ScopeHierarchy(
        instance_id="inst-1", client_id="cli-1", user_id="user-1", internal_user_id="iuser-1"
    )


def _manager(llm=None):
    m = RetrievalManager(scope_resolver=MagicMock(), cache_provider=AsyncMock(), llm_client=llm)
    m._get_maca_settings = AsyncMock(return_value=RM.MACARetrievalSettings())
    return m


class TestAccurateModeWithoutAnLLM:
    async def test_the_splitter_and_templates_write_the_subqueries(self):
        m = _manager(llm=None)
        out = await m._generate_retrieval_subqueries(COMPOUND, "accurate")
        assert out["queries"][0] == COMPOUND
        assert "Do I prefer tea in the morning ?" in out["queries"]
        assert "Do I prefer coffee in the morning ?" in out["queries"]
        assert any(q.startswith("What are the user's preferences") for q in out["queries"])

    async def test_relationship_types_are_gone(self):
        out = await _manager()._generate_retrieval_subqueries(COMPOUND, "accurate")
        assert set(out) == {"queries"}

    async def test_a_failed_llm_call_keeps_the_templates_only_as_cloud_always_did(self):
        llm = _CountingLLM(fail=True)
        out = await _manager(llm)._generate_retrieval_subqueries(COMPOUND, "accurate")
        assert llm.calls >= 1
        assert "Do I prefer tea in the morning ?" not in out["queries"]
        assert any(q.startswith("What are the user's preferences") for q in out["queries"])

    async def test_a_question_that_does_not_split_gets_only_the_templates(self):
        out = await _manager()._generate_retrieval_subqueries("Where do I live?", "accurate")
        assert out["queries"][0] == "Where do I live?"
        assert all(q.startswith(("What are", "What relevant", "What past")) or q == "Where do I live?"
                   for q in out["queries"])

    async def test_fast_mode_is_not_split_here(self):
        # FAST splits in _fetch_fast, on the caller's queries; this method stays templates only.
        out = await _manager()._generate_retrieval_subqueries(COMPOUND, "fast")
        assert "Do I prefer tea in the morning ?" not in out["queries"]


class TestAFetchMakesNoLLMCall:
    async def test_accurate_fetch_with_no_llm_client_reaches_the_search_with_split_parts(self):
        m = _manager(llm=None)
        m._batch_encode_queries = AsyncMock(
            side_effect=lambda qs: [np.full(4, float(i + 1)) for i, _ in enumerate(qs)]
        )
        m.fetch_short_term_context = AsyncMock(return_value=([], None))
        m.fetch_long_term_context = AsyncMock(return_value=[
            ContextItem(item_id="m1", content="The user prefers tea in the morning",
                        context_type="facts", source="vector", similarity_score=0.8, confidence=0.8)
        ])
        m._apply_temporal_window_filter = AsyncMock(side_effect=lambda items, *a, **k: items)
        m._apply_jev_rerank = AsyncMock(side_effect=lambda items, *a, **k: items)
        d = ContextDecision(
            decision_id="d", trigger_type=TriggerType.EXPLICIT_REQUEST, instance_id="inst-1",
            user_id="user-1", conversation_id="conv-1", retrieval_mode=RetrievalMode.ACCURATE,
            context_types=["facts"], search_queries=[COMPOUND], token_budget=2000,
        )
        bundle = ContextBundle(bundle_id="b", decision_id="d")
        await m._fetch_accurate(d, _scope(), bundle)
        searched = [q for c in m.fetch_long_term_context.call_args_list for q in c.kwargs["queries"]]
        assert "Do I prefer tea in the morning ?" in searched
        assert m.llm_client is None

    def test_the_provider_builds_no_llm_client_when_the_edition_has_none(self, monkeypatch):
        from synap.cloud.application.shared import providers

        monkeypatch.setattr(edition, "retrieval_llm", False)
        assert providers.get_llm_provider() is None

    def test_the_fast_path_never_touches_an_llm_client(self):
        llm = _CountingLLM()
        m = _manager(llm)
        out = asyncio.run(m._generate_retrieval_subqueries("hello there friend", "fast"))
        assert out["queries"][0] == "hello there friend"
        assert llm.calls == 0


class TestHQEIsAHook:
    async def test_with_no_hook_registered_the_question_search_does_nothing(self, monkeypatch):
        vector = MagicMock()
        vector.batch_search_questions = AsyncMock(return_value=[[]])
        m = RetrievalManager(scope_resolver=MagicMock(), cache_provider=AsyncMock(), vector_provider=vector)
        edition.reset_hooks()
        try:
            out = await m._search_questions(_scope(), ["q"], [np.ones(4)], top_k=5)
        finally:
            edition.load_cloud_hooks_for_test()
        assert out == []
        vector.batch_search_questions.assert_not_called()

    def test_cloud_registers_the_search(self):
        assert edition.is_registered("retrieval_question_search")


class _Result:
    def first(self):
        return None


class _Session:
    def __init__(self, calls):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, stmt, params):
        self._calls.append(" ".join(str(stmt).split()))
        return _Result()


class TestTheStoredScopePath:
    def _sql(self, monkeypatch, scope_levels):
        from synap.cloud.application.sdk_api import sdk_context

        calls = []
        monkeypatch.setattr(
            "synap.cloud.database.connection.get_session_factory",
            lambda: (lambda: _Session(calls)),
        )
        monkeypatch.setattr(edition, "scope_levels", scope_levels)
        asyncio.run(sdk_context._resolve_scope_from_conversation("3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c", "inst"))
        return calls[0]

    def test_core_does_not_select_it(self, monkeypatch):
        sql = self._sql(monkeypatch, ("user", "customer", "install"))
        assert "scope_path" not in sql
        assert "SELECT user_id, customer_id, NULL FROM conversation" in sql

    def test_cloud_still_selects_it(self, monkeypatch):
        sql = self._sql(monkeypatch, None)
        assert "SELECT user_id, customer_id, scope_path FROM conversation" in sql


class TestRetrievalReadsTheFileInCore:
    async def test_no_ranking_block_gives_the_retrieval_defaults(self, monkeypatch):
        class _Loader:
            async def get_raw_config(self, instance_id=None):
                return {"metadata": {}}

        monkeypatch.setattr(edition, "config_source", "file")
        monkeypatch.setattr(
            "synap.cloud.memory_config.loader.get_config_loader", lambda: _Loader()
        )
        s = await RM._load_maca_config("any-instance")
        assert s.ranking_signals == {"recency": 0.10, "relevance": 0.70, "importance": 0.20}

    async def test_a_ranking_block_in_the_file_wins(self, monkeypatch):
        class _Loader:
            async def get_raw_config(self, instance_id=None):
                return {"retrieval_config": {"ranking": {"primary_signals": {
                    "relevance": 0.5, "importance": 0.25, "recency": 0.25}}}}

        monkeypatch.setattr(edition, "config_source", "file")
        monkeypatch.setattr(
            "synap.cloud.memory_config.loader.get_config_loader", lambda: _Loader()
        )
        s = await RM._load_maca_config("any-instance")
        assert s.ranking_signals["relevance"] == 0.5

    async def test_a_loader_failure_is_not_swallowed(self, monkeypatch):
        class _Loader:
            async def get_raw_config(self, instance_id=None):
                raise RuntimeError("boom")

        monkeypatch.setattr(edition, "config_source", "file")
        monkeypatch.setattr(
            "synap.cloud.memory_config.loader.get_config_loader", lambda: _Loader()
        )
        with pytest.raises(RuntimeError):
            await RM._load_maca_config("any-instance")
