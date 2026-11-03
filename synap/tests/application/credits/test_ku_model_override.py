"""The knowledge-update classifier's model is selectable, and defaults to today's.

Pair classification is a four-way label with a ~34-token answer. Measured over
a sample of real traffic, only about one call in ten changed
any data; the rest answered "unrelated". Running that on the flagship model is
the single clearest cost lever in the pipeline.

`_get_llm_client` pins gpt-4o and ignores its `agent_type` argument, so there
was no way to move one step without moving all of them. These tests pin the new
`model` escape hatch, and pin that the default did not move.
"""

from __future__ import annotations

import os

import pytest


def _client_model(monkeypatch, env=None):
    """Build the wrapper and read back the model it would force."""
    captured = {}

    class _FakeManager:
        async def complete(self, **kwargs):
            captured.update(kwargs)

            class _R:
                content = "{}"
                input_tokens = output_tokens = 0
                cached_input_tokens = 0
                model = kwargs.get("force_model")
            return _R()

    import synap.cloud.application.ingestion.startup as startup

    monkeypatch.setattr(startup, "get_llm_manager", lambda: _FakeManager(), raising=False)
    return captured, startup


class TestKnowledgeUpdateModelIsSelectable:
    def test_default_is_unchanged(self):
        """No env var set must mean exactly the model that ran before."""
        assert os.environ.get("SYNAP_KU_CLASSIFY_MODEL") in (None, "", "gpt-4o")

    def test_get_llm_client_accepts_a_model_override(self):
        import inspect

        from synap.cloud.application.ingestion.startup import _get_llm_client

        sig = inspect.signature(_get_llm_client)
        assert "model" in sig.parameters, (
            "the per-call-site model override is gone; every ingestion step is "
            "pinned to one model again"
        )
        assert sig.parameters["model"].default is None

    def test_the_override_reaches_the_provider_call(self, monkeypatch):
        """A guard against the override becoming decorative, which is exactly
        what happened to `agent_type`."""
        import inspect

        from synap.cloud.application.ingestion import startup

        src = inspect.getsource(startup._get_llm_client)
        assert "_forced_model = model" in src
        assert "_forced_model or" in src, (
            "the model argument is accepted but never used to pick the model"
        )

    def test_agent_type_is_still_documented_as_ignored(self):
        """If someone wires the tier up for real, this test should fail and make
        them re-read the quality note rather than ship it silently."""
        import inspect

        from synap.cloud.application.ingestion.startup import _get_llm_client

        doc = inspect.getdoc(_get_llm_client) or ""
        assert "IS CURRENTLY IGNORED" in doc


_ENV = ("SYNAP_KU_CLASSIFY_MODEL", "SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "EVAL_PROVIDER",
        "SYNAP_LLM_MODEL_KNOWLEDGE_UPDATE")


async def _ku_client(monkeypatch, env):
    """The client startup hands the knowledge-update detector, built by the real startup code."""
    import types

    from synap.cloud.agents.ingestion.pipeline import knowledge_update
    from synap.cloud.application.ingestion import startup
    from synap.cloud.database import connection

    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    # Stages 1 to 4 start no workers; the detector is still built.
    for stage in ("STAGE1", "STAGE23", "STAGE4"):
        monkeypatch.setenv(f"{stage}_WORKER_COUNT", "0")
    built = {}

    class _Detector:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(knowledge_update, "KnowledgeUpdateDetector", _Detector)
    monkeypatch.setattr(startup, "_get_storage_coordinator", lambda **_: types.SimpleNamespace(_vector=object()))
    monkeypatch.setattr(connection, "get_session_factory", lambda: None)
    await startup._start_stage_workers(redis_client=None, status_tracker=None, reporting=None)
    assert "llm_client" in built, "startup no longer builds the knowledge-update detector"
    return built["llm_client"]


async def _model_on_cloud(monkeypatch, env):
    """The model a Cloud knowledge-update call forces on the provider."""
    import types

    from synap.cloud.infrastructure import llm

    sent = {}

    class _Manager:
        async def complete(self, **kwargs):
            sent.update(kwargs)
            return types.SimpleNamespace(content="{}", input_tokens=0, output_tokens=0, model=kwargs["force_model"])

    monkeypatch.setattr(llm, "get_llm_manager", lambda: _Manager())
    client = await _ku_client(monkeypatch, env)
    await client.complete(prompt="classify")
    return sent["force_model"]


class TestKnowledgeUpdateModelChoice:
    async def test_cloud_default_is_gpt4o(self, monkeypatch):
        assert await _model_on_cloud(monkeypatch, {}) == "gpt-4o"

    async def test_blank_env_var_keeps_the_default(self, monkeypatch):
        assert await _model_on_cloud(monkeypatch, {"SYNAP_KU_CLASSIFY_MODEL": "  "}) == "gpt-4o"

    async def test_cloud_env_var_overrides_the_default(self, monkeypatch):
        env = {"SYNAP_KU_CLASSIFY_MODEL": "gpt-4o-mini"}
        assert await _model_on_cloud(monkeypatch, env) == "gpt-4o-mini"

    async def test_core_provider_setting_wins_over_the_env_var(self, monkeypatch):
        """With the Core settings active the manager overrides the forced model, so the
        env var does not decide; the step's own Core model does."""
        import fakeredis

        from synap.cloud.infrastructure.llm import LLMConfigManager, reset_llm_manager
        from synap.cloud.infrastructure.llm import manager as manager_module
        from synap.cloud.infrastructure.llm import settings
        from synap.tests.infrastructure.llm.test_ollama_provider import FakeOllama

        fake = FakeOllama()
        fake.reply = {"message": {"role": "assistant", "content": "{}"}, "done": True,
                      "prompt_eval_count": 1, "eval_count": 1}
        env = {"SYNAP_LLM_PROVIDER": "ollama", "SYNAP_LLM_MODEL": "llama3.1:8b",
               "SYNAP_LLM_MODEL_KNOWLEDGE_UPDATE": "qwen2.5:3b",
               "SYNAP_KU_CLASSIFY_MODEL": "gpt-4o-mini", "OLLAMA_BASE_URL": fake.url}
        try:
            for name, value in env.items():
                monkeypatch.setenv(name, value)
            settings.reset_for_tests()
            reset_llm_manager()
            monkeypatch.setattr(manager_module, "_llm_manager", LLMConfigManager(fakeredis.FakeAsyncRedis()))
            client = await _ku_client(monkeypatch, env)
            await client.complete(prompt="classify")
            assert [body["model"] for _, body in fake.requests] == ["qwen2.5:3b"]
        finally:
            fake.close()
            reset_llm_manager()
            settings.reset_for_tests()
