"""One provider and one model for everything."""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import fakeredis
import pytest

from synap.cloud.infrastructure.llm import LLMConfigManager
from synap.cloud.infrastructure.llm import settings as S
from synap.cloud.infrastructure.llm.models import LLMError, LLMProvider, LLMResponse, ProviderError
from synap.cloud.infrastructure.llm.providers.ollama_provider import OllamaProvider

ENV = ("SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "OLLAMA_BASE_URL", "OPENAI_API_KEY", "OPENAI_API_KEYS",
       "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY") + tuple(f"SYNAP_LLM_MODEL_{s}" for s in S.STEPS)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    S.reset_for_tests()
    yield
    S.reset_for_tests()


def _reply(model: str, provider=LLMProvider.OLLAMA) -> LLMResponse:
    return LLMResponse(content="ok", model=model, provider=provider, input_tokens=1, output_tokens=1, total_tokens=2)


def _manager(monkeypatch, **env) -> LLMConfigManager:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    S.reset_for_tests()
    return LLMConfigManager(fakeredis.FakeAsyncRedis())


def _fake(manager, provider=LLMProvider.OLLAMA, **kw):
    fake = MagicMock()
    fake.complete = AsyncMock(side_effect=kw.get("side_effect"), return_value=_reply("x", provider))
    manager._providers[provider] = fake
    return fake


# --- the settings -------------------------------------------------------------------------------

def test_with_nothing_set_nothing_is_active():
    assert S.load({}) is None


@pytest.mark.parametrize("env,words", [
    ({"SYNAP_LLM_PROVIDER": "ollama"}, "SYNAP_LLM_MODEL is not set"),
    ({"SYNAP_LLM_MODEL": "llama3.1:8b"}, "SYNAP_LLM_PROVIDER is not set"),
    ({"SYNAP_LLM_MODEL_VISION": "llava"}, "SYNAP_LLM_MODEL_VISION is set, but"),
    ({"SYNAP_LLM_PROVIDER": "groq", "SYNAP_LLM_MODEL": "x"}, "anthropic, ollama, openai"),
    ({"SYNAP_LLM_PROVIDER": "gemini", "SYNAP_LLM_MODEL": "x"}, "not a provider Core uses"),
])
def test_half_set_or_unsupported_settings_fail_with_the_reason(env, words):
    with pytest.raises(S.LLMSettingsError, match=words):
        S.load(env)


def test_a_step_can_use_its_own_model():
    cfg = S.load({"SYNAP_LLM_PROVIDER": "Ollama", "SYNAP_LLM_MODEL": "llama3.1:8b", "SYNAP_LLM_MODEL_VISION": "llava:13b"})
    assert cfg.provider == LLMProvider.OLLAMA
    assert cfg.model_for("vision") == "llava:13b" and cfg.model_for("VISION") == "llava:13b"
    assert cfg.model_for("INGESTION") == "llama3.1:8b" and cfg.model_for(None) == "llama3.1:8b"
    assert cfg.all_models() == {"llama3.1:8b", "llava:13b"}


# --- Cloud does not change ---------------------------------------------------------------------------

def test_with_no_settings_the_manager_behaves_as_before(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://somewhere:11434")  # alone, it must not switch anything on
    m = _manager(monkeypatch)
    assert m._single is None
    assert LLMProvider.OLLAMA not in m._providers


async def test_with_no_settings_a_forced_call_goes_where_it_was_forced(monkeypatch):
    m = _manager(monkeypatch)
    openai = _fake(m, LLMProvider.OPENAI)
    openai.complete.return_value = _reply("gpt-4o", LLMProvider.OPENAI)
    await m.complete([{"role": "user", "content": "x"}], force_provider=LLMProvider.OPENAI, force_model="gpt-4o", step="VISION")
    assert openai.complete.await_args.kwargs["model"] == "gpt-4o"


# --- one provider and model for everything --------------------------------------------------------------

def test_only_the_named_provider_is_started(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-unused")  # a key for another provider changes nothing
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1:8b")
    assert list(m._providers) == [LLMProvider.OLLAMA]
    assert isinstance(m._providers[LLMProvider.OLLAMA], OllamaProvider)


def test_the_registry_accepts_a_model_it_does_not_list(monkeypatch):
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1:8b", SYNAP_LLM_MODEL_VISION="llava:13b")
    for name in ("llama3.1:8b", "llava:13b"):
        assert m.get_registry().get_model_info(LLMProvider.OLLAMA, name).name == name


async def test_a_call_that_forced_openai_gpt_4o_runs_on_the_configured_model(monkeypatch):
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1:8b")
    ollama = _fake(m)
    r = await m.complete([{"role": "user", "content": "x"}],
                         force_provider=LLMProvider.OPENAI, force_model="gpt-4o", allow_cross_provider_fallback=True)
    assert ollama.complete.await_args.kwargs["model"] == "llama3.1:8b"
    assert r.content == "ok"


@pytest.mark.parametrize("step,expected", [
    ("INGESTION", "llama3.1:8b"), ("GRAPH", "llama3.1:8b"), ("KNOWLEDGE_UPDATE", "qwen2.5:3b"),
    ("COMPACTION", "llama3.1:8b"), ("VISION", "llava:13b"), (None, "llama3.1:8b"),
])
async def test_each_step_gets_its_override_or_the_default(monkeypatch, step, expected):
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1:8b",
                 SYNAP_LLM_MODEL_VISION="llava:13b", SYNAP_LLM_MODEL_KNOWLEDGE_UPDATE="qwen2.5:3b")
    ollama = _fake(m)
    await m.complete([{"role": "user", "content": "x"}], step=step)
    assert ollama.complete.await_args.kwargs["model"] == expected


async def test_a_failing_provider_is_never_replaced_by_another(monkeypatch):
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1:8b")
    ollama = _fake(m, side_effect=ProviderError("ollama down", "ollama", 500))
    other = _fake(m, LLMProvider.OPENAI)  # present, but not the one the developer chose
    with pytest.raises(LLMError):
        await m.complete([{"role": "user", "content": "x"}], allow_cross_provider_fallback=True)
    assert ollama.complete.await_count >= 1 and other.complete.await_count == 0


async def test_a_failing_model_is_never_replaced_by_another_model_of_the_same_provider(monkeypatch):
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="anthropic", SYNAP_LLM_MODEL="claude-sonnet-4-5", ANTHROPIC_API_KEY="sk-ant-test")
    m.get_registry().register_model(LLMProvider.ANTHROPIC, "claude-haiku-4-5")  # the registry knows another one
    anthropic = _fake(m, LLMProvider.ANTHROPIC, side_effect=ProviderError("overloaded", "anthropic", 500))
    with pytest.raises(LLMError):
        await m.complete([{"role": "user", "content": "x"}])
    tried = {c.kwargs["model"] for c in anthropic.complete.await_args_list}
    assert tried == {"claude-sonnet-4-5"}  # claude-haiku-4-5 is in the registry but the developer never chose it


async def test_streaming_uses_the_configured_model_too(monkeypatch):
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1:8b")

    async def gen(**kw):
        assert kw["model"] == "llama3.1:8b"
        yield "a"
    fake = MagicMock()
    fake.stream = lambda **kw: gen(**kw)
    m._providers[LLMProvider.OLLAMA] = fake
    out = [c async for c in m.stream([{"role": "user", "content": "x"}], force_provider=LLMProvider.GROQ, force_model="gpt-oss")]
    assert out == ["a"]


def test_anthropic_is_started_from_its_key_whatever_the_registry_file_says(monkeypatch):
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="anthropic", SYNAP_LLM_MODEL="claude-sonnet-4-5", ANTHROPIC_API_KEY="sk-ant-test")
    assert list(m._providers) == [LLMProvider.ANTHROPIC]  # the registry file has it disabled


async def test_a_missing_key_is_reported_at_start_not_at_the_first_ingest(monkeypatch, caplog):
    with caplog.at_level(logging.ERROR):
        m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="openai", SYNAP_LLM_MODEL="gpt-4.1")
    assert "llm_provider_unavailable provider=openai" in caplog.text and "OpenAI API key is required" in caplog.text
    with pytest.raises(LLMError):
        await m.complete([{"role": "user", "content": "x"}])


def test_groq_and_gemini_are_not_needed_to_import_or_build_the_manager(monkeypatch):
    import sys
    for mod in ("synap.cloud.infrastructure.llm.providers.groq_provider", "synap.cloud.infrastructure.llm.providers.gemini_provider"):
        monkeypatch.setitem(sys.modules, mod, None)  # as if the files were not in the build
    m = _manager(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1:8b")
    assert list(m._providers) == [LLMProvider.OLLAMA]
    LLMConfigManager(fakeredis.FakeAsyncRedis())  # and the Cloud path degrades to a warning, not a crash
