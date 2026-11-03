"""Ingestion, compaction and image reads on one configured model, end to end.

Each test runs the real step code, through the real manager and the real Ollama provider, to a
stand-in Ollama that records the request. What it proves: no step reaches OpenAI, whatever that
step forces on `main`, and each step lands on the model the developer set for it.
"""
from __future__ import annotations

import base64
import io

import fakeredis
import pytest

from synap.cloud.infrastructure.llm import LLMConfigManager, reset_llm_manager
from synap.cloud.infrastructure.llm import manager as manager_module
from synap.cloud.infrastructure.llm import settings as S
from synap.cloud.infrastructure.llm.models import LLMProvider
from synap.tests.infrastructure.llm.test_ollama_provider import FakeOllama

ENV = ("SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "OPENAI_API_KEY", "OPENAI_API_KEYS", "SYNAP_VISION_MODEL",
       "SYNAP_KU_CLASSIFY_MODEL", "EVAL_PROVIDER") + tuple(f"SYNAP_LLM_MODEL_{s}" for s in S.STEPS)


@pytest.fixture
def ollama(monkeypatch):
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    fake = FakeOllama()
    fake.reply = {"message": {"role": "assistant", "content": '{"ok": true}'}, "done": True,
                  "prompt_eval_count": 5, "eval_count": 2}
    monkeypatch.setenv("SYNAP_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("SYNAP_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("SYNAP_LLM_MODEL_VISION", "llava:13b")
    monkeypatch.setenv("SYNAP_LLM_MODEL_KNOWLEDGE_UPDATE", "qwen2.5:3b")
    monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
    S.reset_for_tests()
    reset_llm_manager()
    manager_module._llm_manager = LLMConfigManager(fakeredis.FakeAsyncRedis())
    yield fake
    fake.close()
    reset_llm_manager()
    S.reset_for_tests()


def _models_sent(fake) -> list:
    return [body["model"] for _, body in fake.requests]


async def test_the_ingestion_client_runs_on_the_configured_model_not_gpt_4o(ollama):
    from synap.cloud.application.ingestion.startup import _get_llm_client
    # the same three clients ingestion builds: extraction, the graph, and the knowledge-update classifier
    out = await _get_llm_client("ingestion_agent").complete(prompt="extract", json_mode=True)
    await _get_llm_client("ingestion_agent_light", purpose="er").complete(prompt="graph")
    await _get_llm_client("ingestion_agent_light", model="gpt-4o", step="KNOWLEDGE_UPDATE").complete(prompt="classify")
    assert out == '{"ok": true}'
    assert _models_sent(ollama) == ["llama3.1:8b", "llama3.1:8b", "qwen2.5:3b"]


async def test_a_gemini_eval_setting_cannot_move_ingestion_off_the_configured_model(ollama, monkeypatch):
    from synap.cloud.application.ingestion.startup import _get_llm_client
    monkeypatch.setenv("EVAL_PROVIDER", "gemini")
    await _get_llm_client("ingestion_agent").complete(prompt="extract")
    assert _models_sent(ollama) == ["llama3.1:8b"]


async def test_an_image_read_goes_to_the_vision_model_with_the_image_attached(ollama):
    from PIL import Image
    from synap.cloud.application.ingestion import multimodal
    multimodal._description_cache.clear()
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    ollama.reply["message"]["content"] = "A red square."
    text = await multimodal.describe_image(buf.getvalue(), "image/png", client_id="c", instance_id="i")
    assert text == "A red square."
    path, body = ollama.requests[0]
    assert body["model"] == "llava:13b"
    assert len(body["messages"][0]["images"]) == 1
    base64.b64decode(body["messages"][0]["images"][0], validate=True)  # a real base64 image, not a link


async def test_compaction_writes_on_the_configured_model(ollama):
    from synap.cloud.application.compaction.passes.base import CompactionProfile
    from synap.cloud.application.compaction.v3.engine import CompactionWriter, UpdateResult
    from synap.cloud.infrastructure.llm import get_llm_manager
    # a profile as Cloud defines it: forced to OpenAI gpt-4o, allowed to fall back to another provider
    profile = CompactionProfile(name="cloud", provider=LLMProvider.OPENAI, model="gpt-4o",
                                reasoning_token_overhead=0, cross_provider_fallback=True, jev_check=False)
    writer = CompactionWriter(get_llm_manager(), profile)
    out = await writer._call_json("system", "user", 1000, UpdateResult(record=None))
    assert out == {"ok": True}
    assert _models_sent(ollama) == ["llama3.1:8b"]


async def test_nothing_reaches_openai(ollama, monkeypatch):
    from synap.cloud.application.ingestion.startup import _get_llm_client
    monkeypatch.setenv("OPENAI_API_KEY", "a-key-that-must-never-be-used")
    manager = manager_module._llm_manager
    assert LLMProvider.OPENAI not in manager._providers
    await _get_llm_client("ingestion_agent").complete(prompt="x")
    assert all(p == "/api/chat" for p, _ in ollama.requests)
