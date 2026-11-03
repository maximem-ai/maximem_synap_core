"""The generic OpenAI-compatible provider: any server that speaks the OpenAI chat API."""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fakeredis
import pytest

from synap.cloud.infrastructure.llm import LLMConfigManager, reset_llm_manager
from synap.cloud.infrastructure.llm import manager as manager_module
from synap.cloud.infrastructure.llm import settings as S
from synap.cloud.infrastructure.llm.models import LLMProvider
from synap.cloud.infrastructure.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from synap.cloud.infrastructure.llm.providers.openai_provider import OpenAIProvider

ENV = ("SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "SYNAP_LLM_BASE_URL", "SYNAP_LLM_API_KEY",
       "OPENAI_API_KEY", "OPENAI_API_KEYS") + tuple(f"SYNAP_LLM_MODEL_{s}" for s in S.STEPS)


class FakeServer:
    """An OpenAI-style server that records the path, the Authorization header and the body."""

    def __init__(self):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append((self.path, self.headers.get("Authorization"), body))
                data = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for v in ENV:
        monkeypatch.delenv(v, raising=False)
    S.reset_for_tests()
    yield
    S.reset_for_tests()
    reset_llm_manager()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


# --- the settings ---------------------------------------------------------------------------------

def test_the_server_address_is_required_and_only_allowed_for_this_provider():
    with pytest.raises(S.LLMSettingsError, match="SYNAP_LLM_BASE_URL is not set"):
        S.load({"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "m"})
    with pytest.raises(S.LLMSettingsError, match="only used with"):
        S.load({"SYNAP_LLM_PROVIDER": "ollama", "SYNAP_LLM_MODEL": "m", "SYNAP_LLM_BASE_URL": "http://x/v1"})
    with pytest.raises(S.LLMSettingsError, match="SYNAP_LLM_BASE_URL is set, but"):
        S.load({"SYNAP_LLM_BASE_URL": "http://x/v1"})
    cfg = S.load({"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "m", "SYNAP_LLM_BASE_URL": "http://x/v1"})
    assert cfg.base_url == "http://x/v1"


@pytest.mark.parametrize("var,value", [
    ("SYNAP_LLM_PROVIDER", "add-your-provider"), ("SYNAP_LLM_MODEL", "add-your-model"),
    ("SYNAP_LLM_BASE_URL", "add-the-api-address"), ("SYNAP_LLM_MODEL_VISION", "add-a-model-that-reads-images"),
])
def test_a_value_left_as_the_example_placeholder_is_refused_by_name(var, value):
    env = {"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "m", "SYNAP_LLM_BASE_URL": "http://x/v1"}
    env[var] = value
    with pytest.raises(S.LLMSettingsError, match=f"{var} still holds the placeholder"):
        S.load(env)


# --- the provider ----------------------------------------------------------------------------------

async def test_requests_go_to_the_configured_server_with_its_key_only(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-that-must-not-travel")
    monkeypatch.setenv("OPENAI_API_KEYS", "k1-must-not-travel,k2-must-not-travel")
    monkeypatch.setenv("SYNAP_LLM_API_KEY", "key-for-this-server")
    p = OpenAICompatibleProvider(server.url)
    r = await p.complete([{"role": "user", "content": "hi"}], model="some-model", json_mode=True, max_tokens=300)
    path, auth, body = server.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer key-for-this-server"
    assert body["model"] == "some-model" and body["max_tokens"] == 300 and body["response_format"] == {"type": "json_object"}
    assert r.content == '{"ok": true}' and r.total_tokens == 6
    assert "must-not-travel" not in json.dumps(server.seen)
    await p.close()


async def test_a_server_that_needs_no_key_gets_a_harmless_placeholder(server):
    p = OpenAICompatibleProvider(server.url)
    await p.complete([{"role": "user", "content": "hi"}], model="m")
    assert server.seen[0][1] == "Bearer not-needed"
    await p.close()


@pytest.mark.parametrize("url", ["", "   ", "not a url", "ftp://host/v1", "http://"])
def test_a_bad_address_is_refused(url):
    with pytest.raises(ValueError, match="SYNAP_LLM_BASE_URL"):
        OpenAICompatibleProvider(url)


def test_a_key_sent_over_plain_http_to_another_machine_is_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        OpenAICompatibleProvider("http://models.example.com/v1", api_key="k")
    assert "unencrypted" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        OpenAICompatibleProvider("http://localhost:8000/v1", api_key="k")
        OpenAICompatibleProvider("https://models.example.com/v1", api_key="k")
    assert "unencrypted" not in caplog.text


def test_the_openai_provider_itself_is_untouched():
    # Cloud's provider still reads OpenAI's keys; the new one is a separate class, in a separate file.
    assert OpenAICompatibleProvider.__mro__[1] is OpenAIProvider
    assert OpenAICompatibleProvider.__init__ is not OpenAIProvider.__init__


# --- through the manager and the ingestion client ---------------------------------------------------------

async def test_ingestion_runs_on_the_configured_server_and_models(server, monkeypatch):
    from synap.cloud.application.ingestion.startup import _get_llm_client
    monkeypatch.setenv("SYNAP_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SYNAP_LLM_MODEL", "main-model")
    monkeypatch.setenv("SYNAP_LLM_MODEL_KNOWLEDGE_UPDATE", "small-model")
    monkeypatch.setenv("SYNAP_LLM_BASE_URL", server.url)
    monkeypatch.setenv("SYNAP_LLM_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-that-must-not-travel")
    S.reset_for_tests()
    reset_llm_manager()
    manager_module._llm_manager = LLMConfigManager(fakeredis.FakeAsyncRedis())
    assert list(manager_module._llm_manager._providers) == [LLMProvider.OPENAI]
    assert isinstance(manager_module._llm_manager._providers[LLMProvider.OPENAI], OpenAICompatibleProvider)

    await _get_llm_client("ingestion_agent").complete(prompt="extract")
    await _get_llm_client("ingestion_agent_light", model="gpt-4o", step="KNOWLEDGE_UPDATE").complete(prompt="classify")
    assert [b["model"] for _, _, b in server.seen] == ["main-model", "small-model"]
    assert {a for _, a, _ in server.seen} == {"Bearer k"}
