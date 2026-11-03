"""The Ollama provider, against a stand-in Ollama that records what it is sent."""
from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from synap.cloud.infrastructure.llm.models import LLMProvider, ProviderError
from synap.cloud.infrastructure.llm.providers.ollama_provider import OllamaProvider


class FakeOllama:
    """Answers /api/chat and /api/tags the way Ollama does, and keeps every request body."""

    def __init__(self):
        self.requests, self.reply, self.status = [], {}, 200
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"models": []}) if self.path == "/api/tags" else self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if outer.status >= 400:
                    return self._send(outer.status, outer.reply)
                if body.get("stream"):
                    lines = outer.reply["stream"]
                    return self._send(200, ("\n".join(json.dumps(x) for x in lines) + "\n").encode(),
                                      "application/x-ndjson")
                self._send(200, outer.reply)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def ollama():
    f = FakeOllama()
    f.reply = {"message": {"role": "assistant", "content": "hello"}, "done": True,
               "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 3}
    yield f
    f.close()


@pytest.fixture
async def provider(ollama):
    p = OllamaProvider(base_url=ollama.url)
    yield p
    await p.close()


async def test_a_completion_returns_text_and_token_counts(provider, ollama):
    r = await provider.complete([{"role": "user", "content": "hi"}], model="llama3.1:8b")
    assert r.content == "hello" and r.provider == LLMProvider.OLLAMA and r.model == "llama3.1:8b"
    assert (r.input_tokens, r.output_tokens, r.total_tokens) == (12, 3, 15)
    assert r.finish_reason == "stop"


async def test_the_context_window_and_limits_are_sent_with_every_request(provider, ollama):
    await provider.complete([{"role": "user", "content": "hi"}], model="m", temperature=0.2, max_tokens=500)
    path, body = ollama.requests[0]
    assert path == "/api/chat" and body["stream"] is False
    assert body["options"] == {"temperature": 0.2, "num_ctx": 16384, "num_predict": 500}


async def test_the_window_can_be_set(ollama, monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    p = OllamaProvider(base_url=ollama.url)
    await p.complete([{"role": "user", "content": "x"}], model="m")
    body = ollama.requests[0][1]
    assert body["options"]["num_ctx"] == 32768 and body["keep_alive"] == "30m"
    await p.close()


async def test_json_mode_asks_for_json(provider, ollama):
    await provider.complete([{"role": "user", "content": "x"}], model="m", json_mode=True)
    assert ollama.requests[0][1]["format"] == "json"
    await provider.complete([{"role": "user", "content": "x"}], model="m")
    assert "format" not in ollama.requests[1][1]


async def test_a_cut_off_reply_is_marked_as_length(provider, ollama):
    ollama.reply["done_reason"] = "length"
    r = await provider.complete([{"role": "user", "content": "x"}], model="m")
    assert r.finish_reason == "length"


async def test_images_move_to_the_images_field(provider, ollama):
    raw = b"\x89PNG-bytes"
    b64 = base64.b64encode(raw).decode()
    await provider.complete([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
        {"type": "text", "text": "Describe this."},
    ]}], model="llava")
    sent = ollama.requests[0][1]["messages"][0]
    assert sent == {"role": "user", "content": "Describe this.", "images": [b64]}


async def test_an_image_link_is_refused_rather_than_fetched(provider):
    with pytest.raises(ProviderError, match="data: URLs"):
        await provider.complete([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest"}}]}], model="llava")


async def test_a_model_that_is_not_pulled_says_how_to_pull_it(provider, ollama):
    ollama.status, ollama.reply = 404, {"error": "model 'nope' not found"}
    with pytest.raises(ProviderError, match="ollama pull nope") as e:
        await provider.complete([{"role": "user", "content": "x"}], model="nope")
    assert e.value.status_code == 404 and e.value.provider == "ollama"


async def test_an_unreachable_ollama_says_where_it_looked_and_what_to_try():
    p = OllamaProvider(base_url="http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(ProviderError, match="host.docker.internal") as e:
        await p.complete([{"role": "user", "content": "x"}], model="m")
    assert "http://127.0.0.1:9" in str(e.value)
    await p.close()


async def test_tool_calls_come_back_in_the_shape_the_engine_reads(provider, ollama):
    ollama.reply = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "lookup", "arguments": {"q": "x"}}}]}, "done": True}
    r = await provider.complete([{"role": "user", "content": "x"}], model="m", tools=[{"type": "function"}])
    call = r.tool_calls[0]
    assert call["type"] == "function" and call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"q": "x"}


async def test_streaming_yields_chunks_and_reports_usage(provider, ollama):
    ollama.reply = {"stream": [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 7, "eval_count": 2},
    ]}
    sink = {}
    chunks = [c async for c in provider.stream([{"role": "user", "content": "x"}], model="m", _usage_sink=sink)]
    assert chunks == ["Hel", "lo"] and sink == {"input_tokens": 7, "output_tokens": 2}


async def test_any_model_the_developer_names_is_accepted(provider):
    assert provider.is_model_supported("some-model:7b")
    info = provider.get_model_info("some-model:7b")
    assert info.provider == LLMProvider.OLLAMA and info.pricing_input_per_1k == 0.0
    assert provider.get_rate_limits().rpm > 100_000


async def test_health_check_reflects_whether_ollama_answers(provider, ollama):
    assert await provider.health_check() is True
    down = OllamaProvider(base_url="http://127.0.0.1:9")
    assert await down.health_check() is False
    await down.close()


async def test_no_api_key_is_needed():
    p = OllamaProvider()
    assert p.base_url == "http://host.docker.internal:11434"
    await p.close()
