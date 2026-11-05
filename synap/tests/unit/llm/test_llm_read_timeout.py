"""LLM calls have a bounded read timeout, and a timeout is its own error type.

Before: httpx.Timeout(None, connect=10.0), so a request that never answered held
an ingestion worker forever. After: read/write are bounded (env-configurable),
and a timeout raises ProviderTimeoutError, which neither marks the API key
unhealthy nor (in the manager) counts toward model availability.
"""
import os

# This repository is Synap Core: the manager, Stage 4 and the start-up check
# load Core's prompt pack and hooks only when the edition says so.
os.environ.setdefault("SYNAP_EDITION", "core")

import asyncio
import socket
import time

import httpx
import pytest

from synap.cloud.infrastructure.llm.models import ProviderError, ProviderTimeoutError
from synap.cloud.infrastructure.llm.providers import base as base_mod
from synap.cloud.infrastructure.llm.providers.openai_provider import OpenAIProvider


def test_default_timeouts(monkeypatch):
    monkeypatch.delenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SYNAP_LLM_WRITE_TIMEOUT_SECONDS", raising=False)
    t = base_mod.llm_http_timeout()
    assert (t.connect, t.read, t.write, t.pool) == (10.0, 300.0, 60.0, None)


@pytest.mark.parametrize("raw,expected", [("20", 20.0), ("2.5", 2.5), ("none", None), ("off", None), ("0", None), ("", 300.0)])
def test_read_timeout_from_env(monkeypatch, raw, expected):
    monkeypatch.setenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", raw)
    assert base_mod.llm_http_timeout().read == expected


@pytest.mark.parametrize("raw", ["abc", "-1", "5m"])
def test_bad_value_stops_startup(monkeypatch, raw):
    monkeypatch.setenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", raw)
    with pytest.raises(ValueError, match="SYNAP_LLM_READ_TIMEOUT_SECONDS"):
        base_mod.llm_http_timeout()


def test_timeout_error_is_a_provider_error_with_408():
    e = ProviderTimeoutError("openai", "ReadTimeout")
    assert isinstance(e, ProviderError) and e.status_code == 408 and "timeout" in str(e).lower()


class _SilentServer:
    """Accepts TCP connections, reads the request, never answers."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.conns = []

    def accept_forever(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self.conns.append(c)

    def close(self):
        for c in self.conns:
            c.close()
        self.sock.close()


@pytest.fixture
def silent_server():
    import threading
    s = _SilentServer()
    threading.Thread(target=s.accept_forever, daemon=True).start()
    yield s
    s.close()


async def test_hung_provider_raises_timeout_within_limit(monkeypatch, silent_server):
    monkeypatch.setenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    p = OpenAIProvider(api_key="test-key", base_url=f"http://127.0.0.1:{silent_server.port}/v1")
    t0 = time.monotonic()
    with pytest.raises(ProviderTimeoutError):
        await p.complete(messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert time.monotonic() - t0 < 5
    await p.close()


async def test_timeout_does_not_mark_key_unhealthy(monkeypatch, silent_server):
    monkeypatch.setenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("OPENAI_API_KEYS", "key-a,key-b")
    p = OpenAIProvider(base_url=f"http://127.0.0.1:{silent_server.port}/v1")
    assert p._key_rotator is not None
    with pytest.raises(ProviderTimeoutError):
        await p.complete(messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    # No failure recorded against any key (record_failure was never called).
    assert all(st.failed_calls == 0 and st.is_healthy for st in p._key_rotator._stats.values())
    await p.close()


async def test_hung_stream_raises_timeout(monkeypatch, silent_server):
    monkeypatch.setenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    p = OpenAIProvider(api_key="test-key", base_url=f"http://127.0.0.1:{silent_server.port}/v1")
    with pytest.raises(ProviderTimeoutError):
        async for _ in p.stream(messages=[{"role": "user", "content": "hi"}], model="gpt-4o"):
            pass
    await p.close()


# --- other providers, request and stream paths ---------------------------------

def _make(kind, port):
    url = f"http://127.0.0.1:{port}"
    if kind == "anthropic":
        from synap.cloud.infrastructure.llm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key="test-key", base_url=url + "/v1"), "claude-sonnet-4-5"
    if kind == "ollama":
        from synap.cloud.infrastructure.llm.providers.ollama_provider import OllamaProvider
        return OllamaProvider(base_url=url), "llama3"
    if kind == "openai-compatible":
        from synap.cloud.infrastructure.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
        return OpenAICompatibleProvider(base_url=url + "/v1", api_key="test-key"), "some-model"
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["anthropic", "ollama", "openai-compatible"])
@pytest.mark.parametrize("path", ["complete", "stream"])
async def test_every_provider_times_out(monkeypatch, silent_server, kind, path):
    monkeypatch.setenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEYS", raising=False)
    p, model = _make(kind, silent_server.port)
    msgs = [{"role": "user", "content": "hi"}]
    t0 = time.monotonic()
    with pytest.raises(ProviderTimeoutError):
        if path == "complete":
            await p.complete(messages=msgs, model=model)
        else:
            async for _ in p.stream(messages=msgs, model=model):
                pass
    assert time.monotonic() - t0 < 5
    await p.close()


async def test_connect_timeout_keeps_the_old_counted_path(monkeypatch):
    """Failing to connect is not a slow answer: it stays a plain ProviderError."""
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    p = OpenAIProvider(api_key="test-key", base_url="http://127.0.0.1:9/v1")

    def boom(request):
        raise httpx.ConnectTimeout("connect timed out", request=request)

    await p._http_client.aclose()
    p._http_client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    with pytest.raises(ProviderError) as ei:
        await p.complete(messages=[{"role": "user", "content": "hi"}], model="gpt-4o")
    assert not isinstance(ei.value, ProviderTimeoutError)
    await p.close()


# --- manager: timeouts are not counted toward availability ---------------------

def _manager_with(provider_exc):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from synap.cloud.infrastructure.llm.manager import LLMConfigManager
    from synap.cloud.infrastructure.llm.models import LLMProvider

    m = LLMConfigManager.__new__(LLMConfigManager)
    m._single = None
    m._providers = {}
    info = SimpleNamespace(name="gpt-4o", provider=LLMProvider.OPENAI)
    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=provider_exc)

    async def _stream(**kw):
        raise provider_exc
        yield  # pragma: no cover

    provider.stream = _stream
    m._registry = MagicMock()
    m._registry.get_model_info = MagicMock(return_value=info)
    m._selector = MagicMock()
    m._selector.get_fallback_model = AsyncMock(return_value=None)
    m._tracker = MagicMock()
    m._tracker.record_error = AsyncMock()
    m._tracker.record_request = AsyncMock()
    m._tracker.mark_unavailable = AsyncMock()
    m.get_provider = MagicMock(return_value=provider)
    return m, LLMProvider.OPENAI


async def _call_complete(m, prov):
    return await m.complete(messages=[{"role": "user", "content": "hi"}],
                            force_provider=prov, force_model="gpt-4o")


async def test_manager_complete_timeout_not_counted():
    from synap.cloud.infrastructure.llm.models import LLMTimeoutError
    m, prov = _manager_with(ProviderTimeoutError("openai", "ReadTimeout"))
    with pytest.raises(LLMTimeoutError) as ei:
        await _call_complete(m, prov)
    assert isinstance(ei.value.__cause__, ProviderTimeoutError)
    m._tracker.record_error.assert_not_called()
    m._tracker.mark_unavailable.assert_not_called()


async def test_manager_complete_other_error_still_counted():
    from synap.cloud.infrastructure.llm.models import LLMError, LLMTimeoutError
    m, prov = _manager_with(ProviderError("boom", "openai", 500))
    with pytest.raises(LLMError) as ei:
        await _call_complete(m, prov)
    assert not isinstance(ei.value, LLMTimeoutError)
    m._tracker.record_error.assert_awaited()


async def test_manager_stream_timeout_not_counted():
    from synap.cloud.infrastructure.llm.models import LLMTimeoutError
    m, prov = _manager_with(ProviderTimeoutError("openai", "ReadTimeout"))
    with pytest.raises(LLMTimeoutError):
        async for _ in m.stream(messages=[{"role": "user", "content": "hi"}],
                                force_provider=prov, force_model="gpt-4o"):
            pass
    m._tracker.record_error.assert_not_called()


def test_invalid_timeout_stops_core_startup():
    from synap.cloud.shared import core_startup
    env = {"SYNAP_LLM_PROVIDER": "openai", "SYNAP_LLM_MODEL": "gpt-4o",
           "SYNAP_LLM_WRITE_TIMEOUT_SECONDS": "5m"}
    msgs = [str(e) for e in core_startup.problems(env)]
    assert any("SYNAP_LLM_WRITE_TIMEOUT_SECONDS" in m for m in msgs), msgs
    env["SYNAP_LLM_WRITE_TIMEOUT_SECONDS"] = "none"
    assert not any("TIMEOUT" in str(e) for e in core_startup.problems(env))
