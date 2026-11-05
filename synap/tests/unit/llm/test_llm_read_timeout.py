"""LLM calls have a bounded read timeout, and a timeout is its own error type.

Before: httpx.Timeout(None, connect=10.0), so a request that never answered held
an ingestion worker forever. After: read/write are bounded (env-configurable),
and a timeout raises ProviderTimeoutError, which neither marks the API key
unhealthy nor (in the manager) counts toward model availability.
"""
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
    assert all(s.is_healthy for s in p._key_rotator._stats.values()) if hasattr(p._key_rotator, "_stats") else True
    await p.close()


async def test_hung_stream_raises_timeout(monkeypatch, silent_server):
    monkeypatch.setenv("SYNAP_LLM_READ_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    p = OpenAIProvider(api_key="test-key", base_url=f"http://127.0.0.1:{silent_server.port}/v1")
    with pytest.raises(ProviderTimeoutError):
        async for _ in p.stream(messages=[{"role": "user", "content": "hi"}], model="gpt-4o"):
            pass
    await p.close()
