"""guardrails.rate_limits.max_retrieval_calls_per_minute on the context fetch routes.

The setting was accepted and read by nothing. These drive the real router with a
fake Redis, a fake config loader and an injected clock, never a real service.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import synap.cloud.application.sdk_api.retrieval_rate_limit as rl
from synap.cloud.application.sdk_api.readiness import require_instance_ready
from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.application.sdk_api.sdk_context import get_retrieval_manager, router
from synap.cloud.memory_config.config_schema import ConfigMetadata, MemoryArchitectureConfig

FETCH_ROUTES = [
    ("/v1/context/conversation/fetch", {"conversation_id": ""}),
    ("/v1/context/user/fetch", {"user_id": "u1"}),
    ("/v1/context/customer/fetch", {"customer_id": "d60af8d5-9db5-40e0-9d3f-91086d939717"}),
    ("/v1/context/client/fetch", {}),
]
ROUTE = FETCH_ROUTES[0]


def _config(**guardrails) -> MemoryArchitectureConfig:
    return MemoryArchitectureConfig(
        metadata=ConfigMetadata(
            client_id="c", instance_id="i", config_version="1.0.0",
            generated_at=datetime.now(timezone.utc), generated_by="test",
        ),
        **guardrails,
    )


def _limited(n) -> MemoryArchitectureConfig:
    return _config(guardrails={"rate_limits": {"max_retrieval_calls_per_minute": n}})


class _Pipe:
    def __init__(self, redis):
        self._r, self._ops = redis, []

    def incr(self, key):
        self._ops.append(("incr", key))

    def expire(self, key, ttl):
        self._ops.append(("expire", key, ttl))

    async def execute(self):
        if self._r.down:
            raise ConnectionError("redis is down")
        out = []
        for op in self._ops:
            if op[0] == "incr":
                self._r.counts[op[1]] = self._r.counts.get(op[1], 0) + 1
                out.append(self._r.counts[op[1]])
            else:
                self._r.ttls[op[1]] = op[2]
                out.append(True)
        return out


class _FakeRedis:
    def __init__(self, down=False):
        self.counts, self.ttls, self.down, self.pipelines = {}, {}, down, 0

    def pipeline(self):
        self.pipelines += 1
        return _Pipe(self)


class _Loader:
    def __init__(self, config=None, fails=False):
        self.config, self.fails, self.asked = config, fails, []

    async def get_config(self, instance_id):
        self.asked.append(instance_id)
        if self.fails:
            raise RuntimeError("config store unreachable")
        return self.config


class _EmptyManager:
    pass


class _Env:
    def __init__(self, monkeypatch, config=None, *, switch=True, redis_down=False, loader_fails=False):
        monkeypatch.delenv("SYNAP_GUARDRAIL_RETRIEVAL_RATE_INSTANCES", raising=False)
        if switch:
            monkeypatch.setenv("SYNAP_GUARDRAIL_RETRIEVAL_RATE", "true")
        else:
            monkeypatch.delenv("SYNAP_GUARDRAIL_RETRIEVAL_RATE", raising=False)
        self.redis = _FakeRedis(down=redis_down)
        self.loader = _Loader(config, fails=loader_fails)
        self.now = 1_000_000 * 60 + 10.0  # ten seconds into a minute
        self.instance_id = "inst_a"

        async def _redis():
            return self.redis

        monkeypatch.setattr(rl, "_get_redis", _redis)
        monkeypatch.setattr(rl, "_now", lambda: self.now)
        monkeypatch.setattr(
            "synap.cloud.memory_config.loader.get_config_loader", lambda: self.loader,
        )
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.dependency_overrides[get_sdk_auth] = lambda: SDKAuthContext(
            client_id="cli_test", instance_id=self.instance_id, credential_id=uuid.uuid4(),
        )
        self.app.dependency_overrides[require_instance_ready] = lambda: None
        self.app.dependency_overrides[get_retrieval_manager] = lambda: _EmptyManager()
        self.http = TestClient(self.app, raise_server_exceptions=False)

    def call(self, route=ROUTE):
        return self.http.post(route[0], json=route[1])


def test_switch_off_makes_no_redis_call_and_reads_no_config(monkeypatch):
    env = _Env(monkeypatch, _limited(1), switch=False)
    for _ in range(5):
        assert env.call().status_code == 200
    assert env.redis.pipelines == 0 and env.redis.counts == {}
    assert env.loader.asked == []


def test_under_the_limit_passes(monkeypatch):
    env = _Env(monkeypatch, _limited(3))
    assert [env.call().status_code == 200 for _ in range(3)] == [True, True, True]
    assert list(env.redis.counts.values()) == [3]
    assert env.loader.asked == ["inst_a", "inst_a", "inst_a"]


@pytest.mark.parametrize("route", FETCH_ROUTES, ids=[r[0] for r in FETCH_ROUTES])
def test_over_the_limit_is_429_with_retry_after_on_every_fetch_route(monkeypatch, route):
    env = _Env(monkeypatch, _limited(2))
    assert env.call(route).status_code != 429
    assert env.call(route).status_code != 429
    resp = env.call(route)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "50"  # 10s into the minute
    detail = resp.json()["detail"]
    assert "limit of 2 per minute" in detail and "Retry in 50 seconds" in detail


def test_the_four_routes_share_one_count(monkeypatch):
    env = _Env(monkeypatch, _limited(3))
    codes = [env.call(r).status_code == 429 for r in FETCH_ROUTES]
    assert codes == [False, False, False, True]


def test_the_window_resets_next_minute(monkeypatch):
    env = _Env(monkeypatch, _limited(1))
    assert env.call().status_code == 200
    assert env.call().status_code == 429
    env.now += 60
    assert env.call().status_code == 200
    assert len(env.redis.counts) == 2
    assert set(env.redis.ttls.values()) == {120}


def test_retry_after_is_the_seconds_left_in_the_minute(monkeypatch):
    env = _Env(monkeypatch, _limited(1))
    env.now = 1_000_000 * 60 + 59.5
    env.call()
    assert env.call().headers["Retry-After"] == "1"
    env.now = 1_000_001 * 60 + 0.0  # first second of the next minute
    env.call()
    assert env.call().headers["Retry-After"] == "60"


def test_two_instances_do_not_share_a_counter(monkeypatch):
    env = _Env(monkeypatch, _limited(1))
    assert env.call().status_code == 200
    assert env.call().status_code == 429
    env.instance_id = "inst_b"
    assert env.call().status_code == 200
    assert len(env.redis.counts) == 2


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_config(), id="nothing_written"),
        pytest.param(_config(guardrails={"rate_limits": {"max_memories_per_user_per_day": 5}}), id="other_rate_limit_only"),
        pytest.param(_limited(0), id="zero"),
        pytest.param(_limited(-4), id="negative"),
        pytest.param(None, id="no_config_at_all"),
    ],
)
def test_setting_absent_or_zero_means_no_limit(monkeypatch, config):
    env = _Env(monkeypatch, config)
    assert all(env.call().status_code == 200 for _ in range(5))
    assert env.redis.pipelines == 0


def test_redis_down_allows_the_call_and_warns(monkeypatch, caplog):
    env = _Env(monkeypatch, _limited(1), redis_down=True)
    with caplog.at_level("WARNING", logger=rl.logger.name):
        assert all(env.call().status_code == 200 for _ in range(3))
    assert any("retrieval_rate_limit_redis_failed" in r.message for r in caplog.records)


def test_a_config_read_failure_allows_the_call(monkeypatch):
    env = _Env(monkeypatch, _limited(1), loader_fails=True)
    assert all(env.call().status_code == 200 for _ in range(3))
    assert env.redis.pipelines == 0


def test_the_per_instance_switch_works_for_a_named_instance(monkeypatch):
    env = _Env(monkeypatch, _limited(1), switch=False)
    monkeypatch.setenv("SYNAP_GUARDRAIL_RETRIEVAL_RATE_INSTANCES", "inst_a")
    env.call()
    assert env.call().status_code == 429
    env.instance_id = "inst_other"
    assert env.call().status_code == 200
    assert env.call().status_code == 200
