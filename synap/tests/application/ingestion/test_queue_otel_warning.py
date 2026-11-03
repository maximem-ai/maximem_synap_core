"""With no tracing set up (every Core install) the "no trace context" warning is true for every
document. It is said once per process, not once per document."""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.application.ingestion import queue as queue_module
from synap.cloud.application.ingestion.queue import IngestionQueue


def _redis():
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, 1])
    redis = MagicMock()
    redis.pipeline.return_value = pipe
    return redis


def _request():
    return SimpleNamespace(
        scope=SimpleNamespace(client_id=uuid4()), request_id="r",
        model_dump=lambda mode="json": {})


@pytest.mark.asyncio
async def test_the_missing_trace_context_is_warned_about_once(monkeypatch, caplog):
    monkeypatch.setattr(queue_module, "_warned_otel_context_empty", False)
    monkeypatch.setattr(queue_module.otel_propagate, "inject", lambda carrier: None)
    q = IngestionQueue(_redis())
    priority = next(iter(queue_module.IngestPriority))
    with caplog.at_level(logging.WARNING, logger=queue_module.logger.name):
        for _ in range(5):
            await q.enqueue(uuid4(), _request(), priority)
    warned = [r for r in caplog.records if "ingestion_queue_otel_context_empty" in r.getMessage()]
    assert len(warned) == 1


@pytest.mark.asyncio
async def test_a_present_trace_context_is_not_warned_about(monkeypatch, caplog):
    monkeypatch.setattr(queue_module, "_warned_otel_context_empty", False)
    monkeypatch.setattr(queue_module.otel_propagate, "inject", lambda carrier: carrier.update(traceparent="x"))
    q = IngestionQueue(_redis())
    priority = next(iter(queue_module.IngestPriority))
    with caplog.at_level(logging.WARNING, logger=queue_module.logger.name):
        await q.enqueue(uuid4(), _request(), priority)
    assert not [r for r in caplog.records if "otel_context_empty" in r.getMessage()]
