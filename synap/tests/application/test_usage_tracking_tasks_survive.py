"""A usage-tracking task that is garbage-collected takes a billing row with it.

There are TWO usage-tracking schedulers and only one had this fix:

  providers._schedule_llm_usage   strong-ref'd via _LLM_USAGE_TASKS, and
                                  covered by test_detached_metering.py
  manager._fire_track_llm_usage   bare loop.create_task(...), no reference

`manager.complete()` is the path EVERY consolidation LLM call takes, so the
unfixed one is the one consolidation uses. The loop holds tasks only WEAKLY, so
such a task can be collected mid-write, its exception swallowed and nothing
logged. Those tasks are the billing record for every LLM call.

Failures must also be visible: under burst the telemetry pool can refuse
connections (TooManyConnectionsError).
Fire-and-forget is right for the answer path; losing the billing row silently
is not.
"""
import asyncio
import logging

import pytest

from synap.cloud.infrastructure.llm import manager as mgr


class _Tracker:
    def __init__(self, boom=None):
        self.boom = boom
        self.calls = 0

    async def track_llm_usage(self, **kw):
        self.calls += 1
        await asyncio.sleep(0)
        if self.boom:
            raise self.boom


def _fire(monkeypatch, tracker):
    m = mgr.LLMConfigManager.__new__(mgr.LLMConfigManager)
    monkeypatch.setattr(m, "_get_usage_tracker", lambda: tracker, raising=False)
    mgr.LLMConfigManager._fire_track_llm_usage(
        m, client_id="c", instance_id="i", input_tokens=1, output_tokens=1,
        model="gpt-4o-mini", operation="consolidation.nap.merge_eval",
        user_id=None, conversation_id=None, correlation_id="run-1")


@pytest.mark.asyncio
async def test_the_task_is_held_strongly_while_in_flight(monkeypatch):
    """The regression: an unreferenced task can be collected mid-write."""
    mgr._USAGE_TRACK_TASKS.clear()
    t = _Tracker()
    _fire(monkeypatch, t)
    assert len(mgr._USAGE_TRACK_TASKS) == 1, (
        "the in-flight usage task is referenced nowhere, so it can be "
        "garbage-collected mid-write and the billing row lost")
    await asyncio.sleep(0.01)
    assert t.calls == 1


@pytest.mark.asyncio
async def test_the_reference_is_released_when_it_finishes(monkeypatch):
    """Holding forever would leak one entry per LLM call the platform makes."""
    mgr._USAGE_TRACK_TASKS.clear()
    _fire(monkeypatch, _Tracker())
    await asyncio.sleep(0.01)
    assert mgr._USAGE_TRACK_TASKS == set()


@pytest.mark.asyncio
async def test_a_failed_usage_write_is_logged_not_swallowed(monkeypatch, caplog):
    """Uses the SAME log name as the other scheduler, so one grep finds both."""
    mgr._USAGE_TRACK_TASKS.clear()
    with caplog.at_level(logging.WARNING):
        _fire(monkeypatch, _Tracker(boom=RuntimeError("TooManyConnectionsError")))
        await asyncio.sleep(0.01)
    assert any("cloud_usage_track_llm_usage_failed" in r.message
               for r in caplog.records), [r.message for r in caplog.records]


@pytest.mark.asyncio
async def test_a_cancelled_usage_write_is_also_reported(monkeypatch, caplog):
    mgr._USAGE_TRACK_TASKS.clear()
    _fire(monkeypatch, _Tracker())
    task = next(iter(mgr._USAGE_TRACK_TASKS))
    with caplog.at_level(logging.WARNING):
        task.cancel()
        await asyncio.sleep(0.01)
    assert any("cancelled" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_scheduling_never_raises_into_the_llm_result(monkeypatch):
    """Whatever else changes, metering must not be able to fail an LLM call."""
    mgr._USAGE_TRACK_TASKS.clear()

    class Exploding:
        def track_llm_usage(self, **kw):
            raise RuntimeError("tracker blew up")

    _fire(monkeypatch, Exploding())          # must not raise
