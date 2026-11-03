"""Regression tests for detached analytics/metering writes.

Two retrieval-path fixes moved billing/telemetry DB writes OFF the response
critical path via fire-and-forget tasks, while guaranteeing the metering still
runs and can never fail the read it measures:

  * ``retrieval_manager._run_detached``  — fetch metering + credit charge.
  * ``providers._schedule_llm_usage``    — per-LLM-call usage tracking.

These tests lock in the two safety properties of that change:
  1. The work does NOT execute inline on the caller's path (detached).
  2. It still runs to completion with the correct arguments, is strong-ref'd
     while pending (not GC'd), and swallows its own errors so a metering
     failure never propagates into the retrieval/LLM result.
"""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import synap.cloud.application.client_api.retrieval_manager as RM
import synap.cloud.application.shared.providers as providers
from synap.cloud.shared.scope import ScopeHierarchy


# === retrieval_manager._run_detached ===


async def test_run_detached_runs_off_the_critical_path():
    """The coroutine must not execute inline, but must still run once drained."""
    before = set(RM._DETACHED_TASKS)
    ran = {"n": 0}

    async def work():
        ran["n"] += 1

    RM._run_detached(work())

    # create_task schedules but does not run the coroutine before the next await,
    # so the caller returns without paying the metering cost.
    assert ran["n"] == 0, "detached coroutine must not run inline on the caller's path"

    scheduled = list(RM._DETACHED_TASKS - before)
    assert len(scheduled) == 1, "pending task must be strong-ref'd so it is not GC'd"

    await asyncio.gather(*scheduled)
    assert ran["n"] == 1, "detached metering must still run to completion"

    # done-callback discards the strong ref; let it fire before asserting no leak.
    await asyncio.sleep(0)
    assert set(RM._DETACHED_TASKS) == before, "task must be discarded after completion"


# === providers._schedule_llm_usage ===


async def test_schedule_llm_usage_detaches_and_forwards_kwargs():
    """Usage tracking is not awaited inline, but still runs with the exact kwargs."""
    before = set(providers._LLM_USAGE_TASKS)
    tracker = MagicMock()
    tracker.track_llm_usage = AsyncMock()

    kwargs = dict(
        client_id="cli_1",
        instance_id="inst_1",
        input_tokens=100,
        output_tokens=20,
        model="openai/gpt-oss-120b",
        operation="retrieval.llm_filter",
        correlation_id="corr-1",
    )

    providers._schedule_llm_usage(tracker, **kwargs)

    assert tracker.track_llm_usage.await_count == 0, "tracking must not be awaited inline"

    scheduled = list(providers._LLM_USAGE_TASKS - before)
    assert len(scheduled) == 1, "pending usage task must be strong-ref'd"

    await asyncio.gather(*scheduled)
    tracker.track_llm_usage.assert_awaited_once_with(**kwargs)


async def test_schedule_llm_usage_swallows_tracker_errors():
    """A telemetry failure must never propagate into the LLM result path."""
    before = set(providers._LLM_USAGE_TASKS)
    tracker = MagicMock()
    tracker.track_llm_usage = AsyncMock(side_effect=RuntimeError("telemetry db down"))

    # Scheduling itself must not raise.
    providers._schedule_llm_usage(
        tracker,
        client_id="cli_1",
        instance_id="inst_1",
        input_tokens=1,
        output_tokens=1,
        model="openai/gpt-oss-120b",
        operation="retrieval.llm",
        correlation_id="x",
    )

    scheduled = list(providers._LLM_USAGE_TASKS - before)
    results = await asyncio.gather(*scheduled, return_exceptions=True)

    assert all(not isinstance(r, Exception) for r in results), (
        "tracker error must be swallowed inside the detached task, not surfaced"
    )
    tracker.track_llm_usage.assert_awaited_once()


# === S1-step-2: the three retrieval arms detach their per-arm usage write ===
#
# Before this change each arm (vector / unified-graph / neo4j-discovery) did
# ``await self._track_*_queries(...)`` on its own response critical path — an
# INSERT+COMMIT the caller paid for even though ``latency_ms`` is computed
# BEFORE the call and nothing downstream reads the write. S1-step-2 routes each
# site through ``_run_detached`` (the same helper the context-fetch write on
# ``fetch_long_term_context`` already uses).
#
# Each test drives a real arm to a live track call site with ``_run_detached``
# stubbed to *capture, not schedule*, and asserts the arm handed the tracker
# coroutine to ``_run_detached`` instead of awaiting it inline — then drains the
# captured coroutine to prove the metering still runs with the right arguments.


def _scope() -> ScopeHierarchy:
    return ScopeHierarchy(
        instance_id="inst-1",
        client_id="cli-1",
        customer_id="cust-1",
        user_id="user-1",
        internal_user_id="internal-user-1",
    )


def _manager() -> RM.RetrievalManager:
    """A RetrievalManager whose arm trackers are AsyncMocks.

    Swapping the two ``_track_*`` coroutines lets us observe whether the arm
    awaits them inline (the old, blocking behaviour) or hands them to
    ``_run_detached`` (the fix) without touching the real usage tracker or its
    telemetry DB.
    """
    m = RM.RetrievalManager()
    m._track_vector_queries = AsyncMock()
    m._track_graph_queries = AsyncMock()
    return m


def _empty_vector_provider():
    """Vector provider returning an empty result set per query embedding."""
    vec = AsyncMock()

    async def _batch_search(embeddings, scope=None, top_k=None, filters=None, scope_level=None, **kw):
        return [[] for _ in embeddings]

    vec.batch_search = AsyncMock(side_effect=_batch_search)
    return vec


async def _assert_detached(track_mock, captured, expected_operation, **expected_kwargs):
    """Assert the arm scheduled exactly one telemetry write off its critical
    path (not awaited inline) and that it still runs with the right args."""
    assert len(captured) == 1, "arm must schedule exactly one detached telemetry write"
    assert asyncio.iscoroutine(captured[0]), "the detached payload must be the tracker coroutine"
    assert track_mock.await_count == 0, (
        "telemetry write must NOT be awaited on the arm's response critical path"
    )

    # Drain it: the metering still runs to completion with the exact arguments.
    await captured[0]
    assert track_mock.await_count == 1, "detached telemetry must still run once drained"
    kwargs = track_mock.call_args.kwargs
    assert kwargs["operation"] == expected_operation
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value, f"{key}={kwargs.get(key)!r} != expected {value!r}"


async def test_vector_arm_detaches_its_usage_write():
    """`_direct_similarity_search` must not block on its telemetry commit."""
    m = _manager()
    m.vector = _empty_vector_provider()
    scope = _scope()

    captured: list = []
    with patch.object(RM, "_run_detached", MagicMock(side_effect=captured.append)):
        await m._direct_similarity_search(
            scope=scope,
            queries=["hello world"],
            query_vectors=[np.array([0.1] * 8, dtype=np.float32)],
            context_types=["facts"],
        )

    await _assert_detached(
        m._track_vector_queries,
        captured,
        "retrieval.direct_similarity_search",
        scope=scope,
        query_count=1,
    )


async def test_unified_graph_arm_detaches_its_usage_write():
    """`_unified_graph_traversal` (success path) must detach its telemetry commit."""
    m = _manager()
    m.vector = object()  # non-None so the arm doesn't skip before tracking
    scope = _scope()

    captured: list = []
    with patch.object(RM, "_run_detached", MagicMock(side_effect=captured.append)):
        # Empty adjacency -> the arm's "no adjacency data" early-return path,
        # which still emits the (now detached) usage write.
        await m._unified_graph_traversal(
            scope=scope,
            queries=["hello"],
            query_vectors=[np.array([0.1] * 8, dtype=np.float32)],
            context_types=["facts"],
            adjacency={"entity_ids": [], "entity_matrix": None},
        )

    await _assert_detached(
        m._track_graph_queries,
        captured,
        "retrieval.unified_graph_traversal",
        scope=scope,
        query_count=1,
    )


async def test_unified_graph_arm_detaches_write_on_error():
    """The unified-graph except-branch write must detach too (status=error)."""
    m = _manager()
    m.vector = object()
    # adjacency=None -> the arm loads it -> this raises -> except-branch track.
    m._get_adjacency_matrix = AsyncMock(side_effect=RuntimeError("neo4j down"))
    scope = _scope()

    captured: list = []
    with patch.object(RM, "_run_detached", MagicMock(side_effect=captured.append)):
        await m._unified_graph_traversal(
            scope=scope,
            queries=["hello"],
            query_vectors=[np.array([0.1] * 8, dtype=np.float32)],
            context_types=["facts"],
        )

    await _assert_detached(
        m._track_graph_queries,
        captured,
        "retrieval.unified_graph_traversal",
        scope=scope,
        query_count=1,
        status="error",
    )


async def test_neo4j_arm_detaches_its_usage_write():
    """`_neo4j_graph_discovery` — the accurate-mode bottleneck — must detach."""
    m = _manager()
    # A single embedding-less node with no edges: enough to get past the
    # "no discovered nodes" early return and reach the (detached) usage write,
    # without building any ContextItems.
    node = types.SimpleNamespace(id="n1", node_type="Entity", properties={"canonical_name": "Foo"})
    res = types.SimpleNamespace(nodes=[node], edges=[])
    m.graph = MagicMock()
    m.graph.combined_name_traverse = AsyncMock(return_value=res)
    scope = _scope()

    captured: list = []
    with patch.object(RM, "_run_detached", MagicMock(side_effect=captured.append)):
        await m._neo4j_graph_discovery(
            scope=scope,
            anchor_names=["Foo"],
            query_vectors=[np.array([0.1] * 8, dtype=np.float32)],
        )

    await _assert_detached(
        m._track_graph_queries,
        captured,
        "retrieval.neo4j_graph_discovery",
        scope=scope,
        query_count=1,
    )
