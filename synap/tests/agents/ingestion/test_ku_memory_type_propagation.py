"""Verify the memory_type field on KU action dataclasses is
plumbed end-to-end from ChromaDB metadata to memory_actions rows.

The detector fills `memory_type` from the existing memory's ChromaDB metadata
when building action objects. The coordinator's three KU paths
(execute_supersessions / execute_skips / execute_enrichments) must then pass
that value into insert_memory_action, replacing the historical NULL.

This file is intentionally narrow and self-contained: it does NOT depend on
the deprecate-with-link or merge code so it can ship
independently.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    EnrichmentAction,
    SkipAction,
    SupersessionAction,
)
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim


_async = pytest.mark.asyncio


def _make_scope():
    from synap.cloud.agents.ingestion.models import Scope
    return Scope(
        client_id=uuid4(),
        instance_id=uuid4(),
        customer_id=uuid4(),
        user_id=uuid4(),
    )


def _make_coordinator(rdbms_mock: AsyncMock) -> StorageCoordinator:
    attach_retire_shim(rdbms_mock)
    return StorageCoordinator(
        rdbms=rdbms_mock,
        graph=None,
        vector=None,
        embedding_client=SimpleNamespace(dimension=768),
    )


def test_action_dataclasses_default_memory_type_to_none():
    """Backward compat: existing call sites that don't yet pass memory_type
    should still construct successfully and surface None."""
    s = SupersessionAction(
        old_memory_id="o", old_memory_content="o",
        new_memory_id="n", new_memory_content="n",
        reason="r",
    )
    n = SkipAction(
        new_memory_id="n", new_memory_content="n",
        existing_memory_id="e", existing_memory_content="e",
        reason="r",
    )
    e = EnrichmentAction(
        existing_memory_id="e", existing_memory_content="e",
        new_memory_id="n", new_memory_content="n",
        reason="r",
    )
    assert s.memory_type is None
    assert n.memory_type is None
    assert e.memory_type is None


@_async
async def test_supersession_carries_memory_type_to_changelog():
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True

    coord = _make_coordinator(rdbms)
    action = SupersessionAction(
        old_memory_id=str(uuid4()),
        old_memory_content="prefers tea",
        new_memory_id=str(uuid4()),
        new_memory_content="prefers coffee",
        reason="updated taste",
        memory_type="preferences",
    )

    await coord.execute_supersessions(
        actions=[action],
        scope=_make_scope(),
        http_correlation_id="req_a",
        ingestion_id="ing-a",
    )

    rdbms.insert_memory_action.assert_awaited_once()
    kw = rdbms.insert_memory_action.await_args.kwargs
    assert kw["action"] == "SUPERSEDE"
    assert kw["memory_type"] == "preferences"


@_async
async def test_skip_carries_memory_type_to_changelog():
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True

    coord = _make_coordinator(rdbms)
    action = SkipAction(
        new_memory_id=str(uuid4()),
        new_memory_content="duplicate of existing",
        existing_memory_id=str(uuid4()),
        existing_memory_content="canonical version",
        reason="exact duplicate",
        memory_type="facts",
    )

    await coord.execute_skips(
        actions=[action],
        scope=_make_scope(),
        http_correlation_id="req_b",
        ingestion_id="ing-b",
    )

    rdbms.insert_memory_action.assert_awaited_once()
    kw = rdbms.insert_memory_action.await_args.kwargs
    assert kw["action"] == "NONE"
    assert kw["memory_type"] == "facts"


@_async
async def test_enrichment_carries_memory_type_to_changelog():
    rdbms = AsyncMock()

    coord = _make_coordinator(rdbms)
    action = EnrichmentAction(
        existing_memory_id=str(uuid4()),
        existing_memory_content="yoga 2x/week",
        new_memory_id=str(uuid4()),
        new_memory_content="yoga 2x/week including hot yoga on Saturdays",
        reason="adds qualifier",
        memory_type="preferences",
    )

    await coord.execute_enrichments(
        actions=[action],
        scope=_make_scope(),
        http_correlation_id="req_c",
        ingestion_id="ing-c",
    )

    rdbms.insert_memory_action.assert_awaited_once()
    kw = rdbms.insert_memory_action.await_args.kwargs
    assert kw["action"] == "ENRICH"
    assert kw["memory_type"] == "preferences"


@_async
async def test_missing_memory_type_falls_through_as_none():
    """Defensive: actions constructed without memory_type (e.g. unit tests
    or legacy call sites) still write a row, just with memory_type=None.
    The schema allows NULL so this must not blow up the changelog write."""
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True

    coord = _make_coordinator(rdbms)
    action = SupersessionAction(
        old_memory_id=str(uuid4()),
        old_memory_content="x",
        new_memory_id=str(uuid4()),
        new_memory_content="y",
        reason="r",
    )

    await coord.execute_supersessions(
        actions=[action],
        scope=_make_scope(),
        http_correlation_id="req_d",
        ingestion_id="ing-d",
    )

    kw = rdbms.insert_memory_action.await_args.kwargs
    assert kw["memory_type"] is None
