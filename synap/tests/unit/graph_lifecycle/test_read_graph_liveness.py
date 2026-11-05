"""The accurate-mode graph arm serves a graph statement only while a live memory supports it.

Drives `_neo4j_graph_discovery` with a fake graph holding the 4b live data
(WORKS_AT Acme, HAS_ROLE data engineer, REPORTS_TO Priya, all in references
mode, plus one unbacked ABOUT edge) and a fake supporter loader in place of
Postgres. Before the read check, "the user works at Acme Corp" and "the user
has role data engineer" were served after the user resigned.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import SupporterRow
from synap.cloud.application.client_api import retrieval_manager as rm
from synap.cloud.application.client_api.retrieval_manager import (
    MACARetrievalSettings,
    RetrievalManager,
)
from synap.cloud.shared import memory_liveness
from synap.cloud.shared.scope import ScopeHierarchy
from synap.providers.graph.models import Edge, GraphResult, Node


CLIENT = str(UUID("11111111-1111-5111-8111-111111111111"))
CUST = str(UUID("22222222-2222-5222-8222-222222222222"))
ME = str(UUID("66666666-6666-5666-8666-666666666666"))
VEC = [1.0, 0.0, 0.0, 0.0]

M_3C1F = "3c1f4817-d3c8-4ad1-8fee-b26d9cfbe082"  # works as data engineer at Acme, SUPERSEDED by 0874
M_58BE = "58bee2aa-8733-4d93-b167-38e0d6210781"  # manager at Acme is Priya, live
M_6336 = "6336821f-d3b7-4068-901e-ff93bc331d65"  # started at Acme, live episode
M_0874 = "0874ae8a-d213-44a2-8b72-9bd5cc83ffed"  # resigned from Acme, live episode
M_DEAD = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"  # hard-deleted


def _d(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


ROWS = {
    M_3C1F: SupporterRow(id=M_3C1F, superseded_by=M_0874, temporal_category="temporal_fact",
                         document_created_at=_d(2026, 9, 6),
                         superseder_document_created_at=_d(2026, 11, 3),
                         superseder_event_start=_d(2026, 10, 26)),
    M_58BE: SupporterRow(id=M_58BE, temporal_category="temporal_fact", document_created_at=_d(2026, 9, 6)),
    M_6336: SupporterRow(id=M_6336, is_episode=True, temporal_category="episode",
                         document_created_at=_d(2026, 9, 6)),
    M_0874: SupporterRow(id=M_0874, is_episode=True, temporal_category="episode",
                         document_created_at=_d(2026, 11, 3)),
}

OWN = {"customer_id": CUST, "user_id": ME}


def _node(key, name, etype, memory_ids=None, **extra):
    props = {"canonical_name": name, "entity_type": etype, "entity_id": f"e-{key}",
             "embedding": list(VEC), **OWN, **extra}
    if memory_ids is not None:
        props["memory_ids"] = list(memory_ids)
    return Node(id=f"n-{key}", node_type=etype, properties=props)


def _edge(eid, a, b, etype, memory_ids=None, **extra):
    props = {"embedding": list(VEC), "confidence": 0.9, **OWN, **extra}
    if memory_ids is not None:
        props["memory_ids"] = list(memory_ids)
    return Edge(id=eid, from_id=f"n-{a}", to_id=f"n-{b}", edge_type=etype, properties=props)


def _graph_result():
    nodes = [
        _node("user", ME, "person", is_writer=True),
        _node("acme", "Acme Corp", "organization", memory_ids=[M_3C1F, M_58BE, M_6336, M_0874]),
        _node("priya", "Priya", "person", memory_ids=[M_58BE, M_6336]),
        _node("role", "data engineer", "role", memory_ids=[M_3C1F]),
        _node("doc4", "document number 4", "document"),
    ]
    edges = [
        _edge("edge-works-at", "user", "acme", "WORKS_AT", [M_3C1F, M_58BE, M_6336]),
        _edge("edge-has-role", "user", "role", "HAS_ROLE", [M_3C1F, M_6336]),
        _edge("edge-reports-to", "user", "priya", "REPORTS_TO", [M_58BE, M_6336]),
        _edge("edge-about", "doc4", "acme", "ABOUT"),
        _edge("edge-deleted-only", "priya", "role", "KNOWS", [M_DEAD]),
        _edge("edge-retired", "priya", "acme", "WORKS_AT", [M_58BE], retired_at="2026-11-05T00:00:00Z"),
    ]
    return GraphResult(nodes=nodes, edges=edges)


class _FakeGraph:
    def __init__(self, result):
        self._result = result

    async def combined_name_traverse(self, *, namespace, anchor_names, **kw):
        return self._result if namespace == CUST else GraphResult(nodes=[], edges=[])


def _manager():
    m = RetrievalManager(vector_provider=AsyncMock(), postgres_provider=AsyncMock(),
                         graph_provider=_FakeGraph(_graph_result()))
    m._current_maca = MACARetrievalSettings()
    m._track_graph_queries = AsyncMock()
    return m


def _scope():
    return ScopeHierarchy(instance_id=str(uuid4()), client_id=CLIENT, customer_id=CUST, user_id=ME)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_SERVE_RELATIONSHIPS", "true")
    monkeypatch.delenv("SYNAP_GRAPH_PER_SCOPE", raising=False)
    monkeypatch.delenv("SYNAP_GRAPH_NODE_MEMORIES", raising=False)
    monkeypatch.delenv("SYNAP_GRAPH_EDGE_MEMORY_IDS", raising=False)
    monkeypatch.setenv("SYNAP_GRAPH_SERVE_CHECKS_LIVENESS", "true")
    calls = []

    async def fake_loader(ids, session=None):
        calls.append(list(ids))
        return {i: ROWS[i] for i in ids if i in ROWS}

    monkeypatch.setattr(memory_liveness, "load_supporter_rows", fake_loader)
    return calls


async def _discover():
    faults: list = []
    token = rm._RETRIEVAL_FAULTS.set(faults)
    try:
        items = await _manager()._neo4j_graph_discovery(
            _scope(), ["Acme Corp"], [np.array(VEC, dtype=np.float32)])
    finally:
        rm._RETRIEVAL_FAULTS.reset(token)
    return items, faults


def _by_id(items):
    return {i.item_id: i for i in items}


async def test_stale_relationships_dropped_live_ones_served(env):
    items, faults = await _discover()
    got = _by_id(items)
    # Closed by T2 (the user resigned): not served.
    assert "edge-works-at" not in got
    assert "edge-has-role" not in got
    # Only support hard-deleted: closed by T1.
    assert "edge-deleted-only" not in got
    # Stored as retired by the write side: not served.
    assert "edge-retired" not in got
    # Positive controls in the same run: live and unbacked relationships served.
    assert "edge-reports-to" in got
    assert got["edge-reports-to"].graph_kind == "relationship"
    assert set(got["edge-reports-to"].backing_memory_ids) == {M_58BE, M_6336}
    assert "edge-about" in got
    assert got["edge-about"].backing_memory_ids == []
    assert faults == []
    # One liveness query for the whole discovery.
    assert len(env) == 1
    assert set(env[0]) >= {M_3C1F, M_58BE, M_6336, M_0874, M_DEAD}


async def test_entity_lines_follow_the_same_rule(env):
    items, _ = await _discover()
    got = _by_id(items)
    # "data engineer" was backed only by the superseded memory.
    assert "e-role" not in got
    # The writer node is always open; Acme and Priya have current support.
    assert "e-user" in got and got["e-user"].graph_kind == "entity"
    assert "e-acme" in got and "e-priya" in got
    assert set(got["e-priya"].backing_memory_ids) == {M_58BE, M_6336}
    # Unbacked node served.
    assert "e-doc4" in got


async def test_liveness_failure_drops_graph_statements_and_marks_degraded(env, monkeypatch):
    async def broken(ids, session=None):
        raise ConnectionError("postgres unreachable")

    monkeypatch.setattr(memory_liveness, "load_supporter_rows", broken)
    items, faults = await _discover()
    kinds = {i.graph_kind for i in items}
    assert "relationship" not in kinds and "entity" not in kinds
    assert any(f.startswith(rm._GRAPH_LIVENESS_FAULT) for f in faults)


async def test_flag_off_serves_as_before_and_never_asks(env, monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_SERVE_CHECKS_LIVENESS", "false")
    items, faults = await _discover()
    got = _by_id(items)
    assert {"edge-works-at", "edge-has-role", "edge-reports-to", "edge-about"} <= set(got)
    assert got["edge-works-at"].backing_memory_ids is None
    assert env == []
    assert faults == []


async def test_flag_defaults_on_only_in_core(monkeypatch):
    from synap.cloud.shared import edition
    monkeypatch.delenv("SYNAP_GRAPH_SERVE_CHECKS_LIVENESS", raising=False)
    monkeypatch.delenv("SYNAP_GRAPH_SERVE_CHECKS_LIVENESS_INSTANCES", raising=False)
    monkeypatch.setattr(edition, "is_core", True)
    assert rm._graph_serve_checks_liveness("i") is True
    monkeypatch.setattr(edition, "is_core", False)
    assert rm._graph_serve_checks_liveness("i") is False


async def test_legacy_copied_memories_filtered_by_liveness(env, monkeypatch):
    legacy = Edge(
        id="edge-legacy", from_id="n-user", to_id="n-acme", edge_type="WORKS_AT",
        properties={**OWN, "memory_contents": json.dumps([
            {"memory_id": M_3C1F, "content": "The user works as a data engineer at Acme Corp.",
             "memory_type": "facts", "embedding": list(VEC), "is_current": True, "is_deleted": False, **OWN},
            {"memory_id": M_58BE, "content": "The user's manager at Acme Corp is Priya.",
             "memory_type": "facts", "embedding": list(VEC), "is_current": True, "is_deleted": False, **OWN},
        ])},
    )
    result = _graph_result()
    result.edges.append(legacy)
    m = _manager()
    m.graph = _FakeGraph(result)
    faults: list = []
    token = rm._RETRIEVAL_FAULTS.set(faults)
    try:
        items = await m._neo4j_graph_discovery(_scope(), ["Acme Corp"], [np.array(VEC, dtype=np.float32)])
    finally:
        rm._RETRIEVAL_FAULTS.reset(token)
    got = _by_id(items)
    assert M_3C1F not in got  # superseded copy no longer served
    assert M_58BE in got and got[M_58BE].graph_kind == "memory"
    assert got[M_58BE].backing_memory_ids == [M_58BE]


async def test_liveness_fault_keeps_response_out_of_cache():
    m = RetrievalManager(vector_provider=AsyncMock(), postgres_provider=AsyncMock(), graph_provider=None)
    m.cache = AsyncMock()
    bundle = AsyncMock()
    bundle.item_count = 3
    bundle.degradation_level = None
    token = rm._RETRIEVAL_FAULTS.set([f"{rm._GRAPH_LIVENESS_FAULT} failed: x"])
    try:
        await m._cache_result("k", bundle)
    finally:
        rm._RETRIEVAL_FAULTS.reset(token)
    m.cache.set.assert_not_awaited()


def test_fast_mode_never_reaches_the_liveness_check():
    """Fast mode never runs graph discovery (use_graph requires accurate), and the
    liveness check is only ever constructed inside graph discovery."""
    import inspect
    src = inspect.getsource(rm)
    assert src.count("_GraphLiveness.load(") == 1
    discovery = inspect.getsource(RetrievalManager._neo4j_graph_discovery)
    assert "_GraphLiveness.load(" in discovery
    assert 'and mode == "accurate"' in inspect.getsource(rm)


async def test_released_ids_are_loaded_and_keep_a_released_edge_closed(env, monkeypatch):
    released = _edge("edge-released", "user", "acme", "EMPLOYED_BY", [M_58BE, M_6336],
                     released_memory_ids=[M_3C1F])
    result = _graph_result()
    result.edges.append(released)
    m = _manager()
    m.graph = _FakeGraph(result)
    items = await m._neo4j_graph_discovery(_scope(), ["Acme Corp"], [np.array(VEC, dtype=np.float32)])
    assert "edge-released" not in _by_id(items)
    assert "edge-reports-to" in _by_id(items)
    assert M_3C1F in env[-1]
