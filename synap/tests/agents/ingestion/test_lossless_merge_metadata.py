"""Lossless metadata resolution for ENRICH-merged memories.

Three layers under test:
  1. resolve_merge_metadata — pure function, every field policy.
  2. APPEND_ONLY guardrail in synthesize_merges — episodes/emotions skip merge.
  3. execute_merges — verify the resolved fields land on insert_memory_action,
     vector upsert metadata, the SQL INSERT params, and entity-link copies.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    EnrichmentAction,
    KnowledgeUpdateDetector,
    MergeAction,
    resolve_merge_metadata,
)
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim


pytestmark = pytest.mark.asyncio


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(),
        instance_id=uuid4(),
        customer_id=uuid4(),
        user_id=uuid4(),
    )


def _t(month: int, year: int = 2023) -> datetime:
    return datetime(year, month, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1. resolve_merge_metadata — field-by-field
# ---------------------------------------------------------------------------


def test_resolve_memory_type_takes_modal():
    contributors = [
        {"memory_type": "preferences"},
        {"memory_type": "preferences"},
        {"memory_type": "facts"},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["primary_memory_type"] == "preferences"


def test_resolve_memory_type_uses_hierarchy_on_tie():
    # 1 of each → modal-tied → hierarchy says temporal_facts > facts > preferences
    contributors = [
        {"memory_type": "facts"},
        {"memory_type": "preferences"},
        {"memory_type": "temporal_facts"},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["primary_memory_type"] == "temporal_facts"


def test_resolve_category_returns_modal_only():
    contributors = [
        {"category": "career"},
        {"category": "career"},
        {"category": "hobbies"},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["category"] == "career"


def test_resolve_category_returns_none_on_tie():
    contributors = [{"category": "career"}, {"category": "hobbies"}]
    out = resolve_merge_metadata(contributors)
    assert out["category"] is None


def test_resolve_subcategory_only_among_matching_category():
    contributors = [
        {"category": "career", "subcategory": "job"},
        {"category": "career", "subcategory": "job"},
        {"category": "hobbies", "subcategory": "music"},  # ignored — wrong cat
    ]
    out = resolve_merge_metadata(contributors)
    assert out["category"] == "career"
    assert out["subcategory"] == "job"


def test_resolve_confidence_uses_independent_or():
    # 0.9 + 0.5 → 1 - (0.1 * 0.5) = 0.95
    contributors = [{"confidence": 0.9}, {"confidence": 0.5}]
    out = resolve_merge_metadata(contributors)
    assert out["confidence"] == pytest.approx(0.95, abs=1e-3)


def test_resolve_confidence_capped_at_99():
    # 0.99 + 0.99 + 0.99 would be ~0.9999... — cap at 0.99
    contributors = [{"confidence": 0.99}] * 3
    out = resolve_merge_metadata(contributors)
    assert out["confidence"] == 0.99


def test_resolve_temporal_category_episode_demotes_to_temporal_fact():
    contributors = [
        {"temporal_category": "episode"},
        {"temporal_category": "perpetual"},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["temporal_category"] == "temporal_fact"
    assert out["is_episode"] is False


def test_resolve_temporal_category_temporal_fact_wins_over_perpetual():
    contributors = [
        {"temporal_category": "perpetual"},
        {"temporal_category": "temporal_fact"},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["temporal_category"] == "temporal_fact"


def test_resolve_temporal_category_all_perpetual_stays_perpetual():
    contributors = [
        {"temporal_category": "perpetual"},
        {"temporal_category": "perpetual"},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["temporal_category"] == "perpetual"


def test_resolve_valid_from_is_min_across_contributors():
    contributors = [
        {"valid_from": _t(5)},
        {"valid_from": _t(1)},
        {"valid_from": _t(7)},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["valid_from"] == _t(1)


def test_resolve_valid_from_falls_back_to_event_date():
    contributors = [
        {"valid_from": None, "event_date": _t(3)},
        {"valid_from": None, "event_date": _t(1)},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["valid_from"] == _t(1)


def test_resolve_valid_until_null_if_any_unbounded():
    contributors = [
        {"valid_until": None},   # still valid → wins
        {"valid_until": _t(8)},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["valid_until"] is None


def test_resolve_valid_until_max_if_all_bounded():
    contributors = [
        {"valid_until": _t(6)},
        {"valid_until": _t(9)},
        {"valid_until": _t(3)},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["valid_until"] == _t(9)


def test_resolve_temporal_confidence_inherits_from_chosen_valid_from_contributor():
    earliest_with_high_conf = {"valid_from": _t(1), "temporal_confidence": 0.95}
    later_with_low_conf = {"valid_from": _t(5), "temporal_confidence": 0.2}
    out = resolve_merge_metadata([later_with_low_conf, earliest_with_high_conf])
    assert out["valid_from"] == _t(1)
    assert out["temporal_confidence"] == 0.95


def test_resolve_pii_obfuscated_or_across_contributors():
    out = resolve_merge_metadata([
        {"pii_obfuscated": False},
        {"pii_obfuscated": True},
    ])
    assert out["pii_obfuscated"] is True


def test_resolve_dct_range_min_and_max():
    contributors = [
        {"document_created_at": _t(7)},
        {"document_created_at": _t(1)},
        {"document_created_at": _t(5)},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["document_created_at"] == _t(1)
    assert out["document_last_observed_at"] == _t(7)


def test_resolve_extracted_at_is_min():
    contributors = [
        {"extracted_at": _t(7)},
        {"extracted_at": _t(2)},
    ]
    out = resolve_merge_metadata(contributors)
    assert out["extracted_at"] == _t(2)


def test_resolve_handles_all_none_gracefully():
    """Older contributors may have no metadata; resolver shouldn't crash."""
    out = resolve_merge_metadata([{}, {}])
    assert out["primary_memory_type"] is None
    assert out["category"] is None
    assert out["subcategory"] is None
    assert out["confidence"] == 0.0
    assert out["temporal_category"] == "perpetual"
    assert out["valid_from"] is None
    assert out["valid_until"] is None
    assert out["document_created_at"] is None
    assert out["pii_obfuscated"] is False


def test_resolve_empty_contributors_returns_empty_dict():
    assert resolve_merge_metadata([]) == {}


# ---------------------------------------------------------------------------
# 2. APPEND_ONLY guardrail in synthesize_merges
# ---------------------------------------------------------------------------


async def test_synthesize_merges_skips_clusters_with_episode_contributor():
    """Even with merge synthesis on, a cluster containing an episode/emotion
    must NOT produce a MergeAction — the type-aware hierarchy is the primary
    line of defense, this is the secondary guardrail when that flag is off."""
    detector = KnowledgeUpdateDetector(
        vector=AsyncMock(),
        llm_client=AsyncMock(),
    )

    # Two pairs forming a cluster of 3, where one member is an EPISODE.
    enrichments = [
        EnrichmentAction(
            existing_memory_id=str(uuid4()),
            existing_memory_content="user attended Concert A",
            new_memory_id=str(uuid4()),
            new_memory_content="user attended Concert A on Saturday",
            reason="adds detail",
            memory_type="episodes",         # ← APPEND_ONLY
            new_memory_type="facts",
        ),
    ]
    # No second pair — this is just testing the guardrail on a tiny cluster.
    # synthesize_merges needs ≥ 2 members per cluster, so add a paired memory.
    enrichments.append(
        EnrichmentAction(
            existing_memory_id=enrichments[0].new_memory_id,
            existing_memory_content="user attended Concert A on Saturday",
            new_memory_id=str(uuid4()),
            new_memory_content="user attended Concert A on the Saturday after work",
            reason="adds when",
            memory_type="facts",
            new_memory_type="facts",
        ),
    )

    out = await detector.synthesize_merges(enrichments=enrichments, scope=_scope())
    assert out == [], "episodes must not merge"


async def test_synthesize_merges_proceeds_when_no_append_only_present():
    """Sanity: facts + preferences cluster goes through the merge pipeline.
    We mock _call_merge_llm so the test doesn't require an LLM."""
    detector = KnowledgeUpdateDetector(
        vector=AsyncMock(),
        llm_client=AsyncMock(),
    )

    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    enrichments = [
        EnrichmentAction(
            existing_memory_id=a,
            existing_memory_content="user enjoys hiking on weekends",
            new_memory_id=b,
            new_memory_content="user enjoys hiking with friends every weekend",
            reason="adds friends",
            memory_type="preferences",
            new_memory_type="preferences",
        ),
        EnrichmentAction(
            existing_memory_id=b,
            existing_memory_content="user enjoys hiking with friends every weekend",
            new_memory_id=c,
            new_memory_content="user enjoys hiking with friends every weekend, especially in the Bay Area",
            reason="adds where",
            memory_type="preferences",
            new_memory_type="preferences",
        ),
    ]

    long_merged = (
        "user enjoys hiking with friends on weekends, especially in the Bay Area, "
        "as a recurring social and outdoor activity"
    )
    detector._call_merge_llm = AsyncMock(return_value={
        "merged_content": long_merged,
        "rationale": "combined frequency, company, location",
    })

    out = await detector.synthesize_merges(enrichments=enrichments, scope=_scope())
    assert len(out) == 1
    assert out[0].merged_content == long_merged
    assert sorted(out[0].contributor_memory_ids) == sorted([a, b, c])


# ---------------------------------------------------------------------------
# 3. execute_merges with full resolved metadata flowing into INSERT/upsert
# ---------------------------------------------------------------------------


async def test_execute_merges_writes_resolved_fields_into_insert():
    """The merged memory's INSERT params must carry every resolved field
    (memory_type, category, confidence, temporal fields, DCT range, etc.)
    instead of the old hard-coded defaults. ALSO verifies the ChromaDB
    upsert hits the underlying provider via asyncio.to_thread (regression
    test: an earlier version called a non-existent vector.upsert method
    and silently dropped every merged memory)."""
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True
    # Simulate AsyncVectorWrapper: it has _get_namespace + _provider.upsert.
    # We DO NOT add a top-level .upsert because the production wrapper
    # doesn't have one — using one would mask the regression we just fixed.
    # provider.upsert is a SYNC callable because asyncio.to_thread runs
    # sync callables in a worker thread.
    provider = SimpleNamespace(upsert=MagicMock(return_value=1))
    vector = SimpleNamespace(
        _get_namespace=lambda scope: "synap_test_namespace",
        _provider=provider,
    )
    embed = SimpleNamespace(
        embed_batch=AsyncMock(return_value=[[0.1] * 768])
    )
    attach_retire_shim(rdbms)
    coord = StorageCoordinator(
        rdbms=rdbms, graph=None, vector=vector,
        embedding_client=SimpleNamespace(dimension=768),
    )

    contrib_a, contrib_b = str(uuid4()), str(uuid4())
    merge = MergeAction(
        contributor_memory_ids=[contrib_a, contrib_b],
        contributor_contents=["A content", "B content"],
        contributor_source_documents=["doc-a", "doc-b"],
        merged_content="merged richer content carrying every detail",
        rationale="combined A and B",
        # Resolved fields (would normally come from _attach_metadata_to_merges):
        primary_memory_type="preferences",
        category="hobbies",
        subcategory="dance",
        confidence=0.95,
        temporal_category="temporal_fact",
        valid_from=_t(1),
        valid_until=_t(8),
        temporal_confidence=0.85,
        is_episode=False,
        pii_obfuscated=True,
        document_created_at=_t(1),
        document_last_observed_at=_t(7),
        extracted_at=_t(1),
        contributor_entity_ids=["entity-1", "entity-2"],
    )

    # Patch the raw-SQL session block to capture INSERT params.
    captured = {}
    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, stmt, params):
            captured["params"] = params
            captured["sql"] = str(stmt)
        async def commit(self): pass
    fake_factory = lambda: _FakeSession()

    with patch(
        "synap.cloud.database.connection.get_session_factory",
        return_value=fake_factory,
    ):
        out = await coord.execute_merges(
            merges=[merge],
            scope=_scope(),
            embedding_client=embed,
            http_correlation_id="req_test",
            ingestion_id="ing-test",
        )

    # Resolved fields ended up in the INSERT.
    p = captured["params"]
    assert p["memory_type"] == "preferences"
    assert p["category"] == "hobbies"
    assert p["subcategory"] == "dance"
    assert p["confidence"] == 0.95
    assert p["temporal_category"] == "temporal_fact"
    assert p["valid_from"] == _t(1)
    assert p["valid_until"] == _t(8)
    assert p["temporal_confidence"] == 0.85
    assert p["is_episode"] is False
    assert p["pii_obfuscated"] is True
    assert p["document_created_at"] == _t(1)
    assert p["document_last_observed_at"] == _t(7)

    # Entity links were copied (one insert_entity_reference per contributor entity).
    assert rdbms.insert_entity_reference.await_count == 2
    entity_args = [c.kwargs["entity_id"] for c in rdbms.insert_entity_reference.await_args_list]
    assert sorted(entity_args) == ["entity-1", "entity-2"]

    # ChromaDB upsert metadata mirrors the resolved fields. The upsert is
    # routed through asyncio.to_thread(provider.upsert, namespace, [doc]),
    # so the SECOND positional arg is the doc list.
    assert provider.upsert.call_count == 1, "merged vector must reach provider.upsert"
    upsert_args = provider.upsert.call_args.args
    namespace, docs = upsert_args[0], upsert_args[1]
    assert namespace == "synap_test_namespace"
    chroma_meta = docs[0].metadata
    assert chroma_meta["memory_type"] == "preferences"
    assert chroma_meta["category"] == "hobbies"
    assert chroma_meta["confidence"] == 0.95
    assert chroma_meta["temporal_category"] == "temporal_fact"
    assert chroma_meta["valid_from"] == _t(1).isoformat()
    assert chroma_meta["valid_until"] == _t(8).isoformat()
    assert chroma_meta["document_created_at"] == _t(1).isoformat()
    assert chroma_meta["document_last_observed_at"] == _t(7).isoformat()
    assert chroma_meta["entities_mentioned"] == "entity-1,entity-2"

    # ENRICH action rows carry resolved memory_type.
    enrich_calls = [
        c for c in rdbms.insert_memory_action.await_args_list
        if c.kwargs.get("action") == "ENRICH"
    ]
    assert len(enrich_calls) == 2
    for c in enrich_calls:
        assert c.kwargs["memory_type"] == "preferences"

    # merged_records returned for HQE re-run carries the resolved fields too.
    assert len(out) == 1
    assert out[0]["memory_type"] == "preferences"
    assert sorted(out[0]["entity_ids"]) == ["entity-1", "entity-2"]
    assert out[0]["document_created_at"] == _t(1)


async def test_execute_merges_skips_deprecation_when_chroma_upsert_fails():
    """Regression test for the data-loss bug where every
    merge silently failed at the ChromaDB upsert step (calling a non-existent
    `vector.upsert` method) yet still went on to deprecate + hard-delete the
    contributors. Net effect was facts gone from retrieval.

    With the fix, if the merged vector can't land in ChromaDB, contributors
    must NOT be deprecated — they remain searchable until the next attempt."""
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True

    # Simulate a vector wrapper whose provider.upsert RAISES.
    provider = SimpleNamespace(
        upsert=MagicMock(side_effect=RuntimeError("chroma is down")),
    )
    vector = SimpleNamespace(
        _get_namespace=lambda scope: "synap_test_namespace",
        _provider=provider,
        delete_memory_vectors=AsyncMock(),
    )
    embed = SimpleNamespace(embed_batch=AsyncMock(return_value=[[0.1] * 768]))
    attach_retire_shim(rdbms)
    coord = StorageCoordinator(
        rdbms=rdbms, graph=None, vector=vector,
        embedding_client=SimpleNamespace(dimension=768),
    )

    contrib = str(uuid4())
    merge = MergeAction(
        contributor_memory_ids=[contrib, str(uuid4())],
        contributor_contents=["aaaaaaaa", "bbbbbbbb"],
        contributor_source_documents=[],
        merged_content="merged content longer than the longest contributor input",
        rationale="x",
        primary_memory_type="facts",
    )

    captured = {}
    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, stmt, params): captured["params"] = params
        async def commit(self): pass
    fake_factory = lambda: _FakeSession()

    with patch(
        "synap.cloud.database.connection.get_session_factory",
        return_value=fake_factory,
    ):
        await coord.execute_merges(
            merges=[merge],
            scope=_scope(),
            embedding_client=embed,
            http_correlation_id="req_y",
            ingestion_id="ing-y",
        )

    # The PG insert still went through (lock-step recovery on next merge attempt).
    assert "params" in captured
    # But contributors were NEITHER soft-deleted NOR hard-deleted from ChromaDB.
    rdbms.soft_delete_memory.assert_not_awaited()
    vector.delete_memory_vectors.assert_not_awaited()
    # And no ENRICH action rows were emitted (those would imply a successful merge).
    enrich_calls = [
        c for c in rdbms.insert_memory_action.await_args_list
        if c.kwargs.get("action") == "ENRICH"
    ]
    assert enrich_calls == []


async def test_execute_merges_falls_back_safely_with_unresolved_metadata():
    """If a caller forgets to call _attach_metadata_to_merges, the SQL must
    still produce a valid row (with safe defaults) rather than crashing."""
    rdbms = AsyncMock()
    rdbms.soft_delete_memory.return_value = True
    provider = SimpleNamespace(upsert=MagicMock(return_value=1))
    vector = SimpleNamespace(
        _get_namespace=lambda scope: "synap_test_namespace",
        _provider=provider,
        delete_memory_vectors=AsyncMock(),
    )
    attach_retire_shim(rdbms)
    coord = StorageCoordinator(
        rdbms=rdbms, graph=None, vector=vector,
        embedding_client=SimpleNamespace(dimension=768),
    )

    merge = MergeAction(
        contributor_memory_ids=[str(uuid4()), str(uuid4())],
        contributor_contents=["a", "b"],
        contributor_source_documents=[],
        merged_content="some merged content longer than the longest contributor input",
        rationale="x",
        # All metadata fields default to safe NULL/zero values.
    )

    captured = {}
    class _FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, stmt, params): captured["params"] = params
        async def commit(self): pass
    fake_factory = lambda: _FakeSession()

    embed = SimpleNamespace(embed_batch=AsyncMock(return_value=[[0.1] * 768]))
    with patch(
        "synap.cloud.database.connection.get_session_factory",
        return_value=fake_factory,
    ):
        out = await coord.execute_merges(
            merges=[merge],
            scope=_scope(),
            embedding_client=embed,
            http_correlation_id="req_x",
            ingestion_id="ing-x",
        )

    p = captured["params"]
    assert p["memory_type"] == "facts"        # default
    assert p["category"] is None
    assert p["temporal_category"] == "perpetual"
    assert p["pii_obfuscated"] is False
    assert p["valid_from"] is None
    assert len(out) == 1
