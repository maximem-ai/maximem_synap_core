"""Tests for KU v2 chunk-fetch fan-out across memory_source_documents.

Verifies that:
1. Merged memories (NULL source_document_id, N rows in memory_source_documents)
   get chunks from EVERY contributor doc, not zero.
2. Non-merged memories preserve current single-doc behavior.
3. Items whose item_id is not a UUID (graph items) silently fall back to
   their in-row source_document_id.
4. Chunks are deduplicated across siblings sharing a source.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from synap.cloud.agents.models.bundles import ContextBundle, ContextItem
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
from synap.cloud.shared.scope import ScopeHierarchy


pytestmark = pytest.mark.asyncio


def _scope() -> ScopeHierarchy:
    return ScopeHierarchy(
        instance_id=str(uuid4()),
        client_id=str(uuid4()),
        customer_id=str(uuid4()),
        user_id=str(uuid4()),
        internal_user_id="internal-user-1",
    )


def _bundle_with(items: list[ContextItem]) -> ContextBundle:
    return ContextBundle(
        bundle_id="b-test",
        decision_id="d-test",
        items_by_type={"facts": items},
    )


def _vector_returning(doc_to_chunks: dict[str, list[str]]) -> AsyncMock:
    vector = AsyncMock()

    async def _get_chunks(scope, source_document_id: str, limit: int = 50):
        return [{"content": c} for c in doc_to_chunks.get(source_document_id, [])]

    vector.get_chunks_by_document_id = AsyncMock(side_effect=_get_chunks)
    return vector


def _postgres_returning(rows: list[dict]) -> AsyncMock:
    pg = AsyncMock()
    pg.fetch_all = AsyncMock(return_value=rows)
    return pg


def _make_manager(*, vector=None, postgres=None) -> RetrievalManager:
    return RetrievalManager(
        vector_provider=vector,
        postgres_provider=postgres,
    )


async def test_lookup_source_documents_filters_non_uuid_ids():
    """Graph item_ids (often non-UUID) must not poison the ANY(uuid[]) cast."""
    pg = _postgres_returning([])
    manager = _make_manager(postgres=pg)

    # mix of valid UUIDs and a graph-style id
    valid_id = str(uuid4())
    result = await manager._lookup_source_documents_for_memories(
        [valid_id, "entity:abc", "neo4j-123", ""]
    )

    pg.fetch_all.assert_awaited_once()
    args = pg.fetch_all.await_args.args
    # the params dict should only contain the valid uuid
    assert args[1] == {"ids": [valid_id]}
    assert result == {}


async def test_lookup_source_documents_returns_empty_when_no_uuids():
    """Skip the SQL round-trip entirely when nothing to look up."""
    pg = _postgres_returning([])
    manager = _make_manager(postgres=pg)

    result = await manager._lookup_source_documents_for_memories(
        ["entity:abc", "graph-1"]
    )
    pg.fetch_all.assert_not_awaited()
    assert result == {}


async def test_lookup_source_documents_groups_rows_by_memory():
    """Merged memory has multiple rows — collapse into one entry."""
    mid = str(uuid4())
    pg = _postgres_returning([
        {"memory_id": mid, "source_document_id": "doc-a"},
        {"memory_id": mid, "source_document_id": "doc-b"},
        {"memory_id": mid, "source_document_id": "doc-c"},
    ])
    manager = _make_manager(postgres=pg)

    result = await manager._lookup_source_documents_for_memories([mid])
    assert sorted(result[mid]) == ["doc-a", "doc-b", "doc-c"]


async def test_chunk_enrichment_fans_out_for_merged_memory():
    """A merged memory with NULL source_document_id but 3 contributor docs
    must receive evidence drawn from those docs."""
    merged_id = str(uuid4())
    item = ContextItem(
        item_id=merged_id,
        content="user enjoys yoga 3x per week",
        context_type="facts",
        source="vector",
        # The merged memory leaves source_document_id NULL on purpose —
        # the join table is the source of truth.
        source_document_id=None,
    )

    pg = _postgres_returning([
        {"memory_id": merged_id, "source_document_id": "doc-a"},
        {"memory_id": merged_id, "source_document_id": "doc-b"},
    ])
    vector = _vector_returning({
        "doc-a": ["I started yoga twice a week last spring."],
        "doc-b": ["Now I'm doing yoga three times a week, it's been great."],
    })

    manager = _make_manager(vector=vector, postgres=pg)
    bundle = _bundle_with([item])

    await manager._enrich_with_chunk_evidence(
        bundle=bundle,
        search_queries=["yoga frequency"],
        scope=_scope(),
        max_items_to_enrich=3,
    )

    # Both contributor docs should have been fetched.
    fetched_docs = {
        call.kwargs["source_document_id"]
        for call in vector.get_chunks_by_document_id.await_args_list
    }
    assert fetched_docs == {"doc-a", "doc-b"}
    # And evidence should land on the item.
    assert item.source_evidence is not None
    assert len(item.source_evidence) >= 1


async def test_chunk_enrichment_falls_back_to_in_row_doc_when_msd_empty():
    """Non-merged legacy memory: the lookup returns no rows for it (e.g.
    pre-backfill straggler), so we should still fetch chunks via the
    in-row source_document_id."""
    item = ContextItem(
        item_id=str(uuid4()),
        content="legacy memory",
        context_type="facts",
        source="vector",
        source_document_id="legacy-doc",
    )

    pg = _postgres_returning([])  # nothing in the join table
    vector = _vector_returning({
        "legacy-doc": ["The legacy memory is about something."],
    })

    manager = _make_manager(vector=vector, postgres=pg)
    bundle = _bundle_with([item])

    await manager._enrich_with_chunk_evidence(
        bundle=bundle,
        search_queries=["legacy memory"],
        scope=_scope(),
    )

    fetched_docs = [
        call.kwargs["source_document_id"]
        for call in vector.get_chunks_by_document_id.await_args_list
    ]
    assert fetched_docs == ["legacy-doc"]
    assert item.source_evidence is not None


async def test_chunk_enrichment_skips_items_with_no_resolvable_doc():
    """Item with no source_document_id and no row in the join table — skip."""
    item = ContextItem(
        item_id=str(uuid4()),
        content="orphan",
        context_type="facts",
        source="vector",
        source_document_id=None,
    )
    pg = _postgres_returning([])
    vector = _vector_returning({})

    manager = _make_manager(vector=vector, postgres=pg)
    bundle = _bundle_with([item])

    await manager._enrich_with_chunk_evidence(
        bundle=bundle,
        search_queries=["query"],
        scope=_scope(),
    )

    vector.get_chunks_by_document_id.assert_not_awaited()
    assert item.source_evidence is None


async def test_chunk_enrichment_dedupes_chunk_across_siblings():
    """Two items sharing the same source doc must not both attach the same
    chunk text — the seen_chunks set should suppress duplicates."""
    doc = "shared-doc"
    a = ContextItem(
        item_id=str(uuid4()),
        content="memory A",
        context_type="facts",
        source="vector",
        source_document_id=doc,
    )
    b = ContextItem(
        item_id=str(uuid4()),
        content="memory B",
        context_type="facts",
        source="vector",
        source_document_id=doc,
    )

    pg = _postgres_returning([
        {"memory_id": a.item_id, "source_document_id": doc},
        {"memory_id": b.item_id, "source_document_id": doc},
    ])
    # Only one chunk in the doc — second item should not also attach it.
    vector = _vector_returning({doc: ["The only chunk in this document."]})

    manager = _make_manager(vector=vector, postgres=pg)
    bundle = _bundle_with([a, b])

    await manager._enrich_with_chunk_evidence(
        bundle=bundle,
        search_queries=["memory"],
        scope=_scope(),
    )

    enriched = [it for it in (a, b) if it.source_evidence]
    assert len(enriched) == 1, "exactly one sibling should win the shared chunk"
