"""The batched chunk fetch must return exactly what the per-document one did.

This is a round-trip optimisation, not a behaviour change, so the test that
matters is equivalence: same documents, same chunks, same order, same
per-document cap. With ten documents, one `$in` query is roughly 8x faster than ten
concurrent single fetches, so the win is real; this
file exists to make sure the win is free.

The starvation guard is the subtle part and gets its own case. One global limit
can let a single document consume the whole allowance, which the per-document
shape can never do, so the wrapper falls back when the batched read comes back
full.
"""
from __future__ import annotations

import pytest

from synap.cloud.agents.ingestion.storage.vector_wrapper import AsyncVectorWrapper


class _Doc:
    def __init__(self, id_, content, meta):
        self.id = id_
        self.content = content
        self.metadata = meta


class _Provider:
    """Records what it was asked for, so the test can assert the round trips."""

    def __init__(self, rows, cap=None):
        self._rows = rows
        self.calls = []
        self.cap = cap

    def get_all(self, namespace, where, limit):
        self.calls.append((namespace, where, limit))
        key = where.get("source_document_id")
        if isinstance(key, dict) and "$in" in key:
            wanted = set(key["$in"])
        else:
            wanted = {key}
        out = [d for d in self._rows if d.metadata["source_document_id"] in wanted]
        n = self.cap if self.cap is not None else limit
        return out[:n]


def _wrapper(provider):
    w = AsyncVectorWrapper.__new__(AsyncVectorWrapper)
    w._provider = provider
    w._get_chunks_namespace = lambda scope: "ns_chunks"
    return w


def _rows():
    out = []
    for doc in ("d1", "d2", "d3"):
        for i in (2, 0, 1):          # deliberately out of order
            out.append(_Doc(f"{doc}-{i}", f"{doc} chunk {i}",
                            {"source_document_id": doc, "chunk_index": i}))
    return out


@pytest.mark.asyncio
async def test_batched_returns_the_same_as_one_call_per_document():
    ids = ["d1", "d2", "d3"]
    single = _wrapper(_Provider(_rows()))
    per_doc = {}
    for d in ids:
        per_doc[d] = await single.get_chunks_by_document_id(
            scope=None, source_document_id=d, limit=50)

    batched = _wrapper(_Provider(_rows()))
    got = await batched.get_chunks_by_document_ids(
        scope=None, source_document_ids=ids, limit_per_document=50)

    assert list(got) == ids
    for d in ids:
        assert [r["id"] for r in got[d]] == [r["id"] for r in per_doc[d]]
        assert [r["content"] for r in got[d]] == [r["content"] for r in per_doc[d]]


@pytest.mark.asyncio
async def test_it_really_is_one_round_trip():
    """The whole point. A batched call that still fans out has saved nothing."""
    p = _Provider(_rows())
    await _wrapper(p).get_chunks_by_document_ids(
        scope=None, source_document_ids=["d1", "d2", "d3"], limit_per_document=50)
    assert len(p.calls) == 1, f"expected one query, got {len(p.calls)}"
    where = p.calls[0][1]
    assert "$in" in where["source_document_id"]


@pytest.mark.asyncio
async def test_chunks_come_back_ordered_by_chunk_index():
    got = await _wrapper(_Provider(_rows())).get_chunks_by_document_ids(
        scope=None, source_document_ids=["d1"], limit_per_document=50)
    assert [r["metadata"]["chunk_index"] for r in got["d1"]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_the_per_document_cap_is_preserved():
    got = await _wrapper(_Provider(_rows())).get_chunks_by_document_ids(
        scope=None, source_document_ids=["d1", "d2"], limit_per_document=2)
    assert all(len(v) <= 2 for v in got.values())


@pytest.mark.asyncio
async def test_a_full_batched_read_falls_back_rather_than_truncating_silently():
    """One global limit can starve a document; the per-document shape cannot.

    When the batched read returns exactly what was asked for, truncation is
    possible and undetectable from the rows alone, so the wrapper must re-read
    per document instead of returning a quietly short answer.
    """
    ids = ["d1", "d2", "d3"]
    # cap the provider at exactly limit_per_document * len(ids)
    p = _Provider(_rows(), cap=3 * len(ids))
    got = await _wrapper(p).get_chunks_by_document_ids(
        scope=None, source_document_ids=ids, limit_per_document=3)
    assert len(p.calls) == 1 + len(ids), (
        "expected the batched attempt plus one fallback read per document, "
        f"got {len(p.calls)} calls")
    for d in ids:
        assert got[d], f"{d} came back empty after the fallback"


@pytest.mark.asyncio
async def test_no_documents_asks_nothing():
    p = _Provider(_rows())
    assert await _wrapper(p).get_chunks_by_document_ids(
        scope=None, source_document_ids=[], limit_per_document=50) == {}
    assert p.calls == []
