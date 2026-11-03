"""Every Chroma collection we create must ask for a durable index.

Chroma writes a collection's HNSW index to disk only once the collection has
accumulated `hnsw:sync_threshold` vectors. Until then the index exists solely
in Chroma's in-memory pool. That pool defaults to 65,536 entries and so never
evicts at our scale, which is why nothing has ever been lost, and it is also
why memory grows with every collection touched and never comes back.

Measured on the Chroma image in use at dim 1536: a collection is readable
after being evicted from the pool only if it holds at least the threshold many
vectors. Typical collections are small, with a median of a few vectors and a
25th percentile of 1, so 2 (the lowest value Chroma's validator accepts) is the
only setting that covers the fleet.

These tests pin the behaviour to the call the adapter actually makes, rather
than to the source text, so they still fail if the key is renamed or moved.
"""

from typing import Any, Dict, List, Optional

import pytest

from synap.providers.vector.adapters.chromadb import (
    _HNSW_SYNC_THRESHOLD,
    ChromaDBAdapter,
)

SYNC_THRESHOLD_KEY = "hnsw:sync_threshold"

# Chroma's own validator refuses anything below this.
CHROMA_MINIMUM_THRESHOLD = 2

# From a sample of real collections. A threshold above
# the 25th percentile leaves a quarter of the fleet with no index on disk.
PROD_COLLECTION_SIZE_P25 = 1


class _RecordingClient:
    """Stands in for a chromadb client and remembers what it was asked for."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None,
                                 **kwargs: Any) -> Any:
        self.calls.append({"name": name, "metadata": metadata or {}})
        return object()


def _adapter_with(client: _RecordingClient) -> ChromaDBAdapter:
    adapter = ChromaDBAdapter.__new__(ChromaDBAdapter)
    adapter._use_fallback_memory = False
    adapter._collection_cache = {}
    adapter._get_client = lambda: client  # type: ignore[assignment]
    return adapter


def _metadata_for_a_new_collection() -> Dict[str, Any]:
    client = _RecordingClient()
    _adapter_with(client).create_namespace("some_tenant_scope", dimension=1536)
    assert client.calls, (
        "create_namespace made no collection call at all, so the assertions "
        "below would pass without checking anything."
    )
    return client.calls[0]["metadata"]


def test_new_collections_ask_for_a_durable_index() -> None:
    metadata = _metadata_for_a_new_collection()
    assert SYNC_THRESHOLD_KEY in metadata, (
        f"create_namespace did not pass {SYNC_THRESHOLD_KEY}. Without it Chroma "
        "uses 1000, and a collection below that size has no index on disk at "
        "all: it is readable only while it sits in the in-memory pool."
    )


def test_threshold_is_low_enough_to_cover_the_fleet() -> None:
    metadata = _metadata_for_a_new_collection()
    threshold = metadata[SYNC_THRESHOLD_KEY]
    assert isinstance(threshold, int), f"{SYNC_THRESHOLD_KEY} must be an integer"
    assert threshold >= CHROMA_MINIMUM_THRESHOLD, (
        f"Chroma's validator refuses a threshold below {CHROMA_MINIMUM_THRESHOLD}; "
        f"the collection would be rejected at create time."
    )
    assert threshold <= PROD_COLLECTION_SIZE_P25 + 1, (
        f"a threshold of {threshold} leaves every collection smaller than it "
        "without an index on disk. Production's 25th percentile is "
        f"{PROD_COLLECTION_SIZE_P25} vectors, so anything above "
        f"{PROD_COLLECTION_SIZE_P25 + 1} strands a quarter of the fleet."
    )


def test_the_constant_and_the_call_agree() -> None:
    """A constant nobody reads is the failure mode this catches."""
    metadata = _metadata_for_a_new_collection()
    assert metadata[SYNC_THRESHOLD_KEY] == _HNSW_SYNC_THRESHOLD, (
        "the value sent to Chroma does not match _HNSW_SYNC_THRESHOLD, so "
        "editing the constant would not change what production does."
    )


def test_the_existing_hnsw_settings_are_still_sent() -> None:
    """The new key must not have displaced the ones already relied on."""
    metadata = _metadata_for_a_new_collection()
    for key in ("dimension", "hnsw:space", "hnsw:search_ef"):
        assert key in metadata, f"{key} is no longer passed at collection create"
    assert metadata["hnsw:space"] == "cosine"


def test_fallback_memory_mode_makes_no_collection_call() -> None:
    client = _RecordingClient()
    adapter = _adapter_with(client)
    adapter._use_fallback_memory = True
    adapter._memory_namespaces = {}
    adapter.create_namespace("scope", dimension=1536)
    assert client.calls == [], "fallback mode should not touch the chroma client"
