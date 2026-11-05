"""Stage 4 applies the graph releases KU queued only AFTER the document's graph
write (dossier 5.2 ordering note), and never when that write failed."""
from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

from synap.cloud.agents.ingestion.pipeline import stage4_organization
from synap.cloud.agents.ingestion.pipeline.stage4_organization import Stage4Organizer


class FakeStorage:
    def __init__(self):
        self.released = []
        self._pending_graph_releases = {"ing1": ["queued"]}

    async def release_pending_graph(self, ingestion_id):
        self.released.append(ingestion_id)
        self._pending_graph_releases.pop(ingestion_id, None)
        return 1


def _organizer(storage):
    org = Stage4Organizer.__new__(Stage4Organizer)
    org._storage = storage
    return org


async def test_release_runs_after_a_successful_graph_write():
    storage = FakeStorage()
    await _organizer(storage)._release_graph_after_write(
        SimpleNamespace(entity_ids=[], relationship_ids=[]), "ing1")
    assert storage.released == ["ing1"]


async def test_failed_graph_write_leaves_releases_to_the_outbox():
    storage = FakeStorage()
    await _organizer(storage)._release_graph_after_write(RuntimeError("neo4j down"), "ing1")
    assert storage.released == []
    # Dropped from memory (the outbox rows stay pending for the worker).
    assert storage._pending_graph_releases == {}


async def test_release_error_does_not_fail_ingestion():
    storage = FakeStorage()

    async def boom(ingestion_id):
        raise RuntimeError("graph down")

    storage.release_pending_graph = boom
    await _organizer(storage)._release_graph_after_write(SimpleNamespace(), "ing1")


def test_hook_sits_after_the_graph_write_and_after_ku():
    """In organize_and_persist: KU (which queues releases) comes first, then the
    graph write is awaited, then the hook. Never before the graph write."""
    src = inspect.getsource(stage4_organization.Stage4Organizer.organize_and_persist)
    ku = src.index("execute_supersessions(")
    graph_write = src.index("self._persist_graph_safe(")
    awaited = src.index("await asyncio.gather(")
    hook = src.index("self._release_graph_after_write(graph_result")
    assert ku < graph_write < awaited < hook
    assert len(re.findall(r"_release_graph_after_write\(", src)) == 1
