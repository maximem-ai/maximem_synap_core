"""StorageCoordinator.execute_deletions — Gap 6 soft-delete execution.

A deletion must: deprecate the target in RDBMS, drop it from the vector index
(so retrieval stops surfacing it now), and write a DELETE changelog row with
no replacement — never a hard destroy. Uses lightweight fakes; no DB/Chroma.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.pipeline.knowledge_update import DeletionAction
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim


class _FakeRDBMS:
    def __init__(self, deprecate_ok=True):
        self.deprecated: list = []
        self.actions: list = []
        self._ok = deprecate_ok

    async def set_deprecated_at(self, memory_id):
        self.deprecated.append(memory_id)
        return self._ok

    async def insert_memory_action(self, **kw):
        self.actions.append(kw)


class _FakeVector:
    def __init__(self):
        self.deleted: list = []

    async def delete_memory_vectors(self, memory_ids, scope):
        self.deleted.extend(memory_ids)


def _coordinator(rdbms, vector):
    # Bypass __init__ (needs graph/embedding we don't exercise here) and set
    # only the two collaborators execute_deletions touches.
    c = StorageCoordinator.__new__(StorageCoordinator)
    c._rdbms = rdbms
    attach_retire_shim(rdbms)
    c._vector = vector
    return c


def _action():
    return DeletionAction(
        old_memory_id="mem-old",
        old_memory_content="the user's home address is 12 Oak St",
        new_memory_id="mem-new",
        new_memory_content="please delete my saved home address",
        reason="explicit retraction",
    )


_SCOPE = Scope(
    client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(), user_id=uuid4()
)


class TestExecuteDeletions:
    async def test_empty_returns_zero(self):
        rdbms, vector = _FakeRDBMS(), _FakeVector()
        coord = _coordinator(rdbms, vector)
        assert await coord.execute_deletions(actions=[], scope=_SCOPE) == 0
        assert rdbms.deprecated == []
        assert vector.deleted == []

    async def test_deprecates_and_drops_vector(self):
        rdbms, vector = _FakeRDBMS(), _FakeVector()
        coord = _coordinator(rdbms, vector)
        n = await coord.execute_deletions(actions=[_action()], scope=_SCOPE)
        assert n == 1
        assert rdbms.deprecated == ["mem-old"]      # target deprecated
        assert vector.deleted == ["mem-old"]        # removed from retrieval

    async def test_changelog_row_is_delete_with_no_replacement(self):
        rdbms, vector = _FakeRDBMS(), _FakeVector()
        coord = _coordinator(rdbms, vector)
        await coord.execute_deletions(
            actions=[_action()], scope=_SCOPE, http_correlation_id="corr-1"
        )
        assert len(rdbms.actions) == 1
        row = rdbms.actions[0]
        assert row["action"] == "DELETE"
        assert row["memory_id"] == "mem-old"
        assert row["replaced_by_memory_id"] is None   # nothing replaces it
        assert row["content"] is None                 # deleted, no new text
        assert row["previous_content"] == "the user's home address is 12 Oak St"

    async def test_the_changelog_is_written_even_without_a_correlation_id(self):
        """This used to assert the opposite, and the opposite was the bug.

        memory_actions.correlation_id is NOT NULL, so an ingest arriving
        without a request id would have violated the constraint, and the
        answer was to skip the audit row. That made the trail thinnest exactly
        where the caller was least ordinary, and a retirement with no trail
        row is a memory no reversal can ever restore. One is synthesised now.
        """
        rdbms, vector = _FakeRDBMS(), _FakeVector()
        coord = _coordinator(rdbms, vector)
        await coord.execute_deletions(actions=[_action()], scope=_SCOPE)
        assert len(rdbms.actions) == 1, "the retirement left no way to undo it"
        row = rdbms.actions[0]
        assert row["action"] == "DELETE"
        assert row["previous_content"] == "the user's home address is 12 Oak St"

    async def test_deprecate_noop_still_drops_vector(self):
        # If the row was already deprecated, still remove the lingering vector.
        rdbms, vector = _FakeRDBMS(deprecate_ok=False), _FakeVector()
        coord = _coordinator(rdbms, vector)
        n = await coord.execute_deletions(actions=[_action()], scope=_SCOPE)
        assert n == 1
        assert vector.deleted == ["mem-old"]
