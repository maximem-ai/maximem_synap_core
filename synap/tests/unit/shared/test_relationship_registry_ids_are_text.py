"""asyncpg takes the instance id of the relationship type registry as text, not as a UUID (found at the first Docker boot)."""
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap.cloud.database.repositories.relationship_type_registry_repo import RelationshipTypeRegistryRepository


def _repo():
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock())
    return RelationshipTypeRegistryRepository(session), session


async def test_upsert_binds_the_instance_id_as_text():
    repo, session = _repo()
    instance = uuid.uuid4()
    await repo.upsert(instance, "prefers", [0.1, 0.2])
    params = session.execute.await_args.args[1]
    assert params["instance_id"] == str(instance) and isinstance(params["instance_id"], str)
    assert params["relationship_type"] == "PREFERS"
