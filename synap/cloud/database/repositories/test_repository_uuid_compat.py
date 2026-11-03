from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid5, NAMESPACE_URL

import pytest

from synap.cloud.database.repositories.client_repo import ClientRepository
from synap.cloud.database.repositories.instance_repo import InstanceRepository


pytestmark = pytest.mark.asyncio


async def test_create_client_sets_deterministic_uuid_before_flush():
    session = SimpleNamespace(
        add=Mock(),
        flush=AsyncMock(),
        refresh=AsyncMock(),
    )
    repo = ClientRepository(session)

    client = await repo.create_client(
        client_id="cli_test123",
        client_name="Test Client",
        client_short_description="Test Client",
        client_context={"org_id": "org_123"},
    )

    session.add.assert_called_once()
    added_client = session.add.call_args.args[0]
    assert client is added_client
    assert isinstance(added_client.uuid, UUID)
    # UUID must be deterministic: uuid5(NAMESPACE_URL, client_id)
    assert added_client.uuid == uuid5(NAMESPACE_URL, "cli_test123")
    assert added_client.client_id == "cli_test123"
    session.flush.assert_awaited_once()
    session.refresh.assert_awaited_once_with(added_client)


async def test_create_instance_sets_deterministic_uuid_before_flush():
    session = SimpleNamespace(
        add=Mock(),
        flush=AsyncMock(),
        refresh=AsyncMock(),
    )
    repo = InstanceRepository(session)

    instance = await repo.create_instance(
        instance_id="inst_test123",
        client_id="cli_test123",
        instance_name="sdk-host",
        instance_metadata={"environment": "dev"},
        enforce_limit=False,  # this test is about the uuid; the limit has its own tests
    )

    session.add.assert_called_once()
    added_instance = session.add.call_args.args[0]
    assert instance is added_instance
    assert isinstance(added_instance.uuid, UUID)
    # UUID must be deterministic: uuid5(NAMESPACE_URL, instance_id)
    assert added_instance.uuid == uuid5(NAMESPACE_URL, "inst_test123")
    assert added_instance.instance_id == "inst_test123"
    session.flush.assert_awaited_once()
    session.refresh.assert_awaited_once_with(added_instance)
