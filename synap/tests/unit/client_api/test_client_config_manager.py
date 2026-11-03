"""Tests for ClientConfigManager."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from synap.cloud.application.client_api.client_config_manager import ClientConfigManager
from synap.cloud.application.models.errors import NotFoundError, ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config_row(config_key: str, config_value: dict, version: str = "1"):
    """Create a mock ClientConfigORM row."""
    row = MagicMock()
    row.config_key = config_key
    row.config_value = config_value
    row.version = version
    return row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.get_all_for_client = AsyncMock(return_value=[])
    repo.get_all_for_instance = AsyncMock(return_value=[])
    repo.get_by_key = AsyncMock(return_value=None)
    repo.upsert = AsyncMock()
    repo.delete_by_key = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def manager(mock_repo):
    session = AsyncMock()
    mgr = ClientConfigManager(session)
    mgr._repo = mock_repo
    return mgr


# ===========================================================================
# get_config
# ===========================================================================

class TestGetConfig:
    @pytest.mark.asyncio
    async def test_client_level_only(self, manager, mock_repo):
        mock_repo.get_all_for_client.return_value = [
            _make_config_row("theme", {"value": "dark"}),
            _make_config_row("locale", {"value": "en"}),
        ]

        result = await manager.get_config("cli_abc123")

        assert result == {
            "theme": {"value": "dark"},
            "locale": {"value": "en"},
        }
        mock_repo.get_all_for_client.assert_awaited_once_with("cli_abc123")
        mock_repo.get_all_for_instance.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_instance_merge_overrides_client(self, manager, mock_repo):
        mock_repo.get_all_for_client.return_value = [
            _make_config_row("theme", {"value": "light"}),
            _make_config_row("locale", {"value": "en"}),
        ]
        mock_repo.get_all_for_instance.return_value = [
            _make_config_row("theme", {"value": "dark"}),
        ]

        result = await manager.get_config("cli_abc123", instance_id="inst_xyz789")

        assert result["theme"] == {"value": "dark"}
        assert result["locale"] == {"value": "en"}

    @pytest.mark.asyncio
    async def test_empty_config(self, manager, mock_repo):
        result = await manager.get_config("cli_abc123")
        assert result == {}

    @pytest.mark.asyncio
    async def test_validation_empty_client_id(self, manager):
        with pytest.raises(ValidationError):
            await manager.get_config("")


# ===========================================================================
# update_config
# ===========================================================================

class TestUpdateConfig:
    @pytest.mark.asyncio
    async def test_upsert_multiple_keys(self, manager, mock_repo):
        mock_repo.get_all_for_client.return_value = [
            _make_config_row("theme", {"value": "dark"}),
            _make_config_row("locale", {"value": "fr"}),
        ]

        result = await manager.update_config(
            "cli_abc123",
            {"theme": {"value": "dark"}, "locale": {"value": "fr"}},
        )

        assert mock_repo.upsert.await_count == 2
        assert result["theme"] == {"value": "dark"}
        assert result["locale"] == {"value": "fr"}

    @pytest.mark.asyncio
    async def test_upsert_with_instance_id(self, manager, mock_repo):
        mock_repo.get_all_for_client.return_value = []
        mock_repo.get_all_for_instance.return_value = [
            _make_config_row("key1", "val1"),
        ]

        await manager.update_config(
            "cli_abc123",
            {"key1": "val1"},
            instance_id="inst_xyz789",
        )

        mock_repo.upsert.assert_awaited_once_with(
            client_id="cli_abc123",
            config_key="key1",
            config_value="val1",
            instance_id="inst_xyz789",
        )

    @pytest.mark.asyncio
    async def test_validation_empty_config(self, manager):
        with pytest.raises(ValidationError):
            await manager.update_config("cli_abc123", {})

    @pytest.mark.asyncio
    async def test_validation_empty_client_id(self, manager):
        with pytest.raises(ValidationError):
            await manager.update_config("", {"k": "v"})


# ===========================================================================
# store_config
# ===========================================================================

class TestStoreConfig:
    @pytest.mark.asyncio
    async def test_store_single_key(self, manager, mock_repo):
        await manager.store_config("cli_abc123", "theme", {"value": "dark"})

        mock_repo.upsert.assert_awaited_once_with(
            client_id="cli_abc123",
            config_key="theme",
            config_value={"value": "dark"},
            instance_id=None,
        )

    @pytest.mark.asyncio
    async def test_store_with_instance_id(self, manager, mock_repo):
        await manager.store_config(
            "cli_abc123", "theme", {"value": "dark"}, instance_id="inst_xyz789"
        )

        mock_repo.upsert.assert_awaited_once_with(
            client_id="cli_abc123",
            config_key="theme",
            config_value={"value": "dark"},
            instance_id="inst_xyz789",
        )

    @pytest.mark.asyncio
    async def test_validation_empty_key(self, manager):
        with pytest.raises(ValidationError):
            await manager.store_config("cli_abc123", "", "value")

    @pytest.mark.asyncio
    async def test_validation_empty_client_id(self, manager):
        with pytest.raises(ValidationError):
            await manager.store_config("", "key", "value")


# ===========================================================================
# delete_config
# ===========================================================================

class TestDeleteConfig:
    @pytest.mark.asyncio
    async def test_delete_existing_key(self, manager, mock_repo):
        mock_repo.delete_by_key.return_value = True

        await manager.delete_config("cli_abc123", "theme")

        mock_repo.delete_by_key.assert_awaited_once_with(
            client_id="cli_abc123",
            config_key="theme",
            instance_id=None,
        )

    @pytest.mark.asyncio
    async def test_delete_with_instance_id(self, manager, mock_repo):
        mock_repo.delete_by_key.return_value = True

        await manager.delete_config("cli_abc123", "theme", instance_id="inst_xyz789")

        mock_repo.delete_by_key.assert_awaited_once_with(
            client_id="cli_abc123",
            config_key="theme",
            instance_id="inst_xyz789",
        )

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises_not_found(self, manager, mock_repo):
        mock_repo.delete_by_key.return_value = False

        with pytest.raises(NotFoundError):
            await manager.delete_config("cli_abc123", "nonexistent")

    @pytest.mark.asyncio
    async def test_validation_empty_key(self, manager):
        with pytest.raises(ValidationError):
            await manager.delete_config("cli_abc123", "")

    @pytest.mark.asyncio
    async def test_validation_empty_client_id(self, manager):
        with pytest.raises(ValidationError):
            await manager.delete_config("", "key")
