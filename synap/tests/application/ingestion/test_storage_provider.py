"""The conversation-promotion path must back up what it ingests (it used a no-op stub)."""
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from synap.cloud.application.ingestion.storage_provider import FileStoreStorageProvider

SCOPE = SimpleNamespace(client_id="c", instance_id="i", customer_id="u")


def _provider(store):
    with patch("synap.cloud.application.ingestion.storage_provider.get_file_store",
               return_value=store):
        return FileStoreStorageProvider()


@pytest.mark.asyncio
async def test_persist_original_writes_text_to_store():
    store = SimpleNamespace(write=AsyncMock())
    iid = uuid.uuid4()
    await _provider(store).persist_original(iid, "user: hi", SCOPE)
    kw = store.write.await_args.kwargs
    assert kw["path"] == f"ingestion/c/i/u/{iid}.bin"
    assert kw["content"] == b"user: hi"


@pytest.mark.asyncio
async def test_persist_original_file_keeps_the_raw_bytes():
    store = SimpleNamespace(write=AsyncMock())
    iid = uuid.uuid4()
    await _provider(store).persist_original_file(iid, b"%PDF-1.7", "Report.PDF", SCOPE)
    kw = store.write.await_args.kwargs
    assert kw["path"] == f"ingestion/c/i/u/{iid}.original.pdf"
    assert kw["content"] == b"%PDF-1.7"


@pytest.mark.asyncio
async def test_unbuildable_file_store_is_loud_not_silent(caplog):
    from synap.cloud.application.ingestion.storage_provider import StorageUnavailableError
    with patch("synap.cloud.application.ingestion.storage_provider.get_file_store",
               side_effect=RuntimeError("no s3")), caplog.at_level("ERROR"):
        p = FileStoreStorageProvider()
    assert "ingestion_backup_unavailable" in caplog.text
    with pytest.raises(StorageUnavailableError):
        await p.persist_original(uuid.uuid4(), "x", SCOPE)
    with pytest.raises(StorageUnavailableError):
        await p.persist_original_file(uuid.uuid4(), b"x", "a.pdf", SCOPE)


@pytest.mark.asyncio
async def test_empty_bucket_is_caught_at_startup_not_at_first_write(caplog):
    from synap.cloud.application.ingestion.storage_provider import StorageUnavailableError
    store = SimpleNamespace(_bucket="", write=AsyncMock())
    with caplog.at_level("ERROR"):
        p = _provider(store)
    assert "S3_BUCKET is not set" in caplog.text
    with pytest.raises(StorageUnavailableError):
        await p.persist_original(uuid.uuid4(), "x", SCOPE)
    store.write.assert_not_awaited()


@pytest.mark.asyncio
async def test_manager_logs_an_error_when_the_original_cannot_be_saved(caplog):
    """The retry loop in IngestionManager turns the raise into a loud ERROR line."""
    from synap.cloud.application.ingestion.manager import IngestionManager
    with patch("synap.cloud.application.ingestion.storage_provider.get_file_store",
               side_effect=RuntimeError("no s3")):
        provider = FileStoreStorageProvider()
    mgr = IngestionManager.__new__(IngestionManager)
    mgr._storage = provider
    with patch("asyncio.sleep", new=AsyncMock()), caplog.at_level("ERROR"):
        await mgr._persist_original_with_retry(uuid.uuid4(), "x", SCOPE, "req-1")
    assert "ingestion_original_persist_failed" in caplog.text


def test_promotion_manager_is_not_built_with_a_stub():
    from synap.cloud.application.shared import providers
    mgr = providers._get_ingestion_manager(AsyncMock())
    assert isinstance(mgr._storage, FileStoreStorageProvider)


@pytest.mark.asyncio
async def test_backup_helper_saves_file_and_swallows_failure():
    from synap.cloud.application.ingestion.storage_provider import backup_uploaded_file
    iid = uuid.uuid4()
    ok = SimpleNamespace(_storage=SimpleNamespace(persist_original_file=AsyncMock()))
    await backup_uploaded_file(ok, iid, b"pdf", "a.pdf", SCOPE)
    ok._storage.persist_original_file.assert_awaited_once()
    bad = SimpleNamespace(_storage=SimpleNamespace(
        persist_original_file=AsyncMock(side_effect=RuntimeError("s3 down"))))
    await backup_uploaded_file(bad, iid, b"pdf", "a.pdf", SCOPE)  # must not raise
    skipped = SimpleNamespace(_storage=SimpleNamespace(persist_original_file=AsyncMock()))
    await backup_uploaded_file(skipped, iid, None, "", SCOPE)  # pasted text: no file
    skipped._storage.persist_original_file.assert_not_awaited()
