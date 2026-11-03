"""The local-disk file store."""
from __future__ import annotations

import asyncio
import io
import os
import uuid
from types import SimpleNamespace

import pytest

from synap.providers.file import LocalDiskAdapter, get_file_store, reset_file_store
from synap.providers.file.adapters.local import UnsafeKeyError
from synap.providers.file.models import FileQuery
from synap.providers.file import registry

SCOPE = SimpleNamespace(client_id="c1", instance_id="i1", customer_id="u1")


@pytest.fixture
def store(tmp_path):
    s = LocalDiskAdapter({"root": str(tmp_path / "originals")})
    s.connect()
    yield s
    s.disconnect()


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAP_FILE_STORE", "local")
    monkeypatch.setenv("SYNAP_ORIGINALS_DIR", str(tmp_path / "originals"))
    for v in ("S3_BUCKET", "AWS_REGION", "S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(v, raising=False)
    reset_file_store()
    yield tmp_path / "originals"
    reset_file_store()


# --- the basics -------------------------------------------------------------------------------

async def test_write_then_download_roundtrips(store):
    meta = await store.write("ingestion/c/i/u/1.bin", b"hello", {"ingestion_id": "1"})
    assert meta.size_bytes == 5 and meta.key == "ingestion/c/i/u/1.bin"
    assert store.download("ingestion/c/i/u/1.bin").read() == b"hello"


async def test_the_key_becomes_folders_under_the_root(store, tmp_path):
    await store.write("conversations/c/i/u/conv-1.json", b"{}")
    assert (tmp_path / "originals" / "conversations" / "c" / "i" / "u" / "conv-1.json").read_bytes() == b"{}"


async def test_overwrite_replaces_and_leaves_no_temporary_files(store, tmp_path):
    await store.write("a/b.bin", b"one")
    await store.write("a/b.bin", b"two")
    assert store.download("a/b.bin").read() == b"two"
    assert [p.name for p in (tmp_path / "originals" / "a").iterdir()] == ["b.bin"]


async def test_concurrent_writes_to_one_key_leave_one_whole_file(store):
    a, b = b"A" * 200_000, b"B" * 200_000
    await asyncio.gather(*[store.write("k/x.bin", a if i % 2 else b) for i in range(20)])
    assert store.download("k/x.bin").read() in (a, b)


def test_upload_from_a_stream(store):
    store.upload("u/s.bin", io.BytesIO(b"stream"), content_type="text/plain")
    assert store.download("u/s.bin").read() == b"stream"


def test_missing_files_behave_like_s3(store):
    with pytest.raises(FileNotFoundError):
        store.download("nope/x.bin")
    assert store.exists("nope/x.bin") is False
    assert store.get_metadata("nope/x.bin") is None
    store.delete("nope/x.bin")  # deleting what is not there is fine


async def test_delete_removes_the_file(store):
    await store.write("d/x.bin", b"1")
    assert store.exists("d/x.bin")
    store.delete("d/x.bin")
    assert not store.exists("d/x.bin")


async def test_metadata_has_size_time_and_a_guessed_type(store):
    await store.write("m/report.pdf", b"%PDF-1.7")
    meta = store.get_metadata("m/report.pdf")
    assert meta.size_bytes == 8 and meta.content_type == "application/pdf"
    assert meta.last_modified.tzinfo is not None and meta.custom_metadata is None
    await store.write("m/blob.bin", b"x")
    assert store.get_metadata("m/blob.bin").content_type == "application/octet-stream"


def test_there_are_no_presigned_urls(store):
    with pytest.raises(NotImplementedError):
        store.get_presigned_url("a/b.bin")


# --- listing ----------------------------------------------------------------------------------

async def test_list_filters_by_prefix_and_is_sorted(store):
    for k in ("ingestion/c/i/u/2.bin", "ingestion/c/i/u/1.bin", "ingestion/c/i/v/9.bin", "conversations/c/i/u/a.json"):
        await store.write(k, b"x")
    got = [f.key for f in store.list_files(FileQuery(prefix="ingestion/c/i/u/")).files]
    assert got == ["ingestion/c/i/u/1.bin", "ingestion/c/i/u/2.bin"]
    assert len(store.list_files(FileQuery()).files) == 4


async def test_list_pages_with_a_continuation_token(store):
    for n in range(5):
        await store.write(f"p/{n}.bin", b"x")
    first = store.list_files(FileQuery(prefix="p/", limit=2))
    assert [f.key for f in first.files] == ["p/0.bin", "p/1.bin"] and first.is_truncated
    second = store.list_files(FileQuery(prefix="p/", limit=2, continuation_token=first.continuation_token))
    assert [f.key for f in second.files] == ["p/2.bin", "p/3.bin"]
    last = store.list_files(FileQuery(prefix="p/", limit=2, continuation_token=second.continuation_token))
    assert [f.key for f in last.files] == ["p/4.bin"] and not last.is_truncated


async def test_listing_hides_files_being_written(store, tmp_path):
    await store.write("q/real.bin", b"x")
    (tmp_path / "originals" / "q" / ".synap-tmp-abc").write_bytes(b"half")
    assert [f.key for f in store.list_files(FileQuery(prefix="q/")).files] == ["q/real.bin"]


# --- keys come from request ids, so they are checked -------------------------------------------

@pytest.mark.parametrize("key", [
    "../escape.bin", "a/../../escape.bin", "/etc/passwd", "a//b.bin", "a/./b.bin", "", "a/b\0.bin",
    "ingestion/../../../../etc/cron.d/x", "a/.synap-tmp-x",
])
async def test_unsafe_keys_are_refused_and_nothing_is_written(store, tmp_path, key):
    with pytest.raises(UnsafeKeyError):
        await store.write(key, b"pwned")
    with pytest.raises(UnsafeKeyError):
        store.download(key)
    assert not (tmp_path / "escape.bin").exists()


async def test_a_symlink_inside_the_root_cannot_lead_out(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / "originals" / "link")
    with pytest.raises(UnsafeKeyError):
        await store.write("link/x.bin", b"pwned")
    assert list(outside.iterdir()) == []


async def test_ids_that_look_odd_but_are_plain_are_fine(store):
    key = f"ingestion/client-1/{uuid.uuid4()}/user.name+tag@x.com/{uuid.uuid4()}.bin"
    await store.write(key, b"ok")
    assert store.download(key).read() == b"ok"


# --- connecting ----------------------------------------------------------------------------------

def test_no_folder_set_is_a_clear_error():
    with pytest.raises(ValueError, match="SYNAP_ORIGINALS_DIR"):
        LocalDiskAdapter({"root": ""}).connect()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_a_folder_that_cannot_be_written_fails_at_connect(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        with pytest.raises(PermissionError, match="not writable"):
            LocalDiskAdapter({"root": str(ro)}).connect()
    finally:
        ro.chmod(0o700)


def test_connect_creates_the_folder(tmp_path):
    s = LocalDiskAdapter({"root": str(tmp_path / "new" / "deep")})
    s.connect()
    assert (tmp_path / "new" / "deep").is_dir()


def test_using_it_before_connect_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="Not connected"):
        LocalDiskAdapter({"root": str(tmp_path)}).exists("a/b")


# --- choosing the adapter --------------------------------------------------------------------------

def test_local_is_chosen_by_the_setting(local_env):
    cfg = registry._load_config()
    assert cfg["adapter"] == "local" and cfg["root"] == str(local_env)
    assert isinstance(get_file_store(), LocalDiskAdapter)


def test_an_unknown_store_name_is_refused_with_the_choices(monkeypatch):
    monkeypatch.setenv("SYNAP_FILE_STORE", "gcs")
    with pytest.raises(ValueError, match="choose one of: local"):
        registry._load_config()


# --- an upload and a compaction both leave files, with no S3 ---------------------

async def test_upload_and_conversation_backup_leave_files_on_disk_with_no_s3(local_env):
    from synap.cloud.application.ingestion.storage_provider import FileStoreStorageProvider
    provider = FileStoreStorageProvider()
    assert provider._unavailable_reason is None  # not "S3_BUCKET is not set"

    iid = uuid.uuid4()
    await provider.persist_original(iid, "user: I moved to Pune", SCOPE)
    await provider.persist_original_file(iid, b"%PDF-1.7 raw", "Report.PDF", SCOPE)
    # the conversation backup, as compaction writes it: read what is there, merge, write back
    key = "conversations/c1/i1/u1/conv-9.json"
    assert await provider.read_object(key) is None
    await provider.write_object(key, b'{"messages": [1]}')
    assert await provider.read_object(key) == b'{"messages": [1]}'
    await provider.write_object(key, b'{"messages": [1, 2]}')

    assert (local_env / f"ingestion/c1/i1/u1/{iid}.bin").read_bytes() == b"user: I moved to Pune"
    assert (local_env / f"ingestion/c1/i1/u1/{iid}.original.pdf").read_bytes() == b"%PDF-1.7 raw"
    assert (local_env / key).read_bytes() == b'{"messages": [1, 2]}'


# --- startup: an unwritable folder stops the server, it is not found at the first ingest -----------

def test_an_unwritable_originals_folder_stops_startup_with_the_reason(tmp_path, monkeypatch):
    import os
    import stat

    import pytest as _pytest
    from synap.providers.file import registry

    folder = tmp_path / "originals"
    folder.mkdir()
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read-only
    if os.access(folder, os.W_OK):
        _pytest.skip("this user can write to a read-only folder (running as root)")
    registry.reset_file_store()
    monkeypatch.setenv("SYNAP_FILE_STORE", "local")
    monkeypatch.setenv("SYNAP_ORIGINALS_DIR", str(folder))
    try:
        with _pytest.raises(PermissionError, match="not writable"):
            registry.check_file_store_at_startup()
        assert registry._instance is None  # the half-built store was not kept
    finally:
        folder.chmod(stat.S_IRWXU)
        registry.reset_file_store()
