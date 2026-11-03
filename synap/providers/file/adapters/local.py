"""Local-disk file store: the copies of ingested text go to a folder, not S3.

Same keys as the S3 adapter, for example ``ingestion/<client>/<instance>/<customer>/<id>.bin``,
laid out as folders under one root. Selected with ``SYNAP_FILE_STORE=local`` and the folder in
``SYNAP_ORIGINALS_DIR``.

A key is built from ids that arrive in requests, and S3 treats it as a flat name. On disk it is a
path, so every key is checked before it touches the filesystem: no absolute path, no ``.`` or ``..``
or empty segment, no NUL, and the resolved path must still be inside the root after symlinks.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import secrets
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional

from ..interface import FileStoreInterface
from ..models import FileListResult, FileMetadata, FileQuery

_TMP_PREFIX = ".synap-tmp-"


class UnsafeKeyError(ValueError):
    """The key would leave the root, or is not a plain relative path."""


class LocalDiskAdapter(FileStoreInterface):
    """Files under one folder.

    Not stored: the content type and custom metadata S3 keeps with an object. The content type is
    guessed from the extension, and custom metadata reads back as None. Nothing in the engine reads
    either; the values passed in are the same ones the file name already carries.
    """

    def __init__(self, config: dict):
        """Config keys: ``root`` (required): the folder, for example the mounted ``originals/``."""
        self.config = config
        self._root_setting = (config.get("root") or "").strip()
        self._root: Optional[Path] = None

    # --- lifecycle ---------------------------------------------------------------------------

    def connect(self) -> None:
        if self._root is not None:
            return
        if not self._root_setting:
            raise ValueError("SYNAP_ORIGINALS_DIR is not set: the local file store has no folder to write to")
        root = Path(self._root_setting)
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f"{_TMP_PREFIX}probe-{secrets.token_hex(4)}"
        try:
            probe.write_bytes(b"")  # fail now, at startup, not at the first ingest
        except OSError as e:
            raise PermissionError(f"the folder {root} is not writable by this process: {e}") from None
        finally:
            probe.unlink(missing_ok=True)
        self._root = root.resolve()

    def disconnect(self) -> None:
        self._root = None

    def _need_root(self) -> Path:
        if self._root is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._root

    # --- keys -------------------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        root = self._need_root()
        if not isinstance(key, str) or not key or "\0" in key or key.startswith("/"):
            raise UnsafeKeyError("not a plain relative key")
        parts = key.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise UnsafeKeyError("a key may not hold an empty, '.' or '..' segment")
        if any(p.startswith(_TMP_PREFIX) for p in parts):
            raise UnsafeKeyError("that name is reserved")
        path = root.joinpath(*parts)
        resolved = path.resolve()  # follows symlinks, even in the folders above the file
        if root != resolved and root not in resolved.parents:
            raise UnsafeKeyError("the key resolves outside the folder")
        return path

    def _key(self, path: Path) -> str:
        return path.relative_to(self._need_root()).as_posix()

    # --- file operations ----------------------------------------------------------------------

    def _write_bytes(self, key: str, content: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{_TMP_PREFIX}{secrets.token_hex(6)}")
        try:
            with tmp.open("wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # a reader sees the old file or the whole new one, never half
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def upload(self, key: str, data: BinaryIO, content_type: str = None, metadata: dict = None) -> FileMetadata:
        data.seek(0)
        self._write_bytes(key, data.read())
        return self.get_metadata(key)

    async def write(self, path: str, content: bytes, metadata: dict = None) -> FileMetadata:
        # off the event loop: this is a disk write, and S3's version blocks it too
        await asyncio.to_thread(self._write_bytes, path, content)
        return await asyncio.to_thread(self.get_metadata, path)

    def download(self, key: str) -> BinaryIO:
        path = self._path(key)
        try:
            return BytesIO(path.read_bytes())
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            raise FileNotFoundError(f"File not found: {key}") from None

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def get_metadata(self, key: str) -> Optional[FileMetadata]:
        path = self._path(key)
        try:
            st = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        if not path.is_file():
            return None
        return FileMetadata(
            key=key,
            size_bytes=st.st_size,
            content_type=mimetypes.guess_type(key)[0] or "application/octet-stream",
            last_modified=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            custom_metadata=None,
        )

    def list_files(self, query: FileQuery) -> FileListResult:
        root = self._need_root()
        prefix = query.prefix or ""
        # start the walk at the deepest whole folder in the prefix, not at the root
        start = root
        if "/" in prefix:
            head = prefix.rsplit("/", 1)[0]
            try:
                start = self._path(head)
            except UnsafeKeyError:
                return FileListResult(files=[])
        keys = []
        if start.is_dir():
            for dirpath, dirnames, filenames in os.walk(start):
                dirnames.sort()
                for name in filenames:
                    if name.startswith(_TMP_PREFIX):
                        continue
                    key = self._key(Path(dirpath) / name)
                    if key.startswith(prefix):
                        keys.append(key)
        keys.sort()
        if query.continuation_token:
            keys = [k for k in keys if k > query.continuation_token]
        page = keys[: query.limit]
        truncated = len(keys) > len(page)
        files = [m for m in (self.get_metadata(k) for k in page) if m is not None]
        return FileListResult(
            files=files,
            continuation_token=page[-1] if truncated and page else None,
            is_truncated=truncated,
        )

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        raise NotImplementedError(
            "The local file store has no presigned URLs: the files are in the originals folder.")
