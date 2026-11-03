"""S3-backed StorageProvider shared by every IngestionManager.

There used to be two: this one on the SDK routes, and a do-nothing stub on the
compaction-promotion path, so promoted conversations were never backed up.
"""

import asyncio
import logging
import uuid as _uuid
from typing import Any, Dict, List, Optional, Union

from synap.providers.file import get_file_store

logger = logging.getLogger(__name__)


class StorageUnavailableError(RuntimeError):
    """The original could not be saved because no file store is configured."""


def _prefix(scope: Any) -> str:
    return f"ingestion/{scope.client_id}/{scope.instance_id}/{scope.customer_id}"


class FileStoreStorageProvider:
    """Persists the original ingested content to the configured file store."""

    def __init__(self):
        self._file_store = None
        self._unavailable_reason: Optional[str] = None
        try:
            store = get_file_store()
        except Exception as exc:
            self._unavailable_reason = f"file store could not be built: {exc}"
        else:
            # get_file_store() succeeds with an empty bucket; the first write
            # would be the first sign. Check here so it is known at startup.
            if not getattr(store, "_bucket", "x"):
                self._unavailable_reason = "S3_BUCKET is not set"
            else:
                self._file_store = store
        if self._unavailable_reason:
            logger.error(
                "ingestion_backup_unavailable reason=%s: originals will NOT be "
                "saved until this is fixed", self._unavailable_reason,
            )

    def _require_store(self):
        if self._file_store is None:
            raise StorageUnavailableError(self._unavailable_reason or "no file store")
        return self._file_store

    async def persist_original(
        self,
        ingestion_id: _uuid.UUID,
        content: Union[str, bytes],
        scope: Any,
    ) -> None:
        store = self._require_store()
        payload = content.encode("utf-8") if isinstance(content, str) else content
        await store.write(
            path=f"{_prefix(scope)}/{ingestion_id}.bin",
            content=payload,
            metadata={"ingestion_id": str(ingestion_id)},
        )

    async def persist_original_file(
        self,
        ingestion_id: _uuid.UUID,
        raw: bytes,
        filename: str,
        scope: Any,
    ) -> None:
        """Save an uploaded file as received (the PDF, not its extracted text)."""
        store = self._require_store()
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        await store.write(
            path=f"{_prefix(scope)}/{ingestion_id}.original{ext}",
            content=raw,
            metadata={"ingestion_id": str(ingestion_id), "filename": filename},
        )

    async def read_object(self, path: str) -> Optional[bytes]:
        """The object's bytes, or None if it does not exist yet."""
        store = self._require_store()
        try:
            return await asyncio.to_thread(lambda: store.download(path).read())
        except FileNotFoundError:
            return None

    async def write_object(self, path: str, content: bytes,
                           metadata: Optional[Dict[str, str]] = None) -> None:
        await self._require_store().write(path=path, content=content, metadata=metadata)

    async def persist_short_term(
        self,
        ingestion_id: _uuid.UUID,
        content: str,
        scope: Any,
        session_id: str,
        qualification_result: Optional[Dict[str, Any]],
    ) -> None:
        # Only long-term ingestion pathways use this provider.
        return None

    async def get_short_term_by_session(
        self,
        session_id: str,
        scope: Any,
    ) -> List[Dict[str, Any]]:
        return []


async def backup_uploaded_file(manager: Any, ingestion_id: Any, raw: Optional[bytes],
                               filename: str, scope: Any) -> None:
    """Keep the original upload beside its extracted text, never failing the ingest.

    Ingestion is already queued when this runs, so a lost backup is logged, not raised.
    """
    if raw is None:
        return
    try:
        await manager._storage.persist_original_file(
            ingestion_id=ingestion_id, raw=raw, filename=filename, scope=scope,
        )
    except Exception:
        logger.error("upload_original_file_backup_failed ingestion_id=%s",
                     ingestion_id, exc_info=True)
