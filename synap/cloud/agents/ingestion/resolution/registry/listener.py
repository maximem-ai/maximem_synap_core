"""Keeps this worker's registry index in step with the other workers'.

Moved from synap-er, which no longer owns storage. Every write goes out as a
Postgres NOTIFY (see write_through); this listens and applies it locally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Awaitable, Callable, Optional

import asyncpg

from synap_er.memory.index_manager import RegistryIndexManager, ScopeKey

from synap.cloud.agents.ingestion.resolution.registry.models import RegistryEntry
from synap.cloud.agents.ingestion.resolution.registry.write_through import NOTIFY_CHANNEL

logger = logging.getLogger(__name__)



class ChangeListener:
    """Subscribes to PostgreSQL ``LISTEN`` and applies changes to the local index.

    Runs as a background :class:`asyncio.Task`.  On receiving a ``NOTIFY``:

    * **add / update** — fetches the full ``RegistryEntry`` from DB via the
      injected *fetch_entry_fn*, then calls
      :pymethod:`RegistryIndexManager.add_entry` or ``update_entry``.
    * **remove** — calls :pymethod:`RegistryIndexManager.remove_entry`.

    Parameters
    ----------
    dsn:
        asyncpg-compatible PostgreSQL DSN
        (e.g. ``"postgresql://user:pass@host/db"``).
    index_manager:
        The local in-memory index to keep in sync.
    fetch_entry_fn:
        Async callable ``(canonical_id) -> RegistryEntry | None``
        used to retrieve the full row from the database.
    """

    def __init__(
        self,
        dsn: str,
        index_manager: RegistryIndexManager,
        fetch_entry_fn: Callable[[uuid.UUID], Awaitable[Optional[RegistryEntry]]],
    ) -> None:
        self._dsn = dsn
        self._index = index_manager
        self._fetch_entry = fetch_entry_fn
        self._connection: Optional[asyncpg.Connection] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start listening in a background asyncio task."""
        self._running = True
        self._task = asyncio.create_task(self._listen_loop())
        logger.info("ChangeListener started on channel '%s'", NOTIFY_CHANNEL)

    async def stop(self) -> None:
        """Stop the listener and close the dedicated connection."""
        self._running = False
        if self._connection and not self._connection.is_closed():
            await self._connection.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ChangeListener stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """Connect, subscribe, and keep the connection alive."""
        while self._running:
            try:
                self._connection = await asyncpg.connect(self._dsn)
                await self._connection.add_listener(
                    NOTIFY_CHANNEL, self._on_notification
                )
                logger.info("Subscribed to LISTEN on '%s'", NOTIFY_CHANNEL)

                # Hold connection open
                while self._running:
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "ChangeListener connection error; reconnecting in 5 s"
                )
                await asyncio.sleep(5)
            finally:
                if self._connection and not self._connection.is_closed():
                    try:
                        await self._connection.remove_listener(
                            NOTIFY_CHANNEL, self._on_notification
                        )
                        await self._connection.close()
                    except Exception:
                        pass

    def _on_notification(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        """asyncpg synchronous callback — schedules the async handler."""
        asyncio.create_task(self._handle_notification(payload))

    async def _handle_notification(self, payload: str) -> None:
        """Parse the NOTIFY payload and update the local index."""
        try:
            data = json.loads(payload)
            action: str = data["action"]
            canonical_id = uuid.UUID(data["canonical_id"])
            scope: str = data["scope"]
            scope_id: str | None = data.get("scope_id")
            scope_key: ScopeKey = (scope, scope_id)

            if action == "remove":
                self._index.remove_entry(canonical_id, scope_key)
                logger.debug("Removed entry %s from index", canonical_id)

            elif action in ("add", "update"):
                entry = await self._fetch_entry(canonical_id)
                if entry is None:
                    logger.warning(
                        "Entry %s not found in DB for action '%s'",
                        canonical_id,
                        action,
                    )
                    return
                if action == "add":
                    self._index.add_entry(entry)
                else:
                    self._index.update_entry(entry)
                logger.debug(
                    "Applied '%s' for entry %s", action, canonical_id
                )

        except Exception:
            logger.exception("Failed to handle notification: %s", payload)
