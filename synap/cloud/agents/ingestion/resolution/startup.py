"""Startup for entity resolution.

Synap owns the registry (resolution/registry): this loads it from the database
into the in-memory index synap-er reads, and starts the listener that keeps
the index current. synap-er is given the index and nothing else: no database
URL, no session of its own.
"""

import logging
import os
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from synap_er.config import PipelineSettings
from synap_er.config import Settings as ERSettings
from synap_er.memory.index_manager import RegistryIndexManager
from synap_er.pipeline.resolver import PipelineResolver

from synap.cloud.agents.ingestion.resolution.registry import (
    ChangeListener,
    RegistryEntry,
    WriteThroughManager,
)

logger = logging.getLogger(__name__)


class EntityResolutionComponents:
    """Holds initialized synap-er components (singleton per process)."""

    def __init__(self) -> None:
        self.settings: Optional[ERSettings] = None
        self.index_manager: Optional[RegistryIndexManager] = None
        self.resolver: Optional[PipelineResolver] = None
        self.write_through: Optional[WriteThroughManager] = None
        self.listener: Optional[ChangeListener] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        # The global dictionary of equivalent names; adjudicated mode only.
        self.dictionary: Optional[object] = None

    @property
    def initialized(self) -> bool:
        return self.resolver is not None


# Module-level singleton
_components = EntityResolutionComponents()


def get_er_components() -> EntityResolutionComponents:
    """Get the global synap-er components singleton."""
    return _components


def listener_dsn(database_url: str) -> str:
    """The plain-asyncpg DSN the ChangeListener connects with.

    Split out so it can be tested by CALLING it. An sslmode already on the URL
    always wins. Otherwise one is chosen, because asyncpg's own default
    (prefer) can hang against a server that silently drops the SSL upgrade
    request rather than refusing it.

    ⚠ The default here used to be a hardcoded ``disable``, and that is not a
    safe default: it silently broke live registry updates wherever
    Postgres has no pg_hba entry for an unencrypted connection. The index still
    loaded at boot and ingestion still worked, so the only
    symptom was ChangeListener reconnecting every 5s forever — the "healthy,
    versioned, silently doing nothing" shape. Against such a server, ``disable`` is refused
    with "no pg_hba.conf entry ...
    no encryption"; ``require`` and ``prefer`` both connect.

    ``disable`` stays the DEFAULT only so this change cannot alter the
    environments the hang was observed in. Anywhere the database requires TLS
    must set ``SYNAP_ER_LISTENER_SSLMODE`` (a deployment with TLS sets it to ``require``).
    """
    url = database_url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    dsn = url.replace("postgresql+asyncpg://", "postgresql://")
    if "sslmode" not in dsn:
        sslmode = os.environ.get("SYNAP_ER_LISTENER_SSLMODE", "").strip() or "disable"
        sep = "&" if "?" in dsn else "?"
        dsn += f"{sep}sslmode={sslmode}"
        logger.info(
            f"synap-er ChangeListener sslmode={sslmode} "
            f"(set SYNAP_ER_LISTENER_SSLMODE to change; a database that requires "
            f"TLS refuses 'disable' and the listener then retries forever)"
        )
    return dsn


async def init_entity_resolution(
    session_factory: async_sessionmaker[AsyncSession],
    database_url: str,
) -> EntityResolutionComponents:
    """Initialize all synap-er components.

    Called from FastAPI lifespan. Reuses the existing DB session_factory
    and database connection — synap-er does NOT create a separate DB.

    Args:
        session_factory: Existing SQLAlchemy async session factory.
        database_url: The application's database URL (postgresql+asyncpg:// format).
    """
    global _components

    # Ensure asyncpg format for SQLAlchemy
    asyncpg_url = database_url
    if asyncpg_url.startswith("postgres://"):
        asyncpg_url = asyncpg_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif asyncpg_url.startswith("postgresql://") and "+asyncpg" not in asyncpg_url:
        asyncpg_url = asyncpg_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # Raw asyncpg DSN for ChangeListener (needs plain postgresql://, and an
    # sslmode the server will actually accept — see listener_dsn).
    raw_dsn = listener_dsn(database_url)

    # 1. Resolution settings. synap-er is not told where the database is.
    settings = ERSettings(
        pipeline=PipelineSettings(
            confidence_threshold=0.85,
        ),
    )

    # 2. Create RegistryIndexManager (empty in-memory)
    index_manager = RegistryIndexManager()

    # 3. Cold-load existing entries from DB
    async with session_factory() as session:
        stmt = select(RegistryEntry).where(RegistryEntry.is_deleted == False)  # noqa: E712
        result = await session.execute(stmt)
        entries = list(result.scalars().all())

    if entries:
        index_manager.build(entries)
        logger.info("Loaded %d registry entries into synap-er index", len(entries))
    else:
        logger.info("No existing registry entries found; synap-er index is empty")

    # 4. Create PipelineResolver
    dictionary = await _load_dictionary(session_factory)
    resolver_kwargs = {}
    if dictionary is not None:
        resolver_kwargs["name_equivalents"] = dictionary
    resolver_kwargs.update(_review_queue_kwargs(session_factory))
    resolver = PipelineResolver(
        index_manager=index_manager,
        settings=settings,
        **resolver_kwargs,
    )

    # 5. Create WriteThroughManager
    write_through = WriteThroughManager(
        session_factory=session_factory,
        index_manager=index_manager,
    )

    # 6. Create and start ChangeListener (background LISTEN/NOTIFY)
    async def fetch_entry(canonical_id):
        async with session_factory() as s:
            return await s.get(RegistryEntry, canonical_id)

    listener = ChangeListener(
        dsn=raw_dsn,
        index_manager=index_manager,
        fetch_entry_fn=fetch_entry,
    )
    await listener.start()

    # 7. Store in singleton
    _components.settings = settings
    _components.index_manager = index_manager
    _components.resolver = resolver
    _components.write_through = write_through
    _components.listener = listener
    _components.session_factory = session_factory
    _components.dictionary = dictionary

    logger.info("synap-er entity resolution initialized")
    return _components


def _review_queue_kwargs(session_factory) -> dict:
    """How the resolver is told where its legacy path queues a low score.

    synap-er 0.4 takes a sink and writes nothing itself. Older releases owned
    the review table and took the session factory to write it.
    """
    import inspect

    from synap.cloud.shared import edition

    parameters = inspect.signature(PipelineResolver.__init__).parameters
    if "review_sink" in parameters:
        # Cloud queues a low score for the review page. Core has no review page,
        # so it passes no sink and the resolver skips the write.
        sink = edition.call("review_queue_write", session_factory)
        return {"review_sink": sink} if sink is not None else {}
    return {"session_factory": session_factory}


async def _load_dictionary(session_factory) -> Optional[object]:
    """The global dictionary, or None outside adjudicated mode.

    A load failure (the table is not there yet) leaves it empty rather than
    stopping entity resolution: it only ever adds candidates.
    """
    from .global_dictionary import (
        GlobalNameDictionary,
        PostgresNameEquivalentStore,
        global_dictionary_enabled,
    )
    from .mode import adjudicated_mode

    if not (adjudicated_mode() and global_dictionary_enabled()):
        return None
    import inspect

    if "name_equivalents" not in inspect.signature(PipelineResolver.__init__).parameters:
        logger.warning("synap-er is older than 0.3.0; the global dictionary is off")
        return None
    store = PostgresNameEquivalentStore(session_factory)
    dictionary = GlobalNameDictionary(store)
    try:
        dictionary.load(await store.load_all())
        logger.info("Loaded %d global name equivalents", len(dictionary))
    except Exception:
        logger.warning("Could not load er_name_equivalents; starting empty", exc_info=True)
    return dictionary


async def shutdown_entity_resolution() -> None:
    """Stop the ChangeListener. Called from FastAPI lifespan shutdown."""
    global _components
    if _components.listener:
        await _components.listener.stop()
        logger.info("synap-er ChangeListener stopped")
