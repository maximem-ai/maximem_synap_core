"""First-boot seeding of Core's one client and one instance.

``seed_default_identity(session)`` creates the default client and the default
instance if they are missing and touches nothing that exists. Safe to run on
every start.

Rows the engine expects, for a user whose id comes from the request:

* ``client`` and ``instance`` (this module creates them). ``IdMapper`` only
  looks these up and refuses an unknown id; the instance has a foreign key to
  the client.
* ``"user"`` and ``customer`` rows are NOT seeded. ``IdMapper.resolve_user`` and
  ``resolve_customer`` create them on first sight of an id (an upsert), so they
  exist by the time ingestion reaches the profile merge, which finds them by
  (client_id, uuid5_hash).

The instance goes through ``InstanceRepository.create_instance``, so the
instance cap check (``shared/instance_limits.py``) runs as for any instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from synap.cloud.application.sdk_api.core_identity import (
    DEFAULT_CLIENT_ID,
    DEFAULT_INSTANCE_ID,
)
from synap.cloud.shared import edition

logger = logging.getLogger(__name__)

CLIENT_NAME = "Synap Core"
INSTANCE_NAME = "default"

# The two scoping modes Core accepts and the relationship each maps
# to. ``load_isolation`` falls back to the instance's ``user_relationship`` when
# no config exists, and reads the same table, so the two cannot disagree.
_RELATIONSHIP_FOR_MODE = {"strict": "b2b", "equals_customer": "b2c"}
_MODE_FOR_RELATIONSHIP = {v: k for k, v in _RELATIONSHIP_FOR_MODE.items()}


class ScopingModeChanged(RuntimeError):
    """The config names another scoping mode than the one this install holds memories under."""


def default_isolation() -> str:
    """The mode a config that leaves ``user_context_isolation`` out runs under: the schema's own default."""
    from synap.cloud.memory_config.config_schema import ScopingConfig

    return ScopingConfig.model_fields["user_context_isolation"].default


async def _holds_memories(session: Any) -> bool:
    """Whether any memory has been stored (Core has one client, so any row is this install's)."""
    from sqlalchemy import text

    result = await session.execute(text("SELECT 1 FROM memories LIMIT 1"))
    return result.first() is not None


@dataclass(frozen=True)
class SeedResult:
    client_created: bool
    instance_created: bool


def relationship_for_mode(isolation: Optional[str]) -> Optional[str]:
    if isolation is None or isolation == "":
        return None
    if isolation not in _RELATIONSHIP_FOR_MODE:
        raise ValueError(
            f"user_context_isolation={isolation!r} is not supported by Synap Core. "
            "Use 'strict' (business apps: users sit under a customer) or "
            "'equals_customer' (consumer apps: each user is their own customer)."
        )
    return _RELATIONSHIP_FOR_MODE[isolation]


def _is_conflict(exc: Exception) -> bool:
    from sqlalchemy.exc import IntegrityError

    return isinstance(exc, IntegrityError)


async def seed_default_identity(session: Any, *, isolation: Optional[str] = None) -> SeedResult:
    """Create the default client and instance if missing. Idempotent.

    *isolation* is the YAML's ``user_context_isolation``; when the YAML leaves it
    out (None), the mode the config runs under by default is used, so the mode
    is held and checked the same way. It is written to the
    new instance as its ``user_relationship``. A different mode on an existing
    instance is accepted only while no memory has been stored; once memories
    exist it raises ``ScopingModeChanged``, which stops the server at start.
    """
    from synap.cloud.database.repositories.client_repo import ClientRepository
    from synap.cloud.database.repositories.instance_repo import InstanceRepository

    left_out = isolation is None or isolation == ""
    if left_out:
        isolation = default_isolation()
    relationship = relationship_for_mode(isolation)
    clients = ClientRepository(session)
    instances = InstanceRepository(session)

    client_created = False
    if await clients.get_by_client_id(DEFAULT_CLIENT_ID) is None:
        try:
            async with session.begin_nested():
                await clients.create_client(
                    client_id=DEFAULT_CLIENT_ID,
                    client_name=CLIENT_NAME,
                    client_short_description="The one client of this Synap Core install.",
                    client_context={"edition": "core"},
                )
            client_created = True
        except Exception as exc:  # noqa: BLE001
            # Another worker seeded it first: fine. Anything else is real.
            if not _is_conflict(exc) or await clients.get_by_client_id(DEFAULT_CLIENT_ID) is None:
                raise

    instance_created = False
    existing = await instances.get_by_instance_id(DEFAULT_INSTANCE_ID)
    if existing is None:
        metadata: dict = {"edition": "core"}
        if relationship:
            metadata["user_relationship"] = relationship
        try:
            async with session.begin_nested():
                await instances.create_instance(
                    instance_id=DEFAULT_INSTANCE_ID,
                    client_id=DEFAULT_CLIENT_ID,
                    instance_name=INSTANCE_NAME,
                    instance_metadata=metadata,
                    agent_type="sdk",
                )
            instance_created = True
        except Exception as exc:  # noqa: BLE001
            # InstanceLimitReached and everything else propagates; only a lost
            # race against another worker is fine.
            if not _is_conflict(exc) or await instances.get_by_instance_id(DEFAULT_INSTANCE_ID) is None:
                raise
    elif relationship:
        meta = dict(getattr(existing, "instance_metadata", None) or {})
        # An instance seeded before the mode was recorded ran under the default.
        have = meta.get("user_relationship") or _RELATIONSHIP_FOR_MODE.get(default_isolation())
        if have and have != relationship:
            if await _holds_memories(session):
                note = " (the default, because the file leaves it out)" if left_out else ""
                raise ScopingModeChanged(
                    f"config/memory.yml sets user_context_isolation to {isolation!r}{note}, but this install already "
                    f"holds memories stored under {_MODE_FOR_RELATIONSHIP.get(have, have)!r}. Changing the "
                    f"scoping mode after memories exist is not supported: those memories would become "
                    f"unreachable. Set it back to {_MODE_FOR_RELATIONSHIP.get(have, have)!r}, or start from "
                    f"nothing (docker compose -f deploy/docker-compose.yml down -v, and remove originals/)."
                )
            # Nothing has been stored yet, so the mode can still be chosen.
            meta["user_relationship"] = relationship
            existing.instance_metadata = meta
            if hasattr(session, "flush"):
                await session.flush()
            logger.info(
                "The scoping mode was changed from %s to %s before any memory was stored.",
                _MODE_FOR_RELATIONSHIP.get(have, have), isolation,
            )

    # From here a reload of the config that names another mode is refused (file_loader).
    from synap.cloud.memory_config.file_loader import pin_scoping_mode

    effective = relationship
    if existing is not None:
        effective = (getattr(existing, "instance_metadata", None) or {}).get("user_relationship") or relationship
    pin_scoping_mode(_MODE_FOR_RELATIONSHIP.get(effective) if effective else None)

    if client_created or instance_created:
        logger.info(
            "seeded the default identity client_created=%s instance_created=%s",
            client_created, instance_created,
        )
    return SeedResult(client_created=client_created, instance_created=instance_created)


async def _yaml_isolation() -> Optional[str]:
    """``storage_architecture.scoping.user_context_isolation`` from the Core config file, or None."""
    from synap.cloud.memory_config.loader import get_config_loader

    raw = await get_config_loader().get_raw_config()
    scoping = ((raw or {}).get("storage_architecture") or {}).get("scoping") or {}
    value = scoping.get("user_context_isolation")
    return value if isinstance(value, str) and value else None


async def seed_on_startup(*, isolation: Optional[str] = None) -> Optional[SeedResult]:
    """Seed in its own session and commit. A no-op unless the edition uses a fixed identity.

    For ``main.py``'s startup, called with no edition check of its own. With
    *isolation* None the scoping mode is read from the YAML (Core only; Cloud
    returns before it looks).
    """
    if edition.identity != "fixed":
        return None
    if isolation is None:
        isolation = await _yaml_isolation()
    from synap.cloud.database.connection import get_db_session

    async with get_db_session() as session:
        return await seed_default_identity(session, isolation=isolation)
