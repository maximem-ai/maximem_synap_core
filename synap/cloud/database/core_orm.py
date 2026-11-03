"""SQLAlchemy ORM models for the tables Core ships: client, instance, client_config.

These classes moved here from ``synap.cloud.auth.models.orm`` (the ``auth`` folder
stays behind in Core, so a Core tree could not import its own database layer).
They are defined once, here. ``synap.cloud.auth.models.orm`` imports them back, so
the old import path gives the same objects.

It sits beside ``connection.py``, not in ``models/``, because importing the
``models`` package also registers the connector tables (Google Drive, Zendesk),
which Core does not need.

``ClientORM.credentials`` and ``InstanceORM.credentials`` are NOT declared here:
the ``credential`` table is not a Core table, so the relationships to
``CredentialORM`` are attached by ``synap.cloud.auth.models.orm`` right after
``CredentialORM`` is defined. The mapped relationships Cloud ends up with are the
same as before.
"""

from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column

from synap.cloud.database.connection import Base
from synap.cloud.database.types import db_enum


# =============================================================================
# Enums (matching SQL enum types from migrations)
# =============================================================================

class EntityStatus(str, Enum):
    """Status for Client table (001_enums.sql)."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SOFT_DELETED = "soft_deleted"


class InstanceStatus(str, Enum):
    """Status for Instance table (001_enums.sql)."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROVISIONING = "provisioning"
    SUSPENDED = "suspended"


class ClientORM(Base):
    """SQLAlchemy ORM model for client (tenant organization).

    Maps to: 002_setup_entities.sql + 032_identity_mapping_alignment.sql -> client table.

    The ``uuid`` column stores a deterministic uuid5(NAMESPACE_URL, client_id)
    so that it matches what ``parse_or_deterministic_uuid()`` computes.
    """
    __tablename__ = "client"

    uuid: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        unique=True,
    )
    client_id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )
    client_name: Mapped[str] = mapped_column(Text, nullable=False)
    client_website: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    client_short_description: Mapped[str] = mapped_column(Text, nullable=False)
    client_context: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Which region runs this client's scheduled credit work; see migration 145
    # and synap/cloud/shared/region.py. Replicated with the row.
    home_region: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[EntityStatus] = mapped_column(
        db_enum(EntityStatus, "entity_status"),
        nullable=False,
        default=EntityStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    instances: Mapped[List["InstanceORM"]] = relationship(
        "InstanceORM",
        back_populates="client",
        cascade="all, delete-orphan",
    )


class InstanceORM(Base):
    """SQLAlchemy ORM model for SDK instance.

    Maps to: 002_setup_entities.sql + 032_identity_mapping_alignment.sql -> instance table.

    The ``uuid`` column stores a deterministic uuid5(NAMESPACE_URL, instance_id)
    so that it matches what ``parse_or_deterministic_uuid()`` computes.
    """
    __tablename__ = "instance"

    uuid: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        unique=True,
    )
    instance_id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )
    client_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("client.client_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    instance_name: Mapped[str] = mapped_column(Text, nullable=False)
    parent_instance_id: Mapped[Optional[str]] = mapped_column(
        Text,
        ForeignKey("instance.instance_id", ondelete="SET NULL"),
        nullable=True,
    )
    instance_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False)
    agent_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[InstanceStatus] = mapped_column(
        db_enum(InstanceStatus, "instance_status"),
        nullable=False,
        default=InstanceStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    client: Mapped["ClientORM"] = relationship(
        "ClientORM",
        back_populates="instances",
    )


class ClientConfigORM(Base):
    """SQLAlchemy ORM model for client/instance configuration.

    Maps to: 002_setup_entities.sql -> client_config table.
    SDK-accessible key-value configuration per client or instance.
    """
    __tablename__ = "client_config"

    config_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    client_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("client.client_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    instance_id: Mapped[Optional[str]] = mapped_column(
        Text,
        ForeignKey("instance.instance_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    config_key: Mapped[str] = mapped_column(Text, nullable=False)
    config_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False, default="1")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# Cloud only: the credential relationships of ClientORM and InstanceORM are attached
# by synap.cloud.auth.models.orm. Load it here so that in Cloud they exist before
# any mapper can configure, whichever module a process imports first. In a Core tree
# the auth folder is absent and this does nothing. Only "auth is missing" is
# swallowed; any other import error in Cloud still raises. (Circular on purpose:
# that module imports the classes above, which are all defined by this point.)
try:
    import synap.cloud.auth.models.orm  # noqa: E402,F401
except ModuleNotFoundError as _exc:
    if not (_exc.name or "").startswith("synap.cloud.auth"):
        raise
