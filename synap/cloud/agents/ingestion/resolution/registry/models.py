"""The registry's tables.

``registry_entries`` holds one row per canonical entity, scoped to the world,
a client, a customer or a user. ``review_queue_entries`` is the older review
queue, one pending row per unresolved name. Both were created by migration
010 and are unchanged; these models moved here from synap-er, which no longer
owns storage.

They have a declarative base of their own, as they did in synap-er, so they
stay out of the application's metadata.
"""

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, MetaData, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from synap.cloud.agents.ingestion.resolution.registry.types import EntityType, Scope, Source

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)


class TimestampMixin:
    """created_at and updated_at, set by the database."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class RegistryEntry(TimestampMixin, Base):
    """A canonical entity within a scoped registry.

    Registries inherit in a strict precedence chain:
    World-data -> Client -> Customer -> User.
    Inheritance is confidence-weighted.
    """

    __tablename__ = "registry_entries"

    canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    canonical_name: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[EntityType] = mapped_column(
        String(32), nullable=False
    )
    scope: Mapped[Scope] = mapped_column(String(16), nullable=False)
    scope_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True, comment="ID of the client/customer/user"
    )
    aliases: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text), nullable=True, default=list
    )
    source: Mapped[Source] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True, default=dict
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True,
        comment="Pre-computed sentence-transformer embedding",
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_registry_scope_type", "scope", "scope_id", "entity_type"),
        Index("ix_registry_canonical_name", "canonical_name"),
        Index("ix_registry_scope_name", "scope", "scope_id", "canonical_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<RegistryEntry {self.canonical_name!r} "
            f"scope={self.scope} type={self.entity_type}>"
        )


class ReviewStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MERGED = "MERGED"


class ReviewQueueEntry(TimestampMixin, Base):
    """An entity mention queued for human review.

    Created when the resolution pipeline produces results below the
    confidence threshold.  Contains the mention text, top-3 scored
    candidates, and source context for reviewer decision-making.
    """

    __tablename__ = "review_queue_entries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_mention: Mapped[str] = mapped_column(String(512), nullable=False)
    candidate_canonicals: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, comment="Top-3 scored candidates as JSON array"
    )
    source_document_ref: Mapped[str | None] = mapped_column(
        String(1024), nullable=True, comment="Reference to source document"
    )
    context_text: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Surrounding text context for the mention"
    )
    scope: Mapped[Scope] = mapped_column(String(16), nullable=False)
    scope_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[ReviewStatus] = mapped_column(
        String(16), nullable=False, default=ReviewStatus.PENDING
    )
    resolved_canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("registry_entries.canonical_id"),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    resolved_canonical: Mapped[RegistryEntry | None] = relationship(
        "RegistryEntry", foreign_keys=[resolved_canonical_id], lazy="selectin"
    )

    __table_args__ = (
        Index("ix_review_queue_status", "status"),
        Index("ix_review_queue_scope", "scope", "scope_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ReviewQueueEntry {self.entity_mention!r} "
            f"status={self.status}>"
        )
