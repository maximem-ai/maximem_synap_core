"""Ingestion output models."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from .enums import (
    ContentNature,
    IngestStatus,
    MemoryDestination,
    MemoryType,
)

logger = logging.getLogger(__name__)

# The only three values `temporal_category` may hold. The column is
# varchar(20), every reader compares against these three strings, and
# `is_episode` is derived from "episode", so a fourth value is never useful:
# it cannot be matched, and if it is longer than 20 characters the INSERT
# fails and the whole document is dead-lettered: "value too long for type
# character varying(20)" kills a document after stage 4 has already done all
# of its work, while invented values that happen to fit are quietly stored.
# The extractor is told to emit one of these three; anything
# else is a model or parse failure, and an invalid value is worth no more than
# a missing one, so it falls back to the same default a missing key gets.
TEMPORAL_CATEGORIES = ("perpetual", "temporal_fact", "episode")


class ExtractedEntity(BaseModel):
    """An entity extracted from content."""
    entity_id: str
    entity_type: str  # From MACA domain_hints.key_entities
    name: str
    canonical_name: Optional[str] = None  # After canonicalization
    # Every other form the same text uses for this entity ("Bob", "IBM").
    # `name` is the most complete one. Filled by the v2 extraction prompt only.
    aliases: List[str] = Field(default_factory=list)
    # Names the entity is widely known by, from the model's general knowledge
    # and NOT from the text ("Consolidated Edison" for "Con Ed"). Only ever a
    # reason to look for a candidate; never stored as an alias on its own.
    known_as: List[str] = Field(default_factory=list)
    # Who or what the entity is, in a phrase, as this text states it
    # ("designer on the Data Product team; she"). Filled by the v2 prompt.
    about: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_span: Optional[str] = None  # Where in content


class ExtractedRelationship(BaseModel):
    """A relationship between entities."""
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # From MACA domain_hints.key_relationships
    properties: Dict[str, Any] = Field(default_factory=dict)


class ExtractedMemory(BaseModel):
    """A single memory extracted from content."""
    memory_id: UUID
    memory_type: MemoryType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    destination: MemoryDestination  # Based on threshold

    # Every type whose extraction call claimed this same fact, for a row that
    # coalescing merged. `memory_type` above stays the single primary, so every
    # existing reader keeps working unchanged; this records that ONE row answers
    # for several types instead of five rows restating one fact.
    #
    # Empty is the normal case and means "just memory_type": every row that was
    # not merged, which is most of them.
    type_labels: List[str] = Field(default_factory=list)

    # For a row the coalesce step produced: what was folded into it, and the
    # model's reason for calling them one fact. Written to memory_actions as a
    # COALESCE entry beside the row's CREATE entry.
    #
    # It has to travel on the memory because coalescing runs before persistence,
    # so the rows it folded away never become database rows and there is nothing
    # else left to point an audit trail at.
    #   {"reason": "...", "contributors": ["...", "..."], "types": ["facts", ...]}
    coalesced_from: Optional[Dict[str, Any]] = None

    # Category context
    category: Optional[str] = None
    subcategory: Optional[str] = None

    # Source tracking
    source_document_id: str
    source_span: Optional[str] = None
    # Speaker who asserted this memory in the source conversation
    # ("user" | "assistant" | "system"). None = unknown: non-conversation
    # documents, legacy rows, or ENABLE_SPEAKER_ATTRIBUTION off.
    assertion_source: Optional[str] = None

    # Entity links
    entities_mentioned: List[str] = Field(default_factory=list)

    # PII handling
    pii_obfuscated: bool = False
    pii_fields_obfuscated: List[str] = Field(default_factory=list)
    # Did a detector run on the document this came from, and did it find
    # anything? None means no detector ran, which is the honest state for
    # every row written before the ingress gate shipped. Stamped by the
    # stage 2/3 worker from the gate's verdict, not decided here.
    pii_detected: Optional[bool] = None

    # Temporal classification
    temporal_category: str = "perpetual"  # perpetual | temporal_fact | episode
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    temporal_confidence: float = 0.0  # 0.0-1.0

    @field_validator("temporal_category", mode="before")
    @classmethod
    def _known_temporal_category(cls, value: Any) -> str:
        """Keep one of TEMPORAL_CATEGORIES; anything else becomes the default.

        Every writer takes this field straight off the model (the Postgres
        insert, the vector metadata, the graph node), so this is the one place
        that can hold the line for all of them. Case and surrounding space are
        forgiven because the legacy text parser lowercases and the JSON parser
        does not; an invented value is not, and it is logged so a prompt
        regression shows up as a rate instead of as a dead-lettered document.
        """
        if value is None:
            return "perpetual"
        text = str(value).strip().lower()
        if text in TEMPORAL_CATEGORIES:
            return text
        logger.warning(
            "extraction_temporal_category_invalid value=%r using=perpetual",
            str(value)[:60],
        )
        return "perpetual"

    # WHEN the event happened, as a window (migration 154). Separate from
    # valid_from/valid_until, which say how long a STATE held. "Bought a cat
    # last month" said on 23 Mar: event_start 2023-02-01, event_end 2023-02-28,
    # event_precision "month". Set by extraction, checked against DCT by
    # event_window.validate_event_windows before storage.
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None

    # When the date is only knowable relative to ANOTHER event ("three months
    # before the wedding"), the relationship goes here and event_start stays
    # NULL until something can settle it (migration 155). Without this the
    # extractor's only options are to anchor the phrase on the conversation date
    # and store a wrong one, or to drop the relationship entirely.
    #   {"anchor_text": "best friend's wedding",
    #    "offset": {"n": 3, "unit": "months", "direction": "before"},
    #    "source_phrase": "had to book three months in advance"}
    event_anchor: Optional[Dict[str, Any]] = None

    # Episode-specific
    is_episode: bool = False
    episode_metadata: Optional[Dict[str, Any]] = None

    # Hypothetical questions for improved retrieval
    hypothetical_questions: List[str] = Field(default_factory=list)

    # Metadata
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    document_created_at: Optional[datetime] = None
    extraction_trace_id: Optional[UUID] = None


class IngestResponse(BaseModel):
    """Response from single document ingestion."""
    request_id: str
    ingestion_id: UUID
    status: IngestStatus
    document_id: str

    # Results (populated when completed)
    memories_created: int = 0
    candidates_created: int = 0
    entities_created: int = 0
    relationships_created: int = 0
    memory_ids: List[str] = Field(default_factory=list)

    # Timing
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Errors
    error_message: Optional[str] = None
    error_stage: Optional[str] = None


class IngestBatchResponse(BaseModel):
    """Response from batch ingestion."""
    request_id: str
    batch_id: UUID
    status: IngestStatus

    # Counts
    total_documents: int
    succeeded: int = 0
    failed: int = 0

    # Per-document results
    document_results: List[IngestResponse] = Field(default_factory=list)

    # Timing
    queued_at: datetime
    completed_at: Optional[datetime] = None


class IngestionTrace(BaseModel):
    """Audit trace for an ingestion operation."""
    trace_id: UUID
    ingestion_id: UUID
    stage: str

    # Input/Output hashes (for privacy)
    input_hash: str
    output_hash: Optional[str] = None

    # Decisions made
    qualification_decision: Optional[str] = None  # "process" | "skip"
    content_nature: Optional[ContentNature] = None
    categories_assigned: List[str] = Field(default_factory=list)
    memory_types_extracted: List[str] = Field(default_factory=list)
    entities_extracted: int = 0

    # PII
    pii_detected: bool = False
    pii_fields_detected: List[str] = Field(default_factory=list)

    # Confidence
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

    # Performance
    processing_time_ms: int = 0
    llm_tokens_used: int = 0
    llm_model_used: Optional[str] = None

    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Errors
    error: Optional[str] = None
