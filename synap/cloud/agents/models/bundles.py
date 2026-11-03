"""Context bundle models."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum


class DegradationLevel(Enum):
    """Indicates retrieval quality degradation."""
    NONE = "none"          # Full retrieval succeeded
    PARTIAL = "partial"    # Some stores failed, partial results
    MINIMAL = "minimal"    # Only cache/fallback available
    FAILED = "failed"      # Complete failure


@dataclass
class ContextItem:
    """
    Individual context item retrieved from stores.

    Items are categorized by type and include relevance metadata.
    """
    item_id: str
    content: str
    context_type: str           # facts | preferences | episodes | emotions | temporal
    source: str                 # "vector" | "graph" | "cache" | "postgres"

    # Relevance scoring
    similarity_score: float = 0.0    # Cosine similarity from vector search
    relevance_score: float = 0.0     # Final relevance after reranking
    confidence: float = 0.0          # Original confidence from ingestion

    # Metadata
    scope: str = "user"              # user | customer | client
    entity_id: Optional[str] = None  # Related entity if from graph
    entity_name: Optional[str] = None  # Entity canonical name (for Neo4j resolution)
    created_at: Optional[datetime] = None

    # Temporal metadata
    event_date: Optional[datetime] = None         # when the event happened: event_start ONLY (the valid_from and DCT fallbacks were removed 2026-10-10)
    valid_until: Optional[datetime] = None
    # The rest of the event window (migration 154). event_date is its start.
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None         # day|week|weekend|month|season|year|range|approximate
    # True when ``event_date`` is NOT a date anything told us the event happened
    # on: no event_start and no valid_from, so the read path fell back to the
    # source document's own date. It therefore says when the user SPOKE, not
    # when the thing happened. Measured on the lme3 failure scopes, 2,164 of
    # 3,259 served memories (66.4%) are in this state, and the renderer used to
    # print them indistinguishably from a real event date. Two consumers care:
    # the date prefix says "said" instead of asserting the date, and the
    # temporal window filter never drops on a date we invented.
    # When the user SAID it, as opposed to when it happened. The source
    # document's own date (DCT), carried whether or not an event window exists.
    #
    # Why both dates have to travel together: a memory with a stored window
    # renders bare, "(2022) ... five sessions", and a memory without one renders
    # "(said May 11, 2023) ... three sessions". Served side by side those are two
    # statements about one event, and the only thing that can order them is when
    # each was SAID — which the first row was not printing. In one case
    # the five-sessions row was spoken on Oct 30, 2023, five months
    # after the three-sessions row, and both readers answered three because
    # "May 11, 2023" looks later than "2022".
    said_at: Optional[datetime] = None
    temporal_category: Optional[str] = None       # perpetual | temporal_fact | episode
    temporal_confidence: float = 0.0

    # Graph traversal metadata
    hop_distance: int = 0                         # BFS hop distance from anchor (0 = direct match)

    # Source tracking (for chunk evidence retrieval)
    source_document_id: Optional[str] = None      # Links back to ingested conversation/document

    # Chunk evidence (populated by retrieval-time enrichment)
    source_evidence: Optional[List[str]] = None    # Top BM25-scored sentences from source chunks

    # Where a stored PII placeholder was swapped back for the real value in
    # `content`, as [{"start", "end", "field_type"}]. Written by the read path
    # on the way out (see privacy/read_path.reveal_objects) so a client UI can
    # mark the swap instead of showing a revealed value as though it had never
    # been protected. A real field rather than a stashed attribute because
    # anything serialising this item has to know the offsets exist: a step that
    # rewrites `content` without moving them puts every highlight in the wrong
    # place, which is precisely what the date prefix does.
    pii_revealed: Optional[List[dict]] = None

    # KU v2 lossless-merge: latest contributor DCT for merged memories.
    # Equals `created_at` for non-merged memories. When the two differ,
    # the date prefix renderer shows a range like "(Jan 2023 – Jul 2023)".
    document_last_observed_at: Optional[datetime] = None

    # Provenance (speaker attribution): who asserted the underlying fact —
    # "user" | "assistant" | "system" | None (unknown/legacy). Written to
    # Chroma metadata + memories.assertion_source by ingestion since
    # migration 071; threaded here so retrieval can discount or label
    # assistant-inferred items.
    assertion_source: Optional[str] = None

    # The name of the custom category this memory was extracted under, read from
    # the stored `category`. Set only when it equals one of the instance's enabled custom
    # categories and SYNAP_CUSTOM_CATEGORIES is on, so an old non-empty `category` is never
    # served as one. Carried through to_dict/from_dict for the 300 s cache.
    custom_category: Optional[str] = None

    # Which real-world thing this item is about, and how that thing is named
    # (migration 159). Two items sharing a subject_id are two statements about
    # one thing, which is what a reader counting "how many pieces of furniture"
    # has to know and could previously only guess. Both None for every memory
    # ingestion did not group, which is most of them and was all of them before.
    #
    # Filled from Postgres at read time rather than carried in Chroma metadata,
    # because a memory joins a subject AFTER its own vector is written: the
    # partner's row is updated too, and updating Chroma metadata for a row
    # written in an earlier request is a backfill problem this avoids entirely.
    subject_id: Optional[str] = None
    subject_label: Optional[str] = None

    # The caller's search queries that found this item, in the order they were
    # sent. A fetch can carry several, and they all come back as one set, so
    # without this a caller cannot tell which of its questions an item answers.
    # A result of a subquery or template the server wrote carries the caller
    # query it was written from. Empty when the fetch named no query.
    matched_queries: List[str] = field(default_factory=list)

    # The number this item was ranked by, as published to the caller in
    # `metadata.rank_score`: higher is better, and sorting on it descending
    # reproduces the served order. Set by `assemble_context_bundle`.
    #
    # A declared field, not only the stashed `_final_score`, because a stash does
    # not survive `to_dict`/`from_dict`: an L1 cache hit or an anticipation
    # bundle came back with no score at all, so the same memory carried
    # `rank_score` live and lost it within the 300s TTL. None until ranked.
    rank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content": self.content,
            "context_type": self.context_type,
            "source": self.source,
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "confidence": self.confidence,
            "scope": self.scope,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "created_at": self.created_at.isoformat() if self.created_at and hasattr(self.created_at, 'isoformat') else self.created_at,
            "event_date": self.event_date.isoformat() if self.event_date and hasattr(self.event_date, 'isoformat') else self.event_date,
            "valid_until": self.valid_until.isoformat() if self.valid_until and hasattr(self.valid_until, 'isoformat') else self.valid_until,
            "event_end": self.event_end.isoformat() if self.event_end and hasattr(self.event_end, 'isoformat') else self.event_end,
            "event_precision": self.event_precision,
            # MUST be carried. The cache round-trips every item through
            # to_dict/from_dict, and a missing key here defaults the flag
            # to False on the way back. That had two effects: the date
            # prefix rendered bare on a cache hit and "(said ...)" live,
            # for the same memory within the 300s TTL; and worse, the
            # temporal window filter reads this flag to decide what it is
            # allowed to drop, so a cache hit turned a borrowed date into
            # a real one and the filter could delete the answer. Found by
            # the benchmark team, 2026-10-01.
            # Carried for the same reason as the flag above: the cache
            # round-trips every item through to_dict/from_dict, and a key
            # missing here renders the prefix one way live and another way on a
            # cache hit, for the same memory inside the 300s TTL.
            "said_at": self.said_at.isoformat() if self.said_at and hasattr(self.said_at, "isoformat") else self.said_at,
            "temporal_category": self.temporal_category,
            "temporal_confidence": self.temporal_confidence,
            "hop_distance": self.hop_distance,
            "source_document_id": self.source_document_id,
            "source_evidence": self.source_evidence,
            "document_last_observed_at": (
                self.document_last_observed_at.isoformat()
                if self.document_last_observed_at and hasattr(self.document_last_observed_at, "isoformat")
                else self.document_last_observed_at
            ),
            "assertion_source": self.assertion_source,
            "custom_category": self.custom_category,
            "subject_id": self.subject_id,
            "subject_label": self.subject_label,
            # Carried, like the two flags above: a key missing here comes back
            # empty on a cache hit, and the per-query share of the results is
            # decided from it.
            "matched_queries": list(self.matched_queries),
            "rank_score": self.rank_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextItem":
        return cls(
            item_id=data["item_id"],
            content=data["content"],
            context_type=data["context_type"],
            source=data["source"],
            similarity_score=data.get("similarity_score", 0.0),
            relevance_score=data.get("relevance_score", 0.0),
            confidence=data.get("confidence", 0.0),
            scope=data.get("scope", "user"),
            entity_id=data.get("entity_id"),
            entity_name=data.get("entity_name"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            event_date=datetime.fromisoformat(data["event_date"]) if data.get("event_date") else None,
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            event_end=datetime.fromisoformat(data["event_end"]) if data.get("event_end") else None,
            event_precision=data.get("event_precision"),
            said_at=datetime.fromisoformat(data["said_at"]) if data.get("said_at") else None,
            temporal_category=data.get("temporal_category"),
            temporal_confidence=data.get("temporal_confidence", 0.0),
            hop_distance=data.get("hop_distance", 0),
            source_document_id=data.get("source_document_id"),
            source_evidence=data.get("source_evidence"),
            document_last_observed_at=(
                datetime.fromisoformat(data["document_last_observed_at"])
                if data.get("document_last_observed_at") else None
            ),
            assertion_source=data.get("assertion_source"),
            custom_category=data.get("custom_category"),
            subject_id=data.get("subject_id"),
            subject_label=data.get("subject_label"),
            matched_queries=list(data.get("matched_queries") or []),
            rank_score=data.get("rank_score"),
        )


@dataclass
class ConversationContext:
    """Current session context from compaction + raw buffer.

    Kept as a coherent structured block, NOT atomized into ContextItems.
    Populated only when a conversation_id is present in the request.
    """
    summary: Optional[str] = None
    current_state: Dict[str, Any] = field(default_factory=dict)
    key_extractions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    recent_turns: List[Dict[str, Any]] = field(default_factory=list)
    compaction_id: Optional[str] = None
    compacted_at: Optional[str] = None
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "current_state": self.current_state,
            "key_extractions": self.key_extractions,
            "recent_turns": self.recent_turns,
            "compaction_id": self.compaction_id,
            "compacted_at": self.compacted_at,
            "conversation_id": self.conversation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        return cls(
            summary=data.get("summary"),
            current_state=data.get("current_state", {}),
            key_extractions=data.get("key_extractions", {}),
            recent_turns=data.get("recent_turns", []),
            compaction_id=data.get("compaction_id"),
            compacted_at=data.get("compacted_at"),
            conversation_id=data.get("conversation_id"),
        )


@dataclass
class ContextBundle:
    """
    Immutable result of executing a ContextDecision.

    The bundle is categorized by context_type for easy SDK consumption.
    Includes degradation info for graceful handling of partial failures.
    """
    bundle_id: str
    decision_id: str

    # Items categorized by type
    items_by_type: Dict[str, List[ContextItem]] = field(default_factory=dict)
    # Structure: {"facts": [ContextItem, ...], "preferences": [...], ...}

    # Token accounting
    total_tokens: int = 0
    token_budget: int = 0
    budget_exceeded: bool = False

    # Retrieval metadata
    retrieval_mode: str = "fast"
    sources_queried: List[str] = field(default_factory=list)  # ["vector", "graph", "cache"]

    # Degradation info
    degradation_level: DegradationLevel = DegradationLevel.NONE
    warnings: List[str] = field(default_factory=list)
    # WHY the retrieval degraded, in a closed vocabulary a caller can branch on:
    # "timeout" | "overloaded" | "unavailable" | "internal".
    #
    # Separate from ``warnings`` because warnings are prose for a human and this
    # is for code. It exists because the prose was the only signal and it named
    # our internals: a failed embedding call served the literal string "embed
    # failed after 2 attempts:" to clients, which tells them which service of
    # ours broke and nothing they can act on. The caller needs to know that the
    # answer is thin and whether retrying is worth it; that is this field.
    failure_code: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_time_ms: int = 0

    # Cache
    cache_key: Optional[str] = None
    cache_hit: bool = False

    # Anticipation metadata (propagated to SDK for BM25 matching)
    search_queries: List[str] = field(default_factory=list)
    search_keywords: List[str] = field(default_factory=list)  # Expanded vocabulary keywords

    # Conversation context (current session from compaction, separate from long-term items)
    conversation_context: Optional[ConversationContext] = None

    # The rung this bundle was retrieved at, written the way a caller names one
    # (`scope_rung.rung_key_from_rungs`). None for every request that resolves
    # no chain, which is almost all of them.
    #
    # Deliberately absent from `to_dict` and `from_dict`. It describes the
    # REQUEST, not the content, and the two serialisers feed the HTTP response
    # body and the L1 cache entry, neither of which should grow a field for it:
    # the L1 key already carries the chain, so a cached entry can only ever be
    # returned to a request at the same rung, and `fetch_context` restamps from
    # that request's own scope on both the hit and the miss path. Carrying it
    # through the cache instead would make the stamp say where the FIRST caller
    # stood, which is the one thing it must never say.
    scope_rung: Optional[str] = None

    # Every served item's id once, in the order the engine ranked them across
    # types, chunks included (see `assemble_context_bundle`). `items_by_type`
    # stays the source of the items themselves; this is only their order, read
    # by `response_shape="ranked"`. Empty for a bundle built before it existed,
    # and the reader then sorts on `rank_score` instead.
    ranked_item_ids: List[str] = field(default_factory=list)

    @property
    def all_items(self) -> List[ContextItem]:
        """Flatten all items into single list."""
        items = []
        for type_items in self.items_by_type.values():
            items.extend(type_items)
        return items

    @property
    def item_count(self) -> int:
        """Total number of items across all types."""
        return sum(len(items) for items in self.items_by_type.values())

    def get_items(self, context_type: str) -> List[ContextItem]:
        """Get items of specific type."""
        return self.items_by_type.get(context_type, [])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "bundle_id": self.bundle_id,
            "decision_id": self.decision_id,
            "items_by_type": {
                ctx_type: [item.to_dict() for item in items]
                for ctx_type, items in self.items_by_type.items()
            },
            "total_tokens": self.total_tokens,
            "token_budget": self.token_budget,
            "budget_exceeded": self.budget_exceeded,
            "retrieval_mode": self.retrieval_mode,
            "sources_queried": self.sources_queried,
            "degradation_level": self.degradation_level.value,
            "warnings": self.warnings,
            "failure_code": self.failure_code,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, 'isoformat') else self.created_at,
            "retrieval_time_ms": self.retrieval_time_ms,
            "cache_hit": self.cache_hit,
            "item_count": self.item_count,
            "search_queries": self.search_queries,
            "search_keywords": self.search_keywords,
            "conversation_context": self.conversation_context.to_dict() if self.conversation_context else None,
            "ranked_item_ids": list(self.ranked_item_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextBundle":
        items_by_type = {}
        for ctx_type, item_dicts in data.get("items_by_type", {}).items():
            items_by_type[ctx_type] = [ContextItem.from_dict(d) for d in item_dicts]

        return cls(
            bundle_id=data["bundle_id"],
            decision_id=data["decision_id"],
            items_by_type=items_by_type,
            total_tokens=data.get("total_tokens", 0),
            token_budget=data.get("token_budget", 0),
            budget_exceeded=data.get("budget_exceeded", False),
            retrieval_mode=data.get("retrieval_mode", "fast"),
            sources_queried=data.get("sources_queried", []),
            degradation_level=DegradationLevel(data.get("degradation_level", "none")),
            warnings=data.get("warnings", []),
            # Absent in bundles cached by a build that predates the field; None
            # is the same thing the field means when nothing failed.
            failure_code=data.get("failure_code"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            retrieval_time_ms=data.get("retrieval_time_ms", 0),
            cache_key=data.get("cache_key"),
            cache_hit=data.get("cache_hit", False),
            search_queries=data.get("search_queries", []),
            search_keywords=data.get("search_keywords", []),
            conversation_context=ConversationContext.from_dict(data["conversation_context"]) if data.get("conversation_context") else None,
            ranked_item_ids=list(data.get("ranked_item_ids") or []),
        )

    def to_sdk_response(self) -> Dict[str, Any]:
        """
        Simplified format for SDK response.
        Items grouped by type, minimal metadata.
        """
        return {
            "bundle_id": self.bundle_id,
            "items": {
                ctx_type: [{"content": item.content, "confidence": item.confidence} for item in items]
                for ctx_type, items in self.items_by_type.items()
            },
            "total_tokens": self.total_tokens,
            "degradation": self.degradation_level.value if self.degradation_level != DegradationLevel.NONE else None,
            "conversation_context": self.conversation_context.to_dict() if self.conversation_context else None,
        }
