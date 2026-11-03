"""MACA config schema models."""

import copy
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import (
    ChunkingStrategy,
    CompactionStrategy,
    ExtractionAggressiveness,
    Priority,
    RetrievalStrategy,
    StorageMode,
)
from .value_types import describe as _describe_value, find_value_problems

logger = logging.getLogger(__name__)


# --- Role semantics ---


class RoleSemantics(BaseModel):
    """Who is the Client, Customer, and User for this instance type.

    These descriptions clarify the real-world actors behind the abstract
    scoping hierarchy (client → customer → user) for a given instance type.
    Defined here, not in the MACA agent's ``inputs.py``, so the schema does not
    import the MACA folder; ``inputs.py`` re-exports it.
    """
    client_description: str = Field(..., description="Who the 'Client' is (the company building/owning the agent)")
    customer_description: str = Field(..., description="Who the 'Customer' is (the entity the agent ultimately serves)")
    user_description: str = Field(..., description="Who the 'User' is (the person who directly interacts with the agent)")


# --- Memory Categories ---


class MemoryCategoryConfig(BaseModel):
    """Configuration for a memory category."""
    enabled: bool = True
    priority: Optional[Priority] = None
    extraction_aggressiveness: Optional[ExtractionAggressiveness] = None
    retention_days: Optional[int] = None
    examples: List[str] = Field(default_factory=list)
    rationale: Optional[str] = None
    # What to look for when extracting THIS category, in this instance's words.
    # agent_hints.ingestion_agent.focus_areas is one list for the whole
    # instance, and MACA writes items that name several memory types at once,
    # so the facts call was reading guidance about preferences and dates. When
    # a category carries its own list, its call uses that instead.
    focus_areas: List[str] = Field(default_factory=list)


class CustomCategoryConfig(BaseModel):
    """Configuration for a custom memory category."""
    name: str
    enabled: bool = True
    priority: Priority = Priority.MEDIUM
    description: str
    examples: List[str] = Field(default_factory=list)


class MemoryCategoriesConfig(BaseModel):
    """All memory category configurations."""
    facts: MemoryCategoryConfig = Field(default_factory=MemoryCategoryConfig)
    preferences: MemoryCategoryConfig = Field(default_factory=MemoryCategoryConfig)
    episodes: MemoryCategoryConfig = Field(default_factory=MemoryCategoryConfig)
    emotions: MemoryCategoryConfig = Field(default_factory=MemoryCategoryConfig)
    temporal_events: MemoryCategoryConfig = Field(default_factory=MemoryCategoryConfig)
    custom_categories: List[CustomCategoryConfig] = Field(default_factory=list)


# --- Memory Extraction ---


class ExtractionTriggers(BaseModel):
    """What triggers memory extraction.

    Only ``on_session_end`` is left. It is read by the conversation tracker to
    let one agent opt out of compacting a conversation when its session ends
    (``application/conversation/tracker.py``). The other three switches never
    drove anything and were removed.
    """
    on_session_end: bool = True


class ExtractionFilters(BaseModel):
    """Filters for extraction quality."""
    min_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    require_entity_grounding: bool = True
    # DEPRECATED. Declared only so live configs that carry the key still load;
    # nothing reads it and MACA no longer writes it. It never drove anything:
    # the knowledge-update step is built without the config and gates on its own
    # constant (knowledge_update.py SIMILARITY_THRESHOLD), so a config saying
    # 0.85 ran at that constant. A cosine cutoff depends on the embedding
    # model, which a client document says nothing about, so it is ours to
    # derive. See synap/docs/architecture/similarity_threshold_calibrator.md.
    # No range check, so an ignored value can never be what rejects a config;
    # excluded from dumps, so a re-saved config stops carrying it.
    similarity_threshold: Optional[float] = Field(None, exclude=True)


class DomainHints(BaseModel):
    """Domain-specific extraction hints."""
    industry: Optional[str] = None
    key_entities: List[str] = Field(default_factory=list)
    key_relationships: List[str] = Field(default_factory=list, description="Relationship types to focus on. Must be UPPERCASE (e.g., WORKS_AT, MANAGES).")
    # How many entities one document may yield and which to keep first. Unset
    # means no limit and no ranking: every named entity of an allowed type is
    # returned. Read by the v2 entity prompt only; a scope-level extractor sets
    # its own.
    max_entities: Optional[int] = Field(default=None, ge=1)
    entity_priority: Optional[str] = None


class MemoryExtractionConfig(BaseModel):
    """Complete extraction configuration."""
    triggers: ExtractionTriggers = Field(default_factory=ExtractionTriggers)
    filters: ExtractionFilters = Field(default_factory=ExtractionFilters)
    domain_hints: DomainHints = Field(default_factory=DomainHints)


# --- Storage Architecture ---


class VectorConfig(BaseModel):
    """Vector store configuration."""
    enabled: bool = True
    use_for: List[str] = Field(default_factory=lambda: ["semantic_search", "similarity_matching"])
    embedding_model: str = "default"
    index_type: str = "hnsw"


class GraphConfig(BaseModel):
    """Graph store configuration."""
    enabled: bool = True
    use_for: List[str] = Field(default_factory=lambda: ["entity_relationships", "temporal_reasoning"])
    graph_type: str = "property_graph"
    entity_canonicalization: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "merge_similar_entities": True, "similarity_threshold": 0.9}
    )


class GraphSchemaConfig(BaseModel):
    """Configuration for a single graph schema within an instance."""
    schema_name: str
    schema_type: str = "knowledge"  # GraphSchemaType values; see value_types.py
    enabled: bool = True
    use_for: List[str] = Field(default_factory=lambda: ["entity_relationships"])
    graph_type: str = "property_graph"
    entity_canonicalization: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "merge_similar_entities": True, "similarity_threshold": 0.9}
    )
    node_types: List[str] = Field(default_factory=list)
    edge_types: List[str] = Field(default_factory=list, description="Graph edge types for this schema. Must be UPPERCASE (e.g., WORKS_AT, USES).")
    description: Optional[str] = None


class ScopingConfig(BaseModel):
    """Data scoping configuration."""
    # `primary_scope` used to live here. It was parsed, validated, echoed in a
    # few status payloads, and never once used to change behaviour. Nested
    # scoping replaces the question it was asking: which level owns a memory is
    # now the ladder's `tenant_of_record` and `identity` roles, and which level
    # a particular fact belongs at is IACS's decision per fact rather than one
    # setting per instance. Carrying a dead scoping field into a larger scoping
    # system compounds the confusion, so it is gone (Z-05).
    #
    # Stored MACA configs still carry the key. Pydantic ignores extra keys, so
    # they keep parsing and no migration is needed.
    user_context_isolation: str = "strict"  # UserContextIsolation values; see value_types.py
    allow_cross_user_retrieval: bool = False
    allow_cross_customer_retrieval: bool = False
    # False by default, not True: this is the exact field the original
    # documented case got wrong (a B2C support agent inherited
    # allow_cross_instance_for_subagents=true from a wrong template pick).
    # It is opt-in cross-instance sharing, the same shape of risk as
    # allow_cross_user_retrieval/allow_cross_customer_retrieval just above,
    # both of which already defaulted safe. This one didn't, and a later
    # real run reproduced the same wrong value again -- not via the old
    # bug's mechanism (the template pick was correct this time), but
    # because both the Pydantic default and the DESIGN prompt's own
    # guidance ("typically true") pushed the model toward it regardless of
    # what the document said.
    allow_cross_instance_for_subagents: bool = False
    # Consumed at runtime, despite what this comment used to say. It is read in
    # `retrieval_manager._load_maca_config` and drives the scope filter, the
    # graph traversal and the rollup branches. The comment here claimed it was
    # dead, which is the kind of note that gets someone to delete a field and
    # break retrieval.
    #
    # What is NOT built is the ingestion half it also described: aggregating
    # user-level material into customer knowledge after ingestion. That is a
    # separate feature and IACS is the thing that would do it.
    roll_up_to_customer: bool = True
    # MACA scoping recommendations — set by MACA when it evaluates
    # cross-user/cross-customer sharing and flagged for human review
    cross_user_recommendation: Optional[str] = None
    cross_customer_recommendation: Optional[str] = None


class StorageArchitectureConfig(BaseModel):
    """Complete storage architecture."""
    primary_mode: StorageMode = StorageMode.HYBRID
    vector_config: VectorConfig = Field(default_factory=VectorConfig)
    graph_config: GraphConfig = Field(default_factory=GraphConfig)
    graph_schemas: List[GraphSchemaConfig] = Field(default_factory=list)
    scoping: ScopingConfig = Field(default_factory=ScopingConfig)
    # A-01. The scope ladder MACA proposed from the client's use-case document.
    #
    # A plain dict rather than a typed model on purpose: it crosses to the
    # dashboard unchanged and nothing in the workflow branches on its shape, so
    # a schema here would only be a second place to keep in step.
    #
    # ⚠ It has to be DECLARED, though. `_assemble_config` was already setting
    # this key and Pydantic was dropping it at validation, because extra fields
    # are ignored by default. The write ran, the config saved, and the panel
    # showed nothing, with no error anywhere. Declaring the field is the whole
    # fix.
    #
    # This is a PROPOSAL, never configuration. A-03 keeps the real ladder in
    # `scope_level` rows precisely so identity does not ride a document that
    # regeneration rewrites. What lives here is the suggestion and its
    # reasoning, and it SHOULD be replaced on a rerun, because that is a fresh
    # reading of the same document.
    proposed_ladder: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def sync_graph_fields(self) -> "StorageArchitectureConfig":
        """If graph_schemas is populated, sync graph_config.enabled accordingly."""
        if self.graph_schemas:
            self.graph_config.enabled = any(gs.enabled for gs in self.graph_schemas)
        return self


# --- Retrieval ---


class RetrievalModeConfig(BaseModel):
    """Configuration for a retrieval mode."""
    enabled: bool = True
    max_latency_ms: int = 100
    strategy: RetrievalStrategy = RetrievalStrategy.CACHE_FIRST
    use_graph_traversal: bool = False
    fallback: Optional[str] = None


class RankingConfig(BaseModel):
    """Retrieval ranking configuration."""
    primary_signals: Dict[str, float] = Field(
        default_factory=lambda: {"recency": 0.3, "relevance": 0.5, "importance": 0.2}
    )
    boost_categories: Dict[str, float] = Field(default_factory=dict)
    decay_function: str = "exponential"
    decay_half_life_days: int = 30


class AnticipationConfig(BaseModel):
    """Proactive retrieval configuration.

    Section 5 of the Anticipation Rebuild added the runtime knobs below.
    Until Section 8's unified reasoning loop lands, the existing four
    spawn paths continue to use hardcoded defaults — these fields are
    read by the reasoning loop when it ships.
    """

    # Existing knobs (preserved; defaults unchanged).
    enabled: bool = True
    prefetch_on_session_start: bool = True
    prefetch_categories: List[str] = Field(default_factory=lambda: ["facts", "preferences"])
    max_prefetch_items: int = 20

    # === Section 5 — declarative runtime knobs ===============================
    # These fields control behavior that today is hardcoded inside
    # ``_spawn_llm_anticipation_prefetch`` and friends. Section 8's reasoning
    # loop reads them so tenants can tune anticipation per instance.

    fire_on_signals: List[str] = Field(
        default_factory=lambda: ["session_start", "assistant_message"],
        description=(
            "Which signal types should the agent consider firing anticipation "
            "on. Allowed values: session_start, user_message, assistant_message, "
            "tool_call, tool_result."
        ),
    )
    token_budget_per_prefetch: int = Field(
        default=1500,
        ge=100,
        le=20000,
        description="Token budget passed to RetrievalManager.fetch_context per anticipation candidate.",
    )
    max_results_per_prefetch: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Max items returned by RetrievalManager.fetch_context per anticipation candidate.",
    )
    mode: str = Field(
        default="fast",
        description="Retrieval mode for anticipation prefetches. One of: fast, accurate.",
    )
    cooldown_seconds: int = Field(
        default=5,
        ge=0,
        le=3600,
        description=(
            "Minimum interval between two anticipations on the same conversation. "
            "Zero disables the cooldown."
        ),
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Drop LLM anticipation hints whose confidence is below this floor.",
    )
    pattern_priority_floor: str = Field(
        default="low",
        description=(
            "Lowest pattern priority that participates in candidate generation. "
            "Patterns below this floor are suppressed even if matched."
        ),
    )

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in {"fast", "accurate"}:
            raise ValueError(f"mode must be 'fast' or 'accurate', got {v!r}")
        return v

    @field_validator("pattern_priority_floor")
    @classmethod
    def _validate_priority_floor(cls, v: str) -> str:
        if v not in {"low", "medium", "high"}:
            raise ValueError(
                f"pattern_priority_floor must be 'low', 'medium', or 'high', got {v!r}"
            )
        return v

    @field_validator("fire_on_signals")
    @classmethod
    def _validate_fire_on_signals(cls, v: List[str]) -> List[str]:
        allowed = {"session_start", "user_message", "assistant_message", "tool_call", "tool_result"}
        bad = [s for s in v if s not in allowed]
        if bad:
            raise ValueError(
                f"fire_on_signals contains unknown signal types: {bad}. "
                f"Allowed: {sorted(allowed)}"
            )
        return v


class ContextBudgetConfig(BaseModel):
    """Context window budget allocation."""
    max_tokens: int = 4000
    reserve_for_conversation: int = 2000
    reserve_for_memories: int = 1500
    reserve_for_org_context: int = 500


class AnticipationPattern(BaseModel):
    """Pattern for anticipatory retrieval.

    scenario: Natural language description of when this pattern applies.
              Used for semantic matching at runtime.
    context_types: Memory categories to prefetch.
    search_queries: Specific retrieval queries to execute.
    priority: Urgency of prefetch (affects ordering).
    activation_hints: Optional structured hints that let the agent activate
              this pattern deterministically rather than via the LLM alone.
              All fields are optional; presence of any field is treated as a
              hint contribution. The reasoning loop (Section 8) combines these
              with the LLM's signal classification when scoring patterns.
    """
    scenario: str
    context_types: List[str]
    search_queries: List[str] = Field(default_factory=list)
    priority: str = "medium"
    activation_hints: Optional["PatternActivationHints"] = Field(
        default=None,
        description="Optional structured triggers. None = LLM-only activation.",
    )
    scope_hint: Optional[str] = Field(
        default=None,
        description=(
            "Explicit scope tier for the prefetched bundle: 'user' | 'customer' "
            "| 'client'. None triggers a heuristic in the reasoning loop:\n"
            "  - context_types {preferences, episodes, emotions} → 'user'\n"
            "  - search_queries mentioning policy/general/overview → 'client'\n"
            "  - otherwise → 'user' (safe default).\n"
            "Drives the SDK funnel lookup: a request with "
            "(user_id, customer_id, client_id) matches bundles stored at "
            "any of those scopes plus the '_any' sentinel — mirroring "
            "RetrievalManager's narrow-to-broad vector filter."
        ),
    )

    @field_validator("scope_hint")
    @classmethod
    def _validate_scope_hint(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"user", "customer", "client"}
        if v not in allowed:
            raise ValueError(
                f"scope_hint must be one of {sorted(allowed)} or None, got {v!r}"
            )
        return v


class PatternActivationHints(BaseModel):
    """Structured triggers used by the reasoning loop to activate a pattern.

    Section 5 schema; consumed by Section 8's reasoning loop.

    Each field is independently optional. The reasoning loop ORs them: any
    matching hint counts as activation evidence. Empty hints object = no
    deterministic activation, fall through to LLM.
    """

    signal_types: List[str] = Field(
        default_factory=list,
        description="Activate when one of these signal types arrives.",
    )
    intent_labels: List[str] = Field(
        default_factory=list,
        description="Activate when the classified user intent matches one of these.",
    )
    keyword_matches: List[str] = Field(
        default_factory=list,
        description="Activate when any of these keywords (case-insensitive) appear in the recent arc.",
    )
    entity_types: List[str] = Field(
        default_factory=list,
        description="Activate when an entity of one of these types is mentioned.",
    )
    affective_states: List[str] = Field(
        default_factory=list,
        description=(
            "Activate when the user shows one of these affective states "
            "(e.g. frustration, confusion, excitement). Set by the upstream classifier."
        ),
    )

    @field_validator("signal_types")
    @classmethod
    def _validate_signal_types(cls, v: List[str]) -> List[str]:
        allowed = {"session_start", "user_message", "assistant_message", "tool_call", "tool_result"}
        bad = [s for s in v if s not in allowed]
        if bad:
            raise ValueError(
                f"signal_types contains unknown values: {bad}. Allowed: {sorted(allowed)}"
            )
        return v


class RetrievalConfig(BaseModel):
    """Complete retrieval configuration."""
    default_mode: str = "fast"
    modes: Dict[str, RetrievalModeConfig] = Field(default_factory=dict)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    anticipation: AnticipationConfig = Field(default_factory=AnticipationConfig)
    context_budget: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)
    fallback_queries: List[str] = Field(default_factory=list)
    # Workstream B lever 2 — consumed by retrieval_manager._reserve_per_type.
    # Guarantees at least this many items of each context_type survive the
    # final trim (0 = single global relevance cut, today's default). Declared
    # here so config round-trips through the schema don't drop the key.
    per_type_floor: int = 0


# --- Compaction ---


class AdaptiveThresholds(BaseModel):
    """Thresholds for adaptive compaction.

    Compaction v3 reads ``target_ratio``, ``idle_minutes`` and
    ``recompaction_idle_minutes`` (``compaction/templates/loader.py``). When to
    compact otherwise follows the agent's model: T = window x 15/128 tokens
    of record plus raw turns, or 30 new messages (compaction/budgets.py).

    ``trigger_at_tokens``, ``trigger_at_messages`` and
    ``incremental_message_delta`` are kept so stored configs round-trip, and
    their defaults match v3's 128K numbers, but compaction does not read them
    until MACA assesses them per agent (COMPACTION_V3_PLAN.md §15).
    ``target_tokens`` and ``preserve_last_n_turns`` are retired in compaction
    v3 and nothing reads them. They stay declared only because MACA's own
    design step still writes them and its rubric (CP5-CP7) still scores them;
    both go when MACA learns to assess compaction.
    """
    trigger_at_tokens: int = 15000
    trigger_at_messages: int = 30
    incremental_message_delta: int = 30
    target_tokens: int = 1500  # retired, unread
    preserve_last_n_turns: int = 3  # retired, unread
    # Each compaction update adds about this share of the new turns' tokens,
    # ±5 points.
    target_ratio: float = 0.30

    # How long a conversation must be inactive before the sweep compacts it.
    # This is the knob a voice agent turns down: its turns arrive seconds apart,
    # so waiting the text default leaves memory unwritten for most of a call.
    idle_minutes: int = 5
    # How long after an EXISTING compaction before inactivity alone justifies
    # re-compacting. Separate from idle_minutes because the question is
    # different — not "has this conversation gone quiet" but "is what we
    # already summarised stale enough to redo".
    recompaction_idle_minutes: int = 5


class CompactionConfig(BaseModel):
    """Context compaction configuration."""
    # Automatic compaction on or off. Off stops the triggers nobody asks for:
    # the size test on a recorded message, the idle sweep, the end of a session. The
    # compact route still works, and promotion to long-term memory is never stopped.
    automatic: bool = True
    strategy: CompactionStrategy = CompactionStrategy.ADAPTIVE
    adaptive_thresholds: AdaptiveThresholds = Field(default_factory=AdaptiveThresholds)
    preservation_priorities: List[str] = Field(default_factory=list)
    summarization_candidates: List[str] = Field(default_factory=list)


# --- Chunking ---


class DocumentTypeChunking(BaseModel):
    """Chunking config for a document type."""
    strategy: ChunkingStrategy
    max_tokens: Optional[int] = None
    max_turns_per_chunk: Optional[int] = None
    overlap_tokens: Optional[int] = None
    overlap_turns: Optional[int] = None
    respect_headers: Optional[bool] = None
    preserve_tables: Optional[bool] = None
    one_record_per_chunk: Optional[bool] = None


class ChunkingConfig(BaseModel):
    """Complete chunking configuration."""
    default_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    by_document_type: Dict[str, DocumentTypeChunking] = Field(default_factory=dict)


# --- Agent Hints ---


class ExtractionPattern(BaseModel):
    """One thing this client's notes must always yield, as an instruction.

    Written by the EXTRACTION PATTERNS step (workflow/extraction_patterns.py),
    read by the graph extraction prompt (ingestion/prompts/extraction/graph_v2).
    """
    name: str
    # One of the fixed categories the generator must cover, or "domain".
    category: str = "domain"
    # Which kinds of note the instruction applies to: "client", "customer", "user".
    applies_to: List[str] = Field(default_factory=list)
    instruction: str


class ExtractionExample(BaseModel):
    """A worked example in the extraction prompt's own input/output format."""
    scope: str
    # Names of the patterns this example shows (not rendered into the prompt).
    shows: List[str] = Field(default_factory=list)
    input: str
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class IngestionAgentHints(BaseModel):
    """Hints for ingestion agent."""
    focus_areas: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    domain_context: Optional[str] = None
    extraction_guidance: Optional[str] = None
    # Written by the EXTRACTION PATTERNS step after DESIGN, not by a DESIGN call.
    # Empty on every config generated before the step existed; the extractor
    # then keeps the prompt's built-in examples.
    #   extraction_scope_notes: what a client / customer / user note is for this client
    #   extraction_patterns:    instructions, each tagged with the note kinds it applies to
    #   extraction_examples:    worked examples, one set per kind of note
    extraction_scope_notes: Dict[str, str] = Field(default_factory=dict)
    extraction_patterns: List[ExtractionPattern] = Field(default_factory=list)
    extraction_examples: List[ExtractionExample] = Field(default_factory=list)


class RetrievalAgentHints(BaseModel):
    """Hints for retrieval agent."""
    anticipation_patterns: List[AnticipationPattern] = Field(default_factory=list)
    context_injection_style: str = "structured"
    ranking_guidance: Optional[str] = None


class CompactionAgentHints(BaseModel):
    """Hints for compaction agent."""
    preserve_signals: List[str] = Field(default_factory=list)


class AgentHints(BaseModel):
    """Hints for all downstream agents."""
    ingestion_agent: IngestionAgentHints = Field(default_factory=IngestionAgentHints)
    retrieval_agent: RetrievalAgentHints = Field(default_factory=RetrievalAgentHints)
    compaction_agent: CompactionAgentHints = Field(default_factory=CompactionAgentHints)


# --- Guardrails ---


class RetentionConfig(BaseModel):
    """Data retention policies."""
    max_memory_age_days: int = 365
    episode_retention_days: int = 90
    conversation_retention_days: int = 30


class ComplianceConfig(BaseModel):
    """Compliance requirements."""
    frameworks: List[str] = Field(default_factory=list)
    prohibited_memory_content: List[str] = Field(default_factory=list)
    required_audit_fields: List[str] = Field(
        default_factory=lambda: ["created_at", "source_document_id", "extraction_confidence"]
    )


class RateLimitsConfig(BaseModel):
    """Rate limiting configuration."""
    max_memories_per_user_per_day: int = 100
    max_retrieval_calls_per_minute: int = 60


class GuardrailsConfig(BaseModel):
    """Complete guardrails configuration."""
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)


# --- Evolution ---


class EvolutionConfig(BaseModel):
    """Config evolution settings."""
    auto_evolution_enabled: bool = False
    regeneration_triggers: List[str] = Field(
        default_factory=lambda: ["use_case_change", "agent_type_change", "compliance_requirement_change"]
    )
    schema_version: str = "1.0.0"
    min_compatible_version: str = "1.0.0"


# --- User Profile ---


class CriticalAttributeConfig(BaseModel):
    """A client-configured critical attribute tracked on the user profile.

    Declared per instance under ``user_profile.critical_attributes``; the
    post-ingestion profile merge extracts/updates these from transcripts
    (spec: Synap_Async_Integration_and_User_Profile_Spec_v1 §5.1).
    Modeled on CustomCategoryConfig; validators follow the in-file
    AnticipationConfig pattern.
    """

    name: str  # snake_case identifier, e.g. "property_preferences"
    type: Literal["string", "number", "boolean", "enum", "list"] = "string"
    description: str = ""  # guides extraction/merge
    allowed_values: Optional[List[str]] = None  # for enum
    required: bool = False  # dashboards flag missing required attrs
    examples: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", v):
            raise ValueError(
                f"critical attribute name must be snake_case "
                f"(lowercase letter, then lowercase/digits/underscores), got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _enum_requires_allowed_values(self) -> "CriticalAttributeConfig":
        if self.type == "enum" and not self.allowed_values:
            raise ValueError(
                f"critical attribute {self.name!r}: type='enum' requires a "
                "non-empty allowed_values list"
            )
        return self


class UserProfileConfig(BaseModel):
    """Per-instance user-profile configuration (spec §5.1).

    Default ``enabled=True`` (product decision 2026-08-07): profiles are ON
    by default. New MACA configs emit an enabled block, and configs that
    predate the feature (no ``user_profile`` key) resolve to enabled at load
    time. With no ``critical_attributes`` configured the Stage-4 merge still
    maintains the overview + extras, so a schema-less tenant gets a useful
    profile out of the box. An instance opts OUT with an explicit
    ``enabled: false`` in its stored config — explicit values always win
    over this default. Declared on MemoryArchitectureConfig so config
    round-trips through the schema (pydantic extra="ignore") don't drop the
    key.
    """

    enabled: bool = True  # ON by default; explicit false in config opts out
    critical_attributes: List[CriticalAttributeConfig] = Field(default_factory=list)
    overview_enabled: bool = True  # maintain a short free-text overview paragraph
    max_profile_tokens: int = Field(1000, ge=100, le=4000)


def user_profile_block_is_meaningful(block: Any) -> bool:
    """Whether a ``user_profile`` block is client-edited state worth preserving.

    A block is "meaningful" when profiles are ``enabled`` OR any
    ``critical_attributes`` are configured. A disabled block with no
    attributes was the pre-2026-08-07 MACA-generated default, so it must
    never override a freshly generated one — with profiles now ON by
    default, this deliberately means such legacy-default blocks get
    replaced (and thereby enabled) at the next regeneration. A client who
    disabled profiles while keeping attributes configured stays meaningful
    and is preserved as-is.

    Accepts a plain dict (the shape stored in ``maca_config.config_content``);
    anything else (``None``, non-mapping) is treated as not meaningful.
    """
    if not isinstance(block, dict):
        return False
    if block.get("enabled") is True:
        return True
    attrs = block.get("critical_attributes")
    return isinstance(attrs, (list, tuple)) and len(attrs) > 0


def carry_forward_user_profile(
    prior_content: Optional[Dict[str, Any]],
    new_content: Dict[str, Any],
) -> Dict[str, Any]:
    """Carry a client-edited ``user_profile`` block into a fresh MACA config.

    MACA config generation and instance promotion build ``user_profile`` from
    scratch, always emitting the schema default (enabled, no attributes).
    Re-running onboarding (or any regeneration/promotion) would therefore
    silently wipe the dashboard-edited block — dropping
    ``critical_attributes`` (spec §5.1, work item §12).

    The prior block is client-edited state, not MACA output, so when it is
    meaningful it **wins wholesale**: it is deep-copied over ``new_content``'s
    ``user_profile``. When the prior block is absent or not meaningful,
    ``new_content`` is left untouched (current behavior preserved).

    Mutates and returns ``new_content``.
    """
    prior_block = prior_content.get("user_profile") if isinstance(prior_content, dict) else None
    if user_profile_block_is_meaningful(prior_block):
        new_content["user_profile"] = copy.deepcopy(prior_block)
    return new_content


# --- Consolidation section (Tech Spec §10; D-009; Implementation Plan H1) ---


class InferenceFieldConfig(BaseModel):
    """One declared inference the cycles compute for this client (H3).

    ``sources`` name data the generic inference node knows how to gather
    (window_memories, curated_memories, recent_corrections, conflict_events,
    anticipation_outcomes, session_metadata). ``output`` decides what the
    result becomes: a memory (CREATE, born-promoted), an annotation (trail
    note on existing memories), or a report item (wake-up report)."""
    name: str
    description: str
    sources: List[str] = Field(default_factory=lambda: ["window_memories"])
    cycle: str = "nap"                    # meditation | nap | sleep
    output: str = "report"                # memory | annotation | report
    enabled: bool = True


class CompressionPolicyConfig(BaseModel):
    """What Sleep's compression may touch (G4 judgment inputs)."""
    protected_categories: List[str] = Field(default_factory=list)
    min_age_days: int = 90
    relevance_ceiling: float = 0.35
    per_category_age_days: Dict[str, int] = Field(default_factory=dict)


class ElevationPolicyConfig(BaseModel):
    """Per-client elevation thresholds (F5); None = ops defaults."""
    min_occurrences: Optional[int] = None
    min_cumulative_confidence: Optional[float] = None
    min_span_days: Optional[int] = None


class SynthesisTargetConfig(BaseModel):
    """One artifact Sleep synthesizes (G3). Profile docs are a different
    feature: targets named or typed 'profile' are skipped by the node."""
    name: str
    description: str = ""
    memory_type: str = "facts"
    output: str = "memory"


class ConsolidationEvolutionConfig(BaseModel):
    """How consolidation may propose changes to this very section (H4)."""
    mutable_sections: List[str] = Field(
        default_factory=lambda: ["inference_fields", "synthesis_targets",
                                 "compression_policy"])
    mode: str = "propose"                 # propose | auto_low_risk (plan-gated)


class ConsolidationSectionConfig(BaseModel):
    """MACA ``consolidation:`` section (D-009): the SEMANTIC knobs.

    Operational knobs (cadence, caps, thresholds) live in the ops config
    table (M7) — never here. ``guidelines`` is per-cycle free text that
    steers cycle prompts and is checked by the judge (H1 amendment).
    Referential rule (fail-loud): every category named here must exist in
    ``memory_categories`` — enforced at config write/activation.
    """
    inference_fields: List[InferenceFieldConfig] = Field(default_factory=list)
    conflict_priors: Dict[str, float] = Field(default_factory=dict)
    compression_policy: CompressionPolicyConfig = Field(
        default_factory=CompressionPolicyConfig)
    elevation: ElevationPolicyConfig = Field(default_factory=ElevationPolicyConfig)
    synthesis_targets: List[SynthesisTargetConfig] = Field(default_factory=list)
    evolution: ConsolidationEvolutionConfig = Field(
        default_factory=ConsolidationEvolutionConfig)
    guidelines: Dict[str, str] = Field(default_factory=dict)


def validate_consolidation_references(config: "MemoryArchitectureConfig") -> List[str]:
    """Referential checks (H1): categories named in the consolidation section
    must exist in the instance architecture. Returns error strings; callers
    at write/activation fail loud on any (ghost-category config rejected)."""
    section = getattr(config, "consolidation", None)
    if section is None:
        return []
    known: set = set()
    categories = getattr(config, "memory_categories", None)
    if categories is not None:
        for builtin in ("facts", "preferences", "episodes", "emotions",
                        "temporal_events"):
            entry = getattr(categories, builtin, None)
            if entry is not None and getattr(entry, "enabled", True):
                known.add(builtin)
        for custom in getattr(categories, "custom_categories", None) or []:
            name = getattr(custom, "name", None)
            if name and getattr(custom, "enabled", True):
                known.add(str(name).lower())
    if not known:
        return []                          # nothing to check against (legacy rows)
    errors = []
    for category in section.compression_policy.protected_categories:
        if str(category).lower() not in known:
            errors.append(f"consolidation.compression_policy.protected_categories: "
                          f"unknown category {category!r}")
    for category in section.compression_policy.per_category_age_days:
        if str(category).lower() not in known:
            errors.append(f"consolidation.compression_policy.per_category_age_days: "
                          f"unknown category {category!r}")
    return errors


# --- Metadata ---


class ConfigMetadata(BaseModel):
    """Config metadata.

    Audit fields are mirrored from the maca_config row's own columns and are
    optional here so that legacy/partial rows don't fail Pydantic validation
    on load. Required-fields used to silently break the listening agent's
    anticipation path: validation failed, the decision layer caught the
    exception and returned None, and the GENERIC prompt forced
    should_prefetch=false on every turn.
    """
    #: The model-owned paths this run did NOT decide, so a schema default
    #: supplied the value. Optional and empty by default: 183 stored configs
    #: predate it and must keep loading, and an empty list honestly means "we
    #: do not know" for those rather than "everything was decided".
    #:
    #: Without this a reader cannot tell ``max_memory_age_days: 365`` chosen
    #: for this client from ``365`` nobody chose, because the schema default is
    #: also 365. Neither could the judge: on the Handa run it reported three
    #: retention values as "selected without client support" and two of them
    #: had never been selected at all. The information already existed, it was
    #: computed at assembly and thrown away after a log line.
    defaulted_fields: List[str] = Field(default_factory=list)

    #: Set only when ``_assemble_config`` could not validate what it built and
    #: fell back to schema defaults overlaid with whichever sections survived.
    #: ``generated_by`` becomes ``maca-v1.0-recovered`` on the same path.
    #:
    #: Both existed in intent before this: the recovery branch set the stamp
    #: inside a dict literal it could never reach, because that literal is the
    #: fallback for a missing ``metadata`` and assembly writes ``metadata``
    #: first. So a recovered config looked exactly like a clean one and the
    #: only trace was a log line. ``recovery_reason`` carries the validation
    #: error that triggered it, truncated; ``recovered_sections`` names the
    #: blocks that were reverted, which is what a reviewer actually needs.
    recovery_reason: Optional[str] = None
    recovered_sections: List[str] = Field(default_factory=list)

    client_id: Optional[str] = None
    instance_id: Optional[str] = None
    config_version: Optional[str] = None
    generated_at: Optional[datetime] = None
    generated_by: str = "maca-v1.0"
    reasoning_trace_id: Optional[str] = None
    # Which of the five archetypes CLASSIFY decided this instance is, carried
    # on the config itself.
    #
    # It is already on ClassificationResult (``models/state.py:39``), but that
    # object lives for the length of one run. Everything downstream that has to
    # treat a knowledge agent differently, the retention ceiling in
    # ``tools/validation_tools.py`` first, is handed a
    # ``MemoryArchitectureConfig`` and nothing else, and the one archetype hint
    # a config ever carried, ``metadata.template_base``, has been removed. So a rule with an archetype carve-out had nothing to read.
    #
    # Optional, and it has to stay optional: this schema is also the read model
    # (``startup.py:374``, ``repository.py:108``) for 183 stored configs, all
    # written before this field existed. ``None`` therefore means "nobody
    # recorded it", never "not a knowledge agent"; see
    # ``enums.retention_ceiling_days`` for which way that is resolved.
    agent_archetype: Optional[str] = None


# --- Complete Config ---


class MemoryArchitectureConfig(BaseModel):
    """
    Complete Memory Architecture Configuration.
    This is the primary output of MACA.
    """
    metadata: ConfigMetadata
    role_semantics: Optional[RoleSemantics] = None
    memory_categories: MemoryCategoriesConfig = Field(default_factory=MemoryCategoriesConfig)
    memory_extraction: MemoryExtractionConfig = Field(default_factory=MemoryExtractionConfig)
    storage_architecture: StorageArchitectureConfig = Field(default_factory=StorageArchitectureConfig)
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    compaction_config: CompactionConfig = Field(default_factory=CompactionConfig)
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    agent_hints: AgentHints = Field(default_factory=AgentHints)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    user_profile: UserProfileConfig = Field(default_factory=UserProfileConfig)
    # D-009: semantic consolidation knobs. Optional so every pre-consolidation
    # config row still validates; readers resolve None to system defaults
    # (existing instances get defaults until their next MACA revision).
    consolidation: Optional[ConsolidationSectionConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _warn_on_values_nothing_understands(cls, data: Any) -> Any:
        """Cloud keeps loading a stored config with a word outside the enums; Core rejects it in core_rules."""
        problems = find_value_problems(data)
        if problems:
            logger.warning(
                "Memory config holds a value nothing understands, so it has no effect:\n%s",
                "\n".join(_describe_value(p) for p in problems),
            )
        # Core refuses a key the schema does not have; Cloud loads the config and says so once.
        # Imported here because core_rules imports this module.
        from .core_rules import find_unknown_keys

        unknown = find_unknown_keys(data)
        if unknown:
            logger.warning(
                "Memory config holds keys the schema does not have, so they have no effect:\n%s",
                "\n".join(message for _, message in unknown),
            )
        return data

    def to_yaml(self, for_judges: bool = False) -> str:
        """Convert to YAML string.

        ``for_judges`` leaves out the generated extraction patterns and examples:
        they are derived from the use case by their own step and checked by code
        (tools/validation_tools.check_extraction_patterns), and every judge call
        carries the whole YAML, so about 15 KB of worked examples would be paid
        for on each of them without changing a verdict.
        """
        data = self.model_dump(mode='json')
        if for_judges:
            ing = data.get("agent_hints", {}).get("ingestion_agent", {})
            for key in ("extraction_scope_notes", "extraction_patterns", "extraction_examples"):
                ing.pop(key, None)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "MemoryArchitectureConfig":
        """Parse from YAML string."""
        data = yaml.safe_load(yaml_content)
        return cls(**data)
