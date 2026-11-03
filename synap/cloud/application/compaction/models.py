"""
Compaction data models.

These models represent the inputs, outputs, and intermediate results
of the compaction pipeline.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from .interfaces import (
    CompactionStrategy,
    CompactionTriggerType,
    ExtractionConfidence,
    DensityLevel
)


# =============================================================================
# INPUT MODELS
# =============================================================================

@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    message_id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    @property
    def token_count(self) -> int:
        """Token count with the server's one counter (tiktoken cl100k_base).

        Was words / 0.75 here, chars / 4 in the DB trigger and tiktoken in
        retrieval: three numbers for one "15K".
        """
        from synap.cloud.shared.tokens import count_tokens

        return max(1, count_tokens(self.content or ""))


@dataclass
class CompactionRequest:
    """Request to compact a conversation."""
    conversation_id: str
    instance_id: str
    messages: List[ConversationMessage]
    strategy: Optional[CompactionStrategy] = None  # Override MACA default
    target_tokens: Optional[int] = None  # Override MACA default
    force: bool = False  # Compact even if under threshold
    trigger_type: CompactionTriggerType = CompactionTriggerType.MANUAL_API

    @property
    def total_tokens(self) -> int:
        """Total token count of all messages."""
        return sum(m.token_count for m in self.messages)


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

@dataclass
class CompactionTemplate:
    """
    Base compaction template loaded from YAML.
    Defines category-specific extraction and compression rules.
    """
    category: str
    version: str

    # Priority extraction rules
    priority_extraction: Dict[str, List[str]]  # high/medium/low -> categories

    # Summarization guidelines
    summarization_guidelines: Dict[str, List[str]]  # focus/compress/preserve

    # Preservation rules
    preservation_rules: Dict[str, List[str]]  # never_compress

    # Special handling instructions
    special_handling: List[str]

    # Compression rates by density
    compression_rates: Dict[str, float]  # high_density, medium_density, low_density


@dataclass
class CompactionConfig:
    """
    Full compaction configuration.
    Merged from base template + MACA customizations.
    """
    # Strategy
    strategy: CompactionStrategy = CompactionStrategy.ADAPTIVE

    # When to compact is model-adaptive (budgets.py): T tokens of record plus
    # raw turns, sized to the agent's context window, or this many new
    # messages, whichever comes first. MACA's trigger_at_tokens and
    # trigger_at_messages are not read: until MACA assesses them per agent
    # (COMPACTION_V3_PLAN.md §15) the stored values are v2's every-few-turns
    # cadence, and honouring them would keep it.
    message_trigger: int = 30

    # Size: each update adds about this share of the new turns' tokens,
    # ±5 points. From MACA adaptive_thresholds.target_ratio.
    target_ratio: float = 0.30

    # Long spans move the record to steering mode (v3/steering.py), which
    # leaves detail to a search of the raw turns in the SDK's local store.
    # From MACA adaptive_thresholds.steering_enabled; turn it off for an
    # instance whose callers have no SDK store (REST only).
    # Retired 20 Oct 2026: on 15K/20K (2 runs each, with search) fidelity 97%/92% correct with 0/8 invented claims; steering 83%/84% with 3/36. Kept behind the flag for later work.
    steering_enabled: bool = False

    # Inactivity triggers, both defaulting to five minutes.
    #
    # idle_minutes keeps the sweep worker's old constructor default.
    # recompaction_idle_minutes does NOT keep the 15-minute literal it
    # replaced: fifteen minutes was long enough that a conversation could sit
    # with a stale summary well past the point anyone would call it current,
    # and nothing chose that number deliberately — it was written inline.
    idle_minutes: int = 5
    recompaction_idle_minutes: int = 5

    # Preservation priorities (from MACA)
    preservation_priorities: List[str] = field(default_factory=lambda: [
        "user_stated_facts",
        "explicit_preferences",
        "current_objective",
        "unresolved_issues",
        "key_decisions"
    ])

    # Summarization candidates (from MACA)
    summarization_candidates: List[str] = field(default_factory=lambda: [
        "chitchat",
        "resolved_issues",
        "repeated_clarifications"
    ])

    # Template (merged)
    template: Optional[CompactionTemplate] = None

    # Extraction categories
    core_extraction_categories: List[str] = field(default_factory=lambda: [
        "explicit_facts",
        "decisions",
        "constraints",
        "user_preferences",
        "open_questions",
        "conflicts_detected"
    ])

    # Additional categories from template
    template_extraction_categories: List[str] = field(default_factory=list)

    # Custom categories from MACA
    custom_extraction_categories: List[str] = field(default_factory=list)

    # Validation threshold
    validation_threshold: float = 0.8

    # Instance context
    instance_id: Optional[str] = None
    instance_category: Optional[str] = None  # From MACA (e.g., "customer_support")


# =============================================================================
# PASS RESULT MODELS
# =============================================================================

@dataclass
class PassResult:
    """Result from a single compaction pass."""
    pass_name: str
    pass_number: int
    success: bool
    output: Dict[str, Any]
    latency_ms: float
    tokens_consumed: int
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class DensityScore:
    """Result from Pass 0: Density Scoring."""
    segment_id: str
    message_range: tuple[int, int]  # (start_idx, end_idx)
    density_score: int  # 1-10
    redundancy_score: int  # 1-10
    criticality_score: int  # 1-10
    density_level: DensityLevel
    recommended_compression_rate: float
    reasoning: str


@dataclass
class ClassificationResult:
    """Result from Pass 1: Classification + Section Planning + Density.

    Pass 1 now does the work of the deprecated Pass 0 (density) too —
    one LLM call instead of two. `density_scores` carries the per-segment
    profile that Pass 2 and Pass 3 use as a verbosity hint.

    `section_plan` carries the markdown H2 headers that Pass 2 and Pass 3
    must emit in their paragraphs, picked per use case from
    ``SECTION_PLANS_BY_CATEGORY`` (see ``passes/section_plans.py``).
    """
    primary_category: str
    subcategory: str
    objective: str
    characteristics: Dict[str, bool]  # temporal_sensitivity, iterative_refinement, etc.
    recommended_templates: List[str]
    special_handling: List[str]
    compression_strategy: str
    # New: section plan + density profile produced by the merged Pass 1.
    section_plan: Dict[str, List[str]] = field(default_factory=dict)
    density_scores: List["DensityScore"] = field(default_factory=list)


@dataclass
class ExtractedFact:
    """A single extracted fact from Pass 2."""
    fact_id: str
    category: str  # explicit_facts, decisions, constraints, etc.
    content: str
    source_messages: List[str]  # message_ids
    confidence: ExtractionConfidence
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "fact_id": self.fact_id,
            "category": self.category,
            "content": self.content,
            "source_messages": self.source_messages,
            "confidence": self.confidence.value if isinstance(self.confidence, ExtractionConfidence) else self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def generate_id(cls, category: str) -> str:
        return f"{category}_{uuid.uuid4().hex[:8]}"


@dataclass
class CodeArtifact:
    """Extracted code artifact from Pass 2."""
    artifact_id: str
    artifact_type: str  # code_block, shell_command, sql_query, etc.
    language: Optional[str]
    content: str
    description: str
    source_messages: List[str]
    status: str  # working_solution, superseded, untested


@dataclass
class ExtractionResult:
    """Full result from Pass 2: Extraction."""
    extraction_metadata: Dict[str, Any]
    explicit_facts: List[ExtractedFact]
    decisions: List[ExtractedFact]
    constraints: List[ExtractedFact]
    attempted_solutions: List[ExtractedFact]
    user_preferences: List[ExtractedFact]
    code_artifacts: List[CodeArtifact]
    open_questions: List[ExtractedFact]
    dependencies: List[ExtractedFact]
    conflicts_detected: List[ExtractedFact]
    reference_pointers: List[Dict[str, Any]]
    emotional_states: List[ExtractedFact] = field(default_factory=list)

    # Template-specific extractions
    custom_extractions: Dict[str, List[ExtractedFact]] = field(default_factory=dict)


@dataclass
class SegmentSummary:
    """Summary of a conversation segment from Pass 3."""
    segment_id: str
    message_range: tuple[int, int]
    timeframe: tuple[datetime, datetime]
    summary: str
    compression_ratio: float
    original_token_count: int
    compressed_token_count: int
    key_moments: List[Dict[str, Any]]
    references_to_extractions: Dict[str, List[str]]
    thread_info: Dict[str, str]


@dataclass
class SummarizationResult:
    """Full result from Pass 3: Summarization."""
    segments: List[SegmentSummary]
    total_original_tokens: int
    total_compressed_tokens: int
    overall_compression_ratio: float


@dataclass
class ValidationQuestion:
    """A test question for Pass 4 validation."""
    question: str
    question_type: str  # factual_recall, chronological, reasoning, status, contextual
    answer: Optional[str] = None
    correct: Optional[bool] = None
    source_citations: List[str] = field(default_factory=list)
    confidence: str = "unknown"


@dataclass
class ValidationResult:
    """Result from Pass 4: Validation."""
    validation_status: str  # passed, failed, retried
    success_rate: float
    total_questions: int
    correct_answers: int
    questions_and_answers: List[ValidationQuestion]
    information_gaps_detected: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]
    recommendation: str


# =============================================================================
# OUTPUT MODELS
# =============================================================================

@dataclass
class NarrativeSummary:
    """Narrative portion of compacted context.

    v2 architecture: two complementary paragraphs.

      - ``factual_paragraph``: section-headed prose about WHAT we know
        (user profile, current state, decisions, preferences, constraints).
        Produced by Pass 2. Verbatim-preserving for IDs/numbers/dates.
      - ``conversational_paragraph``: section-headed prose about WHAT IS
        HAPPENING (timeline, momentum, open threads, emotional context).
        Produced by Pass 3.

    ``overview`` is now derived for backwards compatibility — assembled
    in Pass 5 as the two paragraphs concatenated. Existing consumers that
    read ``overview`` see a single richer string.
    """
    overview: str
    chronological_narrative: List[SegmentSummary]
    current_state: Dict[str, Any]  # status, next_steps, open_issues, pending_dependencies
    factual_paragraph: str = ""
    conversational_paragraph: str = ""


@dataclass
class QualityAssurance:
    """Quality metrics for compacted context."""
    validation_passed: bool
    validation_score: float
    information_completeness: float
    quality_warning: bool = False
    warning_message: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)  # information gaps from validation


@dataclass
class CompactionMetadata:
    """Metadata about the compaction operation."""
    version: int = 1
    compaction_id: str = field(default_factory=lambda: f"compact_{uuid.uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Source info
    conversation_id: str = ""
    instance_id: str = ""
    client_id: str = ""
    source_message_range: Dict[str, Any] = field(default_factory=dict)  # start_id, end_id, timestamps

    # Compression stats
    original_token_count: int = 0
    compacted_token_count: int = 0
    compression_ratio: float = 0.0
    messages_preserved_verbatim: int = 0
    messages_compressed: int = 0

    # Pipeline stats
    trigger_type: CompactionTriggerType = CompactionTriggerType.MANUAL_API
    strategy_used: CompactionStrategy = CompactionStrategy.ADAPTIVE
    total_latency_ms: float = 0.0
    pass_latencies: Dict[str, float] = field(default_factory=dict)
    groq_tokens_consumed: int = 0
    groq_api_calls: int = 0


@dataclass
class CompactionEvent:
    """Event emitted after a successful compaction.

    Consumed by the callbacks shared/providers.py registers: retrieval cache
    invalidation, Redis raw-buffer trim, conversation backup, gRPC push.
    """
    conversation_id: str
    instance_id: str
    compaction_id: str
    compression_ratio: float
    validation_score: float
    is_incremental: bool
    message_count: int
    # Scope fields for downstream routing (cache invalidation, ingestion)
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    client_id: Optional[str] = None
    # The rung the conversation was opened at, root to leaf, or None, for
    # callbacks that route by scope (the conversation backup). Long-term
    # promotion no longer runs from a callback: it has its own schedule
    # (conversation/promotion.py) and resolves scope the same way.
    scope_path: Optional[Dict[str, str]] = None
    # What triggered the compaction, so callbacks can discriminate.
    trigger_type: Optional[CompactionTriggerType] = None


@dataclass
class CompactedContext:
    """
    Final output of the compaction pipeline.
    This is what gets stored in Redis/Postgres and used by RetrievalManager.
    """
    # Metadata
    metadata: CompactionMetadata

    # Classification
    conversation_classification: ClassificationResult

    # Narrative summary
    narrative_summary: NarrativeSummary

    # Structured extractions
    structured_extractions: ExtractionResult

    # Reference index for quick lookup
    reference_index: Dict[str, Any]

    # Verbatim recent messages (last N turns)
    verbatim_recent_messages: List[ConversationMessage]

    # Quality assurance
    quality_assurance: QualityAssurance

    # Density scores (from Pass 0)
    density_scores: Optional[List[DensityScore]] = None

    # Compaction v3. ``record`` is the stored state the next compaction
    # updates (v3/record.py); ``context_block`` is the one framed block every
    # caller receives, REST, retrieval and the SDK alike; ``document`` is the
    # same text without the framing. A row written before v3 has no record,
    # and its conversation is rebuilt from the full history on its next
    # compaction.
    record: Optional[Dict[str, Any]] = None
    context_block: str = ""
    document: str = ""

    @property
    def agreed_actions(self) -> List[Dict[str, Any]]:
        return list((self.record or {}).get("actions") or [])

    @property
    def record_mode(self) -> str:
        """"fidelity" or "steering" (v3/record.py); "" for a pre-v3 row."""
        return str((self.record or {}).get("mode") or ("fidelity" if self.record else ""))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        def serialize_value(obj):
            """Custom serialization for complex types."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (CompactionStrategy, CompactionTriggerType, ExtractionConfidence, DensityLevel)):
                return obj.value
            elif isinstance(obj, tuple):
                # Handle tuples (message_range, timeframe)
                return list(serialize_value(item) for item in obj)
            elif isinstance(obj, list):
                return [serialize_value(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize_value(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                # Handle dataclass instances
                return serialize_value(asdict(obj))
            else:
                return obj

        return serialize_value(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompactedContext":
        """Deserialize from dictionary."""
        def deserialize_datetime(value):
            """Helper to deserialize datetime strings."""
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    return value
            return value

        # Deserialize metadata
        metadata_data = data["metadata"]
        metadata = CompactionMetadata(
            version=int(float(metadata_data.get("version", 1))),
            compaction_id=metadata_data.get("compaction_id", ""),
            created_at=deserialize_datetime(metadata_data.get("created_at")),
            conversation_id=metadata_data.get("conversation_id", ""),
            instance_id=metadata_data.get("instance_id", ""),
            client_id=metadata_data.get("client_id", ""),
            source_message_range=metadata_data.get("source_message_range", {}),
            original_token_count=metadata_data.get("original_token_count", 0),
            compacted_token_count=metadata_data.get("compacted_token_count", 0),
            compression_ratio=metadata_data.get("compression_ratio", 0.0),
            messages_preserved_verbatim=metadata_data.get("messages_preserved_verbatim", 0),
            messages_compressed=metadata_data.get("messages_compressed", 0),
            trigger_type=CompactionTriggerType(metadata_data.get("trigger_type", "manual_api")),
            strategy_used=CompactionStrategy(metadata_data.get("strategy_used", "adaptive")),
            total_latency_ms=metadata_data.get("total_latency_ms", 0.0),
            pass_latencies=metadata_data.get("pass_latencies", {}),
            groq_tokens_consumed=metadata_data.get("groq_tokens_consumed", 0),
            groq_api_calls=metadata_data.get("groq_api_calls", 0)
        )

        # Deserialize classification
        class_data = data["conversation_classification"]
        # Density scores nested under classification (Pass 1 merged)
        density_scores_data = class_data.get("density_scores", []) or []
        density_scores = []
        for ds in density_scores_data:
            try:
                mr = ds.get("message_range", (0, 0))
                density_scores.append(DensityScore(
                    segment_id=ds.get("segment_id", ""),
                    message_range=tuple(mr) if isinstance(mr, list) else mr,
                    density_score=int(ds.get("density_score", 5)),
                    redundancy_score=int(ds.get("redundancy_score", 5)),
                    criticality_score=int(ds.get("criticality_score", 5)),
                    density_level=DensityLevel(ds.get("density_level", "medium")),
                    recommended_compression_rate=float(ds.get("recommended_compression_rate", 0.25)),
                    reasoning=ds.get("reasoning", ""),
                ))
            except Exception:
                # Don't fail full deserialization on a single bad density entry
                pass

        classification = ClassificationResult(
            primary_category=class_data["primary_category"],
            subcategory=class_data["subcategory"],
            objective=class_data["objective"],
            characteristics=class_data.get("characteristics", {}),
            recommended_templates=class_data.get("recommended_templates", []),
            special_handling=class_data.get("special_handling", []),
            compression_strategy=class_data.get("compression_strategy", ""),
            section_plan=class_data.get("section_plan", {}),
            density_scores=density_scores,
        )

        # Deserialize narrative summary
        narrative_data = data["narrative_summary"]
        segment_summaries = []
        for seg_data in narrative_data.get("chronological_narrative", []):
            timeframe_raw = seg_data.get("timeframe", [None, None])
            segment_summaries.append(SegmentSummary(
                segment_id=seg_data["segment_id"],
                message_range=tuple(seg_data["message_range"]),
                timeframe=(
                    deserialize_datetime(timeframe_raw[0]) if timeframe_raw[0] else datetime.now(timezone.utc),
                    deserialize_datetime(timeframe_raw[1]) if timeframe_raw[1] else datetime.now(timezone.utc)
                ),
                summary=seg_data["summary"],
                compression_ratio=seg_data.get("compression_ratio", 0.0),
                original_token_count=seg_data.get("original_token_count", 0),
                compressed_token_count=seg_data.get("compressed_token_count", 0),
                key_moments=seg_data.get("key_moments", []),
                references_to_extractions=seg_data.get("references_to_extractions", {}),
                thread_info=seg_data.get("thread_info", {})
            ))

        narrative_summary = NarrativeSummary(
            overview=narrative_data.get("overview", ""),
            chronological_narrative=segment_summaries,
            current_state=narrative_data.get("current_state", {}),
            factual_paragraph=narrative_data.get("factual_paragraph", ""),
            conversational_paragraph=narrative_data.get("conversational_paragraph", ""),
        )

        # Deserialize extractions
        extract_data = data["structured_extractions"]

        def deserialize_facts(facts_list: List[Dict[str, Any]]) -> List[ExtractedFact]:
            return [
                ExtractedFact(
                    fact_id=f["fact_id"],
                    category=f["category"],
                    content=f["content"],
                    source_messages=f.get("source_messages", []),
                    confidence=ExtractionConfidence(f.get("confidence", "explicit")),
                    timestamp=deserialize_datetime(f.get("timestamp"))
                )
                for f in facts_list
            ]

        def deserialize_artifacts(artifacts_list: List[Dict[str, Any]]) -> List[CodeArtifact]:
            return [
                CodeArtifact(
                    artifact_id=a["artifact_id"],
                    artifact_type=a["artifact_type"],
                    language=a.get("language"),
                    content=a["content"],
                    description=a["description"],
                    source_messages=a.get("source_messages", []),
                    status=a.get("status", "untested")
                )
                for a in artifacts_list
            ]

        custom_extractions = {}
        for key, value in extract_data.get("custom_extractions", {}).items():
            custom_extractions[key] = deserialize_facts(value)

        extractions = ExtractionResult(
            extraction_metadata=extract_data.get("extraction_metadata", {}),
            explicit_facts=deserialize_facts(extract_data.get("explicit_facts", [])),
            decisions=deserialize_facts(extract_data.get("decisions", [])),
            constraints=deserialize_facts(extract_data.get("constraints", [])),
            attempted_solutions=deserialize_facts(extract_data.get("attempted_solutions", [])),
            user_preferences=deserialize_facts(extract_data.get("user_preferences", [])),
            code_artifacts=deserialize_artifacts(extract_data.get("code_artifacts", [])),
            open_questions=deserialize_facts(extract_data.get("open_questions", [])),
            dependencies=deserialize_facts(extract_data.get("dependencies", [])),
            conflicts_detected=deserialize_facts(extract_data.get("conflicts_detected", [])),
            reference_pointers=extract_data.get("reference_pointers", []),
            custom_extractions=custom_extractions
        )

        # Deserialize verbatim messages
        verbatim_messages = [
            ConversationMessage(
                message_id=m["message_id"],
                role=m["role"],
                content=m["content"],
                timestamp=deserialize_datetime(m["timestamp"]),
                metadata=m.get("metadata")
            )
            for m in data.get("verbatim_recent_messages", [])
        ]

        # Deserialize quality assurance
        qa_data = data["quality_assurance"]
        quality_assurance = QualityAssurance(
            validation_passed=qa_data.get("validation_passed", False),
            validation_score=qa_data.get("validation_score", 0.0),
            information_completeness=qa_data.get("information_completeness", 0.0),
            quality_warning=qa_data.get("quality_warning", False),
            warning_message=qa_data.get("warning_message"),
            flags=qa_data.get("flags", []),
            warnings=qa_data.get("warnings", [])
        )

        # Deserialize density scores
        density_scores = None
        if "density_scores" in data and data["density_scores"]:
            density_scores = [
                DensityScore(
                    segment_id=ds["segment_id"],
                    message_range=tuple(ds["message_range"]),
                    density_score=ds["density_score"],
                    redundancy_score=ds["redundancy_score"],
                    criticality_score=ds["criticality_score"],
                    density_level=DensityLevel(ds["density_level"]),
                    recommended_compression_rate=ds["recommended_compression_rate"],
                    reasoning=ds["reasoning"]
                )
                for ds in data["density_scores"]
            ]

        return cls(
            metadata=metadata,
            conversation_classification=classification,
            narrative_summary=narrative_summary,
            structured_extractions=extractions,
            reference_index=data.get("reference_index", {}),
            verbatim_recent_messages=verbatim_messages,
            quality_assurance=quality_assurance,
            density_scores=density_scores,
            record=data.get("record"),
            context_block=data.get("context_block", "") or "",
            document=data.get("document", "") or "",
        )

    def get_injection_format(self, style: str = "structured") -> str:
        """
        Get the compacted context formatted for injection into LLM context.

        Args:
            style: "structured" | "narrative" | "bullet_points"

        Returns:
            Formatted string ready for context injection

        Deprecated: every caller now reads ``context_block``, which this
        returns, whatever the style, for any compaction written by v3. The
        styles below only render rows written before v3.
        """
        if self.context_block:
            return self.context_block
        if style == "structured":
            # Format as structured sections
            sections = []

            # Overview
            sections.append("## Conversation Overview")
            sections.append(self.narrative_summary.overview)
            sections.append("")

            # Key facts — include all (already compressed by pipeline)
            if self.structured_extractions.explicit_facts:
                sections.append("## Key Facts")
                for fact in self.structured_extractions.explicit_facts:
                    sections.append(f"- {fact.content}")
                sections.append("")

            # Current state
            sections.append("## Current State")
            for key, value in self.narrative_summary.current_state.items():
                sections.append(f"**{key.replace('_', ' ').title()}**: {value}")
            sections.append("")

            # Recent context (verbatim, full content)
            if self.verbatim_recent_messages:
                sections.append("## Recent Messages")
                for msg in self.verbatim_recent_messages[-6:]:  # Last 3 turns
                    sections.append(f"**{msg.role}**: {msg.content}")
                sections.append("")

            return "\n".join(sections)

        elif style == "narrative":
            # Format as flowing narrative
            sections = []
            sections.append(self.narrative_summary.overview)
            sections.append("")

            for segment in self.narrative_summary.chronological_narrative:
                sections.append(segment.summary)

            sections.append("")
            sections.append("Current status:")
            for key, value in self.narrative_summary.current_state.items():
                sections.append(f"- {key.replace('_', ' ').title()}: {value}")

            return "\n".join(sections)

        elif style == "bullet_points":
            # Format as bullet points
            sections = []
            sections.append("# Conversation Context")
            sections.append("")

            # Facts — include all
            sections.append("**Key Information:**")
            for fact in self.structured_extractions.explicit_facts:
                sections.append(f"• {fact.content}")
            sections.append("")

            # Decisions — include all
            if self.structured_extractions.decisions:
                sections.append("**Decisions Made:**")
                for decision in self.structured_extractions.decisions:
                    sections.append(f"• {decision.content}")
                sections.append("")

            # Current state
            sections.append("**Current Status:**")
            for key, value in self.narrative_summary.current_state.items():
                sections.append(f"• {key.replace('_', ' ').title()}: {value}")

            return "\n".join(sections)

        else:
            raise ValueError(f"Unknown injection style: {style}")


# =============================================================================
# STORAGE MODELS
# =============================================================================

@dataclass
class CompactedContextRecord:
    """Database record for compacted context (Postgres)."""
    id: str  # UUID
    instance_id: str
    conversation_id: str
    version: int
    compacted_content: Dict[str, Any]  # Full CompactedContext as JSON
    original_token_count: int
    compacted_token_count: int
    compression_ratio: float
    validation_score: float
    trigger_type: str
    created_at: datetime

    # Indexes:
    # - (instance_id, conversation_id, version)
    # - (instance_id, conversation_id, created_at)
