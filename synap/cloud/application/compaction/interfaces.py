"""
Compaction system interfaces and protocols.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum


class CompactionStrategy(str, Enum):
    """Compaction strategy options."""
    AGGRESSIVE = "aggressive"      # Maximum compression, drops content
    BALANCED = "balanced"          # Default, summarizes batches
    CONSERVATIVE = "conservative"  # Preserves more detail
    ADAPTIVE = "adaptive"          # Dynamic based on content density


class CompactionTriggerType(str, Enum):
    """What triggered the compaction."""
    TOKEN_THRESHOLD = "token_threshold"
    MESSAGE_COUNT = "message_count"
    IDLE_PERIOD = "idle_period"
    MANUAL_API = "manual_api"
    BACKGROUND_WORKER = "background_worker"
    # One-shot transcript ingest (POST /v1/conversations/ingest): summary-only
    # compaction — the post-compaction long-term promotion callback must skip
    # it because the full transcript was already ingested by the endpoint.
    TRANSCRIPT_PUSH = "transcript_push"
    # The conversation ended (the client said so, or the same user opened a
    # new one) with turns no compaction had covered yet. Forced, so that the
    # tail is promoted now and not at the idle sweep.
    SESSION_END = "session_end"


class ExtractionConfidence(str, Enum):
    """Confidence level for extracted facts."""
    EXPLICIT = "explicit"    # User directly stated
    INFERRED = "inferred"    # Reasonably inferred from context
    ASSUMED = "assumed"      # Working assumption, not confirmed


class DensityLevel(str, Enum):
    """Information density classification."""
    HIGH = "high"      # Score 8-10: Technical, complex problem-solving
    MEDIUM = "medium"  # Score 4-7: General Q&A, explanations
    LOW = "low"        # Score 1-3: Small talk, pleasantries


class ICompactionPass(ABC):
    """Interface for a single compaction pass."""

    @property
    @abstractmethod
    def pass_name(self) -> str:
        """Name of this pass (e.g., 'density_scoring', 'classification')."""
        pass

    @property
    @abstractmethod
    def pass_number(self) -> int:
        """Pass number (0-5)."""
        pass

    @abstractmethod
    async def execute(
        self,
        input_data: Dict[str, Any],
        config: "CompactionConfig",
        previous_results: Dict[str, Any]
    ) -> "PassResult":
        """
        Execute this pass.

        Args:
            input_data: Input for this pass (messages, previous pass output, etc.)
            config: Compaction configuration (from MACA + template)
            previous_results: Results from previous passes

        Returns:
            PassResult with output data and metadata
        """
        pass


class ICompactionManager(ABC):
    """Interface for the CompactionManager."""

    @abstractmethod
    async def compact(
        self,
        conversation_id: str,
        instance_id: str,
        messages: List["ConversationMessage"],
        config: Optional["CompactionConfig"] = None,
        trigger_type: CompactionTriggerType = CompactionTriggerType.MANUAL_API,
        force: bool = False
    ) -> Optional["CompactedContext"]:
        """
        Execute the full compaction pipeline.

        Args:
            conversation_id: ID of the conversation to compact
            instance_id: Instance ID for config lookup
            messages: List of conversation messages
            config: Optional config override (uses MACA config if not provided)
            trigger_type: What triggered this compaction
            force: Compact even if under threshold

        Returns:
            CompactedContext with all compaction results, or None when the
            compaction is not due (and not forced) and nothing is stored yet
        """
        pass

    @abstractmethod
    async def get_or_create(
        self,
        conversation_id: str,
        instance_id: str,
        messages: List["ConversationMessage"]
    ) -> "CompactedContext":
        """
        Get existing compacted context or create new one if stale/missing.
        """
        pass

    @abstractmethod
    def should_compact(
        self,
        messages: List["ConversationMessage"],
        config: "CompactionConfig",
        last_compaction: Optional["CompactedContext"] = None
    ) -> tuple[bool, CompactionTriggerType]:
        """
        Determine if compaction should be triggered.

        Returns:
            Tuple of (should_compact, trigger_type)
        """
        pass


class ITemplateLoader(ABC):
    """Interface for loading and merging compaction templates."""

    @abstractmethod
    def load_template(self, category: str) -> "CompactionTemplate":
        """Load a base template by category name."""
        pass

    @abstractmethod
    def merge_with_maca(
        self,
        template: "CompactionTemplate",
        maca_config: Dict[str, Any]
    ) -> "CompactionConfig":
        """Merge base template with MACA customizations."""
        pass
