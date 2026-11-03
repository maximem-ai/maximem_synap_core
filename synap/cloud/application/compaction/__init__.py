"""Compaction system for context compression and summarization."""

from .interfaces import (
    CompactionStrategy,
    CompactionTriggerType,
    ExtractionConfidence,
    DensityLevel,
    ICompactionPass,
    ICompactionManager,
    ITemplateLoader
)

from .models import (
    ConversationMessage,
    CompactionRequest,
    CompactionTemplate,
    CompactionConfig,
    PassResult,
    DensityScore,
    ClassificationResult,
    ExtractedFact,
    CodeArtifact,
    ExtractionResult,
    SegmentSummary,
    SummarizationResult,
    ValidationQuestion,
    ValidationResult,
    NarrativeSummary,
    QualityAssurance,
    CompactionMetadata,
    CompactedContext,
    CompactedContextRecord
)

from .templates import TemplateLoader, get_template_loader

from .passes import BaseCompactionPass
from .budgets import (
    AgentBudget,
    CompactionModelLimits,
    agent_budget,
    derive_model_limits,
    resolve_agent_window,
    window_for_model,
)

from .manager import CompactionManager, get_compaction_manager, CompactionError

from .worker import (
    CompactionWorker,
    get_compaction_worker,
    start_compaction_worker,
    stop_compaction_worker
)

from .storage import CompactionStorage

__all__ = [
    # Interfaces and Enums
    "CompactionStrategy",
    "CompactionTriggerType",
    "ExtractionConfidence",
    "DensityLevel",
    "ICompactionPass",
    "ICompactionManager",
    "ITemplateLoader",
    # Input Models
    "ConversationMessage",
    "CompactionRequest",
    # Configuration Models
    "CompactionTemplate",
    "CompactionConfig",
    # Pass Result Models
    "PassResult",
    "DensityScore",
    "ClassificationResult",
    "ExtractedFact",
    "CodeArtifact",
    "ExtractionResult",
    "SegmentSummary",
    "SummarizationResult",
    "ValidationQuestion",
    "ValidationResult",
    # Output Models
    "NarrativeSummary",
    "QualityAssurance",
    "CompactionMetadata",
    "CompactedContext",
    # Storage Models
    "CompactedContextRecord",
    # Template System
    "TemplateLoader",
    "get_template_loader",
    # Pass Infrastructure (v2; profiles live in passes/base.py)
    "BaseCompactionPass",
    # Budgets (model-adaptive triggers and limits)
    "AgentBudget",
    "CompactionModelLimits",
    "agent_budget",
    "derive_model_limits",
    "resolve_agent_window",
    "window_for_model",
    # Manager and Worker
    "CompactionManager",
    "get_compaction_manager",
    "CompactionError",
    "CompactionWorker",
    "get_compaction_worker",
    "start_compaction_worker",
    "stop_compaction_worker",
    # Storage
    "CompactionStorage"
]
