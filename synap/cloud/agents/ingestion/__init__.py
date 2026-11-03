"""Ingestion agent."""

import importlib

from .state import IngestionState

# IngestionAgent (agent.py) and build_graph (graph.py) load on first use: Core
# does not ship those two files. Cloud gets the same objects under the same names.
_LAZY = {"IngestionAgent": ".agent", "build_graph": ".graph"}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


# Models
from .models import (
    # Enums
    ContentNature,
    DocumentSource,
    DocumentType,
    IngestPriority,
    IngestStatus,
    MemoryDestination,
    MemoryType,
    PipelineStage,
    # Input models
    ConvertSessionRequest,
    IngestBatchRequest,
    IngestDocumentRequest,
    IngestFromListeningAgentRequest,
    Scope,
    # Output models
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelationship,
    IngestBatchResponse,
    IngestResponse,
    IngestionTrace,
    # Pipeline models
    CategoryAssignment,
    Stage1Output,
    Stage2Output,
    Stage3Output,
    Stage4Output,
    # State models
    IngestionPipelineState,
)

__all__ = [
    # Agent components
    "IngestionAgent",
    "IngestionState",
    "build_graph",
    # Enums
    "ContentNature",
    "DocumentSource",
    "DocumentType",
    "IngestPriority",
    "IngestStatus",
    "MemoryDestination",
    "MemoryType",
    "PipelineStage",
    # Input models
    "ConvertSessionRequest",
    "IngestBatchRequest",
    "IngestDocumentRequest",
    "IngestFromListeningAgentRequest",
    "Scope",
    # Output models
    "ExtractedEntity",
    "ExtractedMemory",
    "ExtractedRelationship",
    "IngestBatchResponse",
    "IngestResponse",
    "IngestionTrace",
    # Pipeline models
    "CategoryAssignment",
    "Stage1Output",
    "Stage2Output",
    "Stage3Output",
    "Stage4Output",
    # State models
    "IngestionPipelineState",
]
