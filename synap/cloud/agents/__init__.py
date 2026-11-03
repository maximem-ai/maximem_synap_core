"""Autonomous agents and coordination."""

from .models import (
    ContextDecision,
    QualifiedData,
    MemoryArchitecturePlan,
    ContextBundle,
    AgentError,
    LLMUnavailableError,
    MCPServerError,
    RetrievalError,
    IngestionError,
    ArchitectureDesignError,
)

# Agent classes are lazy-loaded to avoid circular imports.
# Import directly from submodules:
#   from synap.cloud.agents.listening_retrieval import ListeningRetrievalAgent
#   from synap.cloud.agents.memory_architecture import MemoryArchitectureCreationAgent
#   from synap.cloud.agents.ingestion import IngestionAgent

_LAZY_IMPORTS = {
    # The registries: Core does not ship the registry folder, so they load on first use.
    "AgentRegistry": ".registry",
    "RoutingMode": ".registry",
    "MCPRegistry": ".registry",
    "MCPServer": ".registry",
    "ListeningRetrievalAgent": ".listening_retrieval",
    "ListeningRetrievalState": ".listening_retrieval",
    "MemoryArchitectureCreationAgent": ".memory_architecture",
    "get_maca_agent": ".memory_architecture",
    "IngestionAgent": ".ingestion",
    "IngestionState": ".ingestion",
}


def __getattr__(name):
    """Lazy-load agent classes on first access."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Registry
    "AgentRegistry",
    "RoutingMode",
    "MCPRegistry",
    "MCPServer",
    # Models
    "ContextDecision",
    "QualifiedData",
    "MemoryArchitecturePlan",
    "ContextBundle",
    "AgentError",
    "LLMUnavailableError",
    "MCPServerError",
    "RetrievalError",
    "IngestionError",
    "ArchitectureDesignError",
    # Agents (lazy-loaded)
    "ListeningRetrievalAgent",
    "ListeningRetrievalState",
    "MemoryArchitectureCreationAgent",
    "get_maca_agent",
    "IngestionAgent",
    "IngestionState",
]
