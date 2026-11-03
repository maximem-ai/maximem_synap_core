"""Agent decision models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum


class RetrievalMode(Enum):
    """Retrieval mode determines query complexity and stores accessed."""
    FAST = "fast"          # No LLM subquery, direct vector+graph search
    ACCURATE = "accurate"  # LLM subquery decomposition + reranking


class TriggerType(Enum):
    """What triggered the context retrieval."""
    EXPLICIT_REQUEST = "explicit_request"  # SDK called fetch_context
    STREAM_EVENT = "stream_event"          # gRPC stream event
    ANTICIPATORY = "anticipatory"          # Proactive cache warming


@dataclass
class ContextDecision:
    """
    Auditable artifact explaining context injection decision.

    This is the output of the decide_node in the ListeningRetrievalAgent.
    It captures WHAT context is needed and WHY, leaving HOW to the Manager.
    """
    decision_id: str

    # Trigger info
    trigger_type: TriggerType
    trigger_payload: Optional[Dict[str, Any]] = None  # Original request/event

    # Scope
    client_id: str = ""
    instance_id: str = ""
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    customer_id: Optional[str] = None

    # What to retrieve
    retrieval_mode: RetrievalMode = RetrievalMode.FAST
    context_types: List[str] = field(default_factory=lambda: ["facts", "preferences", "episodes"])
    search_queries: List[str] = field(default_factory=list)  # Original or decomposed queries
    search_keywords: List[str] = field(default_factory=list)  # Expanded vocabulary for SDK BM25 matching
    filters: Dict[str, Any] = field(default_factory=dict)    # Time range, confidence, etc.
    # The moment the question is being asked, when the caller knows it. A
    # declared field rather than a key in ``filters`` because two steps read it
    # and a stashed key is invisible to whoever writes the third.
    query_date: Optional[datetime] = None

    # Budget
    token_budget: int = 2000
    budget_mode: str = "hard"  # "hard" | "soft"
    max_results: int = 10
    # How many items each search query may contribute, when the fetch carries
    # several. Works inside ``max_results``, which stays the ceiling on the
    # whole response. None, or a fetch with one query: no per-query share, and
    # the best ``max_results`` items overall are returned as they always were.
    max_results_per_query: Optional[int] = None

    # Explanation (template-based for MVP)
    explanation: str = ""

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str = ""

    # SDK-authoritative short-term context. When False the
    # retrieval path skips ST assembly entirely (SDK has it cached
    # locally and will merge client-side). Defaults True for backwards
    # compatibility.
    include_conversation_context: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence and API response."""
        return {
            "decision_id": self.decision_id,
            "trigger_type": self.trigger_type.value,
            "trigger_payload": self.trigger_payload,
            "client_id": self.client_id,
            "instance_id": self.instance_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "customer_id": self.customer_id,
            "retrieval_mode": self.retrieval_mode.value,
            "context_types": self.context_types,
            "search_queries": self.search_queries,
            "search_keywords": self.search_keywords,
            "filters": self.filters,
            "token_budget": self.token_budget,
            "budget_mode": self.budget_mode,
            "max_results": self.max_results,
            "max_results_per_query": self.max_results_per_query,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat(),
            "trace_id": self.trace_id,
            "include_conversation_context": self.include_conversation_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextDecision":
        """Deserialize from dict."""
        return cls(
            decision_id=data["decision_id"],
            trigger_type=TriggerType(data["trigger_type"]),
            trigger_payload=data.get("trigger_payload"),
            client_id=data.get("client_id", ""),
            instance_id=data.get("instance_id", ""),
            conversation_id=data.get("conversation_id"),
            user_id=data.get("user_id"),
            customer_id=data.get("customer_id"),
            retrieval_mode=RetrievalMode(data["retrieval_mode"]),
            context_types=data.get("context_types", ["facts", "preferences", "episodes"]),
            search_queries=data.get("search_queries", []),
            search_keywords=data.get("search_keywords", []),
            filters=data.get("filters", {}),
            token_budget=data.get("token_budget", 2000),
            budget_mode=data.get("budget_mode", "hard"),
            max_results=data.get("max_results", 10),
            max_results_per_query=data.get("max_results_per_query"),
            explanation=data.get("explanation", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            trace_id=data.get("trace_id", ""),
            include_conversation_context=data.get("include_conversation_context", True),
        )


@dataclass
class QualifiedData:
    """Data qualified for persistence (from Ingestion Agent)."""
    qualification_id: str
    should_persist: bool
    classification: str         # fact | preference | episode | temporal | emotion
    raw_input: Dict[str, Any]
    processed_form: Dict[str, Any]
    confidence: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MemoryArchitecturePlan:
    """Memory architecture blueprint for an instance.
    
    Produced by the Memory Architecture agent.
    Defines how memory should be structured for a specific use-case.
    """
    plan_id: str
    instance_id: str
    architecture_type: str               # "vector" | "graph" | "hybrid"
    schema: Dict[str, Any]              # Storage schema definition
    indexing_strategy: Dict[str, Any]   # How to index/organize memory
    retrieval_strategy: Dict[str, Any]  # How to retrieve context
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate architecture plan fields."""
        valid_types = ["vector", "graph", "hybrid"]
        if self.architecture_type not in valid_types:
            raise ValueError(f"Invalid architecture_type: {self.architecture_type}")
