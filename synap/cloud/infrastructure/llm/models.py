"""
LLM data models for the Config Manager.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class ModelType(str, Enum):
    """Model tier classification."""
    FLAGSHIP = "flagship"      # Best quality (GPT-4o, Claude 3.5 Sonnet)
    BALANCED = "balanced"      # Good balance (Llama 3.1 70B)
    ECONOMY = "economy"        # Fast/cheap (GPT-4o-mini, Claude 3 Haiku)


@dataclass
class ModelScores:
    """
    Model capability scores (1-10 scale).
    Used for intelligent model selection.
    """
    speed: int = 5          # Tokens per second
    accuracy: int = 5       # Fact retrieval, hallucination resistance
    reasoning: int = 5      # Complex logic capability
    tool_usage: int = 5     # JSON/function call ability

    def calculate_weighted_score(self, criteria: Dict[str, int]) -> float:
        """
        Calculate weighted score based on selection criteria.

        Args:
            criteria: Dict of {factor: weight} e.g., {"speed": 8, "accuracy": 5}

        Returns:
            Weighted sum of scores
        """
        total = 0.0
        for factor, weight in criteria.items():
            score = getattr(self, factor, 5)
            total += score * weight
        return total


@dataclass
class RateLimits:
    """Rate limits for a provider or model."""
    rpm: int = 10000        # Requests per minute
    tpm: int = 1000000      # Tokens per minute


@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    provider: LLMProvider
    model_type: ModelType
    scores: ModelScores
    rate_limits: RateLimits
    context_window: int = 128000
    max_output_tokens: int = 4096
    supports_streaming: bool = True
    supports_json_mode: bool = True
    supports_tools: bool = True

    # Optional metadata
    description: Optional[str] = None
    pricing_input_per_1k: Optional[float] = None   # $ per 1K input tokens
    pricing_output_per_1k: Optional[float] = None  # $ per 1K output tokens


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: LLMProvider
    enabled: bool = True
    api_key_env_var: str = ""  # Environment variable name for API key
    base_url: Optional[str] = None  # Custom base URL (for proxies)
    rate_limits: RateLimits = field(default_factory=RateLimits)
    models: Dict[str, ModelInfo] = field(default_factory=dict)


@dataclass
class SelectionCriteria:
    """
    Criteria for selecting a model.
    Higher values = more important.
    """
    speed: int = 5
    accuracy: int = 5
    reasoning: int = 5
    tool_usage: int = 5

    def to_dict(self) -> Dict[str, int]:
        return {
            "speed": self.speed,
            "accuracy": self.accuracy,
            "reasoning": self.reasoning,
            "tool_usage": self.tool_usage
        }


@dataclass
class LLMRequest:
    """Request to the LLM."""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    provider: Optional[LLMProvider] = None
    temperature: float = 0.0
    max_tokens: int = 4000
    json_mode: bool = False
    tools: Optional[List[Dict]] = None

    # Selection criteria (if model not specified)
    criteria: Optional[SelectionCriteria] = None
    agent_type: Optional[str] = None  # For agent-specific defaults

    # Metadata
    caller: str = "unknown"
    instance_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    model: str
    provider: LLMProvider

    # Token usage. `input_tokens` is the provider's full prompt count;
    # `cached_input_tokens` is the part of it that hit the provider's prompt
    # cache and is billed at a discount. It is a SUBSET of input_tokens, not
    # an addition to it — do not add the two together.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0

    # Metadata
    latency_ms: float = 0.0
    finish_reason: str = "stop"

    # Tool calls (if any)
    tool_calls: Optional[List[Dict]] = None

    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None

    # Reasoning, where the provider reports it. The count is a subset of
    # `output_tokens` on the providers that bill it as output. The text is
    # kept apart from `content` and is never returned to a caller as output.
    reasoning_tokens: Optional[int] = None
    reasoning_content: Optional[str] = None

    # Provider-side identity of this call, for matching against their logs.
    provider_request_id: Optional[str] = None


@dataclass
class LLMUsageRecord:
    """
    Record of LLM usage for analytics.
    Stored for cost tracking and analysis.
    """
    id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: datetime
    caller: str  # Agent name or component
    instance_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ModelAvailability:
    """
    Availability status of a model.
    Used for rate limit tracking and load balancing.
    """
    provider: LLMProvider
    model: str
    rpm_usage: float = 0.0       # 0.0 to 1.0 (percentage used)
    tpm_usage: float = 0.0       # 0.0 to 1.0 (percentage used)
    is_available: bool = True
    unavailable_until: Optional[datetime] = None
    reason: Optional[str] = None  # Reason for unavailability or load status

    @property
    def availability_factor(self) -> float:
        """
        Calculate availability factor for scoring.

        Returns:
            1.0 = fully available
            0.5 = near capacity (>80% used)
            0.0 = unavailable (100% used or marked unavailable)
        """
        if not self.is_available:
            return 0.0

        max_usage = max(self.rpm_usage, self.tpm_usage)

        if max_usage >= 1.0:
            return 0.0  # Hard block
        elif max_usage > 0.8:
            return 0.5  # Heavy penalty
        elif max_usage > 0.5:
            return 0.9  # Slight penalty
        else:
            return 1.0  # No penalty


# =============================================================================
# Agent Defaults
# =============================================================================

AGENT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "listening_retrieval_agent": {
        "criteria": SelectionCriteria(speed=8, accuracy=5, reasoning=3, tool_usage=7),
        "preferred_models": ["gpt-4o", "llama-3.1-70b"],
        "description": "Fast processing for real-time listening"
    },
    "ingestion_agent": {
        "criteria": SelectionCriteria(speed=3, accuracy=8, reasoning=7, tool_usage=5),
        "preferred_models": ["gpt-4o", "claude-3-5-sonnet"],
        "description": "Accurate extraction of memories (Tier 2+3: preferences, episodes, relationships, temporal)"
    },
    "ingestion_agent_light": {
        "criteria": SelectionCriteria(speed=6, accuracy=7, reasoning=4, tool_usage=4),
        "preferred_models": ["gpt-4o"],
        "description": "Light extraction for Tier 1 types: facts, emotions, entities, categorization"
    },
    "compaction_manager": {
        "criteria": SelectionCriteria(speed=9, accuracy=6, reasoning=5, tool_usage=4),
        "preferred_models": ["gpt-oss-20b"],
        "provider_override": LLMProvider.GROQ,
        "description": "Fast bulk processing for compaction"
    },
    "maca_agent": {
        "criteria": SelectionCriteria(speed=2, accuracy=9, reasoning=10, tool_usage=6),
        "preferred_models": ["gpt-4o", "claude-3-5-sonnet"],
        "description": "High reasoning for architecture decisions"
    },
    "context_decision_agent": {
        "criteria": SelectionCriteria(speed=5, accuracy=8, reasoning=8, tool_usage=6),
        "preferred_models": ["gpt-4o", "claude-3-5-sonnet"],
        "description": "Balanced for context decision making"
    },
    "context_decision_agent_stream": {
        "criteria": SelectionCriteria(speed=8, accuracy=8, reasoning=8, tool_usage=7),
        "preferred_models": ["openai/gpt-oss-120b"],
        "provider_override": LLMProvider.GROQ,
        "description": "Fast context decision for stream events (upgraded from gpt-oss-20b for better reasoning)"
    },
    "anticipation_agent": {
        # AnticipationRuntime's tool-use loop (search_memories,
        # push_bundle_to_sdk, etc.). Per-turn pattern: 4-6 sequential
        # LLM calls. Measured in a load test: switching
        # from OpenAI gpt-4o-mini to Groq gpt-oss-120b is projected to
        # cut per-turn LLM total from ~6.5s to ~1.3-2s and pool
        # throughput from ~6 to ~16-20 events/sec. provider_override
        # is required: the cross-provider selector ignores
        # preferred_models otherwise.
        "criteria": SelectionCriteria(speed=8, accuracy=7, reasoning=6, tool_usage=8),
        "preferred_models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "provider_override": LLMProvider.GROQ,
        "description": "Anticipation runtime tool-use loop — fast Groq with strong tool calling"
    },
    "anticipation_agent_loop": {
        # Alias for the operation tag agent_logic passes as ``operation``.
        # Cost tracker matches on operation when agent_type isn't keyed.
        "criteria": SelectionCriteria(speed=8, accuracy=7, reasoning=6, tool_usage=8),
        "preferred_models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "provider_override": LLMProvider.GROQ,
        "description": "Same as anticipation_agent — aliased for operation-tag selectors"
    },
    "playground_chat": {
        # Public playground demo agent. Groq
        # only — fast token-level streaming for the demo's "thinking" panel.
        # provider_override=GROQ is REQUIRED: the selector ignores
        # preferred_models and scores by criteria across all providers, so
        # without an override it can pick e.g. Gemini (not deployed here)
        # and fail before the first stream chunk arrives.
        # If Groq is down entirely, the selector's line-103 fallback opens
        # the candidate pool back up to other providers; complete()'s built-in
        # cross-provider retry then handles the rest. Anthropic is
        # intentionally not in the chain — playground demos shouldn't pull
        # on the Claude budget.
        "criteria": SelectionCriteria(speed=7, accuracy=7, reasoning=5, tool_usage=4),
        "preferred_models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "gpt-4o"],
        "provider_override": LLMProvider.GROQ,
        "description": "Public playground demo agent — Groq only, OpenAI fallback if Groq down"
    },
}


# =============================================================================
# Exceptions
# =============================================================================

class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMTimeoutError(LLMError):
    """A model call failed because the provider did not answer in time.

    Raised by the manager (complete and stream) when an attempt ended in
    ProviderTimeoutError, so callers that must not carry on without an answer
    (knowledge-update classification) can tell a timeout from a bad answer and
    fail their stage, letting the worker's retry and dead-letter queue run.
    """
    pass


class ProviderError(LLMError):
    """Error from LLM provider."""
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    """The provider did not answer within the configured time limit.

    Kept apart from ProviderError on purpose: a request that is slow or stuck is
    not evidence that the model is down, so it must not count toward the error
    counter that marks a model unavailable (tracker.record_error), nor mark an
    API key unhealthy. The call still fails, so the caller's retry runs.
    """
    def __init__(self, provider: str, phase: str = "read"):
        super().__init__(f"Request timeout ({phase})", provider, 408)
        self.phase = phase


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        super().__init__(f"Rate limit exceeded for {provider}", provider, 429)
        self.retry_after = retry_after


class ModelUnavailableError(LLMError):
    """Model is unavailable."""
    def __init__(self, provider: str, model: str, reason: str = ""):
        super().__init__(f"Model {model} from {provider} is unavailable: {reason}")
        self.provider = provider
        self.model = model


class AllProvidersUnavailableError(LLMError):
    """All providers are unavailable."""
    def __init__(self, retry_after: int = 60):
        super().__init__("All LLM providers are currently unavailable")
        self.retry_after = retry_after


class ConfigurationError(LLMError):
    """Configuration error."""
    pass
