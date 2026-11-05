"""
Base interface for LLM providers.
"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx

from synap.cloud.infrastructure.llm.call_trace import note_request, note_response

from synap.cloud.infrastructure.llm.models import (
    LLMProvider,
    LLMResponse,
    ModelInfo,
    RateLimits
)



def _seconds_from_env(name: str, default: Optional[float], environ=None) -> Optional[float]:
    """Seconds from an env var. 'none', 'off' or '0' means no limit (None).

    Anything else that is not a positive number stops start-up with a message
    naming the variable, so a typo is never silently replaced by a default.
    """
    raw = (os.environ if environ is None else environ).get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("none", "off", "0"):
        return None
    try:
        seconds = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds or 'none', got {raw!r}") from None
    if seconds < 0:
        raise ValueError(f"{name} must not be negative, got {raw!r}")
    return seconds


# The timeouts that mean "the provider accepted the request but did not answer
# in time". They raise ProviderTimeoutError, which neither marks the API key
# unhealthy nor counts toward model availability. A ConnectTimeout is left out
# on purpose: failing to even connect within 10 s means the endpoint is
# unreachable, which is the evidence the availability tracker exists for, so it
# keeps its old path (httpx.RequestError: key failure, counted error).
SLOW_ANSWER_TIMEOUTS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)


LLM_TIMEOUT_VARS = (
    ("SYNAP_LLM_READ_TIMEOUT_SECONDS", 300.0),
    ("SYNAP_LLM_WRITE_TIMEOUT_SECONDS", 60.0),
)


def timeout_setting_problems(environ=None) -> List[str]:
    """One message per invalid timeout variable; checked by Core start-up so a
    typo stops the server instead of being logged as a provider init failure."""
    found = []
    for name, default in LLM_TIMEOUT_VARS:
        try:
            _seconds_from_env(name, default, environ)
        except ValueError as exc:
            found.append(str(exc))
    return found


def llm_http_timeout() -> httpx.Timeout:
    """Timeouts for every LLM HTTP call: connect 10 s, read/write bounded, pool unbounded."""
    return httpx.Timeout(
        connect=10.0,
        read=_seconds_from_env("SYNAP_LLM_READ_TIMEOUT_SECONDS", 300.0),
        write=_seconds_from_env("SYNAP_LLM_WRITE_TIMEOUT_SECONDS", 60.0),
        pool=None,
    )

class ILLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    All provider implementations (OpenAI, Anthropic, Groq) must implement this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> LLMProvider:
        """Return the provider identifier."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        json_mode: bool = False,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Make a completion request.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            model: Model name (e.g., "gpt-4o")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            json_mode: If True, force JSON output
            tools: Optional list of tool definitions for function calling
            **kwargs: Provider-specific options

        Returns:
            LLMResponse with content and metadata

        Raises:
            ProviderError: If the provider returns an error
            RateLimitError: If rate limit is exceeded
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming completion request.

        Args:
            messages: List of chat messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Provider-specific options

        Yields:
            String chunks as they arrive

        Raises:
            ProviderError: If the provider returns an error
            RateLimitError: If rate limit is exceeded
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Return list of available model names for this provider.

        Returns:
            List of model name strings
        """
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.

        Args:
            model: Model name

        Returns:
            ModelInfo if model exists, None otherwise
        """
        pass

    @abstractmethod
    def get_rate_limits(self, model: Optional[str] = None) -> RateLimits:
        """
        Get rate limits for a model or provider.

        Args:
            model: Optional model name. If None, returns provider-level limits.

        Returns:
            RateLimits for the model/provider
        """
        pass

    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by this provider."""
        return model in self.get_available_models()

    async def health_check(self) -> bool:
        """
        Check if the provider is healthy.

        Default implementation tries a minimal completion.
        Providers can override with their own health check.

        Returns:
            True if healthy, False otherwise
        """
        try:
            await self.complete(
                messages=[{"role": "user", "content": "hi"}],
                model=self.default_model,
                max_tokens=5
            )
            return True
        except Exception:
            return False


class BaseLLMProvider(ILLMProvider):
    """
    Base implementation with common functionality.

    Providers should extend this class and implement abstract methods.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, api_keys: Optional[List[str]] = None):
        """
        Initialize provider.

        Args:
            api_key: API key for authentication
            base_url: Optional custom base URL
            api_keys: Optional list of API keys for rotation (overrides api_key)
        """
        self.base_url = base_url
        self._models_cache: Optional[Dict[str, ModelInfo]] = None

        # Key rotation: if multiple keys provided, set up rotator
        if api_keys and len(api_keys) > 1:
            from synap.cloud.infrastructure.llm.key_rotator import LLMKeyRotator
            self._key_rotator = LLMKeyRotator(api_keys=api_keys, calls_per_key=20)
            self.api_key = api_keys[0]  # default for health checks etc.
        else:
            self._key_rotator = None
            self.api_key = api_key

        # Shared HTTP client with connection pooling (avoids TCP+TLS per request).
        # Timeouts come from llm_http_timeout(): connect 10 s, read 300 s
        # (SYNAP_LLM_READ_TIMEOUT_SECONDS), write 60 s
        # (SYNAP_LLM_WRITE_TIMEOUT_SECONDS), pool unlimited. The read limit was
        # once removed because a timeout raised a plain ProviderError, which the
        # manager counts toward model availability: three slow calls in the
        # penalty window took the only model offline for everyone. Now a read or
        # write timeout raises ProviderTimeoutError, which the providers do not
        # hold against the API key and the manager does not count; the call still
        # fails (LLMTimeoutError), so the ingestion stage retries and, after its
        # last try, dead-letters the document instead of hanging a worker forever.
        # Calls that pass their own timeout= (e.g. Ollama's 5 s health check)
        # keep it.
        self._http_client = httpx.AsyncClient(
            timeout=llm_http_timeout(),
            limits=httpx.Limits(
                max_connections=int(os.environ.get("LLM_MAX_CONNECTIONS", "50")),
                max_keepalive_connections=int(os.environ.get("LLM_MAX_KEEPALIVE_CONNECTIONS", "20")),
            ),
        )

    async def close(self):
        """Close the shared HTTP client. Call on shutdown."""
        await self._http_client.aclose()

    def _get_api_key(self) -> str:
        """Get next API key (rotated if multiple keys available)."""
        if self._key_rotator:
            return self._key_rotator.get_next_key()
        return self.api_key

    def _note_request(
        self, kwargs: Dict[str, Any], payload: Dict[str, Any], api_key: Optional[str] = None
    ) -> None:
        """Report the payload about to be sent to the caller's trace, if any.

        The manager passes `_trace_sink` the way it passes `_usage_sink` to
        `stream()`. It is read here and never forwarded to the provider.
        """
        note_request(
            kwargs.get("_trace_sink"), payload, api_key=api_key, base_url=self.base_url
        )

    def _note_response(
        self, kwargs: Dict[str, Any], headers: Any, http_attempts: int = 1
    ) -> None:
        """Report the provider's request id and region to the caller's trace."""
        note_response(kwargs.get("_trace_sink"), headers, http_attempts=http_attempts)

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format."""
        if not messages:
            raise ValueError("Messages cannot be empty")

        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")

    def _normalize_model_name(self, model: str) -> str:
        """
        Normalize model name to provider's expected format.

        Override in subclasses if needed.
        """
        return model

    async def _handle_error(self, response: Any, provider: str) -> None:
        """
        Handle error response from provider.

        Override in subclasses for provider-specific error handling.
        """
        pass
