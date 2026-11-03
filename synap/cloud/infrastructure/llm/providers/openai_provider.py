"""
OpenAI LLM Provider implementation.
"""
import logging
import os
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx

logger = logging.getLogger(__name__)

from synap.cloud.infrastructure.llm.models import (
    LLMProvider,
    LLMResponse,
    ModelInfo,
    ModelScores,
    ModelType,
    RateLimits,
    ProviderError,
    RateLimitError
)
from .base import BaseLLMProvider

# Reasoning-family models (gpt-5, gpt-6, o-series) reject ``max_tokens`` and any
# ``temperature`` but the default. Verified 2026-10-21 against gpt-6-luna: 400
# "'max_tokens' is not supported with this model. Use 'max_completion_tokens'"
# and 400 "'temperature' does not support 0.0 with this model".
_COMPLETION_TOKEN_MODELS = ("gpt-5", "gpt-6", "o1", "o3", "o4")
# Reasoning tokens count against max_completion_tokens, so a budget written for
# gpt-4o (8000) can be spent thinking before the JSON starts. 16000 is the
# budget the entity-resolution harness ran gpt-6-luna with.
_MIN_COMPLETION_TOKENS = int(os.environ.get("SYNAP_OPENAI_MIN_COMPLETION_TOKENS", "16000"))


# Gemini 2.5 and 3 think before they answer, behind an OpenAI-compatible endpoint too, and the thinking
# counts against ``max_tokens``. They accept ``max_tokens`` and a temperature, so only the budget is lifted.
# Verified 2026-11-01 against gemini-3.1-pro-preview: with max_tokens=200 the reply was cut off at the
# limit (11 characters, finish_reason "length"); with 4000 it was complete. Every short JSON call
# (the knowledge-update classifier asks for 200) came back empty or cut off.
_THINKING_BUDGET_MODELS = ("gemini-2.5", "gemini-3")


def _is_thinking_budget_model(model: str) -> bool:
    # A gateway may prefix the vendor ("google/gemini-3.1-pro-preview").
    return model.lower().rsplit("/", 1)[-1].startswith(_THINKING_BUDGET_MODELS)


def error_message(response: Any) -> str:
    """The message in a provider's error body, whatever shape the body has.

    OpenAI sends ``{"error": {"message": ...}}``. Google's OpenAI-compatible endpoint sends the same
    object inside a list (``[{"error": {...}}]``), and a gateway in front of a server may send plain
    text or an HTML page. Reading one of those as the object raised ``AttributeError`` inside the
    handler, which is not an ``LLMError``: the real status (a 503 to retry, a 400 to report) was
    lost and the call failed without the manager's retry.
    """
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - not JSON at all
        return (getattr(response, "text", "") or "")[:300] or "Unknown error"
    if isinstance(body, list):
        body = next((item for item in body if isinstance(item, dict)), {})
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or "Unknown error")
    if isinstance(error, str) and error:
        return error
    if isinstance(body, dict) and body.get("message"):
        return str(body["message"])
    return "Unknown error"


def token_params(model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """The request fields that carry the output budget and the temperature."""
    if model.lower().startswith(_COMPLETION_TOKEN_MODELS):
        return {"max_completion_tokens": max(max_tokens, _MIN_COMPLETION_TOKENS)}
    if _is_thinking_budget_model(model):
        return {"temperature": temperature, "max_tokens": max(max_tokens, _MIN_COMPLETION_TOKENS)}
    return {"temperature": temperature, "max_tokens": max_tokens}


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider.

    Supports: GPT-4o, GPT-4o-mini, GPT-4-turbo, etc.
    """

    # Model definitions
    MODELS = {
        "gpt-4o": ModelInfo(
            name="gpt-4o",
            provider=LLMProvider.OPENAI,
            model_type=ModelType.FLAGSHIP,
            scores=ModelScores(speed=8, accuracy=9, reasoning=9, tool_usage=10),
            rate_limits=RateLimits(rpm=5000, tpm=800000),
            context_window=128000,
            max_output_tokens=4096,
            description="Most capable GPT-4 model",
            # $2.50/$10.00 per 1M (gpt-4o-2024-08-06). Mirror any change in
            # cost_calculator._PRICING — the two drifted apart once already.
            pricing_input_per_1k=0.0025,
            pricing_output_per_1k=0.010
        ),
        "gpt-4-turbo": ModelInfo(
            name="gpt-4-turbo",
            provider=LLMProvider.OPENAI,
            model_type=ModelType.FLAGSHIP,
            scores=ModelScores(speed=7, accuracy=9, reasoning=9, tool_usage=10),
            rate_limits=RateLimits(rpm=5000, tpm=600000),
            context_window=128000,
            max_output_tokens=4096,
            description="GPT-4 Turbo with vision",
            pricing_input_per_1k=0.01,
            pricing_output_per_1k=0.03
        )
    }

    DEFAULT_MODEL = "gpt-4o"
    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        # Try plural env var first (comma-separated keys for rotation)
        api_keys_str = os.environ.get("OPENAI_API_KEYS", "")
        api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()] if api_keys_str else []

        # Fallback to single key
        if not api_keys:
            single = api_key or os.environ.get("OPENAI_API_KEY")
            if single:
                api_keys = [single]

        if not api_keys:
            raise ValueError("OpenAI API key is required (set OPENAI_API_KEYS or OPENAI_API_KEY)")

        super().__init__(api_keys[0], base_url or self.BASE_URL, api_keys=api_keys)
        logger.info("OpenAIProvider initialized with %d API key(s)", len(api_keys))

    @property
    def provider_name(self) -> LLMProvider:
        return LLMProvider.OPENAI

    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

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
        """Make a completion request to OpenAI."""
        self._validate_messages(messages)
        model = self._normalize_model_name(model)

        start_time = time.time()
        api_key = self._get_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            **token_params(model, temperature, max_tokens),
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        self._note_request(kwargs, payload, api_key)

        try:
            response = await self._http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            self._note_response(kwargs, response.headers)

            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 60))
                if self._key_rotator:
                    self._key_rotator.record_failure(api_key, "rate_limited", is_rate_limit=True)
                raise RateLimitError("openai", retry_after)

            if response.status_code >= 400:
                error_msg = error_message(response)
                if self._key_rotator:
                    self._key_rotator.record_failure(api_key, error_msg)
                raise ProviderError(error_msg, "openai", response.status_code)

            data = response.json()

        except httpx.TimeoutException:
            if self._key_rotator:
                self._key_rotator.record_failure(api_key, "timeout")
            raise ProviderError("Request timeout", "openai", 408)
        except httpx.RequestError as e:
            if self._key_rotator:
                self._key_rotator.record_failure(api_key, str(e))
            raise ProviderError(str(e), "openai")

        latency_ms = (time.time() - start_time) * 1000

        if self._key_rotator:
            rate_limit_info = {
                "remaining": response.headers.get("x-ratelimit-remaining-requests"),
            }
            self._key_rotator.record_success(api_key, time.time() - start_time, rate_limit_info)

        choice = data["choices"][0]
        message = choice["message"]

        return LLMResponse(
            content=message.get("content", ""),
            model=model,
            provider=LLMProvider.OPENAI,
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
            # Prompt-cache hits, billed at half rate. Reconciling our reported
            # spend against the OpenAI invoice showed we were ~15% high even
            # after fixing the list price, because this was never read: the
            # pipeline resends large fixed system prompts thousands of times a
            # day, so roughly a third of our input tokens are cache hits.
            cached_input_tokens=(
                data.get("usage", {})
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                or 0
            ),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=message.get("tool_calls"),
            raw_response=data
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        _usage_sink: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion from OpenAI."""
        import json as _json
        self._validate_messages(messages)
        model = self._normalize_model_name(model)

        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            **token_params(model, temperature, max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        self._note_request(kwargs, payload)

        async with self._http_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code == 429:
                raise RateLimitError("openai")

            if response.status_code >= 400:
                error_text = await response.aread()
                raise ProviderError(error_text.decode(), "openai", response.status_code)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    chunk = _json.loads(data)
                    # Final usage chunk has empty choices — capture and skip
                    usage = chunk.get("usage")
                    if usage and _usage_sink is not None:
                        _usage_sink["input_tokens"] = usage.get("prompt_tokens", 0)
                        _usage_sink["output_tokens"] = usage.get("completion_tokens", 0)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content

    def get_available_models(self) -> List[str]:
        return list(self.MODELS.keys())

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        return self.MODELS.get(model)

    def get_rate_limits(self, model: Optional[str] = None) -> RateLimits:
        if model and model in self.MODELS:
            return self.MODELS[model].rate_limits
        # Provider-level defaults
        return RateLimits(rpm=10000, tpm=2000000)
