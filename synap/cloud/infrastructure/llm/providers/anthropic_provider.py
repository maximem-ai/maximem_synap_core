"""
Anthropic (Claude) LLM Provider implementation.
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


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic API provider.

    Supports: Claude 3.5 Sonnet, Claude 3 Haiku, etc.
    """

    MODELS = {
        "claude-3-5-sonnet-20241022": ModelInfo(
            name="claude-3-5-sonnet-20241022",
            provider=LLMProvider.ANTHROPIC,
            model_type=ModelType.FLAGSHIP,
            scores=ModelScores(speed=7, accuracy=10, reasoning=10, tool_usage=9),
            rate_limits=RateLimits(rpm=4000, tpm=400000),
            context_window=200000,
            max_output_tokens=8192,
            description="Most intelligent Claude model",
            pricing_input_per_1k=0.003,
            pricing_output_per_1k=0.015
        ),
        "claude-3-haiku-20240307": ModelInfo(
            name="claude-3-haiku-20240307",
            provider=LLMProvider.ANTHROPIC,
            model_type=ModelType.ECONOMY,
            scores=ModelScores(speed=9, accuracy=7, reasoning=6, tool_usage=8),
            rate_limits=RateLimits(rpm=4000, tpm=400000),
            context_window=200000,
            max_output_tokens=4096,
            description="Fast and affordable Claude",
            pricing_input_per_1k=0.00025,
            pricing_output_per_1k=0.00125
        ),
        "claude-3-opus-20240229": ModelInfo(
            name="claude-3-opus-20240229",
            provider=LLMProvider.ANTHROPIC,
            model_type=ModelType.FLAGSHIP,
            scores=ModelScores(speed=5, accuracy=10, reasoning=10, tool_usage=9),
            rate_limits=RateLimits(rpm=4000, tpm=400000),
            context_window=200000,
            max_output_tokens=4096,
            description="Most powerful Claude for complex tasks",
            pricing_input_per_1k=0.015,
            pricing_output_per_1k=0.075
        )
    }

    # Aliases for convenience
    MODEL_ALIASES = {
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-opus": "claude-3-opus-20240229"
    }

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    BASE_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        # Try plural env var first (comma-separated keys for rotation)
        api_keys_str = os.environ.get("ANTHROPIC_API_KEYS", "")
        api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()] if api_keys_str else []

        # Fallback to single key
        if not api_keys:
            single = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if single:
                api_keys = [single]

        if not api_keys:
            raise ValueError("Anthropic API key is required (set ANTHROPIC_API_KEYS or ANTHROPIC_API_KEY)")

        super().__init__(api_keys[0], base_url or self.BASE_URL, api_keys=api_keys)
        logger.info("AnthropicProvider initialized with %d API key(s)", len(api_keys))

    @property
    def provider_name(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC

    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    def _normalize_model_name(self, model: str) -> str:
        """Convert aliases to full model names."""
        return self.MODEL_ALIASES.get(model, model)

    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple[str, List[Dict]]:
        """
        Convert OpenAI-style messages to Anthropic format.

        Anthropic requires system message to be separate.
        """
        system_message = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": self._convert_content(msg["content"]),
                })

        return system_message, anthropic_messages

    @staticmethod
    def _convert_content(content: Any) -> Any:
        """Text stays as it is. An OpenAI-style ``image_url`` part (what the image reader sends) becomes
        Anthropic's ``image`` block: a base64 source for a data URL, a url source for a web address."""
        if not isinstance(content, list):
            return content
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = str((part.get("image_url") or {}).get("url") or "")
                if url.startswith("data:") and ";base64," in url:
                    header, data = url[5:].split(";base64,", 1)
                    out.append({"type": "image",
                                "source": {"type": "base64", "media_type": header or "image/png", "data": data}})
                else:
                    out.append({"type": "image", "source": {"type": "url", "url": url}})
            else:
                out.append(part)
        return out

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
        """Make a completion request to Anthropic."""
        self._validate_messages(messages)
        model = self._normalize_model_name(model)

        start_time = time.time()
        api_key = self._get_api_key()

        system_message, anthropic_messages = self._convert_messages(messages)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "anthropic-beta": "prompt-caching-2024-07-31",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if system_message:
            payload["system"] = [
                {"type": "text", "text": system_message, "cache_control": {"type": "ephemeral"}}
            ]

        if tools:
            payload["tools"] = self._convert_tools(tools)

        self._note_request(kwargs, payload, api_key)

        try:
            response = await self._http_client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
            )
            self._note_response(kwargs, response.headers)

            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 60))
                if self._key_rotator:
                    self._key_rotator.record_failure(api_key, "rate_limited", is_rate_limit=True)
                raise RateLimitError("anthropic", retry_after)

            if response.status_code >= 400:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                if self._key_rotator:
                    self._key_rotator.record_failure(api_key, error_msg)
                raise ProviderError(error_msg, "anthropic", response.status_code)

            data = response.json()

        except httpx.TimeoutException:
            if self._key_rotator:
                self._key_rotator.record_failure(api_key, "timeout")
            raise ProviderError("Request timeout", "anthropic", 408)
        except httpx.RequestError as e:
            if self._key_rotator:
                self._key_rotator.record_failure(api_key, str(e))
            raise ProviderError(str(e), "anthropic")

        latency_ms = (time.time() - start_time) * 1000

        if self._key_rotator:
            rate_limit_info = {
                "remaining": response.headers.get("anthropic-ratelimit-requests-remaining"),
            }
            self._key_rotator.record_success(api_key, time.time() - start_time, rate_limit_info)

        # Extract content from Anthropic's response format
        content = ""
        tool_calls = None

        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(block)

        return LLMResponse(
            content=content,
            model=model,
            provider=LLMProvider.ANTHROPIC,
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            total_tokens=(
                data.get("usage", {}).get("input_tokens", 0) +
                data.get("usage", {}).get("output_tokens", 0)
            ),
            latency_ms=latency_ms,
            finish_reason=data.get("stop_reason", "end_turn"),
            tool_calls=tool_calls,
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
        """Stream completion from Anthropic."""
        import json as _json
        self._validate_messages(messages)
        model = self._normalize_model_name(model)

        system_message, anthropic_messages = self._convert_messages(messages)

        headers = {
            "x-api-key": self._get_api_key(),
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

        if system_message:
            payload["system"] = system_message

        self._note_request(kwargs, payload)

        async with self._http_client.stream(
            "POST",
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code == 429:
                raise RateLimitError("anthropic")

            if response.status_code >= 400:
                error_text = await response.aread()
                raise ProviderError(error_text.decode(), "anthropic", response.status_code)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = _json.loads(line[6:])
                    event_type = data.get("type")

                    if event_type == "message_start" and _usage_sink is not None:
                        usage = data.get("message", {}).get("usage", {})
                        _usage_sink["input_tokens"] = usage.get("input_tokens", 0)

                    elif event_type == "message_delta" and _usage_sink is not None:
                        usage = data.get("usage", {})
                        _usage_sink["output_tokens"] = usage.get("output_tokens", 0)

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        return anthropic_tools

    def get_available_models(self) -> List[str]:
        return list(self.MODELS.keys()) + list(self.MODEL_ALIASES.keys())

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        model = self._normalize_model_name(model)
        return self.MODELS.get(model)

    def get_rate_limits(self, model: Optional[str] = None) -> RateLimits:
        if model:
            model = self._normalize_model_name(model)
            if model in self.MODELS:
                return self.MODELS[model].rate_limits
        return RateLimits(rpm=4000, tpm=400000)
