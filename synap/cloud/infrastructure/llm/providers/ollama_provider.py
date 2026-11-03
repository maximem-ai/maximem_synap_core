"""Ollama LLM provider: a model the developer runs on their own machine.

Uses Ollama's native chat API, not its OpenAI-compatible one, because only the native API lets
the request set the context window. Ollama's default window is small (2048 or 4096 tokens) and it
cuts a longer prompt without an error, which for ingestion means the end of the conversation is
never read. ``OLLAMA_NUM_CTX`` sets it (default 16384).

    OLLAMA_BASE_URL   where Ollama listens. From inside the Core container the host is
                      http://host.docker.internal:11434 (the compose file maps that name on Linux too).
    OLLAMA_NUM_CTX    context window in tokens for every request
    OLLAMA_KEEP_ALIVE how long the model stays loaded after a call, for example 30m (optional)

There is no API key and no rate limit. Any model the developer has pulled can be named.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from synap.cloud.infrastructure.llm.models import (
    LLMProvider,
    LLMResponse,
    ModelInfo,
    ModelScores,
    ModelType,
    ProviderError,
    RateLimits,
)
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://host.docker.internal:11434"
DEFAULT_NUM_CTX = 16384
_NO_LIMITS = RateLimits(rpm=1_000_000, tpm=1_000_000_000)


def _flatten(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI-style messages to Ollama's: text parts joined, data-URL images moved to ``images``."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            texts, images = [], []
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    texts.append(part.get("text", ""))
                elif kind == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        images.append(url.split(",", 1)[1])
                    else:
                        raise ProviderError(
                            "Ollama needs images sent as data: URLs, not links", "ollama", 400)
            msg: Dict[str, Any] = {"role": m["role"], "content": "\n".join(t for t in texts if t)}
            if images:
                msg["images"] = images
            out.append(msg)
        else:
            out.append({"role": m["role"], "content": content or ""})
    return out


def _tool_calls(message: Dict[str, Any]) -> Optional[List[Dict]]:
    """Ollama's tool calls in the shape the rest of the engine reads (OpenAI's)."""
    calls = message.get("tool_calls")
    if not calls:
        return None
    return [{
        "id": f"call_{i}",
        "type": "function",
        "function": {
            "name": c["function"]["name"],
            "arguments": json.dumps(c["function"].get("arguments") or {}),
        },
    } for i, c in enumerate(calls)]


class OllamaProvider(BaseLLMProvider):
    """A local Ollama server."""

    DEFAULT_MODEL = ""  # there is no sensible default: the developer names the model

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        super().__init__(api_key="", base_url=url)
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))
        self.keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE") or None
        self._known: Dict[str, ModelInfo] = {}
        logger.info("OllamaProvider initialized base_url=%s num_ctx=%d", self.base_url, self.num_ctx)

    @property
    def provider_name(self) -> LLMProvider:
        return LLMProvider.OLLAMA

    @property
    def default_model(self) -> str:
        return self.DEFAULT_MODEL

    # --- requests ---------------------------------------------------------------------------

    def _payload(self, messages, model, temperature, max_tokens, json_mode, tools, stream) -> Dict[str, Any]:
        self._validate_messages(messages)
        options: Dict[str, Any] = {"temperature": temperature, "num_ctx": self.num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens
        payload: Dict[str, Any] = {
            "model": model, "messages": _flatten(messages), "stream": stream, "options": options}
        if json_mode:
            payload["format"] = "json"
        if tools:
            payload["tools"] = tools
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _unreachable(self, e: Exception) -> ProviderError:
        return ProviderError(
            f"Could not reach Ollama at {self.base_url} ({type(e).__name__}). Is it running? "
            "From inside the Core container the host is host.docker.internal; set OLLAMA_BASE_URL "
            "if yours listens elsewhere.", "ollama")

    async def _raise_for(self, response: httpx.Response, model: str) -> None:
        if response.status_code < 400:
            return
        try:
            detail = response.json().get("error") or response.text
        except Exception:
            detail = response.text
        if response.status_code == 404:
            detail = f"{detail} (run `ollama pull {model}` on the machine that runs Ollama)"
        raise ProviderError(str(detail)[:500], "ollama", response.status_code)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        json_mode: bool = False,
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        payload = self._payload(messages, model, temperature, max_tokens, json_mode, tools, stream=False)
        self._note_request(kwargs, payload)
        start = time.time()
        try:
            response = await self._http_client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException:
            raise ProviderError("Request timeout", "ollama", 408)
        except httpx.RequestError as e:
            raise self._unreachable(e)
        self._note_response(kwargs, response.headers)
        await self._raise_for(response, model)
        data = response.json()
        message = data.get("message") or {}
        prompt, produced = data.get("prompt_eval_count", 0) or 0, data.get("eval_count", 0) or 0
        return LLMResponse(
            content=message.get("content", "") or "",
            model=model,
            provider=LLMProvider.OLLAMA,
            input_tokens=prompt,
            output_tokens=produced,
            total_tokens=prompt + produced,
            latency_ms=(time.time() - start) * 1000,
            # "length" is how a cut-off reply is told apart from a finished one
            finish_reason={"stop": "stop", "length": "length"}.get(data.get("done_reason"), "stop"),
            tool_calls=_tool_calls(message),
            raw_response=data,
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        _usage_sink: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(messages, model, temperature, max_tokens, False, None, stream=True)
        self._note_request(kwargs, payload)
        try:
            async with self._http_client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    await self._raise_for(response, model)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise ProviderError(str(chunk["error"])[:500], "ollama")
                    text = (chunk.get("message") or {}).get("content", "")
                    if text:
                        yield text
                    if chunk.get("done") and _usage_sink is not None:
                        _usage_sink["input_tokens"] = chunk.get("prompt_eval_count", 0) or 0
                        _usage_sink["output_tokens"] = chunk.get("eval_count", 0) or 0
        except httpx.RequestError as e:
            raise self._unreachable(e)

    # --- models -------------------------------------------------------------------------------

    def _info(self, model: str) -> ModelInfo:
        if model not in self._known:
            self._known[model] = ModelInfo(
                name=model, provider=LLMProvider.OLLAMA, model_type=ModelType.BALANCED,
                scores=ModelScores(), rate_limits=_NO_LIMITS,
                context_window=self.num_ctx, max_output_tokens=self.num_ctx,
                description="Local model served by Ollama",
                pricing_input_per_1k=0.0, pricing_output_per_1k=0.0,
            )
        return self._known[model]

    def get_available_models(self) -> List[str]:
        return list(self._known)

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        return self._info(model)

    def is_model_supported(self, model: str) -> bool:
        return True  # whatever the developer has pulled

    def get_rate_limits(self, model: Optional[str] = None) -> RateLimits:
        return _NO_LIMITS

    async def health_check(self) -> bool:
        try:
            r = await self._http_client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False
