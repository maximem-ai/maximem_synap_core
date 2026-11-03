"""
LLM Config Manager - Main interface for LLM provider management.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, AsyncGenerator, Any

import redis.asyncio as redis

from .models import (
    LLMProvider,
    LLMResponse,
    ModelInfo,
    SelectionCriteria,
    LLMError,
    ProviderError,
    RateLimitError,
)
from . import call_trace
from .registry import LLMRegistry
from .tracker import LLMUsageTracker
from .selector import LLMModelSelector
from .providers.base import ILLMProvider
from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_compatible_provider import OpenAICompatibleProvider
from . import settings as llm_settings

# The privacy boundary. Imported here rather than called from each caller
# because "every prompt was protected" has to be true of code nobody has
# written yet, and position is the only thing that makes it true.
# Cloud registers the real functions as hooks; Core has no privacy folder
# and the hooks do nothing, so messages and replies pass through.
from synap.cloud.shared import edition


def _guard_messages(messages, operation=None):
    """``(messages, guard)``: the boundary's view of the outgoing messages (Cloud), or unchanged (Core)."""
    if not edition.is_registered("provider_guard_messages"):
        return messages, None
    return edition.call("provider_guard_messages", messages, operation=operation)


def _guard_response(content):
    """The reply with the boundary's labels put back (Cloud), or unchanged (Core)."""
    if not edition.is_registered("provider_guard_response"):
        return content
    return edition.call("provider_guard_response", content)


logger = logging.getLogger(__name__)


# Strong references to in-flight usage-tracking tasks. Without this the loop
# holds them only weakly and a task can vanish mid-write, taking a billing row
# with it and logging nothing. Entries remove themselves on completion. Mirrors
# `providers._LLM_USAGE_TASKS`, which already does this for the OTHER usage
# scheduler; `manager.complete()` -- the path every consolidation LLM call takes
# -- was the one that never got it.
_USAGE_TRACK_TASKS: set = set()


def _log_usage_track_failure(task) -> None:
    """Surface a dropped billing row instead of losing it.

    Same log name as `providers._schedule_llm_usage` emits, so one grep for
    `cloud_usage_track_llm_usage_failed` finds failures from BOTH schedulers.
    Never raises: this is a done-callback, and a reporter that throws helps
    nobody.
    """
    try:
        if task.cancelled():
            logger.warning("cloud_usage_track_llm_usage_failed error=cancelled")
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("cloud_usage_track_llm_usage_failed error=%r", exc)
    except Exception:  # noqa: BLE001 -- a reporter must not become the fault
        pass

class LLMConfigManager:
    """
    Central manager for LLM provider access.

    Provides:
    - Unified interface for all LLM providers
    - Intelligent model selection based on task requirements
    - Rate limit tracking and load balancing
    - Automatic fallback on failures

    Usage:
        manager = LLMConfigManager(redis_client)

        # Simple completion
        response = await manager.complete(
            messages=[{"role": "user", "content": "Hello"}],
            agent_type="listening_retrieval_agent"
        )

        # With specific criteria
        response = await manager.complete(
            messages=[...],
            criteria=SelectionCriteria(speed=10, accuracy=5, reasoning=3, tool_usage=8)
        )

        # Force specific provider
        response = await manager.complete(
            messages=[...],
            force_provider=LLMProvider.GROQ
        )
    """

    MAX_RETRIES = 2

    def __init__(
        self,
        redis_client: redis.Redis,
        config_path: Optional[str] = None
    ):
        """
        Initialize the LLM Config Manager.

        Args:
            redis_client: Async Redis client for usage tracking
            config_path: Optional path to llm_registry.yaml
        """
        self._redis = redis_client
        self._registry = LLMRegistry(config_path)
        self._tracker = LLMUsageTracker(redis_client)
        self._providers: Dict[LLMProvider, ILLMProvider] = {}
        # The selector may only offer providers this process initialised, i.e.
        # those with a key. The registry alone is not that list: it enabled
        # gemini, no deployment had a Google key, and every fallback
        # was routed to a provider that could not answer.
        self._selector = LLMModelSelector(
            self._registry, self._tracker,
            available_providers=lambda: list(self._providers.keys()),
        )
        self._usage_tracker = None  # lazy-imported to avoid circular deps

        # One provider and one model for every call, when SYNAP_LLM_PROVIDER and SYNAP_LLM_MODEL
        # are set. None otherwise, and then nothing below changes.
        self._single = llm_settings.active()
        if self._single:
            for name in self._single.all_models():
                self._registry.register_model(self._single.provider, name)

        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize provider instances for enabled providers."""
        if self._single:
            self._initialize_single_provider()
            return
        enabled = self._registry.get_enabled_providers()

        if LLMProvider.OPENAI in enabled:
            try:
                self._providers[LLMProvider.OPENAI] = OpenAIProvider()
            except ValueError as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")

        if LLMProvider.ANTHROPIC in enabled:
            try:
                self._providers[LLMProvider.ANTHROPIC] = AnthropicProvider()
            except ValueError as e:
                logger.warning(f"Failed to initialize Anthropic provider: {e}")

        # Groq and Gemini are imported only here: their files are not in every build.
        if LLMProvider.GROQ in enabled:
            try:
                from .providers.groq_provider import GroqProvider
                self._providers[LLMProvider.GROQ] = GroqProvider()
            except (ValueError, ImportError) as e:
                logger.warning(f"Failed to initialize Groq provider: {e}")

        if LLMProvider.GEMINI in enabled:
            try:
                from .providers.gemini_provider import GeminiProvider
                self._providers[LLMProvider.GEMINI] = GeminiProvider()
            except (ValueError, ImportError) as e:
                logger.warning(f"Failed to initialize Gemini provider: {e}")

    def _initialize_single_provider(self) -> None:
        """Only the provider the developer named, whatever the registry file enables."""
        wanted = self._single.provider
        classes = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.ANTHROPIC: AnthropicProvider,
            LLMProvider.OLLAMA: OllamaProvider,
        }
        try:
            if self._single.base_url:
                self._providers[wanted] = OpenAICompatibleProvider(self._single.base_url)
            else:
                self._providers[wanted] = classes[wanted]()
        except ValueError as e:
            # No key. Say so at the start, not as "provider not initialized" on the first ingest.
            logger.error(
                "llm_provider_unavailable provider=%s reason=%s: every model call will fail until this is fixed",
                wanted.value, e)

    def _get_usage_tracker(self):
        if self._usage_tracker is None:
            # Cloud's hook returns get_usage_tracker(); Core has none and returns None.
            self._usage_tracker = edition.call("usage_tracking", "tracker")
        return self._usage_tracker

    def _fire_track_llm_usage(
        self,
        client_id: Optional[str],
        instance_id: Optional[str],
        input_tokens: int,
        output_tokens: int,
        model: str,
        operation: Optional[str],
        user_id: Optional[str],
        conversation_id: Optional[str],
        correlation_id: Optional[str],
        cached_input_tokens: int = 0,
    ) -> None:
        """Non-blocking fire-and-forget usage tracking."""
        if not client_id or not instance_id:
            return
        tracker = self._get_usage_tracker()
        if tracker is None:
            return
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(
                tracker.track_llm_usage(
                    client_id=client_id,
                    instance_id=instance_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    operation=operation,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    correlation_id=correlation_id,
                    cached_input_tokens=cached_input_tokens,
                )
            )
            # A bare create_task() is only WEAKLY held by the loop, so the task
            # can be garbage-collected mid-flight and its exception swallowed
            # with nothing logged. These tasks are the billing record for every
            # LLM call, so a collected one is a usage row that silently never
            # existed. The same hazard was already found and fixed for the
            # tripwire publisher in retrieval_manager ("hold a strong ref until
            # done so the task can't be GC'd mid-flight"); this path never got
            # the same treatment.
            _USAGE_TRACK_TASKS.add(task)
            task.add_done_callback(_USAGE_TRACK_TASKS.discard)
            # And a failure must be VISIBLE. Under a burst the telemetry pool
            # can refuse connections (TooManyConnectionsError); fire-and-forget is right for
            # the answer
            # path, but dropping the billing row without a word is not.
            task.add_done_callback(_log_usage_track_failure)
        except Exception as e:
            logger.debug("Failed to schedule usage tracking: %s", e)

    @staticmethod
    def _emit_trace(
        base: Optional[Dict[str, Any]],
        *,
        attempt: int,
        started_at: datetime,
        started: float,
        model: Optional[ModelInfo],
        sink: Dict[str, Any],
        response: Optional[LLMResponse] = None,
        raw_output: Optional[str] = None,
        error: Optional[BaseException] = None,
        fallback_reason: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Emit the trace record for one attempt. `base` is None when no sink
        is registered, in which case this does nothing."""
        if base is None:
            return
        try:
            record = call_trace.LLMCallRecord(**base)
            record.attempt = attempt
            record.started_at = started_at
            record.latency_ms = (time.perf_counter() - started) * 1000
            record.is_fallback = attempt > 0
            record.fallback_reason = fallback_reason
            if model is not None:
                record.provider = model.provider.value
                record.model = model.name
            record.request_params = sink.get("request_params")
            record.key_id = sink.get("key_id")
            record.base_url_host = sink.get("base_url_host")
            record.http_attempts = sink.get("http_attempts")
            record.provider_request_id = sink.get("provider_request_id")
            record.provider_region = sink.get("provider_region")
            if response is not None:
                record.input_tokens = response.input_tokens
                record.output_tokens = response.output_tokens
                record.cached_input_tokens = getattr(response, "cached_input_tokens", 0) or 0
                record.reasoning_tokens = response.reasoning_tokens
                record.reasoning = response.reasoning_content
                record.finish_reason = response.finish_reason
                record.tool_calls = response.tool_calls
                record.output = raw_output
                record.http_status = 200
                if response.provider_request_id:
                    record.provider_request_id = response.provider_request_id
            if error is not None:
                record.status, record.http_status = call_trace.classify_error(error)
                record.error_class = type(error).__name__
                record.error_message = str(error)[:500]
            for name, value in extra.items():
                setattr(record, name, value)
        except Exception:  # noqa: BLE001 -- a trace must not fail the call
            logger.debug("llm_trace_build_failed", exc_info=True)
            return
        call_trace.emit(record)

    async def close(self) -> None:
        """Close all provider HTTP clients. Call on shutdown."""
        for provider in self._providers.values():
            if hasattr(provider, 'close'):
                try:
                    await provider.close()
                except Exception as e:
                    logger.warning("Error closing provider: %s", e)
        logger.info("LLM provider HTTP clients closed")

    def get_provider(self, provider: LLMProvider) -> ILLMProvider:
        """
        Get a specific provider instance.

        Args:
            provider: The provider to get

        Returns:
            Provider instance

        Raises:
            ProviderError: If provider not available
        """
        if provider not in self._providers:
            raise ProviderError(
                f"Provider {provider.value} not available",
                provider=provider.value
            )
        return self._providers[provider]

    async def complete(
        self,
        messages: List[Dict[str, str]],
        criteria: Optional[SelectionCriteria] = None,
        agent_type: Optional[str] = None,
        force_provider: Optional[LLMProvider] = None,
        force_model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        json_mode: bool = False,
        tools: Optional[List[Dict]] = None,
        # Attribution — optional, used for internal cost monitoring
        client_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        conversation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        allow_cross_provider_fallback: bool = True,
        step: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Execute a completion request with automatic model selection.

        Args:
            messages: List of message dicts with 'role' and 'content'
            criteria: Selection criteria (overrides agent_type defaults)
            agent_type: Agent type for default criteria lookup
            force_provider: Force specific provider
            force_model: Force specific model (requires force_provider)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Force JSON output mode
            tools: Tool definitions for function calling
            allow_cross_provider_fallback: When False and force_provider is
                set, a failed call only ever retries on another model of that
                same provider. The default (True) lets a provider error fall
                through to whichever provider scores best, which is what keeps
                retrieval answering when Groq is down, but for a caller whose
                prompts were only ever measured on one provider it swaps in an
                unmeasured model without saying so.
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with completion result

        Raises:
            LLMError: If completion fails after retries
        """
        if self._single:
            # The developer chose one provider and model. What the call site asked for (a forced
            # gpt-4o, a measured pairing, a fallback to another provider) is Cloud's choice.
            force_provider, force_model = self._single.provider, self._single.model_for(step)
            allow_cross_provider_fallback = False

        # The privacy boundary, before anything else. Nothing below this line
        # sees the caller's original list, so a stage that assembled a prompt
        # from raw stored content cannot reach a provider with it. Runs once
        # rather than per retry attempt: the substitution is not idempotent for
        # a value whose replacement contains it.
        messages, _guard = _guard_messages(messages, operation=operation or agent_type)

        # One trace record per attempt, all sharing this call's id. None when
        # no sink is registered, and then nothing below does any trace work.
        trace_base: Optional[Dict[str, Any]] = None
        if call_trace.tracing_enabled():
            trace_base = {
                "call_id": call_trace.new_call_id(),
                "call_kind": "completion",
                "agent_type": agent_type,
                "requested_provider": force_provider.value if force_provider else None,
                "requested_model": force_model,
                "selection_criteria": criteria.to_dict() if criteria else None,
                "requested_params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "json_mode": json_mode,
                    **{k: v for k, v in kwargs.items()
                       if isinstance(v, (str, int, float, bool))},
                },
                "json_mode": json_mode,
                # Inline images are replaced by their type and size: trace
                # bodies are stored, and an image is not prompt text.
                "messages": call_trace.redact_inline_media(messages),
                "tools": tools,
                "guard": _guard,
                **call_trace.resolve_scope(
                    client_id=client_id, instance_id=instance_id, user_id=user_id,
                    conversation_id=conversation_id, correlation_id=correlation_id,
                    operation=operation or agent_type,
                ),
            }

        # Select model
        if force_model and force_provider:
            model_info = self._registry.get_model_info(force_provider, force_model)
            if not model_info:
                raise LLMError(f"Model {force_model} not found for {force_provider.value}")
        else:
            model_info = await self._selector.select_model(
                criteria=criteria,
                agent_type=agent_type,
                force_provider=force_provider,
            )

            if not model_info:
                raise LLMError("No models available for request")

        # Execute with retries
        last_error: Optional[Exception] = None
        attempted_models: List[str] = []
        failures: List[str] = []
        current_model = model_info
        pinned_provider = (
            force_provider if not allow_cross_provider_fallback else None
        )

        for attempt in range(self.MAX_RETRIES + 1):
            trace_sink: Dict[str, Any] = {}
            trace_kwargs = {"_trace_sink": trace_sink} if trace_base is not None else {}
            attempt_started_at = datetime.now(timezone.utc)
            attempt_started = time.perf_counter()
            try:
                try:
                    provider = self.get_provider(current_model.provider)
                    response = await provider.complete(
                        messages=messages,
                        model=current_model.name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        tools=tools,
                        **trace_kwargs,
                        **kwargs
                    )
                except BaseException as exc:
                    # Every way an attempt can end without a response, the
                    # caller's own timeout (a CancelledError) included.
                    self._emit_trace(
                        trace_base, attempt=attempt, started_at=attempt_started_at,
                        started=attempt_started, model=current_model,
                        sink=trace_sink, error=exc,
                        fallback_reason=failures[-1][:200] if failures else None,
                    )
                    raise

                if trace_base is not None:
                    (response.reasoning_tokens,
                     response.reasoning_content) = call_trace.read_reasoning(
                        response.raw_response
                    )
                    # The trace keeps what the provider wrote, before the
                    # guard puts labels back.
                    self._emit_trace(
                        trace_base, attempt=attempt, started_at=attempt_started_at,
                        started=attempt_started, model=current_model,
                        sink=trace_sink, response=response,
                        raw_output=response.content,
                        fallback_reason=failures[-1][:200] if failures else None,
                    )

                # The other half of the boundary. If the outgoing messages
                # carried person-shaped stand-ins, the labels go back in here,
                # so no caller can ever hold a response containing one.
                response.content = _guard_response(response.content)

                # Track rate-limit counters (Redis)
                await self._tracker.record_request(
                    provider=current_model.provider,
                    model=current_model.name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )

                # Track analytics / cost (TimescaleDB, non-blocking)
                self._fire_track_llm_usage(
                    client_id=client_id,
                    instance_id=instance_id,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    model=current_model.name,
                    operation=operation or agent_type,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    correlation_id=correlation_id,
                    cached_input_tokens=getattr(
                        response, "cached_input_tokens", 0
                    ),
                )

                return response

            except RateLimitError as e:
                logger.warning(f"Rate limit hit for {current_model.name}: {e}")
                last_error = e
                failures.append(f"{current_model.name}: {e}")

                # Mark model as temporarily unavailable
                await self._tracker.mark_unavailable(
                    provider=current_model.provider,
                    model=current_model.name,
                    duration_seconds=60,
                )

                # Try fallback
                attempted_models.append(current_model.name)
                fallback = await self._selector.get_fallback_model(
                    original_provider=force_provider,  # Respect provider constraint
                    original_model=current_model.name,
                )
                fallback = self._allowed_fallback(pinned_provider, fallback)

                if fallback and fallback.name not in attempted_models:
                    current_model = fallback
                    logger.info(f"Falling back to {fallback.name}")
                else:
                    break  # No more fallbacks

            except ProviderError as e:
                logger.error(f"Provider error for {current_model.name}: {e}")
                last_error = e
                failures.append(f"{current_model.name}: {e}")

                await self._tracker.record_error(
                    provider=current_model.provider,
                    model=current_model.name,
                    is_rate_limit=False,
                )

                # Try fallback
                attempted_models.append(current_model.name)
                fallback = await self._selector.get_fallback_model(
                    # Another provider is allowed unless the caller pinned one.
                    original_provider=pinned_provider,
                    original_model=current_model.name,
                )
                fallback = self._allowed_fallback(pinned_provider, fallback)

                if fallback and fallback.name not in attempted_models:
                    current_model = fallback
                    logger.info(f"Falling back to {fallback.name}")
                else:
                    break

        # All retries exhausted. Name every attempt, not only the last one:
        # when the primary rejected a request for length and the fallback was
        # a provider with no key, "Last error: Provider gemini not available"
        # hid the real reason for an afternoon.
        if failures:
            raise LLMError("All models failed. " + " | ".join(failures))
        raise LLMError(f"All models failed. Last error: {last_error}")

    def _allowed_fallback(
        self,
        pinned_provider: Optional[LLMProvider],
        fallback: Optional[ModelInfo],
    ) -> Optional[ModelInfo]:
        """The fallback, if it stays on the pinned provider and, when the developer chose the
        models, is one of those models. Another model of the same provider is not
        one they chose and may not be installed."""
        fallback = self._keep_within(pinned_provider, fallback)
        if self._single and fallback and fallback.name not in self._single.all_models():
            logger.warning("Not falling back to %s, it is not a configured model", fallback.name)
            return None
        return fallback

    @staticmethod
    def _keep_within(
        pinned_provider: Optional[LLMProvider],
        fallback: Optional[ModelInfo],
    ) -> Optional[ModelInfo]:
        """Drop a fallback that leaves the provider the caller pinned.

        The selector prefers the same provider but, when that provider has no
        other model to offer, goes on to every other provider. Asking it for
        the same provider is therefore not enough to stay there.
        """
        if pinned_provider and fallback and fallback.provider != pinned_provider:
            logger.warning(
                "Not falling back to %s:%s, caller is pinned to %s",
                fallback.provider.value, fallback.name, pinned_provider.value,
            )
            return None
        return fallback

    async def stream(
        self,
        messages: List[Dict[str, str]],
        criteria: Optional[SelectionCriteria] = None,
        agent_type: Optional[str] = None,
        force_provider: Optional[LLMProvider] = None,
        force_model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        # Attribution — optional, used for internal cost monitoring
        client_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        conversation_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        step: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Execute a streaming completion request.

        Note: Streaming does not support automatic fallback mid-stream.

        Args:
            messages: List of message dicts
            criteria: Selection criteria
            agent_type: Agent type for defaults
            force_provider: Force specific provider
            force_model: Force specific model
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            String chunks of the completion

        Raises:
            LLMError: If streaming fails
        """
        if self._single:
            force_provider, force_model = self._single.provider, self._single.model_for(step)

        # The privacy boundary. Streaming gets the same treatment as
        # completion: a stage that streams is not a stage that may leak.
        messages, _guard = _guard_messages(messages, operation=operation or agent_type)

        trace_base: Optional[Dict[str, Any]] = None
        if call_trace.tracing_enabled():
            trace_base = {
                "call_id": call_trace.new_call_id(),
                "call_kind": "stream",
                "agent_type": agent_type,
                "requested_provider": force_provider.value if force_provider else None,
                "requested_model": force_model,
                "selection_criteria": criteria.to_dict() if criteria else None,
                "requested_params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **{k: v for k, v in kwargs.items()
                       if isinstance(v, (str, int, float, bool))},
                },
                "messages": call_trace.redact_inline_media(messages),
                "guard": _guard,
                **call_trace.resolve_scope(
                    client_id=client_id, instance_id=instance_id, user_id=user_id,
                    conversation_id=conversation_id, correlation_id=correlation_id,
                    operation=operation or agent_type,
                ),
            }

        # Select model
        if force_model and force_provider:
            model_info = self._registry.get_model_info(force_provider, force_model)
            if not model_info:
                raise LLMError(f"Model {force_model} not found")
        else:
            model_info = await self._selector.select_model(
                criteria=criteria,
                agent_type=agent_type,
                force_provider=force_provider,
            )

            if not model_info:
                raise LLMError("No models available for request")

        provider = self.get_provider(model_info.provider)

        # Trace state. A stream ends in one of three ways and only the first
        # reaches the code after the loop: it completes, it raises, or the
        # consumer stops reading (which arrives as GeneratorExit, a
        # BaseException no `except Exception` sees). The `finally` below
        # covers all three.
        trace_sink: Dict[str, Any] = {}
        trace_kwargs = {"_trace_sink": trace_sink} if trace_base is not None else {}
        trace_chunks: List[str] = []
        trace_error: Optional[BaseException] = None
        trace_done = False
        trace_ttft_ms: Optional[float] = None
        stream_started_at = datetime.now(timezone.utc)
        stream_started = time.perf_counter()
        usage_sink: Dict[str, int] = {}

        try:
            async for chunk in provider.stream(
                messages=messages,
                model=model_info.name,
                temperature=temperature,
                max_tokens=max_tokens,
                _usage_sink=usage_sink,
                **trace_kwargs,
                **kwargs
            ):
                if trace_base is not None:
                    if trace_ttft_ms is None:
                        trace_ttft_ms = (time.perf_counter() - stream_started) * 1000
                    trace_chunks.append(chunk)
                yield chunk
            trace_done = True

            # Use actual token counts from provider; fall back to estimation only
            # when the provider didn't populate usage_sink (e.g. Groq/Gemini).
            actual_input = usage_sink.get(
                "input_tokens",
                sum(len(m.get("content", "")) // 4 for m in messages),
            )
            actual_output = usage_sink.get("output_tokens", 0)

            await self._tracker.record_request(
                provider=model_info.provider,
                model=model_info.name,
                input_tokens=actual_input,
                output_tokens=actual_output,
            )

            self._fire_track_llm_usage(
                client_id=client_id,
                instance_id=instance_id,
                input_tokens=actual_input,
                output_tokens=actual_output,
                model=model_info.name,
                operation=operation or agent_type,
                user_id=user_id,
                conversation_id=conversation_id,
                correlation_id=correlation_id,
            )

        except RateLimitError as e:
            trace_error = e
            await self._tracker.mark_unavailable(
                provider=model_info.provider,
                model=model_info.name,
            )
            raise
        except Exception as e:
            trace_error = e
            await self._tracker.record_error(
                provider=model_info.provider,
                model=model_info.name,
            )
            raise LLMError(f"Streaming failed: {e}")
        finally:
            if trace_base is not None:
                if trace_error is None and not trace_done:
                    trace_error = asyncio.CancelledError("stream not read to the end")
                self._emit_trace(
                    trace_base, attempt=0, started_at=stream_started_at,
                    started=stream_started, model=model_info, sink=trace_sink,
                    error=trace_error,
                    output="".join(trace_chunks),
                    ttft_ms=trace_ttft_ms,
                    input_tokens=usage_sink.get(
                        "input_tokens",
                        sum(len(str(m.get("content") or "")) // 4 for m in messages),
                    ),
                    output_tokens=usage_sink.get("output_tokens", 0),
                    tokens_estimated="input_tokens" not in usage_sink,
                    finish_reason="stop" if trace_done else None,
                )

    def get_registry(self) -> LLMRegistry:
        """Get the model registry."""
        return self._registry

    def get_tracker(self) -> LLMUsageTracker:
        """Get the usage tracker."""
        return self._tracker

    async def get_model_status(self, provider: LLMProvider, model: str) -> Dict[str, Any]:
        """
        Get current status of a specific model.

        Args:
            provider: The provider
            model: The model name

        Returns:
            Dict with availability and usage info
        """
        model_info = self._registry.get_model_info(provider, model)
        if not model_info:
            return {"error": "Model not found"}

        availability = await self._tracker.get_availability(
            provider=provider,
            model=model,
            rpm_limit=model_info.rate_limits.rpm,
            tpm_limit=model_info.rate_limits.tpm,
        )

        return {
            "provider": provider.value,
            "model": model,
            "is_available": availability.is_available,
            "availability_factor": availability.availability_factor,
            "rpm_usage": availability.rpm_usage,
            "tpm_usage": availability.tpm_usage,
            "reason": availability.reason,
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_llm_manager: Optional[LLMConfigManager] = None


def get_llm_manager(redis_url: Optional[str] = None) -> LLMConfigManager:
    """Get or create the LLM Config Manager singleton.

    Environment variables:
        REDIS_URL: Redis connection URL (default: redis://localhost:6379)
        LLM_REGISTRY_CONFIG: Path to llm_registry.yaml
    """
    global _llm_manager
    if _llm_manager is None:
        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(url, decode_responses=True)
        config_path = os.environ.get("LLM_REGISTRY_CONFIG")
        _llm_manager = LLMConfigManager(redis_client, config_path)
    return _llm_manager


def reset_llm_manager() -> None:
    """Reset the singleton (useful for testing)."""
    global _llm_manager
    _llm_manager = None
