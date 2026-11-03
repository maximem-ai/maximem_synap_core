"""
Per-call trace hook for the LLM layer.

The manager emits one `LLMCallRecord` per attempt at a model call. What happens
to a record is decided by whoever registers a sink: Cloud registers a writer
(`synap.cloud.observability.llm_traces`), and with no sink registered `emit`
returns at once, so this file adds nothing to a process that never installs
one.

Nothing here may raise into the caller or await. A trace is evidence about a
call, never a reason for the call to fail.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Keys of a provider payload that carry prompt text. They are kept out of
# `request_params`, which lives with the long-retention metadata.
PROMPT_PAYLOAD_KEYS = frozenset(
    {"messages", "tools", "contents", "system", "systemInstruction"}
)

SCOPE_KEYS = (
    "client_id", "instance_id", "customer_id", "user_id", "conversation_id",
    "correlation_id", "operation", "prompt_id", "prompt_version",
    "config_version",
)


@dataclass
class LLMCallRecord:
    """One attempt at one model call."""

    call_id: str
    attempt: int = 0
    call_kind: str = "completion"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Scope
    client_id: Optional[str] = None
    instance_id: Optional[str] = None
    customer_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Intent
    operation: Optional[str] = None
    agent_type: Optional[str] = None
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    config_version: Optional[str] = None

    # Routing
    requested_provider: Optional[str] = None
    requested_model: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    selection_criteria: Optional[Dict[str, Any]] = None

    # Parameters. `requested_params` is what the caller asked for;
    # `request_params` is the payload the adapter built, minus prompt text.
    # The two differ: an adapter may drop or remap a parameter.
    requested_params: Dict[str, Any] = field(default_factory=dict)
    request_params: Optional[Dict[str, Any]] = None
    base_url_host: Optional[str] = None
    key_id: Optional[str] = None

    # Usage
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: Optional[int] = None
    tokens_estimated: bool = False

    # Timing
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    http_attempts: Optional[int] = None

    # Outcome
    status: str = "ok"
    http_status: Optional[int] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None
    provider_request_id: Optional[str] = None
    provider_region: Optional[str] = None

    # Privacy guard summary, as `guard_messages` returned it.
    guard: Optional[Dict[str, Any]] = None

    # Text. `messages` is what was sent (after the privacy guard); `output` is
    # what the provider returned (before the guard restores labels). Held by
    # reference until the sink has dealt with the record.
    json_mode: bool = False
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    output: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


# ── Sink ─────────────────────────────────────────────────────────────────────

_sink: Optional[Callable[[LLMCallRecord], None]] = None
_sink_failures = 0
_last_failure_log = 0.0


def set_trace_sink(sink: Optional[Callable[[LLMCallRecord], None]]) -> None:
    """Register (or with None, remove) the function that receives records.

    The sink is called on the caller's thread and in the caller's context, so
    it must only hand the record off; any real work belongs behind a queue.
    """
    global _sink
    _sink = sink


def tracing_enabled() -> bool:
    return _sink is not None


def emit(record: LLMCallRecord) -> None:
    """Hand a record to the sink. Never raises, never awaits."""
    global _sink_failures, _last_failure_log
    sink = _sink
    if sink is None:
        return
    try:
        sink(record)
    except Exception:  # noqa: BLE001 -- a trace must not fail the call it describes
        _sink_failures += 1
        now = time.monotonic()
        if now - _last_failure_log > 60:
            _last_failure_log = now
            logger.warning(
                "llm_trace_sink_failed total=%d", _sink_failures, exc_info=True
            )


def new_call_id() -> str:
    return str(uuid.uuid4())


# ── Scope ────────────────────────────────────────────────────────────────────

_scope: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "llm_trace_scope", default=None
)


@contextmanager
def trace_scope(**attrs: Any) -> Iterator[None]:
    """Attribute the model calls made inside this block.

    For a caller that knows the client, user or operation but does not pass
    them to the manager. Nested scopes add to the outer one; None values are
    ignored so an inner scope never blanks an outer value.
    """
    merged = dict(_scope.get() or {})
    merged.update({k: v for k, v in attrs.items() if v is not None and k in SCOPE_KEYS})
    token = _scope.set(merged)
    try:
        yield
    finally:
        _scope.reset(token)


def _request_context() -> Dict[str, Any]:
    """The request-scoped ids the logging layer already tracks, when present."""
    try:
        from synap.cloud.logging_config import (
            client_id_ctx, correlation_id_ctx, instance_id_ctx,
        )
    except Exception:  # noqa: BLE001 -- absent in a build without that module
        return {}
    return {
        "client_id": client_id_ctx.get(),
        "instance_id": instance_id_ctx.get(),
        "correlation_id": correlation_id_ctx.get(),
    }


def resolve_scope(**explicit: Any) -> Dict[str, Any]:
    """Explicit arguments win, then `trace_scope`, then the request context."""
    out: Dict[str, Any] = {}
    scoped = _scope.get() or {}
    ambient = _request_context()
    for key in SCOPE_KEYS:
        value = explicit.get(key) or scoped.get(key) or ambient.get(key)
        if value is not None:
            out[key] = str(value)
    return out


# ── Helpers shared by the manager and the adapters ──────────────────────────


def redact_inline_media(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Messages for a trace, with inline (base64 data URL) images replaced.

    A vision call sends the image as a data URL inside a list-of-parts content.
    The trace keeps the instruction and the reply, never the image: only its
    type and size, the same shape `image_extractor` already records. Returns
    the list unchanged when no message has list content, which is every call
    but a vision one.
    """
    if not any(isinstance(m.get("content"), list) for m in messages):
        return messages
    out: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        parts: List[Any] = []
        for part in content:
            url = (
                part.get("image_url", {}).get("url")
                if isinstance(part, dict) and isinstance(part.get("image_url"), dict)
                else None
            )
            if isinstance(url, str) and url.startswith("data:"):
                media_type = url[5:].split(";", 1)[0] or "unknown"
                b64_chars = len(url.split(",", 1)[-1])
                part = {
                    **part,
                    "image_url": {
                        **part["image_url"],
                        "url": f"[image {media_type}, {b64_chars} base64 chars]",
                    },
                }
            parts.append(part)
        out.append({**m, "content": parts})
    return out


def key_fingerprint(api_key: Optional[str]) -> Optional[str]:
    """A stable id for an API key that reveals none of it."""
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def note_request(
    sink: Optional[Dict[str, Any]],
    payload: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Record the payload an adapter is about to send, minus prompt text."""
    if sink is None:
        return
    try:
        sink["request_params"] = {
            k: v for k, v in payload.items() if k not in PROMPT_PAYLOAD_KEYS
        }
        sink["key_id"] = key_fingerprint(api_key)
        if base_url:
            sink["base_url_host"] = base_url.split("://", 1)[-1].split("/", 1)[0]
    except Exception:  # noqa: BLE001
        pass


def note_response(
    sink: Optional[Dict[str, Any]], headers: Any, *, http_attempts: int = 1
) -> None:
    """Record what the provider's response headers say about this call."""
    if sink is None:
        return
    try:
        sink["http_attempts"] = http_attempts
        request_id = headers.get("x-request-id") or headers.get("request-id")
        if request_id:
            sink["provider_request_id"] = str(request_id)
        region = headers.get("x-groq-region")
        if region:
            sink["provider_region"] = str(region)
    except Exception:  # noqa: BLE001
        pass


def read_reasoning(raw: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    """Reasoning token count and text from a provider's raw response.

    Each provider reports these differently and some report only one of them:
    OpenAI-style bodies carry a count under `completion_tokens_details` and,
    on Groq, the text on the message; Anthropic returns `thinking` blocks;
    Gemini returns a count under `usageMetadata` and marks thought parts.
    """
    if not isinstance(raw, dict):
        return None, None
    tokens: Optional[int] = None
    text: Optional[str] = None
    try:
        usage = raw.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        if details.get("reasoning_tokens") is not None:
            tokens = int(details["reasoning_tokens"])

        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("reasoning") or message.get("reasoning_content") or None

        blocks = raw.get("content")
        if isinstance(blocks, list):
            thinking = [
                b.get("thinking", "") for b in blocks
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            if any(thinking):
                text = "\n".join(t for t in thinking if t)

        meta = raw.get("usageMetadata") or {}
        if meta.get("thoughtsTokenCount") is not None:
            tokens = int(meta["thoughtsTokenCount"])
        candidates = raw.get("candidates") or []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            thoughts = [p.get("text", "") for p in parts if p.get("thought")]
            if any(thoughts):
                text = "\n".join(t for t in thoughts if t)
    except Exception:  # noqa: BLE001 -- an unexpected shape is not an error here
        pass
    return tokens, text


def classify_error(exc: BaseException) -> Tuple[str, Optional[int]]:
    """Map an exception from a provider call to a trace status and HTTP code."""
    import asyncio

    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return "cancelled", None
    if status_code == 429:
        return "rate_limited", 429
    if status_code == 408:
        return "timeout", 408
    if status_code is not None or type(exc).__name__ == "ProviderError":
        return "provider_error", status_code
    return "error", None


def record_external_call(
    *,
    call_kind: str,
    provider: str,
    model: Optional[str],
    messages: Optional[List[Dict[str, Any]]] = None,
    output: Optional[str] = None,
    latency_ms: Optional[float] = None,
    error: Optional[BaseException] = None,
    started_at: Optional[datetime] = None,
    **fields: Any,
) -> None:
    """Record a model call that did not go through the manager.

    For the paths that build their own provider client. Scope is resolved the
    same way as for manager calls; any other `LLMCallRecord` field may be
    passed by name.
    """
    if _sink is None:
        return
    try:
        scope = resolve_scope(**{k: fields.pop(k, None) for k in SCOPE_KEYS})
        record = LLMCallRecord(
            call_id=new_call_id(),
            call_kind=call_kind,
            provider=provider,
            model=model,
            messages=messages,
            output=output,
            latency_ms=latency_ms,
            **scope,
        )
        if started_at is not None:
            record.started_at = started_at
        if error is not None:
            record.status, record.http_status = classify_error(error)
            record.error_class = type(error).__name__
            record.error_message = str(error)[:500]
        for name, value in fields.items():
            if hasattr(record, name):
                setattr(record, name, value)
    except Exception:  # noqa: BLE001
        logger.debug("llm_trace_external_record_failed", exc_info=True)
        return
    emit(record)


@contextmanager
def external_call(
    call_kind: str,
    provider: str,
    model: Optional[str],
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    default_operation: Optional[str] = None,
    **fields: Any,
) -> Iterator[Dict[str, Any]]:
    """Time and record a model call made outside the manager.

        with external_call("jev", "typesafe", model, messages=...) as trace:
            answers = await post(...)
            trace["output"] = json.dumps(answers)

    The body may set any `LLMCallRecord` field on the yielded dict. An
    exception is recorded and re-raised unchanged. `default_operation` names
    the call when no surrounding `trace_scope` already has. With no sink
    registered this only yields an empty dict.
    """
    out: Dict[str, Any] = {}
    if _sink is None:
        yield out
        return
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    error: Optional[BaseException] = None
    try:
        yield out
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            merged = {**fields, **out}
            if not resolve_scope(operation=merged.get("operation")).get("operation"):
                merged["operation"] = default_operation
            record_external_call(
                call_kind=call_kind, provider=provider, model=model,
                messages=merged.pop("messages", messages),
                output=merged.pop("output", None),
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error, started_at=started_at, **merged,
            )
        except Exception:  # noqa: BLE001 -- a trace must not fail the call
            logger.debug("llm_trace_external_call_failed", exc_info=True)
