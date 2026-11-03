"""Centralized logging configuration for Synap Cloud.

Provides:
- Environment-aware logging (dev, staging, production)
- JSON structured logging for production
- Human-readable for development
- Correlation ID context injection
- Sensitive data sanitization
- Log level configuration via environment
"""

import json
import logging
import logging.config
import os
import re
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variables for correlation ID tracking
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
instance_id_ctx: ContextVar[Optional[str]] = ContextVar('instance_id', default=None)
client_id_ctx: ContextVar[Optional[str]] = ContextVar('client_id', default=None)


class CorrelationIDFilter(logging.Filter):
    """Inject correlation ID into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation context to log record."""
        trace_id = None
        try:
            from synap.cloud.observability.tracing import current_trace_id_hex

            trace_id = current_trace_id_hex()
        except Exception:
            trace_id = None

        # Preserve explicitly-provided IDs from logger.extra first.
        record.correlation_id = (
            getattr(record, "correlation_id", None)
            or correlation_id_ctx.get()
            or "N/A"
        )
        record.instance_id = (
            getattr(record, "instance_id", None)
            or instance_id_ctx.get()
            or "N/A"
        )
        record.client_id = (
            getattr(record, "client_id", None)
            or client_id_ctx.get()
            or "N/A"
        )
        record.trace_id = getattr(record, "trace_id", None) or trace_id or "N/A"
        return True


class SanitizingFilter(logging.Filter):
    """Sanitize sensitive data from logs."""

    SENSITIVE_KEYS = {
        'password', 'passwd', 'token', 'api_key', 'secret', 'authorization',
        'access_key', 'private_key', 'certificate', 'credential', 'bearer'
    }

    # Redact the VALUE of a sensitive key inside a message, not the message.
    #
    # The old rule was: if any SENSITIVE_KEYS entry appears anywhere in the
    # string and the string contains "=" or ":", replace the entire string with
    # "***REDACTED***". Many log lines were blanked that way, and
    # almost none of them held a secret. Two causes, both verified by calling
    # this filter:
    #
    #   'cert' matched as a substring, so every line containing "certain",
    #   "concert" or "certificate" and a colon went blank. That is how retrieval
    #   HQE_MATCH lines disappeared mid-investigation.
    #
    #   'token' matched "input_tokens=1241 output_tokens=465", so our own LLM
    #   cost telemetry lines were destroyed in full.
    #
    # So: 'cert' is gone in favour of 'certificate', and a message keeps
    # everything except the matched value. A bare number is never redacted,
    # because no credential is an integer and token COUNTS are the thing this
    # rule kept eating.
    _SENSITIVE_PAIR_RE = re.compile(
        r"(?i)"
        r"([\w.\-]{0,32}?"
        r"(?:password|passwd|token|api[_\-]?key|secret|authorization|"
        r"access[_\-]?key|private[_\-]?key|credential|bearer|certificate))"
        # The separator may carry an auth scheme. Without this,
        # "Authorization: Bearer sk-live-..." redacted the word "Bearer" and
        # left the credential standing.
        r"([\"']?\s*[=:]\s*[\"']?(?:Bearer\s+|Basic\s+|Token\s+)?)"
        r"([^\s,;&)}\]\"']+)"
    )
    _NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

    # Credentials that are recognisable by their SHAPE, whatever key (or none)
    # they sit under: the one-time onboarding code and a synap_ API key.
    # A Python dict repr inside an f-string ("{'code': 'syn_ob_...'}") has no
    # key the pair rule above can see, so match the value itself.
    _SECRET_SHAPE_RE = re.compile(
        r"(?i:\b(syn_ob_)[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b)"
        # A real API key is `synap_` + 64 random URL-safe characters. Match that
        # shape only: lower-case prefix (so SYNAP_MACA_BUILD_SWEEPER_ENABLED is
        # left alone), 40 or more URL-safe characters, and at least one digit,
        # one upper-case and one lower-case letter (so `synap_credit_wallet_...`
        # table names and `synap_maca_expected_build_seconds` metric names are
        # left alone: they are all lower case).
        r"|\b(synap_)(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Z])(?=[A-Za-z0-9_\-]*[a-z])"
        r"[A-Za-z0-9_\-]{40,}"
    )

    # Keys where a bare number is a count, not a credential, so the number
    # survives: "tokens": 1241 is our own cost telemetry. Every other sensitive
    # key is redacted whatever its value looks like, because a short numeric
    # password or PIN is a real thing.
    _COUNTABLE_KEYS = ('token',)

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize sensitive data from log record."""
        # Sanitize the message, but NOT when it is a format string with
        # arguments waiting for it.
        #
        # `logger.info("... tokens=%d ...", n)` has a message written by us and
        # a value supplied at the call site. Only the second can hold user
        # data. Redacting the first replaces it with "***REDACTED***" while
        # leaving the args in place, and `getMessage()` then evaluates
        # "***REDACTED***" % args and raises TypeError. Both formatters catch
        # that and fall back to `str(record.msg)`, so the line survives with
        # all of its content gone, no crash and no logging error to notice.
        #
        # Measured 2026-08-21: 29 call sites across 16 files, tripped almost
        # always by the word "token" meaning token COUNT, because SENSITIVE_KEYS
        # matches as a substring. Among them every gRPC auth-failure line, which
        # went blank exactly when someone was debugging an auth problem.
        #
        # Messages with no args are still sanitized: an f-string or a plain
        # string is where user data actually reaches a logger.
        if hasattr(record, 'msg') and not getattr(record, 'args', None):
            record.msg = self._sanitize_value(record.msg)

        # Sanitize args
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                # Through _sanitize_value, not a dict comprehension over the
                # values. Python's logging turns a single Mapping argument into
                # `record.args` itself, so `logger.info("payload=%s", body)`
                # arrives here as the caller's dict, and sanitising only its
                # values never looks at the keys. `{"password": "hunter2"}` came
                # through untouched, which is the one thing this filter exists
                # to stop, and it is the most natural way anyone logs a payload.
                #
                # The same dict nested one level deeper was always handled
                # correctly, because then `record.args` is a tuple and
                # _sanitize_value sees the dict as a value and checks its keys.
                # That is why the existing tests, which cover `record.msg` as a
                # dict, did not catch it.
                record.args = self._sanitize_value(record.args)
            else:
                record.args = tuple(self._sanitize_value(arg) for arg in record.args)

        return True

    def _is_sensitive_key(self, key: Any) -> bool:
        """Return True when a dict key names something that must be redacted.

        Dict keys are not guaranteed to be strings: counters keyed by int,
        tuple, or None reach the logger from ordinary application code. This
        filter runs in front of *every* log call, so a key type it cannot
        handle turns the caller's successful operation into a logged failure.
        Coerce instead of assuming.
        """
        if not isinstance(key, str):
            try:
                key = str(key)
            except Exception:
                return False
        lowered = key.lower()
        return any(s in lowered for s in self.SENSITIVE_KEYS)

    def _must_redact(self, key: Any, value: Any) -> bool:
        """Whether this dict entry's value has to go.

        Sensitive key plus one exemption: a bare number under a token-shaped key
        is a COUNT. ``{"input_tokens": 1241}`` is the LLM cost telemetry we
        write ourselves, and redacting it made spend impossible to audit from
        logs for weeks. Every other sensitive key is redacted whatever the value
        looks like.
        """
        if not self._is_sensitive_key(key):
            return False
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return True
        lowered = str(key).lower()
        return not any(c in lowered for c in self._COUNTABLE_KEYS)

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize sensitive values."""
        if isinstance(value, dict):
            return {
                k: "***REDACTED***" if self._must_redact(k, v)
                else self._sanitize_value(v)
                for k, v in value.items()
            }
        elif isinstance(value, str):
            # Redact the value beside a sensitive key, keep the rest of the line.
            # See _SENSITIVE_PAIR_RE for why this is no longer all-or-nothing.
            def _sub(m: "re.Match") -> str:
                # Same exemption as _must_redact, and only for the same keys:
                # "input_tokens=1241" is a count, "password=1234" is not.
                if self._NUMERIC_RE.match(m.group(3)) and any(
                    c in m.group(1).lower() for c in self._COUNTABLE_KEYS
                ):
                    return m.group(0)
                return f"{m.group(1)}{m.group(2)}***REDACTED***"

            value = self._SENSITIVE_PAIR_RE.sub(_sub, value)
            return self._SECRET_SHAPE_RE.sub(
                lambda m: f"{m.group(1) or m.group(2)}***REDACTED***", value
            )
        return value


class JSONFormatter(logging.Formatter):
    """JSON structured formatter for production / log-aggregation pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)

        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": getattr(record, "correlation_id", "N/A"),
            "instance_id": getattr(record, "instance_id", "N/A"),
            "client_id": getattr(record, "client_id", "N/A"),
            "trace_id": getattr(record, "trace_id", "N/A"),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in human-readable format with colors."""
        correlation_id = getattr(record, 'correlation_id', 'N/A')

        # Truncate correlation ID for display
        corr_id_display = correlation_id[:8] if correlation_id != 'N/A' else 'N/A'

        # Get color for log level
        color = self.COLORS.get(record.levelname, '')

        # Format base message
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        base_format = (
            f"{timestamp} "
            f"{color}{record.levelname:8}{self.RESET} "
            f"[{corr_id_display}] "
            f"{record.name}:{record.funcName}:{record.lineno} - "
            f"{message}"
        )

        # Add exception info if present
        if record.exc_info:
            base_format += "\n" + self.formatException(record.exc_info)

        return base_format


def get_log_level() -> int:
    """Get log level from environment variable.

    Returns:
        Log level constant from logging module
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration based on environment.

    Returns:
        Dictionary configuration for logging.config.dictConfig
    """
    env = os.getenv("ENVIRONMENT", "development").lower()
    log_level = get_log_level()

    formatter_class = "synap.cloud.logging_config.HumanReadableFormatter"

    # Persistent file logging — survives container rebuilds when
    # /app/logs is bind-mounted to a host directory.
    log_dir = os.getenv("LOG_DIR", "/app/logs")
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "default",
            "filters": ["correlation_id", "sanitizing"],
            "stream": "ext://sys.stdout",
        },
    }
    root_handlers = ["console"]

    if os.path.isdir(log_dir) and os.access(log_dir, os.W_OK):
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "default",
            "filters": ["correlation_id", "sanitizing"],
            "filename": os.path.join(log_dir, "synap-cloud.log"),
            "maxBytes": 50 * 1024 * 1024,  # 50MB
            "backupCount": 5,
        }
        root_handlers.append("file")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "correlation_id": {
                "()": "synap.cloud.logging_config.CorrelationIDFilter",
            },
            "sanitizing": {
                "()": "synap.cloud.logging_config.SanitizingFilter",
            },
        },
        "formatters": {
            "default": {
                "()": formatter_class,
            },
        },
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": root_handlers,
        },
        # Set third-party libraries to WARNING to reduce noise
        "loggers": {
            # The in-process SDK sets the "synap.sdk" parent logger to
            # WARNING at construction (config_utils.configure_logging),
            # which silenced every anticipation-cache HIT/MISS decision,
            # so cache behavior was invisible while debugging the
            # echo-poisoning eval failures. An explicitly-configured child
            # level wins over the parent's later setLevel, so this keeps
            # cache decisions visible without touching SDK behavior.
            "synap.sdk.cache.anticipation": {
                "level": os.getenv("SYNAP_SDK_CACHE_LOG_LEVEL", "INFO"),
            },
            "uvicorn": {"level": "INFO"},
            "uvicorn.access": {"level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "fastapi": {"level": "INFO"},
            "sqlalchemy.engine": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "asyncio": {"level": "WARNING"},
        },
    }

    return config


def configure_logging() -> None:
    """Configure logging for the application.

    Call this once at application startup (in main.py lifespan).
    Safe to call multiple times (idempotent).
    """
    config = get_logging_config()
    logging.config.dictConfig(config)


def set_correlation_context(
    correlation_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> None:
    """Set correlation context for current async task.

    Use in middleware to propagate request context to all logs.

    Args:
        correlation_id: Request correlation ID
        instance_id: SDK instance ID
        client_id: Client ID
    """
    if correlation_id:
        correlation_id_ctx.set(correlation_id)
    if instance_id:
        instance_id_ctx.set(instance_id)
    if client_id:
        client_id_ctx.set(client_id)


def clear_correlation_context() -> None:
    """Clear correlation context."""
    correlation_id_ctx.set(None)
    instance_id_ctx.set(None)
    client_id_ctx.set(None)
