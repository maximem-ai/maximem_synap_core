"""Unit tests for logging configuration module."""

import json
import logging
import os
from unittest.mock import patch

import pytest

from synap.cloud.logging_config import (
    CorrelationIDFilter,
    HumanReadableFormatter,
    JSONFormatter,
    SanitizingFilter,
    clear_correlation_context,
    configure_logging,
    get_log_level,
    get_logging_config,
    set_correlation_context,
)


class TestCorrelationIDFilter:
    """Test correlation ID filter."""

    def test_adds_correlation_context_to_record(self):
        """Test that correlation ID filter adds context to log record."""
        # Set correlation context
        set_correlation_context(
            correlation_id="test-corr-123",
            instance_id="inst-456",
            client_id="client-789",
        )

        # Create filter and record
        filter_instance = CorrelationIDFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Apply filter
        result = filter_instance.filter(record)

        # Verify
        assert result is True
        assert record.correlation_id == "test-corr-123"
        assert record.instance_id == "inst-456"
        assert record.client_id == "client-789"

        # Cleanup
        clear_correlation_context()

    def test_uses_na_when_context_not_set(self):
        """Test that filter uses N/A when context is not set."""
        clear_correlation_context()

        filter_instance = CorrelationIDFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        filter_instance.filter(record)

        assert record.correlation_id == "N/A"
        assert record.instance_id == "N/A"
        assert record.client_id == "N/A"


class TestSanitizingFilter:
    """Test sensitive data sanitization filter."""

    def test_sanitizes_dict_with_sensitive_keys(self):
        """Test that sensitive keys in dictionaries are redacted."""
        filter_instance = SanitizingFilter()

        # Create record with sensitive data in message
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg={"password": "secret123", "username": "john"},
            args=(),
            exc_info=None,
        )

        filter_instance.filter(record)

        assert record.msg["password"] == "***REDACTED***"
        assert record.msg["username"] == "john"

    def test_sanitizes_string_with_sensitive_content(self):
        """Test that strings with sensitive content are redacted."""
        filter_instance = SanitizingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Authorization: Bearer token123",
            args=(),
            exc_info=None,
        )

        filter_instance.filter(record)

        # The credential goes; the line stays readable. Blanking the whole line
        # is what used to blank many log lines (see _SENSITIVE_PAIR_RE).
        assert "token123" not in record.msg
        assert record.msg == "Authorization: Bearer ***REDACTED***"

    def test_preserves_non_sensitive_data(self):
        """Test that non-sensitive data is preserved."""
        filter_instance = SanitizingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg={"username": "john", "email": "john@example.com"},
            args=(),
            exc_info=None,
        )

        filter_instance.filter(record)

        assert record.msg["username"] == "john"
        assert record.msg["email"] == "john@example.com"

    def test_handles_non_string_dict_keys(self):
        """Test that non-string dict keys do not raise.

        Regression: the analytics cleanup sweep logs a per-tier summary keyed
        by retention days (int). The key check used to call ``.lower()``
        directly, so the whole log call raised
        ``'int' object has no attribute 'lower'`` and a successful cleanup was
        reported as a failure.
        """
        filter_instance = SanitizingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg={30: 2, 90: 1, 365: 4},
            args=(),
            exc_info=None,
        )

        result = filter_instance.filter(record)

        assert result is True
        assert record.msg == {30: 2, 90: 1, 365: 4}

    def test_handles_exotic_dict_keys(self):
        """Test that None and tuple keys survive sanitization."""
        filter_instance = SanitizingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg={None: "unknown", ("a", "b"): 1, 1.5: "float", True: "bool"},
            args=(),
            exc_info=None,
        )

        result = filter_instance.filter(record)

        assert result is True
        assert record.msg[None] == "unknown"
        assert record.msg[("a", "b")] == 1
        assert record.msg[1.5] == "float"

    def test_redacts_sensitive_key_in_nested_dict_with_int_keys(self):
        """Test that redaction still happens alongside non-string keys."""
        filter_instance = SanitizingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg={7: {"api_key": "sk-live-123", "count": 4}, "events": 12},
            args=(),
            exc_info=None,
        )

        filter_instance.filter(record)

        assert record.msg[7]["api_key"] == "***REDACTED***"
        assert record.msg[7]["count"] == 4
        assert record.msg["events"] == 12

    def test_sanitizes_dict_args_with_non_string_keys(self):
        """Test that a non-string-keyed dict passed as a log arg is safe.

        This is the shape the analytics cleanup sweep logs: a lazily
        formatted message with dict arguments, one of them keyed by int.
        """
        filter_instance = SanitizingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Cleanup completed: %s (paid_tiers=%s)",
            args=({"events": 12}, {30: 2, 90: 1}),
            exc_info=None,
        )

        result = filter_instance.filter(record)

        assert result is True
        assert record.args[1] == {30: 2, 90: 1}
        assert record.getMessage() == (
            "Cleanup completed: {'events': 12} (paid_tiers={30: 2, 90: 1})"
        )


class TestJSONFormatter:
    """Test JSON formatter."""

    def test_formats_record_as_valid_json(self):
        """Test that formatter produces valid JSON."""
        formatter = JSONFormatter()

        set_correlation_context(correlation_id="test-123")

        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )

        # Apply correlation ID filter before formatting
        corr_filter = CorrelationIDFilter()
        corr_filter.filter(record)

        output = formatter.format(record)

        # Verify it's valid JSON
        log_data = json.loads(output)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.module"
        assert log_data["message"] == "Test message"
        assert log_data["correlation_id"] == "test-123"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
        assert "timestamp" in log_data

        clear_correlation_context()

    def test_includes_exception_info(self):
        """Test that exception info is included in JSON."""
        import sys

        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
                func="test_func",
            )

        output = formatter.format(record)
        log_data = json.loads(output)

        assert "exception" in log_data
        assert "ValueError: Test error" in log_data["exception"]


class TestHumanReadableFormatter:
    """Test human-readable formatter."""

    def test_formats_record_with_colors(self):
        """Test that formatter includes ANSI color codes."""
        formatter = HumanReadableFormatter()

        set_correlation_context(correlation_id="test-corr-123")

        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function",
        )

        # Apply correlation ID filter before formatting
        corr_filter = CorrelationIDFilter()
        corr_filter.filter(record)

        output = formatter.format(record)

        # Should contain correlation ID (truncated)
        assert "test-cor" in output or "test-corr" in output

        # Should contain module and function
        assert "test.module" in output
        assert "test_function" in output

        # Should contain message
        assert "Test message" in output

        clear_correlation_context()

    def test_includes_exception_traceback(self):
        """Test that exception traceback is included."""
        formatter = HumanReadableFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
                func="test_func",
            )

        output = formatter.format(record)

        assert "Error occurred" in output
        assert "ValueError" in output or "Traceback" in output


class TestConfigFunctions:
    """Test configuration functions."""

    def test_get_log_level_from_env(self):
        """Test that log level is read from environment."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            level = get_log_level()
            assert level == logging.DEBUG

        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            level = get_log_level()
            assert level == logging.WARNING

    def test_get_log_level_defaults_to_info(self):
        """Test that log level defaults to INFO."""
        with patch.dict(os.environ, {}, clear=True):
            level = get_log_level()
            assert level == logging.INFO

    def test_get_logging_config_dev_environment(self):
        """Test logging config for development environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "LOG_LEVEL": "DEBUG"}):
            config = get_logging_config()

            assert config["version"] == 1
            assert "correlation_id" in config["filters"]
            assert "sanitizing" in config["filters"]
            assert config["root"]["level"] == logging.DEBUG

    def test_get_logging_config_production_uses_human_readable(self):
        """Every environment now uses the human-readable formatter.

        Historically ``get_logging_config`` selected ``JSONFormatter`` when
        ``ENVIRONMENT`` was ``production``/``staging``. That branch was
        intentionally removed in commit d96912a0 ("Remove JSONFormatter for
        simplified logging configuration"), which collapsed the selection to
        an unconditional ``HumanReadableFormatter``. The ``JSONFormatter``
        class is retained (and unit-tested directly in ``TestJSONFormatter``)
        but is no longer wired into the config. This test asserts the current,
        deliberate behavior: all environments use ``HumanReadableFormatter``.
        """
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            config = get_logging_config()

            formatter_class = config["formatters"]["default"]["()"]
            assert "HumanReadableFormatter" in formatter_class

    def test_get_logging_config_dev_uses_human_readable(self):
        """Test that development environment uses human-readable formatter."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            config = get_logging_config()

            formatter_class = config["formatters"]["default"]["()"]
            assert "HumanReadableFormatter" in formatter_class

    def test_configure_logging_is_idempotent(self):
        """Test that configure_logging can be called multiple times safely."""
        # Should not raise an exception
        configure_logging()
        configure_logging()


class TestContextManagement:
    """Test correlation context management."""

    def test_set_and_clear_correlation_context(self):
        """Test setting and clearing correlation context."""
        # Set context
        set_correlation_context(
            correlation_id="test-123",
            instance_id="inst-456",
            client_id="client-789",
        )

        # Verify it's set (by creating a filter and checking)
        filter_instance = CorrelationIDFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)

        assert record.correlation_id == "test-123"

        # Clear context
        clear_correlation_context()

        # Verify it's cleared
        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record2)

        assert record2.correlation_id == "N/A"

    def test_set_partial_context(self):
        """Test setting only some context values."""
        clear_correlation_context()

        set_correlation_context(correlation_id="test-123")

        filter_instance = CorrelationIDFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)

        assert record.correlation_id == "test-123"
        assert record.instance_id == "N/A"
        assert record.client_id == "N/A"

        clear_correlation_context()


class TestAFormatStringIsNotUserData:
    """The filter must not redact a message that still has arguments waiting.

    `logger.info("... tokens=%d ...", n)` has a message written by us and a
    value supplied at the call site. Only the second can hold user data.
    Redacting the first replaces it with the marker while leaving the args in
    place, and getMessage() then evaluates "***REDACTED***" % args and raises
    TypeError. Both formatters catch that and fall back to str(record.msg), so
    the line survives with all of its content gone: no crash, no logging error,
    nothing to notice.

    Measured 2026-08-21: 29 call sites across 16 files, tripped almost always
    by the word "token" meaning token COUNT, since SENSITIVE_KEYS matches as a
    substring. Among them every gRPC auth-failure line, which went blank
    exactly when someone was debugging an auth problem.
    """

    @pytest.mark.parametrize("msg,args,expected", [
        ("stage0_chunking_needed request_id=%s tokens=%d max=%d",
         ("r1", 900, 500),
         "stage0_chunking_needed request_id=r1 tokens=900 max=500"),
        ("Auth failed: reason=expired_credential peer=%s token_prefix=%s",
         ("1.2.3.4", "ab"),
         "Auth failed: reason=expired_credential peer=1.2.3.4 token_prefix=ab"),
        ("API key revoked via dashboard instance_id=%s credential_id=%s",
         ("i-1", "c-1"),
         "API key revoked via dashboard instance_id=i-1 credential_id=c-1"),
    ])
    def test_a_format_string_survives_its_own_sensitive_words(self, msg, args, expected):
        record = logging.LogRecord("t", logging.INFO, "p.py", 1, msg, args, None)
        SanitizingFilter().filter(record)
        assert record.getMessage() == expected

    def test_a_message_with_no_args_is_still_redacted(self):
        """This is where user data actually reaches a logger: an f-string or a
        plain string that someone built from a request."""
        record = logging.LogRecord(
            "t", logging.INFO, "p.py", 1, "Authorization: Bearer token123", (), None
        )
        SanitizingFilter().filter(record)
        assert "token123" not in record.getMessage()
        assert record.getMessage() == "Authorization: Bearer ***REDACTED***"

    def test_arguments_are_still_sanitized(self):
        """The half of the filter that was always right: a secret passed as an
        argument is exactly what this exists to catch."""
        record = logging.LogRecord(
            "t", logging.INFO, "p.py", 1, "credentials=%s",
            ({"password": "hunter2", "user": "alice"},), None
        )
        SanitizingFilter().filter(record)
        rendered = record.getMessage()
        assert "hunter2" not in rendered
        assert "alice" in rendered


class TestTheFilterDoesNotBlankOrdinaryLines:
    """Many log lines were replaced in full with
    "***REDACTED***", and almost none of them held a secret. Two causes, both
    pinned here, plus the credential leak found while fixing them."""

    def _msg(self, text, args=None):
        record = logging.LogRecord("t", logging.INFO, "p.py", 1, text, args, None)
        SanitizingFilter().filter(record)
        return record.msg if args is None else record.getMessage()

    def test_cert_no_longer_matches_inside_an_ordinary_word(self):
        """'cert' was in SENSITIVE_KEYS and matched as a substring, so any line
        with "certain" or "concert" and a colon went blank. That is how the
        retrieval HQE_MATCH lines vanished mid-investigation."""
        line = 'HQE_MATCH score=0.51 question="Which concert?" content="certain behaviors"'
        assert self._msg(line) == line

    def test_a_token_count_is_not_a_token(self):
        """Our own LLM cost telemetry. Blanking this made spend un-auditable
        from logs."""
        line = "cloud_usage_track_llm_usage input_tokens=1241 output_tokens=465"
        assert self._msg(line) == line
        assert self._msg("tokens=500") == "tokens=500"

    def test_a_real_credential_is_still_removed(self):
        for line, secret in (
            ("auth failed token=eyJhbGciOiJIUzI1NiJ9.abc for cli_x", "eyJhbGciOiJIUzI1NiJ9.abc"),
            ("api_key=synap_jn5Trec0M306ZFX1 instance=inst_1", "synap_jn5Trec0M306ZFX1"),
            ('{"access_token": "abc123xyz"}', "abc123xyz"),
            ("private_key=-----BEGIN", "-----BEGIN"),
        ):
            out = self._msg(line)
            assert secret not in out, out
            assert "***REDACTED***" in out

    def test_an_auth_scheme_does_not_shield_the_credential(self):
        """Regression found while fixing the above: redacting the value after
        "Authorization:" removed the word "Bearer" and left the key standing."""
        out = self._msg("Authorization: Bearer sk-svcacct-abcdef123456")
        assert "sk-svcacct-abcdef123456" not in out
        assert out == "Authorization: Bearer ***REDACTED***"

    def test_a_numeric_password_is_still_redacted(self):
        """The count exemption is scoped to token-shaped keys only. A four-digit
        password is a real thing and must not ride out on being numeric."""
        assert "1234" not in self._msg("password=1234")
        assert "9999" not in self._msg("pin_secret=9999")

    def test_a_dict_argument_keeps_its_counts_and_loses_its_secrets(self):
        out = SanitizingFilter()._sanitize_value(
            {"input_tokens": 1241, "token": "eyJabc", "password": 1234}
        )
        assert out["input_tokens"] == 1241
        assert out["token"] == "***REDACTED***"
        assert out["password"] == "***REDACTED***"

    def test_no_line_is_ever_replaced_in_full(self):
        """The property that broke: whatever else happens, the line keeps its
        own words. Every message here contains a sensitive word."""
        for line in (
            'HQE_MATCH question="certain behaviors" content="a Queen concert"',
            "auth failed token=eyJabc for client cli_x",
            "Authorization: Bearer sk-abc",
            "certificate_count=4 renewals=2",
        ):
            assert self._msg(line) != "***REDACTED***", line
