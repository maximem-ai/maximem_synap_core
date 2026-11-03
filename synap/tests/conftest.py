"""Suite-wide fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def _reset_schema_compat_cache():
    """The "does this column exist" probe results are cached per process. Tests
    build databases with and without migrations 166-171 in the same process, so
    forget every probe (and every once-per-process warning) around each test."""
    try:
        from synap.cloud.database import schema_compat
    except Exception:  # pragma: no cover - import problems surface in the tests that need it
        yield
        return
    schema_compat.reset_for_tests()
    yield
    schema_compat.reset_for_tests()


@pytest.fixture
def core_llm_env(monkeypatch):
    """Give a Core run the one LLM it needs, for tests written against Cloud's compaction profiles.

    Core compacts on the configured LLM (SYNAP_LLM_PROVIDER and SYNAP_LLM_MODEL) and refuses to
    start a compaction without one. Cloud picks its profile from its keys, so there this fixture
    does nothing and Cloud's environment is left alone. The values are set with monkeypatch, so
    they are gone when the test ends. The key is a fake: the tests mock every model call.
    """
    from synap.cloud.shared import edition

    # Asked of the environment, not of the module attribute: other tests swap the module's
    # edition values while they run, and a leftover swap must not decide this.
    if edition.resolve(os.environ).compaction_profile != "configured":
        yield
        return
    monkeypatch.setenv("SYNAP_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SYNAP_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    # The settings are read once per process; forget what an earlier test cached, and what this one did.
    from synap.cloud.infrastructure.llm import settings as llm_settings

    llm_settings.reset_for_tests()
    yield
    llm_settings.reset_for_tests()
