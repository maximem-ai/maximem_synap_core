"""The engine's "remote" embedding provider asks a hook for its client, so Core ships no code that can reach a remote service."""
import inspect
import logging

import pytest

from synap.cloud.application.ingestion import startup
from synap.cloud.shared import edition


@pytest.fixture
def no_hooks():
    edition.reset_hooks()
    yield
    edition.load_cloud_hooks_for_test()


def test_with_no_hook_the_remote_provider_uses_the_local_model(no_hooks, monkeypatch, caplog):
    local = object()
    monkeypatch.setattr(startup, "_get_sentence_transformer_client", lambda: local)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "remote")
    with caplog.at_level(logging.WARNING):
        assert startup._get_embedding_client() is local
    assert "no embedding service to use" in caplog.text


def test_the_engine_file_imports_no_remote_client():
    # The scrub rule (scripts/oss/scrub_patterns.list) keeps the service's settings out of every shipped file.
    assert "remote_embedding_client" not in inspect.getsource(startup)
