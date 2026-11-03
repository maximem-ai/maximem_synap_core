"""The in-process embedding loader loads one exact model commit, or refuses.

The model is faked: these tests check what the loader asks for, with no download.
"""
import re
import sys
import types

import pytest

from synap.cloud.application.ingestion import embedding_client as ec
from synap.cloud.application.shared import providers

ARCTIC = "Snowflake/snowflake-arctic-embed-l-v2.0"


@pytest.fixture
def seen(monkeypatch):
    """Fake sentence_transformers; returns the keyword arguments the loader passed."""
    calls = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append({"name": name, **kwargs})

        def get_sentence_embedding_dimension(self):
            return 1024

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeModel))
    monkeypatch.setattr(ec.SentenceTransformerEmbeddingClient, "_model", None)
    for name in ("EMBEDDING_MODEL_REVISION", "EMBEDDING_TRUST_REMOTE_CODE", "EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return calls


def test_the_pin_is_a_full_commit_hash():
    assert re.fullmatch(r"[0-9a-f]{40}", ec.PINNED_EMBEDDING_REVISIONS[ARCTIC])


def test_the_default_model_loads_at_its_pinned_commit_without_remote_code(seen):
    ec.get_sentence_transformer_client()
    assert seen == [{
        "name": ARCTIC,
        "cache_folder": None,
        "revision": ec.PINNED_EMBEDDING_REVISIONS[ARCTIC],
        "trust_remote_code": False,
    }]


def test_an_operator_can_pick_another_commit_with_a_full_hash(seen, monkeypatch):
    other = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", other)
    ec.get_sentence_transformer_client()
    assert seen[0]["revision"] == other


@pytest.mark.parametrize("bad", ["main", "v1", "ac6544c", "g" * 40, "a" * 41])
def test_a_branch_tag_or_short_hash_is_refused(seen, monkeypatch, bad):
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", bad)
    with pytest.raises(ec.UnpinnedEmbeddingModelError):
        ec.get_sentence_transformer_client()
    assert seen == []


def test_a_model_with_no_pin_is_refused_and_never_loaded(seen):
    with pytest.raises(ec.UnpinnedEmbeddingModelError, match="EMBEDDING_MODEL_REVISION"):
        ec.get_sentence_transformer_client(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert seen == []


def test_a_model_with_no_pin_loads_when_the_operator_gives_a_commit(seen, monkeypatch):
    commit = "f" * 40
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", commit)
    ec.get_sentence_transformer_client(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert seen[0]["revision"] == commit


def test_remote_code_is_off_unless_asked_for(seen, monkeypatch):
    monkeypatch.setenv("EMBEDDING_TRUST_REMOTE_CODE", "true")
    ec.get_sentence_transformer_client()
    assert seen[0]["trust_remote_code"] is True


def test_the_refusal_is_not_swallowed_into_a_missing_provider(seen, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(ec.UnpinnedEmbeddingModelError):
        providers.get_embedding_provider()
