"""Model-adaptive triggers and limits (COMPACTION_V3_PLAN.md §4.2, §4.7).

Two of these tests exist to be noticed when the compaction model changes:

* every compaction profile's model must be in the model -> window table, so
  a new model cannot ship with an unknown window;
* G and K are pinned for the current model, so a model change shows up in
  review as a deliberate change to these numbers.

If you changed the compaction model and one of them failed: add the model to
compaction/data/model_windows.json, check the new K and G against plan §4.7,
and update the pinned values.
"""
import os

import pytest

from synap.cloud.application.compaction.budgets import (
    DEFAULT_AGENT_WINDOW,
    agent_budget,
    derive_model_limits,
    resolve_agent_window,
    window_for_model,
)
from synap.cloud.application.compaction.passes.base import COMPACTION_PROFILES, GROQ_PROFILE


@pytest.mark.parametrize("profile", sorted(COMPACTION_PROFILES))
def test_every_compaction_model_has_a_known_window(profile):
    model = COMPACTION_PROFILES[profile].model
    assert window_for_model(model), f"{model} (profile {profile}) is not in model_windows.json"
    derive_model_limits(model)  # raises KeyError for an unknown model


def test_g_and_k_are_pinned_for_the_production_model():
    limits = derive_model_limits(GROQ_PROFILE.model)
    assert GROQ_PROFILE.model == "openai/gpt-oss-120b"
    assert limits.window == 131_072
    assert limits.chunk_tokens == 15_000  # K
    assert limits.record_ceiling == 14_000  # G
    assert limits.describe() == (
        "compaction model openai/gpt-oss-120b · window 131K · budget 32.8K → K=15K, G=14K"
    )


def test_every_call_fits_a_quarter_of_our_window():
    for profile in COMPACTION_PROFILES.values():
        limits = derive_model_limits(profile.model)
        assert 3_000 + 400 + limits.record_ceiling + limits.chunk_tokens <= limits.window * 0.25


@pytest.mark.parametrize("window,trigger,ceiling", [
    (128_000, 15_000, 7_500),
    (200_000, 23_437, 11_718),
    (400_000, 46_875, 14_000),
    (1_000_000, 117_187, 14_000),
])
def test_the_plan_table(window, trigger, ceiling):
    budget = agent_budget(window, derive_model_limits(GROQ_PROFILE.model).record_ceiling)
    assert (budget.trigger_tokens, budget.record_ceiling, budget.message_trigger) == (trigger, ceiling, 30)


@pytest.mark.parametrize("name,window", [
    ("gpt-4o", 128_000),
    ("gpt-4o-2024-08-06", 128_000),
    ("openai/gpt-oss-120b", 131_072),
    ("claude-sonnet-4-5-20250929", 200_000),
    ("models/gemini-2.5-pro", 1_048_576),
    ("gpt-4.1-mini-2025-04-14", 1_047_576),
    ("meta-llama/llama-4-scout-17b-16e-instruct", 131_072),
    ("GPT-5", 400_000),
])
def test_lookup(name, window):
    assert window_for_model(name) == window


def test_detection_precedence():
    assert window_for_model("a-model-nobody-has-heard-of") is None
    assert resolve_agent_window() == DEFAULT_AGENT_WINDOW
    assert resolve_agent_window(model="unknown", instance_window=300_000) == 300_000
    assert resolve_agent_window(model="gpt-5", instance_window=300_000) == 400_000
    assert resolve_agent_window(context_window=64_000, model="gpt-5") == 64_000


def test_the_token_counter_needs_no_network(monkeypatch, tmp_path):
    """The encoding is bundled: an air-gapped install still counts tokens."""
    import synap.cloud.shared.tokens as tokens

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "unwritable" / "x"))
    os.chmod(tmp_path, 0o500)
    try:
        monkeypatch.setattr(tokens, "_bundled_encoding", None)
        counter = tokens.TokenCounter()
        assert counter.encoding is not None and counter.encoding.name == "cl100k_base"
        assert counter.count("Hello world, this is Synap.") == 8
    finally:
        os.chmod(tmp_path, 0o700)


# --- SYNAP_LLM_CONTEXT_WINDOW (found at the first Docker boot: a model the table does not know) -----------

from synap.cloud.application.compaction.budgets import WINDOW_ENV, configured_window  # noqa: E402


def test_the_developer_can_write_down_a_window_for_a_model_the_table_does_not_know():
    assert configured_window({}) is None
    assert configured_window({WINDOW_ENV: ""}) is None
    assert configured_window({WINDOW_ENV: "32768"}) == 32768
    assert configured_window({WINDOW_ENV: " 1_000_000 "}) == 1_000_000
    limits = derive_model_limits("some-new-model", window=configured_window({WINDOW_ENV: "32768"}))
    assert limits.window == 32768 and limits.record_ceiling < derive_model_limits("some-new-model", window=128_000).record_ceiling


@pytest.mark.parametrize("bad", ["lots", "12.5", "-1", "0", "3999"])
def test_a_bad_window_setting_is_refused_with_its_name(bad):
    with pytest.raises(ValueError, match=WINDOW_ENV):
        configured_window({WINDOW_ENV: bad})
