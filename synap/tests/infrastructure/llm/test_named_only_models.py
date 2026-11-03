"""gpt-6-luna: known to the registry so a call can name it, invisible to the selector.

The first SYNAP_ER_MODEL run stored empty graphs: the manager raised
"Model gpt-6-luna not found for openai" and Stage 3 caught it. The tests of that
setting used a fake manager, so nothing checked the real registry file.
"""

from synap.cloud.infrastructure.llm.models import LLMProvider
from synap.cloud.infrastructure.llm.registry import LLMRegistry


def test_the_real_registry_knows_gpt_6_luna_for_a_named_call():
    info = LLMRegistry().get_model_info(LLMProvider.OPENAI, "gpt-6-luna")
    assert info is not None and info.max_output_tokens >= 16000


def test_the_selector_is_never_offered_it():
    registry = LLMRegistry()
    assert "gpt-6-luna" not in [m.name for m in registry.get_models_by_provider(LLMProvider.OPENAI)]
    assert "gpt-6-luna" not in [m.name for m in registry.get_all_models()]


def test_the_models_the_selector_chooses_from_are_unchanged():
    names = {m.name for m in LLMRegistry().get_models_by_provider(LLMProvider.OPENAI)}
    assert {"gpt-4o", "gpt-4o-mini"} <= names


def test_a_model_marked_not_selectable_in_a_yaml_is_named_only(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text("""
defaults: {provider: openai, model: a}
providers:
  openai:
    enabled: true
    models:
      a: {type: flagship}
      b: {type: flagship, selectable: false}
""")
    registry = LLMRegistry(str(path))
    assert [m.name for m in registry.get_models_by_provider(LLMProvider.OPENAI)] == ["a"]
    assert registry.get_model_info(LLMProvider.OPENAI, "b") is not None
    registry.reload()
    assert [m.name for m in registry.get_all_models()] == ["a"]
