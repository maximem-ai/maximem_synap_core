"""Card A11: turning a Cloud-built memory config into one Core's loader accepts."""
import copy

import yaml

from synap.cloud.memory_config import core_rules
from synap.cloud.memory_config.core_convert import to_core_config
from synap.cloud.memory_config.file_loader import MemoryConfigError, parse_and_validate

import pytest

CLOUD_CONFIG = {
    "schema_version": "1.0.0",
    "memory_categories": {"facts": {"enabled": True, "append_only": True, "retention_days": 30}},
    "retrieval_config": {"default_mode": "fast", "anticipation": {"enabled": True}},
    "consolidation": {"enabled": True},
    "storage_architecture": {"scoping": {"allow_cross_user_retrieval": True}, "made_up_key": 1},
}


def _loads(raw):
    return parse_and_validate(yaml.safe_dump(raw), "memory.generated.yml")


def test_a_cloud_config_is_refused_as_it_comes():
    with pytest.raises(MemoryConfigError):
        _loads(CLOUD_CONFIG)


def test_the_converted_config_loads_and_the_fields_are_listed():
    cfg, removed = to_core_config(CLOUD_CONFIG)
    _loads(cfg)
    assert "consolidation" in removed
    assert "retrieval_config.anticipation" in removed
    assert "memory_categories.facts.append_only" in removed
    assert "storage_architecture.scoping.allow_cross_user_retrieval" in removed
    assert "storage_architecture.made_up_key" in removed
    assert cfg["memory_categories"]["facts"] == {"enabled": True}
    assert cfg["retrieval_config"]["default_mode"] == "fast"


def test_the_input_is_not_changed():
    before = copy.deepcopy(CLOUD_CONFIG)
    to_core_config(CLOUD_CONFIG)
    assert CLOUD_CONFIG == before


def test_a_value_is_never_rewritten():
    raw = {"memory_categories": {"facts": {"enabled": True}},
           "storage_architecture": {"scoping": {"user_context_isolation": "maca_suggests"}}}
    cfg, removed = to_core_config(raw)
    assert cfg["storage_architecture"]["scoping"]["user_context_isolation"] == "maca_suggests"
    assert removed == []
    with pytest.raises(MemoryConfigError):
        _loads(cfg)


def test_a_config_with_nothing_to_remove_comes_back_equal():
    raw = {"memory_categories": {"facts": {"enabled": True}}, "retrieval_config": {"default_mode": "fast"}}
    cfg, removed = to_core_config(raw)
    assert cfg == raw and removed == []


def test_not_a_mapping_is_returned_for_the_loader_to_refuse():
    assert to_core_config(["a"]) == (["a"], [])
    assert to_core_config(None) == (None, [])


def test_what_is_left_has_no_problem_the_converter_can_fix():
    cfg, _ = to_core_config(CLOUD_CONFIG)
    assert [m for _, m in core_rules.find_problems(cfg) if "removed from Core" in m or "unknown key" in m] == []
