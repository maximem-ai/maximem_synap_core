"""An unknown key in a memory config: Core refuses it, Cloud loads the config and warns once.

``core_rules`` has the Core side (tested with the other Core rules); these tests pin the Cloud side
and that a field Core removed on purpose is not reported as unknown.
"""
import logging
from datetime import datetime, timezone

from synap.cloud.memory_config.config_schema import ConfigMetadata, MemoryArchitectureConfig
from synap.cloud.memory_config.core_rules import find_problems, find_unknown_keys

META = ConfigMetadata(client_id="c", instance_id="i", config_version="1.0.0",
                      generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc), generated_by="test")


def _build(**extra):
    return MemoryArchitectureConfig(metadata=META, **extra)


def _warnings(caplog):
    return [r for r in caplog.records if "keys the schema does not have" in r.getMessage()]


def test_a_config_with_unknown_keys_loads_and_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger="synap.cloud.memory_config.config_schema"):
        cfg = _build(memory_categories={"facts": {"enabled": True, "bogus_setting": 1}}, odd_top_level=3)
    assert cfg.memory_categories.facts.enabled is True
    records = _warnings(caplog)
    assert len(records) == 1
    text = records[0].getMessage()
    assert "bogus_setting" in text and "odd_top_level" in text


def test_a_clean_config_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="synap.cloud.memory_config.config_schema"):
        _build(memory_categories={"facts": {"enabled": True}})
    assert _warnings(caplog) == []


def test_a_field_core_removed_is_not_called_unknown():
    raw = {"storage_architecture": {"primary_mode": "x"}, "evolution": {}}
    assert find_unknown_keys(raw) == []
    assert len(find_problems(raw)) == 2  # Core still rejects both, as removed


def test_core_still_rejects_what_cloud_only_warns_about():
    raw = {"memory_categories": {"facts": {"enabled": True, "bogus_setting": 1}}}
    assert [p[0] for p in find_unknown_keys(raw)] == [("memory_categories", "facts", "bogus_setting")]
    assert find_problems(raw)


def test_a_config_that_is_not_a_mapping_is_left_to_the_schema():
    assert find_unknown_keys(None) == [] and find_unknown_keys([1]) == []
