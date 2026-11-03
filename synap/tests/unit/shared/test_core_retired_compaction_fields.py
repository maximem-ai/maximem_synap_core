"""Core rejects the five compaction fields that nothing reads, by name."""
import pytest
import yaml

from synap.cloud.memory_config.core_convert import to_core_config
from synap.cloud.memory_config.file_loader import MemoryConfigError, parse_and_validate

FIELDS = ["trigger_at_tokens", "trigger_at_messages", "incremental_message_delta", "target_tokens",
          "preserve_last_n_turns"]


@pytest.mark.parametrize("field", FIELDS)
def test_a_retired_compaction_field_stops_the_load_and_is_named(field):
    text = f"compaction_config:\n  adaptive_thresholds:\n    {field}: 100\n"
    with pytest.raises(MemoryConfigError, match=field):
        parse_and_validate(text)


def test_the_fields_that_are_read_still_load():
    text = "compaction_config:\n  adaptive_thresholds:\n    idle_minutes: 7\n    target_ratio: 0.3\n"
    parse_and_validate(text)


def test_the_cloud_to_core_converter_drops_them_instead_of_failing():
    raw = {"compaction_config": {"adaptive_thresholds": {f: 1 for f in FIELDS}}}
    converted, removed = to_core_config(raw)
    assert "compaction_config.adaptive_thresholds.trigger_at_tokens" in removed
    assert not converted.get("compaction_config", {}).get("adaptive_thresholds")
