"""The sample configs that ship in the Core repo's config/ folder load in Core and say what the docs say."""
from pathlib import Path

import pytest

from synap.cloud.memory_config.file_loader import parse_and_validate

CONFIG = Path(__file__).resolve().parents[4] / "docs" / "oss" / "core_repo" / "config"

pytestmark = pytest.mark.skipif(not CONFIG.is_dir(), reason="the Core repo's config folder is not in this tree")

SAMPLES = {
    "memory.yml": "equals_customer",
    "samples/business-app.yml": "strict",
}


@pytest.mark.parametrize("name,mode", SAMPLES.items())
def test_each_sample_loads_in_core_and_sets_the_documented_scoping_mode(name, mode):
    raw, _config = parse_and_validate((CONFIG / name).read_text(), source=name)
    assert raw["storage_architecture"]["scoping"]["user_context_isolation"] == mode


def test_the_quickstart_sample_is_the_file_the_server_reads():
    assert (CONFIG / "memory.yml").is_file()
