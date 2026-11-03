"""The file store is chosen by the edition module, and a build without the S3 adapter never needs boto3."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from synap.providers.file import LocalDiskAdapter, registry, reset_file_store

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SYNAP_FILE_STORE", raising=False)
    monkeypatch.delenv("SYNAP_ORIGINALS_DIR", raising=False)
    reset_file_store()
    yield
    reset_file_store()


def test_the_edition_choice_decides_when_nothing_is_set(monkeypatch, tmp_path):
    monkeypatch.setattr(registry.edition, "file_store", "local")
    monkeypatch.setenv("SYNAP_ORIGINALS_DIR", str(tmp_path / "originals"))
    assert registry._load_config()["adapter"] == "local"
    assert isinstance(registry.get_file_store(), LocalDiskAdapter)


def test_a_core_style_edition_without_a_folder_stops_instead_of_falling_back_to_s3(monkeypatch):
    monkeypatch.setattr(registry.edition, "file_store", "local")
    with pytest.raises(ValueError, match="SYNAP_ORIGINALS_DIR"):
        registry.get_file_store()
    with pytest.raises(ValueError, match="SYNAP_ORIGINALS_DIR"):
        registry.check_file_store_at_startup()
    assert registry._instance is None


def test_the_setting_still_wins_over_the_edition(monkeypatch, tmp_path):
    monkeypatch.setattr(registry.edition, "file_store", "s3")
    monkeypatch.setenv("SYNAP_FILE_STORE", "local")
    monkeypatch.setenv("SYNAP_ORIGINALS_DIR", str(tmp_path / "originals"))
    assert isinstance(registry.get_file_store(), LocalDiskAdapter)


def test_a_store_whose_module_is_not_in_the_build_is_not_offered(monkeypatch):
    real = importlib.util.find_spec

    def find_spec(name, *a, **kw):
        return None if name.endswith("adapters.s3") else real(name, *a, **kw)

    monkeypatch.setattr(registry.importlib.util, "find_spec", find_spec)
    monkeypatch.setenv("SYNAP_FILE_STORE", "s3")
    with pytest.raises(ValueError, match="not available; choose one of: local$"):
        registry._load_config()


def test_the_file_store_package_loads_without_boto3():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "sys.modules['boto3'] = None\n"
        "sys.modules['botocore'] = None\n"
        "import synap.providers.file\n"
        "from synap.providers.file import registry\n"
        "assert 'synap.providers.file.adapters.s3' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
