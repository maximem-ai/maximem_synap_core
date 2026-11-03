"""The startup warning for a missing vision model, and the marker."""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from synap.cloud.application.ingestion import multimodal as mm
from synap.cloud.infrastructure.llm import settings as llm_settings

VARS = ["SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "SYNAP_LLM_MODEL_VISION", "SYNAP_LLM_BASE_URL",
        "SYNAP_VISION_MODEL"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    llm_settings.reset_for_tests()
    yield
    llm_settings.reset_for_tests()


def _set(monkeypatch, **kv):
    for k, v in kv.items():
        monkeypatch.setenv(k, v)
    llm_settings.reset_for_tests()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "vision model" in r.getMessage()]


def test_cloud_with_the_legacy_settings_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        assert mm.warn_if_no_vision_model() is False
    assert _warnings(caplog) == []


@pytest.mark.parametrize("env", [
    {"SYNAP_LLM_PROVIDER": "openai", "SYNAP_LLM_MODEL": "gpt-4o-mini"},
    {"SYNAP_LLM_PROVIDER": "anthropic", "SYNAP_LLM_MODEL": "claude-x"},
    {"SYNAP_LLM_PROVIDER": "ollama", "SYNAP_LLM_MODEL": "llama3.2-vision"},
    {"SYNAP_LLM_PROVIDER": "ollama", "SYNAP_LLM_MODEL": "llama3.1", "SYNAP_LLM_MODEL_VISION": "llava:13b"},
])
def test_a_configured_vision_model_logs_nothing(monkeypatch, caplog, env):
    _set(monkeypatch, **env)
    with caplog.at_level(logging.WARNING):
        assert mm.warn_if_no_vision_model() is False
    assert _warnings(caplog) == []


@pytest.mark.parametrize("env", [
    {"SYNAP_LLM_PROVIDER": "ollama", "SYNAP_LLM_MODEL": "llama3.1"},
    {"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "mistral-7b",
     "SYNAP_LLM_BASE_URL": "http://localhost:8000/v1"},
])
def test_core_with_a_text_only_model_warns_once(monkeypatch, caplog, env):
    _set(monkeypatch, **env)
    with caplog.at_level(logging.WARNING):
        assert mm.warn_if_no_vision_model() is True
    found = _warnings(caplog)
    assert len(found) == 1 and "could not be read" in found[0].getMessage()


def test_start_ingestion_workers_makes_the_one_call():
    import inspect
    from synap.cloud.application.ingestion import startup
    src = inspect.getsource(startup.start_ingestion_workers)
    assert src.count("warn_if_no_vision_model()") == 1


@pytest.mark.asyncio
async def test_a_model_that_cannot_read_images_leaves_the_marker_and_the_turn_is_recorded(monkeypatch):
    """No vision model: the provider refuses the image. The turn text is still kept."""
    _set(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1")
    manager = SimpleNamespace(complete=AsyncMock(side_effect=RuntimeError("model does not support images")))
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    mm._description_cache.clear()
    with patch("synap.cloud.infrastructure.llm.get_llm_manager", return_value=manager):
        rendered, reads = await mm.read_and_render(
            ["look at this", mm.ImageInput(data=buf.getvalue(), media_type="image/png", filename="shot.png")])
    assert rendered == "look at this\n[image: shot.png (could not be read)]"
    assert [r.attachment()["status"] for r in reads] == ["failed"]
    assert manager.complete.await_args.kwargs["step"] == "VISION"
