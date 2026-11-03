"""An image upload without a vision model is refused with what to set; the vision call names the configured provider."""
import io
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from synap.cloud.application.ingestion import multimodal as mm
from synap.cloud.application.sdk_api import document_extractor as de
from synap.cloud.infrastructure.llm import settings as llm_settings
from synap.cloud.infrastructure.llm.models import LLMProvider

VARS = ["SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "SYNAP_LLM_MODEL_VISION", "SYNAP_LLM_BASE_URL",
        "SYNAP_VISION_MODEL"]


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    llm_settings.reset_for_tests()
    mm._description_cache.clear()
    yield
    llm_settings.reset_for_tests()
    mm._description_cache.clear()


def _set(monkeypatch, **kv):
    for k, v in kv.items():
        monkeypatch.setenv(k, v)
    llm_settings.reset_for_tests()


def _llm():
    manager = NS(complete=AsyncMock(return_value=NS(content="A red square.")))
    return patch("synap.cloud.infrastructure.llm.get_llm_manager", return_value=manager), manager


@pytest.mark.asyncio
async def test_extractor_refuses_an_image_when_the_model_has_no_vision(monkeypatch):
    _set(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1")
    patcher, manager = _llm()
    with patcher:
        with pytest.raises(HTTPException) as e:
            await de.extract_text("shot.png", PNG)
    assert 400 <= e.value.status_code < 500
    assert "SYNAP_LLM_MODEL_VISION" in e.value.detail
    manager.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_route_returns_that_error_for_an_image(monkeypatch):
    from synap.cloud.application.sdk_api import sdk_memories

    _set(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1")
    patcher, manager = _llm()
    with patcher:
        with pytest.raises(HTTPException) as e:
            await sdk_memories.sdk_upload_memory(
                user_id="u", customer_id="u", relationship_type="b2b", scope=None,
                file=UploadFile(file=io.BytesIO(PNG), filename="shot.png"), text=None,
                document_type=None, mode="long-range", metadata=None,
                auth=NS(client_id="c", instance_id="i", correlation_id="x", credential_id="k"),
                ingestion_manager=NS())
    assert e.value.status_code == 422 and "SYNAP_LLM_MODEL_VISION" in e.value.detail
    manager.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_named_vision_model_lets_the_image_through(monkeypatch):
    _set(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1",
         SYNAP_LLM_MODEL_VISION="llava:13b")
    patcher, manager = _llm()
    with patcher:
        assert await de.extract_text("shot.png", PNG) == "A red square."
    manager.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_text_files_are_not_gated(monkeypatch):
    _set(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1")
    assert await de.extract_text("a.txt", b"hello") == "hello"


@pytest.mark.asyncio
async def test_the_vision_call_names_the_configured_provider_and_model(monkeypatch):
    _set(monkeypatch, SYNAP_LLM_PROVIDER="anthropic", SYNAP_LLM_MODEL="claude-x")
    patcher, manager = _llm()
    with patcher:
        await mm.describe_image(PNG, "image/png")
    kw = manager.complete.await_args.kwargs
    assert kw["force_provider"] == LLMProvider.ANTHROPIC
    assert kw["force_model"] == "claude-x"


@pytest.mark.asyncio
async def test_the_vision_call_uses_the_vision_step_model(monkeypatch):
    _set(monkeypatch, SYNAP_LLM_PROVIDER="ollama", SYNAP_LLM_MODEL="llama3.1",
         SYNAP_LLM_MODEL_VISION="llava:13b")
    patcher, manager = _llm()
    with patcher:
        await mm.describe_image(PNG, "image/png")
    kw = manager.complete.await_args.kwargs
    assert (kw["force_provider"], kw["force_model"]) == (LLMProvider.OLLAMA, "llava:13b")


@pytest.mark.asyncio
async def test_without_the_single_provider_setting_the_call_is_openai(monkeypatch):
    patcher, manager = _llm()
    with patcher:
        await mm.describe_image(PNG, "image/png")
    kw = manager.complete.await_args.kwargs
    assert (kw["force_provider"], kw["force_model"]) == (LLMProvider.OPENAI, "gpt-4o")
