"""A vision call's list-of-parts content through the manager.

The vision reader is the first caller that sends `content` as a list (text plus
a base64 data URL). Every layer on the way has to let it through untouched, and
the trace, whose bodies are stored, must not keep the image.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synap.cloud.infrastructure.llm import (
    LLMConfigManager, LLMProvider, LLMResponse, call_trace,
)
from synap.cloud.infrastructure.llm.providers.openai_provider import OpenAIProvider

B64 = "iVBORw0KGgo" + "A" * 4000
URL = f"data:image/png;base64,{B64}"
VISION = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": URL, "detail": "high"}},
        {"type": "text", "text": "Describe it."},
    ],
}]


@pytest.fixture
def records():
    seen = []
    call_trace.set_trace_sink(seen.append)
    yield seen
    call_trace.set_trace_sink(None)


@pytest.fixture
async def manager(redis_client, test_config_path):
    manager = LLMConfigManager(redis_client, test_config_path)
    for provider in manager.get_registry().get_enabled_providers():
        manager._providers[provider] = MagicMock(name=f"stand-in {provider.value}")
    return manager


def _ok():
    return LLMResponse(
        content="A chart.", model="gpt-4o", provider=LLMProvider.OPENAI,
        input_tokens=800, output_tokens=50, total_tokens=850,
    )


class TestRedaction:
    def test_a_data_url_becomes_its_type_and_size(self):
        out = call_trace.redact_inline_media(VISION)
        part = out[0]["content"][0]
        assert part["image_url"]["url"] == f"[image image/png, {len(B64)} base64 chars]"
        assert part["image_url"]["detail"] == "high"
        assert out[0]["content"][1] == {"type": "text", "text": "Describe it."}

    def test_the_callers_messages_are_not_mutated(self):
        call_trace.redact_inline_media(VISION)
        assert VISION[0]["content"][0]["image_url"]["url"] == URL

    def test_text_only_messages_are_returned_as_they_are(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert call_trace.redact_inline_media(msgs) is msgs

    def test_a_remote_url_is_kept(self):
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]}]
        assert call_trace.redact_inline_media(msgs) == msgs


class TestThroughTheManager:
    @pytest.mark.asyncio
    async def test_the_provider_gets_the_image_and_the_trace_does_not(self, manager, records):
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_ok())
        with patch.object(manager, "get_provider", return_value=provider):
            response = await manager.complete(
                messages=VISION, force_provider=LLMProvider.OPENAI, force_model="gpt-4o",
                allow_cross_provider_fallback=False, operation="ingestion.vision",
            )
        assert response.content == "A chart."
        assert provider.complete.await_args.kwargs["messages"] == VISION
        assert len(records) == 1
        assert B64 not in repr(records[0].messages)
        assert records[0].messages[0]["content"][1]["text"] == "Describe it."

    @pytest.mark.asyncio
    async def test_the_privacy_guard_lets_list_content_through(self, manager):
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_ok())
        with patch.object(manager, "get_provider", return_value=provider), \
             patch("synap.cloud.privacy.provider_boundary.tripwire_enabled", return_value=True):
            await manager.complete(messages=VISION, force_provider=LLMProvider.OPENAI,
                                   force_model="gpt-4o")
        assert provider.complete.await_args.kwargs["messages"] == VISION

    @pytest.mark.asyncio
    async def test_usage_is_tracked_against_the_caller(self, manager):
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_ok())
        with patch.object(manager, "get_provider", return_value=provider), \
             patch.object(manager, "_fire_track_llm_usage") as fire:
            await manager.complete(
                messages=VISION, force_provider=LLMProvider.OPENAI, force_model="gpt-4o",
                client_id="cli", instance_id="inst", operation="ingestion.vision",
            )
        kw = fire.call_args.kwargs
        assert (kw["client_id"], kw["instance_id"], kw["operation"]) == (
            "cli", "inst", "ingestion.vision")
        assert (kw["input_tokens"], kw["output_tokens"]) == (800, 50)


def test_the_openai_adapter_accepts_list_content(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    OpenAIProvider()._validate_messages(VISION)


class TestAnthropicImageBlocks:
    """The image reader sends OpenAI-style parts; Anthropic takes its own block."""

    def _convert(self, messages):
        from synap.cloud.infrastructure.llm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider._convert_messages(object.__new__(AnthropicProvider), messages)

    def test_a_data_url_becomes_a_base64_image_block(self):
        _, out = self._convert(VISION)
        assert out[0]["content"][0] == {
            "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B64}}
        assert out[0]["content"][1] == {"type": "text", "text": "Describe it."}

    def test_a_web_address_becomes_a_url_image_block(self):
        _, out = self._convert([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}])
        assert out[0]["content"] == [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]

    def test_plain_text_and_the_system_message_are_unchanged(self):
        system, out = self._convert([{"role": "system", "content": "be brief"},
                                     {"role": "user", "content": "hi"}])
        assert system == "be brief" and out == [{"role": "user", "content": "hi"}]

    def test_the_callers_messages_are_not_changed(self):
        self._convert(VISION)
        assert VISION[0]["content"][0]["type"] == "image_url"
