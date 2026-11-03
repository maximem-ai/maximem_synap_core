"""Anthropic LLM adapter."""

from typing import Dict, Any, List

from anthropic import AsyncAnthropic, APIError, RateLimitError, APIConnectionError

from .base import BaseLLMAdapter


class LLMUnavailableError(Exception):
    """Raised when an LLM provider API is unavailable."""
    pass


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude API adapter.

    Supports Claude 3.5 Sonnet, Claude 3 Haiku, and other Claude models.

    Documentation:
    - https://docs.anthropic.com/claude/reference/messages_post
    """

    def __init__(self, api_key: str):
        """Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key
        """
        self._api_key = api_key
        self._client = AsyncAnthropic(api_key=api_key)

    def _extract_system(self, messages: List[Dict]) -> tuple:
        """Separate system messages from conversation messages.

        Anthropic requires system prompt as a separate parameter, not in
        the messages list.

        Returns:
            (system_text, non_system_messages)
        """
        system_parts = []
        other = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                other.append(msg)
        system_text = "\n".join(system_parts) if system_parts else None
        return system_text, other

    async def complete(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Send completion request to Anthropic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Anthropic-specific parameters (temperature, max_tokens, model, etc.)

        Returns:
            Response dict with 'content', 'usage', 'model', 'finish_reason'.

        Raises:
            LLMUnavailableError: If Anthropic API is unavailable
        """
        model = kwargs.pop("model", "claude-3-5-sonnet-20241022")
        max_tokens = kwargs.pop("max_tokens", 4096)
        system_text, conv_messages = self._extract_system(messages)

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system_text is not None:
            create_kwargs["system"] = system_text

        try:
            response = await self._client.messages.create(**create_kwargs)
        except (APIConnectionError, RateLimitError, APIError) as exc:
            raise LLMUnavailableError(f"Anthropic API error: {exc}") from exc

        content = ""
        for block in response.content:
            if block.type == "text":
                content = block.text
                break

        return {
            "content": content,
            "model": response.model,
            "finish_reason": response.stop_reason or "end_turn",
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }

    async def complete_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict[str, Any]:
        """Send completion with tool definitions to Anthropic.

        Args:
            messages: List of message dicts
            tools: List of tool definitions in Anthropic format
                   (each with 'name', 'description', 'input_schema')
            **kwargs: Anthropic-specific parameters

        Returns:
            Response dict with 'content', 'tool_calls', 'usage', etc.

        Raises:
            LLMUnavailableError: If Anthropic API is unavailable
        """
        model = kwargs.pop("model", "claude-3-5-sonnet-20241022")
        max_tokens = kwargs.pop("max_tokens", 4096)
        system_text, conv_messages = self._extract_system(messages)

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "tools": tools,
            **kwargs,
        }
        if system_text is not None:
            create_kwargs["system"] = system_text

        try:
            response = await self._client.messages.create(**create_kwargs)
        except (APIConnectionError, RateLimitError, APIError) as exc:
            raise LLMUnavailableError(f"Anthropic API error: {exc}") from exc

        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "tool_use",
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
                })

        return {
            "content": content,
            "model": response.model,
            "finish_reason": response.stop_reason or "end_turn",
            "tool_calls": tool_calls if tool_calls else None,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }
