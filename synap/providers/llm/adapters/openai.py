"""OpenAI LLM adapter."""

from typing import Dict, Any, List

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

from .base import BaseLLMAdapter


class LLMUnavailableError(Exception):
    """Raised when an LLM provider API is unavailable."""
    pass


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI API adapter.

    Supports GPT-4o, GPT-4o-mini, and other OpenAI chat models.

    Documentation:
    - https://platform.openai.com/docs/api-reference/chat
    """

    def __init__(self, api_key: str, organization: str = None):
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key
            organization: Optional organization ID
        """
        self._api_key = api_key
        self._organization = organization
        self._client = AsyncOpenAI(api_key=api_key, organization=organization)

    async def complete(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Send completion request to OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: OpenAI-specific parameters (temperature, max_tokens, model, etc.)

        Returns:
            Response dict with 'content', 'usage', 'model', 'finish_reason'.

        Raises:
            LLMUnavailableError: If OpenAI API is unavailable
        """
        model = kwargs.pop("model", "gpt-4o")

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except (APIConnectionError, RateLimitError, APIError) as exc:
            raise LLMUnavailableError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage

        return {
            "content": choice.message.content or "",
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }

    async def complete_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict[str, Any]:
        """Send completion with tool definitions to OpenAI.

        Args:
            messages: List of message dicts
            tools: List of tool definitions in OpenAI format
            **kwargs: OpenAI-specific parameters

        Returns:
            Response dict with 'content', 'tool_calls', 'usage', etc.

        Raises:
            LLMUnavailableError: If OpenAI API is unavailable
        """
        model = kwargs.pop("model", "gpt-4o")

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                **kwargs,
            )
        except (APIConnectionError, RateLimitError, APIError) as exc:
            raise LLMUnavailableError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return {
            "content": choice.message.content or "",
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }
