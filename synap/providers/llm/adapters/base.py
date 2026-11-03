"""Abstract LLM provider adapter."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseLLMAdapter(ABC):
    """Abstract LLM provider adapter.

    Defines the contract for LLM provider implementations. Each adapter
    wraps a specific LLM API (OpenAI, Anthropic, Groq, etc.) and provides
    a unified async interface for completion requests.
    """

    @abstractmethod
    async def complete(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Send completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Response dict with 'content', 'usage', 'model', etc.

        Raises:
            LLMUnavailableError: If provider API is unavailable
        """
        ...

    @abstractmethod
    async def complete_with_tools(self, messages: List[Dict], tools: List[Dict], **kwargs) -> Dict[str, Any]:
        """Send completion with tool definitions.

        Args:
            messages: List of message dicts
            tools: List of tool definitions in provider-specific format
            **kwargs: Provider-specific parameters

        Returns:
            Response dict with 'content', 'tool_calls', 'usage', etc.

        Raises:
            LLMUnavailableError: If provider API is unavailable
        """
        ...
