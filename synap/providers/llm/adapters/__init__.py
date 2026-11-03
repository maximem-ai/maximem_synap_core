"""LLM provider adapters."""

from .base import BaseLLMAdapter
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter

__all__ = [
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GroqAdapter",
]


def __getattr__(name: str):
    # Groq does not ship in Core, so its adapter loads
    # only when asked for. Cloud gets the same name, imported on first use.
    if name == "GroqAdapter":
        from .groq import GroqAdapter

        return GroqAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
