"""LLM provider abstraction and configuration."""

from .config_manager import LLMConfigManager, ModelSelection, LLMUnavailableError
from .tracker import UsageTracker, InMemoryTracker
from .adapters import (
    BaseLLMAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
)

__all__ = [
    "LLMConfigManager",
    "ModelSelection",
    "LLMUnavailableError",
    "UsageTracker",
    "InMemoryTracker",
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GroqAdapter",
]


def __getattr__(name: str):
    # GroqAdapter does not ship in Core, so it loads only when asked for.
    if name == "GroqAdapter":
        from .adapters import GroqAdapter

        return GroqAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
