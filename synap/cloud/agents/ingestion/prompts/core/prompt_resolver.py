"""Core's prompt resolver: there are no per-provider variants in Core.

``resolve`` returns the prompt class it is given. Same names as the Cloud
resolver, so the extraction code calls both the same way.
"""

from typing import Optional, Type


class PromptResolver:
    @classmethod
    def register(cls, provider: str, base_class: Type, variant_class: Type) -> None:
        raise RuntimeError("Core has no per-provider prompt variants.")

    @classmethod
    def resolve(cls, prompt_class: Type, provider: Optional[str] = None) -> Type:
        return prompt_class
