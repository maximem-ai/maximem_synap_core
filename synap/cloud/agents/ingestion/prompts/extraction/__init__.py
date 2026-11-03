"""Extraction prompts for different memory types and entities.

Every module here is a neutral interface (synap/cloud/shared/prompt_switch.py).
The names below are loaded on first use, so importing this package never loads
a prompt that Core does not have.
"""

import importlib

_SOURCES = {
    "FactsExtractionPrompt": "facts",
    "PreferencesExtractionPrompt": "preferences",
    "EpisodesExtractionPrompt": "episodes",
    "EmotionsExtractionPrompt": "emotions",
    "TemporalExtractionPrompt": "temporal",
    "EntityExtractionPrompt": "entities",
    "RelationshipExtractionPrompt": "relationships",
}

__all__ = list(_SOURCES)


def __getattr__(name):
    leaf = _SOURCES.get(name)
    if leaf is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{leaf}"), name)
