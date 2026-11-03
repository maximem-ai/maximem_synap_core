"""One provider and one model for everything.

Cloud scores models and picks one per call, and several steps force a provider. A developer
running Core has one key or one local model. When both settings below are present, every call
the manager makes uses them, whatever the call site asked for; when neither is, nothing here
does anything and Cloud behaves exactly as before.

    SYNAP_LLM_PROVIDER   openai | anthropic | ollama | openai-compatible
    SYNAP_LLM_MODEL      the model every step uses
    SYNAP_LLM_MODEL_<STEP>   optional: a different model for one step
    SYNAP_LLM_BASE_URL, SYNAP_LLM_API_KEY   for openai-compatible: the server's address and key

Core ships no default model. Nothing here, in the example files or in the docs names one as the
thing to use; the developer chooses.

Steps: INGESTION (extraction and the profile merge), GRAPH (entities and relationships),
KNOWLEDGE_UPDATE (the classifier that decides whether a new memory replaces an old one),
COMPACTION, VISION (image descriptions; the model must read images).

The provider stays the one set above for every step, and a failed call is never retried on another
provider: the prompts were written for the model the developer chose.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from .models import LLMProvider

PROVIDER_VAR = "SYNAP_LLM_PROVIDER"
MODEL_VAR = "SYNAP_LLM_MODEL"
STEPS = ("INGESTION", "GRAPH", "KNOWLEDGE_UPDATE", "COMPACTION", "VISION")
# Providers a single-model setup may name. Groq and Gemini do not ship in Core.
PROVIDERS = {"openai": LLMProvider.OPENAI, "anthropic": LLMProvider.ANTHROPIC, "ollama": LLMProvider.OLLAMA,
             "openai-compatible": LLMProvider.OPENAI}
BASE_URL_VAR = "SYNAP_LLM_BASE_URL"
PLACEHOLDER_PREFIX = "add-"  # what .env.example holds until the developer fills it in


class LLMSettingsError(ValueError):
    """The LLM settings are incomplete or name something that cannot be used."""


@dataclass(frozen=True)
class LLMSettings:
    provider: LLMProvider
    model: str
    step_models: Dict[str, str] = field(default_factory=dict)
    # Set for openai-compatible: the server to talk to. The provider is then OpenAI's protocol only.
    base_url: Optional[str] = None

    def model_for(self, step: Optional[str]) -> str:
        return self.step_models.get((step or "").upper(), self.model)

    def all_models(self) -> set:
        return {self.model, *self.step_models.values()}


def load(environ=None) -> Optional[LLMSettings]:
    """The settings, None when they are not in use, an error when they are half there."""
    env = os.environ if environ is None else environ
    provider = (env.get(PROVIDER_VAR) or "").strip().lower()
    model = (env.get(MODEL_VAR) or "").strip()
    overrides = {
        step: (env.get(f"{MODEL_VAR}_{step}") or "").strip()
        for step in STEPS if (env.get(f"{MODEL_VAR}_{step}") or "").strip()
    }
    base_url = (env.get(BASE_URL_VAR) or "").strip()
    for name, value in ((PROVIDER_VAR, provider), (MODEL_VAR, model), (BASE_URL_VAR, base_url),
                        *((f"{MODEL_VAR}_{k}", v) for k, v in overrides.items())):
        if value.lower().startswith(PLACEHOLDER_PREFIX):
            raise LLMSettingsError(f"{name} still holds the placeholder from .env.example ({value!r}): fill it in")
    if not provider and not model:
        if overrides or base_url:
            raise LLMSettingsError(
                f"{', '.join(n for n in (BASE_URL_VAR if base_url else '', *(f'{MODEL_VAR}_{s}' for s in overrides)) if n)}"
                f" is set, but {PROVIDER_VAR} and {MODEL_VAR} are not")
        return None
    if not provider or not model:
        missing = PROVIDER_VAR if not provider else MODEL_VAR
        raise LLMSettingsError(f"{missing} is not set: set both {PROVIDER_VAR} and {MODEL_VAR}, or neither")
    if provider not in PROVIDERS:
        raise LLMSettingsError(
            f"{PROVIDER_VAR}={provider!r} is not a provider Core uses; choose one of: {', '.join(sorted(PROVIDERS))}")
    if provider == "openai-compatible" and not base_url:
        raise LLMSettingsError(f"{BASE_URL_VAR} is not set: the openai-compatible provider needs the server's address")
    if provider != "openai-compatible" and base_url:
        raise LLMSettingsError(f"{BASE_URL_VAR} is only used with {PROVIDER_VAR}=openai-compatible, not {provider}")
    return LLMSettings(PROVIDERS[provider], model, overrides, base_url or None)


_cached: Optional[LLMSettings] = None
_loaded = False


def active() -> Optional[LLMSettings]:
    """The settings for this process, read once."""
    global _cached, _loaded
    if not _loaded:
        _cached = load()
        _loaded = True
    return _cached


def reset_for_tests() -> None:
    global _cached, _loaded
    _cached, _loaded = None, False
