"""Any server that speaks the OpenAI chat API: hosted open-weights providers,
vLLM, LM Studio, a company gateway. Core uses this so that "which model" is configuration.

    SYNAP_LLM_PROVIDER=openai-compatible
    SYNAP_LLM_BASE_URL   the server's API address, for example the part before /chat/completions
    SYNAP_LLM_API_KEY    its key (optional for a server that needs none)

It reuses the OpenAI request and response handling and changes only where the key comes from. It
reads ONLY ``SYNAP_LLM_API_KEY``: never ``OPENAI_API_KEY`` or ``OPENAI_API_KEYS``, which belong to
OpenAI. Pointing the OpenAI provider at another host would send that key there.
"""
from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import urlsplit

from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


class OpenAICompatibleProvider(OpenAIProvider):
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        if not (base_url or "").strip():
            raise ValueError("SYNAP_LLM_BASE_URL is required for the openai-compatible provider")
        url = base_url.strip().rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"SYNAP_LLM_BASE_URL is not a web address: {base_url!r}")
        key = (api_key if api_key is not None else os.environ.get("SYNAP_LLM_API_KEY") or "").strip()
        if key and parts.scheme == "http" and (parts.hostname or "") not in _LOCAL_HOSTS:
            logger.warning(
                "SYNAP_LLM_BASE_URL is http:// to %s: the API key and every prompt travel unencrypted", parts.hostname)
        # BaseLLMProvider, not OpenAIProvider.__init__: that one looks up OPENAI_API_KEY(S).
        BaseLLMProvider.__init__(self, api_key=key or "not-needed", base_url=url)
        logger.info("OpenAICompatibleProvider initialized host=%s key=%s", parts.netloc, "set" if key else "none")
