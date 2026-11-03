"""Core: what the server checks before it starts, so a wrong setting stops it with the reason.

Whether the checks run is a plain value of the edition module (``checks_settings_at_start``); this
file never reads the edition name. Where it is false, ``check_at_startup`` does nothing.

The checks:

* every prompt text Core owes is present;
* ``SYNAP_LLM_PROVIDER`` and ``SYNAP_LLM_MODEL`` are both set and valid;
* ``EMBEDDING_PROVIDER`` is not ``remote``;
* ``EMBEDDING_PROVIDER=openai`` has an ``OPENAI_API_KEY``.
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional

from synap.cloud.shared import edition


class CoreStartupError(RuntimeError):
    """A setting or a prompt text that Core needs is missing or wrong."""


def problems(environ: Optional[Mapping[str, str]] = None) -> List[CoreStartupError]:
    """One error per thing that is wrong. Empty means the server may start."""
    env = os.environ if environ is None else environ
    found: List[CoreStartupError] = []

    from synap.cloud.shared import prompt_switch

    found.extend(CoreStartupError(m) for m in prompt_switch.check_core_pack())

    from synap.cloud.infrastructure.llm import settings as llm_settings

    try:
        if llm_settings.load(env) is None:
            found.append(CoreStartupError(
                f"{llm_settings.PROVIDER_VAR} and {llm_settings.MODEL_VAR} are not set. "
                "Core uses one provider and one model for everything: set both in the environment file."
            ))
    except llm_settings.LLMSettingsError as exc:
        found.append(CoreStartupError(str(exc)))

    provider = (env.get("EMBEDDING_PROVIDER") or "sentence-transformers").strip().lower()
    if provider == "remote":
        found.append(CoreStartupError(
            "EMBEDDING_PROVIDER=remote is not available in Core: the standalone embedding service "
            "does not ship. Use sentence-transformers (the default) or openai."
        ))
    elif provider == "openai" and not (env.get("OPENAI_API_KEY") or "").strip():
        found.append(CoreStartupError(
            "EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY, and it is not set. "
            "Set the key, or use sentence-transformers."
        ))
    return found


def check_at_startup(environ: Optional[Mapping[str, str]] = None) -> None:
    """Raise ``CoreStartupError`` listing everything that is wrong. Does nothing where the edition does not check."""
    if not edition.checks_settings_at_start:
        return
    found = problems(environ)
    if found:
        raise CoreStartupError("Synap Core cannot start: " + " | ".join(str(e) for e in found))
