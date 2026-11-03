"""Compaction triggers and size limits, sized to the models involved.

Two models set the numbers, and they are different models:

* The AGENT's model (the customer's) sets when to compact and how big the
  record may grow. Accuracy holds up to about 25% of a context window; memory
  gets 15K of the 32K that leaves on a 128K window, the smallest in common use,
  and the rest goes to tools, skills, system and user prompts. So:

      trigger T = window x 15/128          (15K at 128K, uncapped above)
      ceiling C = min(T/2, G)              (T/2 keeps >= T/2 of raw turns)

* OUR compaction model sets G and K, because every call to it must stay
  inside the same 25% rule:

      prompt (~3K) + chunk recaps (<= 0.4K) + record (<= G) + chunk (<= K)
          <= 25% of our model's window

  K is T at the 128K default, so a 128K customer's new turns always fit in
  one chunk; G is what is left. Both are worked out from the window of the
  model ``resolve_compaction_profile`` picks, so replacing that model
  re-derives them. Two tests pin the result (see test_compaction_budgets.py),
  and the manager logs it at startup, so a model change cannot slip through
  unnoticed.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TABLE_PATH = Path(__file__).resolve().parent / "data" / "model_windows.json"

# 15K of a 128K window, 25% accuracy budget.
DEFAULT_AGENT_WINDOW = 128_000
TRIGGER_TOKENS_AT_DEFAULT = 15_000
TRIGGER_FRACTION = TRIGGER_TOKENS_AT_DEFAULT / DEFAULT_AGENT_WINDOW
ACCURACY_FRACTION = 0.25
MESSAGE_TRIGGER = 30

# Fixed parts of every writer call, in tokens.
PROMPT_RESERVE = 3_000
RECAP_RESERVE = 400

# Below this, a compaction model's window is too small to split as above
# and K and G share what is left equally.
_MIN_RECORD_CEILING = 4_000


@lru_cache(maxsize=1)
def _table() -> Dict[str, Any]:
    return json.loads(_TABLE_PATH.read_text())


def _lookup(name: str) -> Optional[int]:
    models: Dict[str, int] = _table()["models"]
    if name in models:
        return models[name]
    prefixes = [key for key in models if name.startswith(key)]
    if prefixes:
        return models[max(prefixes, key=len)]
    return None


def window_for_model(model: Optional[str]) -> Optional[int]:
    """The model's context window from the lookup table, or None if unknown.

    Exact name first, then the longest table key the name starts with
    (dated snapshots resolve to their family), then the same after dropping
    any 'provider/' or 'org/' prefix, then the family table.
    """
    if not model:
        return None
    name = model.strip().lower()
    for candidate in dict.fromkeys((name, name.rsplit("/", 1)[-1])):
        found = _lookup(candidate)
        if found:
            return found
    short = name.rsplit("/", 1)[-1]
    families: Dict[str, int] = _table().get("families", {})
    prefixes = [key for key in families if short.startswith(key)]
    return families[max(prefixes, key=len)] if prefixes else None


def resolve_agent_window(
    context_window: Optional[int] = None,
    model: Optional[str] = None,
    instance_window: Optional[int] = None,
) -> int:
    """The agent's window, first value available.

    An explicit window wins, then the model's window from the table (a
    model reported by the SDK or detected by an integration), then the
    instance setting, then 128K.
    """
    if context_window and context_window > 0:
        return int(context_window)
    found = window_for_model(model)
    if found:
        return found
    if instance_window and instance_window > 0:
        return int(instance_window)
    return DEFAULT_AGENT_WINDOW


WINDOW_ENV = "SYNAP_LLM_CONTEXT_WINDOW"


def configured_window(environ: Optional[Dict[str, str]] = None) -> Optional[int]:
    """The context window the developer wrote down for their model, or None when they did not.

    The table only knows models by name, and a self-hosted or new model is never in it. Raises
    ``ValueError`` when the setting is not a whole number of tokens of at least 4,000.
    """
    raw = ((os.environ if environ is None else environ).get(WINDOW_ENV) or "").strip().replace("_", "")
    if not raw:
        return None
    try:
        window = int(raw)
    except ValueError:
        raise ValueError(f"{WINDOW_ENV}={raw!r} is not a whole number of tokens") from None
    if window < 4_000:
        raise ValueError(f"{WINDOW_ENV}={window} is too small: compaction needs a window of at least 4,000 tokens")
    return window


@dataclass(frozen=True)
class CompactionModelLimits:
    """What our compaction model's window allows per call."""
    model: str
    window: int
    budget: int  # 25% of the window
    chunk_tokens: int  # K
    record_ceiling: int  # G

    def describe(self) -> str:
        return (
            f"compaction model {self.model} · window {self.window / 1000:.0f}K · "
            f"budget {self.budget / 1000:.1f}K → K={self.chunk_tokens // 1000}K, "
            f"G={self.record_ceiling // 1000}K"
        )


def derive_model_limits(model: str, window: Optional[int] = None) -> CompactionModelLimits:
    """K and G for our compaction model.

    Raises KeyError when the model is not in the table and no window is
    given: a test calls this for every compaction profile, so a new model
    cannot ship with an unknown window. The manager catches it and falls back
    to the 128K default, with a warning, rather than stop compacting.
    """
    if window is None:
        window = window_for_model(model)
        if window is None:
            raise KeyError(f"compaction model {model!r} is not in {_TABLE_PATH.name}")
    budget = int(window * ACCURACY_FRACTION)
    left = budget - PROMPT_RESERVE - RECAP_RESERVE
    chunk = TRIGGER_TOKENS_AT_DEFAULT
    if left - chunk < _MIN_RECORD_CEILING:
        chunk = left // 2
    ceiling = left - chunk
    # Whole thousands, rounded down, so the logged and pinned values are the
    # ones in force.
    return CompactionModelLimits(
        model=model,
        window=window,
        budget=budget,
        chunk_tokens=max(1_000, chunk // 1000 * 1000),
        record_ceiling=max(1_000, ceiling // 1000 * 1000),
    )


@dataclass(frozen=True)
class AgentBudget:
    """Trigger and ceiling for one conversation."""
    window: int
    trigger_tokens: int  # T
    record_ceiling: int  # C
    message_trigger: int = MESSAGE_TRIGGER


def agent_budget(window: int, record_ceiling_cap: int) -> AgentBudget:
    trigger = int(window * TRIGGER_FRACTION)
    return AgentBudget(
        window=window,
        trigger_tokens=trigger,
        record_ceiling=min(trigger // 2, record_ceiling_cap),
    )
