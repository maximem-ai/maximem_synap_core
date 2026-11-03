"""One switch for the temporal ingestion work (PR #1315).

⚠ This exists because four separate flags was a mistake. The pre-pass, the event
anchor, the prior-context block and the open-anchor resolver were built one at a
time and each got its own pair of environment variables, which is eight names to
reason about for one feature that is only useful switched on as a set:

  - the resolver (4) has nothing to settle unless the anchor (2) is asked for;
  - the anchor (2) leaves rows open unless the resolver (4) runs;
  - the pre-pass (1) and prior context (3) are what make the dates they both
    depend on consistent in the first place.

So the switch is ONE name:

    SYNAP_TEMPORAL_V2=on                          # whole fleet
    SYNAP_TEMPORAL_V2_BY_INSTANCE=inst_abc=on     # one instance only

The per-feature variables still work, and still win, so a single piece can be
turned off without touching the rest when something misbehaves:

    SYNAP_TEMPORAL_V2=on
    SYNAP_RESOLVE_OPEN_ANCHORS=off                # everything but the resolver

Precedence, most specific first:
  1. SYNAP_<FEATURE>_BY_INSTANCE  for this instance
  2. SYNAP_<FEATURE>              fleet-wide
  3. SYNAP_TEMPORAL_V2_BY_INSTANCE for this instance
  4. SYNAP_TEMPORAL_V2            fleet-wide
  5. off

Nothing here reads a flag at import time, so a value set after boot still counts
and tests can monkeypatch the environment.
"""

from __future__ import annotations

import os
from typing import Optional

MASTER_ENV = "SYNAP_TEMPORAL_V2"
MASTER_BY_INSTANCE_ENV = "SYNAP_TEMPORAL_V2_BY_INSTANCE"

_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no"}


def _parse(raw: Optional[str]) -> Optional[bool]:
    """True, False, or None for "nothing was said here"."""
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return None


def _for_instance(env_name: str, instance_id: Optional[object]) -> Optional[bool]:
    """The value this instance is given in an ``id=on,id=off`` list, if any."""
    raw = os.environ.get(env_name, "") or ""
    if instance_id is None or not raw.strip():
        return None
    wanted = str(instance_id).strip().lower()
    for part in raw.split(","):
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == wanted:
            return _parse(value)
    return None


def enabled(flag_env: str, by_instance_env: str,
            instance_id: Optional[object]) -> bool:
    """Whether one feature of the set is on for this instance.

    ⚠ ``instance_id`` has no default on purpose. It had one, and a caller that
    omitted it got the fleet-wide answer for a per-instance flag: every
    extraction prompt resolved SYNAP_EVENT_ANCHOR for instance None, so
    ``SYNAP_TEMPORAL_V2_BY_INSTANCE=<id>=on`` was off at the only place that
    mattered and ``memories.event_anchor`` was NULL on every row ever written.
    Passing None explicitly still means "the fleet"; being unable to say
    nothing is the point.
    """
    for candidate in (
        _for_instance(by_instance_env, instance_id),
        _parse(os.environ.get(flag_env)),
        _for_instance(MASTER_BY_INSTANCE_ENV, instance_id),
        _parse(os.environ.get(MASTER_ENV)),
    ):
        if candidate is not None:
            return candidate
    return False


def master_enabled(instance_id: Optional[object] = None) -> bool:
    """The master switch alone, for logging what an operator actually set."""
    for candidate in (_for_instance(MASTER_BY_INSTANCE_ENV, instance_id),
                      _parse(os.environ.get(MASTER_ENV))):
        if candidate is not None:
            return candidate
    return False


def describe(instance_id: Optional[object] = None) -> dict:
    """Every feature's state plus where the value came from.

    Worth having: the one question an operator asks after setting a flag is
    "is it actually on", and four features reading four names each with a
    fallback is not something to work out by hand.
    """
    from synap.cloud.agents.ingestion.services import (  # local: avoid import cycles
        anchor_resolver, date_prepass, event_anchor, prior_context, relatedness)

    # ⚠ The EFFECTIVE state, not the flag's own value.
    #
    # This function exists because "is it actually on" was unanswerable by hand,
    # and then it gave a wrong answer of its own: it reported
    # resolve_open_anchors True for eight months while every prompt
    # resolved the event-anchor flag for instance None and nothing ever asked the
    # model for an anchor. A switch reporting on while the feature is inert is the
    # defect this whole workstream has been chasing.
    #
    # So the resolver's dependency is reported too. Deciding WHICH stored memory is
    # the event an anchor names is the Jev screen's job and nobody else's, so with
    # that screen off the resolver defers every anchor rather than settling it. Its
    # own flag being on does not make it act.
    screen = relatedness.enabled_for(instance_id)
    resolver_flag = anchor_resolver.enabled_for(instance_id)
    return {
        "master": master_enabled(instance_id),
        "date_prepass": date_prepass.enabled_for(instance_id),
        "event_anchor": event_anchor.enabled_for(instance_id),
        "prior_context": prior_context.enabled_for(instance_id),
        "resolve_open_anchors": resolver_flag and screen,
        "resolve_open_anchors_flag": resolver_flag,
        "candidate_screen": screen,
        "resolve_open_anchors_blocked_by": (
            None if (screen or not resolver_flag) else
            "the candidate screen is off, so every anchor is deferred"),
    }
