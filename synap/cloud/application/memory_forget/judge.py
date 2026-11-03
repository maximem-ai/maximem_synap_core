"""Which stored memories are the thing the user asked to forget.

Asked of JEV (TypeSafe System One) through the shared ``jev_client``. Vector
search goes wide and decides nothing; JEV reads the description against each
candidate and scores it on a stated rubric.

Two choices here differ from the other JEV callers, and both are deliberate.

**Fails closed.** The ingestion screens fall back to the old behaviour when JEV is
down. Here the fallback would be to delete, or to guess, and a wrong delete cannot
be taken back by a retry. An unavailable JEV, a timeout or a malformed reply raises
:class:`JudgeUnavailable`, and the caller deletes nothing.

**Strict.** "Prefers email over phone calls" must survive a request to forget the
phone number. The rubric separates a memory that is ABOUT the same topic from one
that STATES the thing to forget, and only the second is admitted. A missing or
unparsable answer for a candidate is not admitted: silence means keep.

JEV's scores are not stable between identical calls (a prior measurement found 81%
moving). That is why preview exists: a person sees the list before anything goes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from synap.cloud.application.memory_forget.models import Candidate
from synap.cloud.application.memory_forget.prompts.judge import (
    RUBRIC,
    PURPOSE as _PURPOSE,
    rate_instruction,
)

logger = logging.getLogger(__name__)

# One JEV state holds at most this many candidates (its own re-ranker truncates at
# 150), so a longer list is judged in chunks of this size.
BATCH = 100

ADMIT_FROM = 2          # levels 2 and 3
# Higher than the 0.5 the ingestion screens use: this one gates a delete.
ADMIT_P = 0.6


class JudgeUnavailable(Exception):
    """JEV could not give a usable answer, so nothing may be deleted."""


class JudgeNotConfigured(JudgeUnavailable):
    """No Jev API key is set on the server, so forget by description cannot run."""


def build_state(description: str, candidates: Sequence[Candidate]) -> str:
    lines = [f"[{i}] {c.content}" for i, c in enumerate(candidates)]
    return (f"{_PURPOSE}\n\nWHAT THE PERSON ASKED TO FORGET:\n{description}\n\n"
            "MEMORIES STORED ABOUT THIS PERSON:\n" + "\n".join(lines))


def build_questions(count: int) -> Dict[str, Dict[str, Any]]:
    return {f"s{i}": {"type": "score",
                      "instructions": rate_instruction(i),
                      "criteria": RUBRIC}
            for i in range(count)}


def admitted_from_answers(answers: Dict[str, Any], count: int) -> Set[int]:
    """Indices whose P(level >= ADMIT_FROM) reaches ADMIT_P. Silence is a no."""
    out: Set[int] = set()
    for i in range(count):
        answer = answers.get(f"s{i}")
        if not isinstance(answer, dict):
            continue
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            continue
        total = 0.0
        for level in range(ADMIT_FROM, len(RUBRIC)):
            try:
                total += float(probs.get(str(level), 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        if total >= ADMIT_P:
            out.add(i)
    return out


async def pick(description: str, candidates: Sequence[Candidate], *,
               ask_fn=None, correlation_id: Optional[str] = None) -> List[Candidate]:
    """The candidates that really are the thing to forget.

    Raises :class:`JudgeUnavailable` if any batch cannot be judged. A partial
    answer is not returned: half a judged list would delete half and say nothing
    about the rest.
    """
    if not candidates:
        return []
    if ask_fn is None:
        from synap.cloud.shared.jev_client import ask as _ask, jev_key_set
        if not jev_key_set():
            raise JudgeNotConfigured("JEV_API_KEY is not set")

        async def ask_fn(state, questions):
            # One pooled call can hold ~100 candidates, so the client's 3s default
            # would time out, and a timed-out call that JEV already processed would
            # be sent again and billed twice. Wait long, retry at most once.
            return await _ask(state, questions, timeout=30.0, retries=1, for_forget=True)

    picked: List[Candidate] = []
    for start in range(0, len(candidates), BATCH):
        batch = list(candidates[start:start + BATCH])
        try:
            answers = await ask_fn(build_state(description, batch),
                                   build_questions(len(batch)))
        except Exception as exc:  # noqa: BLE001 - every failure means delete nothing
            logger.warning("forget_judge_failed %s: %s correlation_id=%s",
                           type(exc).__name__, str(exc)[:160], correlation_id or "")
            raise JudgeUnavailable(f"{type(exc).__name__}: {exc}") from exc
        admitted = admitted_from_answers(answers or {}, len(batch))
        picked.extend(batch[i] for i in sorted(admitted))
    logger.info("forget_judge candidates=%d picked=%d correlation_id=%s",
                len(candidates), len(picked), correlation_id or "")
    return picked
