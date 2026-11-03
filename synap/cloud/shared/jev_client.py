"""One call to TypeSafe's System One API (Jev).

Older callers may carry their own copy of this request, each with measured
timeouts and fault reporting of its own. New callers should use this module.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

from synap.cloud.infrastructure.llm.call_trace import external_call

JEV_API_URL = os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")


class JevUnavailable(Exception):
    """No key, a timeout, an HTTP error, or a reply without answers."""


def jev_key_set() -> bool:
    return bool(os.environ.get("JEV_API_KEY", "").strip())


def jev_configured() -> bool:
    """Whether the engine's own Jev steps may call Jev: a key is set and this edition runs those steps.

    Core runs none of them, so with a key set this is still False there. Forget by description
    asks :func:`jev_key_set` instead.
    """
    from synap.cloud.shared import edition
    return jev_key_set() and edition.engine_uses_jev


def jev_messages(state: Any, questions: Dict[str, Any]) -> list:
    """A Jev request in the message shape the call traces store."""
    def as_text(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(
            value, default=str, ensure_ascii=False)

    return [
        {"role": "system", "content": as_text(questions)},
        {"role": "user", "content": as_text(state)},
    ]


async def ask(state: Any, questions: Dict[str, Any], *, model: str = "jev-latest",
              timeout: float = 3.0, retries: int = 2, for_forget: bool = False) -> Dict[str, Any]:
    """POST one state with its questions; return the ``answers`` map.

    Retries only what the API documents as transient (429 rate limit, 529
    overloaded, 5xx), honouring ``retry-after``. Raises JevUnavailable
    otherwise, so every caller decides its own fail-open behaviour.
    """
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        raise JevUnavailable("JEV_API_KEY is not set")
    if not for_forget:
        from synap.cloud.shared import edition
        if not edition.engine_uses_jev:
            # In Core the key belongs to forget alone: no other step may spend it.
            raise JevUnavailable("this step does not call Jev in this edition")
    import httpx

    payload = {"state": state, "model": model, "questions": questions}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last: Exception = JevUnavailable("no attempt made")
    # One trace record per ask, however many HTTP attempts it took. The
    # questions are the instruction text, so they go where a system prompt
    # would: calls then group by instruction the way model calls group by
    # prompt.
    with external_call(
        "jev", "typesafe", model, default_operation="jev.ask",
        messages=jev_messages(state, questions),
        requested_params={"timeout": timeout, "retries": retries},
    ) as trace:
        for attempt in range(retries + 1):
            trace["http_attempts"] = attempt + 1
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await asyncio.wait_for(
                        client.post(JEV_API_URL, json=payload, headers=headers), timeout=timeout)
                if response.status_code in (429, 529) or response.status_code >= 500:
                    wait = float(response.headers.get("retry-after") or 0.5 * 2 ** attempt)
                    last = JevUnavailable(f"HTTP {response.status_code}")
                    await asyncio.sleep(min(wait, 5.0))
                    continue
                response.raise_for_status()
                answers = (response.json() or {}).get("answers")
                if not answers:
                    raise JevUnavailable("reply carried no answers")
                trace["output"] = json.dumps(answers, default=str, ensure_ascii=False)
                return answers
            except JevUnavailable:
                raise
            except Exception as exc:
                last = JevUnavailable(f"{type(exc).__name__}: {exc}")
                if attempt < retries:
                    await asyncio.sleep(0.5 * 2 ** attempt)
        raise last
