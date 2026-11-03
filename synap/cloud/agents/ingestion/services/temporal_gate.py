"""Does this document need the temporal rules in its extraction prompt?

Asked of Jev (TypeSafe System One), one message at a time. Replaces the regex in
TemporalTaggingService as the decision; the regex stays as the fallback.

MEASURED (60 LongMemEval sessions, all 707 messages, labels from gpt-4o and
gemini-3.1-pro with 65 disagreements settled by hand, 2026-09-29):

                         right   catches real time references
  regex (before)          61%     79%   (misses "on a Friday", "this summer", "Nov. 20")
  one broad Jev question  75%     91%
  these three questions   91%     98%   (per whole session: 100%)

WHY ONE MESSAGE PER CALL: TypeSafe's own guidance is that accuracy falls as the
state grows with unrelated detail, and a session runs to thousands of tokens.

WHY THREE QUESTIONS: the guidance's central advice is narrow, atomic questions
combined in code. One question per kind of reference (relative, calendar,
duration) beat one broad question by 15 points.

WHY THESE EXAMPLES AND NO MORE: adding "lately", "currently", "soon", "just
got" and "for a while" to them caught 2 more vague references per 707 messages
but said a wrong yes to 42 more (correct 90.5% -> 84.9%), and changed nothing
per whole session (0 missed either way). Kept as approved.

FAILING OPEN: a false yes costs a few hundred prompt tokens; a false no loses
the dates of every memory in the document. So any Jev failure means "include
the rules", and a document too long to check fully falls back to the regex for
the rest.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from synap.cloud.agents.ingestion.prompts.temporal_gate import (
    BACKGROUND as _BACKGROUND,
    QUESTIONS,
)

logger = logging.getLogger(__name__)


_THRESHOLD = float(os.environ.get("SYNAP_TEMPORAL_GATE_THRESHOLD") or "0.5")
_BATCH = int(os.environ.get("SYNAP_TEMPORAL_GATE_BATCH") or "8")
_MAX_MESSAGES = int(os.environ.get("SYNAP_TEMPORAL_GATE_MAX_MESSAGES") or "80")
_TIMEOUT = float(os.environ.get("SYNAP_TEMPORAL_GATE_TIMEOUT_SECS") or "3")
_MAX_CHARS = 6000   # per message; Jev's state budget is 32k tokens
_CHUNK_CHARS = 1500  # size of a slice when the text has no speaker lines

_SPEAKER_RE = re.compile(
    r"^[ \t]*(?:\[[^\]\n]{1,40}\][ \t]*)?([A-Za-z][\w .'\-]{0,30}):[ \t]", re.MULTILINE)


@dataclass
class GateDecision:
    needs_rules: bool
    source: str            # "jev" | "jev_error" | "regex" | "regex_tail" | "empty"
    max_score: Optional[float] = None
    messages_checked: int = 0
    messages_total: int = 0


def split_messages(text: str) -> List[Tuple[str, str]]:
    """'User: ...\\nAssistant: ...' -> [(speaker, text)]. Text without at least
    two speaker lines is cut into paragraph-aligned slices instead."""
    marks = list(_SPEAKER_RE.finditer(text))
    if len(marks) >= 2:
        out = []
        head = text[:marks[0].start()].strip()
        if head:
            out.append(("document", head))
        for m, nxt in zip(marks, marks[1:] + [None]):
            body = text[m.end():nxt.start() if nxt else len(text)].strip()
            if body:
                out.append((m.group(1).strip().lower(), body[:_MAX_CHARS]))
        return out
    out, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) > _CHUNK_CHARS:
            out.append(("document", buf[:_MAX_CHARS]))
            buf = ""
        buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        out.append(("document", buf[:_MAX_CHARS]))
    return out


def _state(speaker: str, message: str, dct: Optional[datetime]):
    date = f"{dct.date().isoformat()} ({calendar.day_name[dct.weekday()]})" if dct else "unknown"
    return {"conversation_date": date, "speaker": speaker, "message": message}


def _regex(text: str, dct: Optional[datetime]) -> bool:
    from synap.cloud.agents.ingestion.services.temporal_tagging_service import TemporalTaggingService
    return TemporalTaggingService().extract_temporal_flags(text=text, dct=dct).has_temporal_content


def jev_gate_enabled() -> bool:
    """On when a Jev key exists, unless SYNAP_TEMPORAL_GATE=regex."""
    from synap.cloud.shared.jev_client import jev_configured
    return os.environ.get("SYNAP_TEMPORAL_GATE", "jev").strip().lower() != "regex" and jev_configured()


async def decide(text: str, dct: Optional[datetime], correlation_id: str = "") -> GateDecision:
    """Whether the document's extraction prompt should carry the temporal rules."""
    if not text or not text.strip():
        return GateDecision(False, "empty")
    if not jev_gate_enabled():
        return GateDecision(_regex(text, dct), "regex")

    from synap.cloud.shared.jev_client import JevUnavailable, ask

    messages = split_messages(text)
    checked, best = 0, 0.0
    for i in range(0, min(len(messages), _MAX_MESSAGES), _BATCH):
        batch = messages[i:min(i + _BATCH, _MAX_MESSAGES)]
        results = await asyncio.gather(
            *[ask(_state(s, m, dct), QUESTIONS, timeout=_TIMEOUT) for s, m in batch],
            return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.warning("temporal_gate_jev_failed correlation_id=%s error=%s -> including rules",
                               correlation_id, res if isinstance(res, JevUnavailable) else repr(res))
                return GateDecision(True, "jev_error", best, checked + 1, len(messages))
            checked += 1
            score = max(float(a.get("noul") or 0.0) for a in res.values())
            best = max(best, score)
        if best >= _THRESHOLD:
            return GateDecision(True, "jev", best, checked, len(messages))
    if len(messages) > _MAX_MESSAGES:
        rest = "\n\n".join(m for _, m in messages[_MAX_MESSAGES:])
        return GateDecision(_regex(rest, dct), "regex_tail", best, checked, len(messages))
    return GateDecision(False, "jev", best, checked, len(messages))
