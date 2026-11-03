"""Let extraction read a document against what we already know about the person.

Today it cannot. Stage 3 receives one document, one conversation date, and nothing
else: a search for any channel carrying the user's existing memories into
extraction returns nothing, and ``analysis_hints`` is client-supplied call-analysis
JSON, not our corpus. So every document is read as if it were the first one we had
ever seen, which is what a vector store does.

That is the difference this closes. A reference like "three months before the
wedding" or "a week after returning from San Francisco" cannot be resolved from the
document alone, and a memory system is supposed to know what it already knows.

Safety, measured before building
--------------------------------
The obvious risk is that the extractor imports a fact from the context and stores
it as if the person had just said it. Tested: three plausible false
facts planted in the context of 21 questions across 3 memory types, 63
opportunities, context up to 119 memories. **None was ever stored.** A second,
stricter check flagged 162 memories for carrying a word absent from the document
and every one turned out to be the detector's own fault — word stems, possessives,
quote-attached tokens, and month names absent because the document says "today"
and the date arrives as DCT.

``synap/tests/ingestion/test_prior_context.py`` keeps that probe as a test, because
it is the only thing standing between us and a silent leak later.

⚠ What is NOT settled: with context the probe produced 12% fewer memories, and 25%
of the no-context arm's content was not carried over at a 0.6 overlap threshold.
Against a 26% run-to-run noise floor that is not separable from noise. Measure
recall with pooled repeats before turning this on anywhere that matters.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from synap.cloud.agents.ingestion.prompts.prior_context import HEADER

logger = logging.getLogger(__name__)

_FLAG_ENV = "SYNAP_STAGE3_PRIOR_CONTEXT"
_BY_INSTANCE_ENV = "SYNAP_STAGE3_PRIOR_CONTEXT_BY_INSTANCE"
_CAP_ENV = "SYNAP_STAGE3_PRIOR_CONTEXT_MAX"
_TRUTHY = {"1", "true", "on", "yes"}
DEFAULT_CAP = 60


def enabled_for(instance_id: Optional[object]) -> bool:
    """Whether the prior-context block is on for this instance.

    One switch for the whole set, ``SYNAP_TEMPORAL_V2``; this feature's own pair
    of variables still overrides it. See services/temporal_flags for precedence.
    """
    from synap.cloud.agents.ingestion.services import temporal_flags
    return temporal_flags.enabled(_FLAG_ENV, _BY_INSTANCE_ENV, instance_id)


def max_memories() -> int:
    try:
        return max(1, int(os.environ.get(_CAP_ENV) or DEFAULT_CAP))
    except (TypeError, ValueError):
        return DEFAULT_CAP


def _iso_day(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.date().isoformat()
    except AttributeError:
        text = str(value).strip()
        return text[:10] or None


def render(memories: Sequence[Dict[str, Any]]) -> str:
    """The context block itself.

    ⚠ Never render a bare date here. "happened" and "said on" are separate
    labelled lines and an unknown event date says UNKNOWN outright, because a
    single date sitting next to a fact reads as the event date even when all we
    know is the day someone mentioned it. That misreading is the defect this whole
    change exists to remove; a context block that reintroduces it would teach the
    extractor the same wrong thing.
    """
    lines: List[str] = []
    for i, m in enumerate(memories, 1):
        start = _iso_day(m.get("event_start"))
        end = _iso_day(m.get("event_end"))
        if start and end and end != start:
            happened = f"{start} to {end}"
        elif start:
            happened = start
        else:
            happened = "UNKNOWN"
        said = _iso_day(m.get("document_created_at")) or "UNKNOWN"
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{i}. {content}\n"
                     f"   happened: {happened}\n"
                     f"   said on:  {said}")
    return "\n".join(lines)


def build_fetcher(session_factory):
    """An ``async fn(scope, document_id) -> str`` returning the rendered block.

    Injected rather than reached for: the extractor is deliberately built with a
    config, an LLM and nothing else, and an absent fetcher means the feature is off
    however the flag is set, rather than Stage 3 quietly acquiring a database
    dependency.
    """
    from sqlalchemy import text

    async def fetch(scope, document_id: Optional[str] = None) -> str:
        user_id = getattr(scope, "user_id", None)
        if session_factory is None or not user_id:
            return ""
        # Most recent first, and never the document being ingested: its own
        # memories are not yet written, but a retry or a re-ingest would otherwise
        # feed the document its own output back as "earlier context".
        # ⚠ The exclusion used to read `(:doc IS NULL OR source_document_id IS
        # DISTINCT FROM :doc)`, and that one clause meant this feature never ran
        # once. A bind parameter alone on the left of `IS NULL`
        # gives Postgres nothing to infer a type from, so asyncpg's prepare step
        # raised `AmbiguousParameterError: could not determine data type of
        # parameter $2` on all 44 documents of the 3 Oct benchmark run. The
        # except below is deliberately non-fatal, so it degraded silently: every
        # call logged a warning and returned "", and extraction simply never saw
        # a prior-context block.
        #
        # Built conditionally rather than cast, so the untypeable comparison does
        # not exist at all and the parameter is only bound when it is used.
        sql = ("SELECT content, event_start, event_end, document_created_at "
               "FROM memories "
               "WHERE user_id = CAST(:uid AS uuid) "
               "  AND superseded_by IS NULL AND deprecated_at IS NULL AND deleted_at IS NULL ")
        params = {"uid": str(user_id), "lim": max_memories()}
        if document_id:
            sql += ("  AND source_document_id IS DISTINCT FROM "
                    "CAST(:doc AS text) ")
            params["doc"] = str(document_id)
        sql += ("ORDER BY document_created_at DESC NULLS LAST, created_at DESC "
                "LIMIT :lim")
        try:
            async with session_factory() as session:
                rows = (await session.execute(text(sql), params)).fetchall()
        except Exception:  # noqa: BLE001 — context is an enhancement, never a gate
            logger.warning("prior_context_fetch_failed user_id=%s", user_id,
                           exc_info=True)
            return ""
        memories = [{"content": r[0], "event_start": r[1], "event_end": r[2],
                     "document_created_at": r[3]} for r in rows]
        block = render(memories)
        logger.info("prior_context user_id=%s memories=%d chars=%d",
                    user_id, len(memories), len(block))
        return block

    return fetch


def wrap(user_prompt: str, context_block: str) -> str:
    """Prepend the context, or return the prompt untouched.

    An empty block leaves the prompt byte-identical to today's, which is what a
    disabled flag, a missing fetcher and a failed read all produce.
    """
    if not context_block:
        return user_prompt
    return HEADER.format(context=context_block) + user_prompt
