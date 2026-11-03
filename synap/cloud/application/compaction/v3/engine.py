"""The v3 compaction engine: one bounded writer call per chunk of new turns.

    record D, new turns S
      -> chunk S at K tokens (one chunk on a 128K agent model)
      -> for each chunk, in order:
           writer call (returns only what changed)
           code checks; on failure ONE repair call that revises the draft
           items still failing are dropped and logged; literals still
           missing are added by code
           merge: unchanged sections copied byte for byte
      -> if the record is over its ceiling C: one roll-up call
      -> new record

Calls: 1 per chunk normally, +1 per repair, +1 for a roll-up. Nothing here
writes storage: the caller stores the new record only if this returns, so a
failed update leaves the previous record in place.
"""
from __future__ import annotations

import json
import re
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from synap.cloud.shared.tokens import count_tokens

from .checks import (
    MIN_UPDATE_TOKENS,
    Problem,
    check_coverage,
    check_promises,
    check_responses,
    check_suggestions,
    check_literals,
    check_quotes,
    check_size,
    check_structure,
    format_problems,
    said_with_context,
)
from .checks import ALWAYS_KEPT_KINDS, linked_entry, only_in_bulk, value_key, value_kind
from .chunker import Turn, chunk_turns, render_turns
from .statuses import LINE_MODES, apply_block, apply_statuses, block_system, block_view, status_view, writer_system
from .prompts import retry as retry_text
from .prompts.writer import (
    DENSITY_GUIDANCE,
    REPAIR_USER,
    ROLLUP_SYSTEM,
    ROLLUP_USER,
    STEERING_SYSTEM,
    STEERING_USER,
    WRITER_SYSTEM,
    WRITER_USER,
)
from .record import (
    STEERING,
    CompactionRecord,
    LedgerChange,
    LedgerEntry,
    MergeNotes,
    UNLABELLED,
    _add_section,
    apply_stated_changes,
    track_statuses,
    inventory_items,
    merge_output,
    normalise,
    split_sentences,
)

REPAIR_KINDS = {"quote", "structure"}

# Where code keeps inventory items the writer left out of its sections.
ALSO_NOTED_HEADER = "Also noted"
# Where code keeps, verbatim, promises the writer left out.
SAID_HEADER = "Promises, as said"
from .render import budgeted_tokens, render_document
from .steering import (
    STEERING_CHUNK_TOKENS,
    STEERING_FROM_TOKENS,
    choose_mode,
    merge_steering,
    roll_up_steering,
    steering_view,
    switch_to_steering,
)

logger = logging.getLogger(__name__)

RECAP_TOKEN_LIMIT = 400
ROLLUP_FLOOR = 0.8  # share of the roll-up target below which it is discarded
JSON_MAX_RETRIES = 2

# (turns of this chunk, rendered record) -> extra repair hints. The manager
# passes the Jev completeness check here on profiles that run it.
OmissionFinder = Callable[[Sequence[Turn], str], Awaitable[List[str]]]


@dataclass
class EngineSettings:
    target_ratio: float = 0.30
    record_ceiling: int = 7_500  # C
    chunk_tokens: int = 15_000  # K
    guidance: str = "(none)"
    # Template density rates (high / medium / low), used only for relative weights.
    density_rates: Dict[str, float] = field(
        default_factory=lambda: {"high_density": 0.35, "medium_density": 0.25, "low_density": 0.10}
    )
    # Steering mode (steering.py), off by default. Retired 20 Oct 2026: on 15K/20K (2 runs each, with search) fidelity 97%/92% correct with 0/8 invented claims; steering 83%/84% with 3/36. Kept behind the flag for later work.
    steering_enabled: bool = False
    steering_from: int = STEERING_FROM_TOKENS
    steering_chunk_tokens: int = STEERING_CHUNK_TOKENS
    # 20 Oct 2026 experiments, off by default.
    # "restate" / "delta": the writer carries every line's status from chunk
    # to chunk (statuses.py); code then writes no statuses of its own.
    status_mode: str = "off"
    # The values code adds when the writer left them out: "full" (each with
    # ~24 words of its sentence), "trim" (8 words, and not a value found
    # only inside pasted logs, code or tables), "kinds" (trim, and only the
    # value kinds in value_kinds, which the memory design names).
    value_net: str = "full"
    value_kinds: tuple = ()
    # A value code adds that names the same thing as one code added from an
    # earlier turn (the same table row, the same words before it) replaces
    # it, keeping the old one as history: "$388,500 → $357,000".
    link_values: bool = False


@dataclass
class UpdateResult:
    record: CompactionRecord
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    chunks: int = 0
    repairs: int = 0
    rollup: bool = False
    dropped: List[str] = field(default_factory=list)
    auto_literals: List[str] = field(default_factory=list)
    auto_items: List[str] = field(default_factory=list)
    skipped_values: List[str] = field(default_factory=list)  # value_net trim/kinds left these out
    linked_values: List[str] = field(default_factory=list)  # link_values: replaced an earlier code-added value
    status_counts: Dict[str, int] = field(default_factory=dict)
    problems_after_repair: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    mode: str = "fidelity"
    switched: bool = False  # this update moved the record to steering


class CompactionWriter:
    """Runs writer, repair and roll-up calls on one compaction profile."""

    def __init__(self, llm_manager, profile):
        self._llm = llm_manager
        self._profile = profile

    # ------------------------------------------------------------------ LLM
    async def _call_json(self, system: str, user: str, max_tokens: int, result: UpdateResult) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if not self._profile.cross_provider_fallback:
            extra["allow_cross_provider_fallback"] = False
        current = system
        last_error: Optional[Exception] = None
        for attempt in range(JSON_MAX_RETRIES + 1):
            try:
                response = await self._llm.complete(
                    messages=[{"role": "system", "content": current}, {"role": "user", "content": user}],
                    force_provider=self._profile.provider,
                    force_model=self._profile.model,
                    step="COMPACTION",
                    temperature=0.0,
                    max_tokens=max_tokens + self._profile.reasoning_token_overhead,
                    json_mode=True,
                    reasoning_effort="medium",
                    **extra,
                )
            except Exception as exc:
                # Groq validates JSON mode itself and answers a malformed
                # generation with an error ("Failed to generate JSON"), not
                # with the text: the same bad-JSON retry applies.
                if "json" not in str(exc).lower() or attempt == JSON_MAX_RETRIES:
                    raise
                last_error = exc
                logger.warning("compaction writer: provider rejected JSON (attempt %d): %s", attempt + 1, exc)
                current = system + retry_text.JSON_RETRY_SUFFIX
                continue
            result.calls += 1
            result.input_tokens += _count(getattr(response, "input_tokens", 0))
            result.output_tokens += _count(getattr(response, "output_tokens", 0))
            try:
                data = _parse_json(response.content)
                if isinstance(data, dict):
                    return data
                raise ValueError("top-level JSON is not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning("compaction writer: bad JSON (attempt %d): %s", attempt + 1, exc)
                current = system + retry_text.JSON_RETRY_SUFFIX
        raise last_error  # type: ignore[misc]

    # --------------------------------------------------------------- update
    async def update(
        self,
        record: Optional[CompactionRecord],
        new_turns: Sequence[Turn],
        settings: EngineSettings,
        known_turns: Optional[Dict[str, str]] = None,
        find_omissions: Optional[OmissionFinder] = None,
    ) -> UpdateResult:
        """Fold ``new_turns`` into ``record`` (None = start a new record).

        ``known_turns`` maps labels to content for turns the record already
        covers, so an action update quoting an older turn can be checked.
        """
        started = time.time()
        current = record.copy() if record is not None else CompactionRecord()
        if settings.status_mode != "off":
            current.status_mode = settings.status_mode
        result = UpdateResult(record=current)
        turn_text: Dict[str, str] = dict(known_turns or {})
        chunks = chunk_turns(new_turns, settings.chunk_tokens)
        result.chunks = len(chunks)
        recaps: List[str] = []
        span_tokens = count_tokens(render_turns(new_turns)) if new_turns else 0
        result.mode = choose_mode(record, span_tokens, settings.steering_enabled, settings.steering_from)
        carried = ""
        if result.mode == STEERING and current.mode != STEERING:
            carried = switch_to_steering(current)
            result.switched = True
        if result.mode == STEERING and settings.steering_chunk_tokens < settings.chunk_tokens:
            chunks = chunk_turns(new_turns, settings.steering_chunk_tokens)
            result.chunks = len(chunks)

        for chunk in chunks:
            for turn in chunk:
                turn_text[turn.label] = (turn_text.get(turn.label, "") + "\n\n" + turn.content).strip() \
                    if turn.part else turn.content
            if result.mode == STEERING:
                current, recap = await self._fold_steering(current, chunk, recaps, settings, turn_text, result,
                                                           carried)
                carried = ""
            else:
                current, recap = await self._fold_chunk(
                    current, chunk, recaps, settings, turn_text, result, find_omissions
                )
            if not current.sections and current.mode != STEERING:
                _fallback_plan(current)
            if recap:
                recaps.append(recap)
                recaps = _trim_recaps(recaps)

        if new_turns:
            covers = dict(current.covers)
            first, last = new_turns[0], new_turns[-1]
            covers.setdefault("start_m", first.label)
            covers.setdefault("start_timestamp", first.timestamp)
            covers["end_m"] = last.label
            covers["end_timestamp"] = last.timestamp
            covers["message_count"] = int(last.label[1:]) if last.label[1:].isdigit() else len(turn_text)
            current.covers = covers

        if count_tokens(render_document(current)) > settings.record_ceiling:
            if current.mode == STEERING:
                last = new_turns[-1].label if new_turns else ""
                result.rollup = roll_up_steering(current, settings.record_ceiling,
                                                 int(last[1:]) if last[1:].isdigit() else 0)
            else:
                current = await self._roll_up(current, settings, result)

        result.record = current
        result.latency_ms = (time.time() - started) * 1000
        return result

    async def _fold_chunk(
        self,
        record: CompactionRecord,
        chunk: Sequence[Turn],
        recaps: List[str],
        settings: EngineSettings,
        turn_text: Dict[str, str],
        result: UpdateResult,
        find_omissions: Optional[OmissionFinder],
    ) -> tuple[CompactionRecord, str]:
        chunk_text = render_turns(chunk)
        chunk_tokens = count_tokens(chunk_text)
        budget = max(MIN_UPDATE_TOKENS, int(chunk_tokens * settings.target_ratio))
        ids: Dict[str, Any] = {}
        system = WRITER_SYSTEM
        if record.status_mode in LINE_MODES:
            view, ids = status_view(record)
            system = writer_system(WRITER_SYSTEM, record.status_mode)
        elif record.status_mode == "block":
            view = block_view(_record_view(record), record)
            system = block_system(WRITER_SYSTEM)
        else:
            view = _record_view(record)
        user = WRITER_USER.format(
            record=view,
            recaps=_recap_block(recaps),
            span=f"{chunk[0].label}–{chunk[-1].label}",
            turns=chunk_text,
            budget=budget,
            density_guidance=_density_guidance(settings.density_rates),
            guidance=settings.guidance,
        )
        # Room for the inventory too: about one line per item, a few items per turn.
        max_tokens = min(16_000, 2_500 + 3 * budget + chunk_tokens // 2 + count_tokens(render_document(record)) // 2
                         + 12 * len(ids))
        timestamps = {t.label: t.timestamp for t in chunk}
        roles = {t.label: ("the user" if t.role == "user" else "the assistant") for t in chunk}
        chunk_turns_text = {t.label: turn_text[t.label] for t in chunk}
        before = budgeted_tokens(record)

        output = await self._call_json(system, user, max_tokens, result)
        trial, problems = self._evaluate(record, output, chunk_turns_text, turn_text, timestamps, chunk_tokens,
                                         before, settings, roles)

        # Only a wrong claim is worth a second call: an unsupported quote or
        # a banned header. A missing value is added by code (_settle), and
        # size is the writer's budget, not a correctness problem.
        if any(p.kind in REPAIR_KINDS for p in problems):
            hints = list(problems)
            if find_omissions is not None and any(p.kind == "size_under" for p in problems):
                try:
                    for hint in await find_omissions(chunk, render_document(trial)):
                        hints.append(Problem("omission", hint))
                except Exception as exc:  # fail-open, like the check itself
                    logger.warning("completeness check failed, repairing without it: %r", exc)
            repair_user = user + "\n\n" + REPAIR_USER.format(
                problems=format_problems(hints), draft=json.dumps(output, ensure_ascii=False)
            )
            result.repairs += 1
            try:
                revised = await self._call_json(system, repair_user, max_tokens, result)
                trial2, problems2 = self._evaluate(record, revised, chunk_turns_text, turn_text, timestamps,
                                                   chunk_tokens, before, settings, roles)
                # Keep the repair unless it made things worse.
                if len(problems2) <= len(problems):
                    output, trial, problems = revised, trial2, problems2
            except Exception as exc:
                logger.warning("compaction repair call failed, keeping the draft: %r", exc)

        if problems:
            trial = self._settle(record, output, problems, chunk_turns_text, turn_text, timestamps, result, roles,
                                 settings)
        _complete_suffixes(trial, chunk_turns_text)
        if trial.status_mode == "block":
            result.status_counts["block_lines"] = apply_block(trial, output)
        if trial.status_mode in LINE_MODES:
            # The writer's statuses are the only ones (statuses.py).
            for key, n in apply_statuses(trial, output, ids, chunk_turns_text, timestamps).items():
                result.status_counts[key] = result.status_counts.get(key, 0) + n
        else:
            apply_stated_changes(trial, chunk_turns_text, timestamps)
        _say_unlabelled(trial, chunk_turns_text)
        if trial.status_mode not in LINE_MODES:
            track_statuses(trial, chunk_turns_text)
        return trial, str(output.get("chunk_recap") or "").strip()

    async def _fold_steering(self, record: CompactionRecord, chunk: Sequence[Turn], recaps: List[str],
                             settings: EngineSettings, turn_text: Dict[str, str], result: UpdateResult,
                             carried: str) -> tuple[CompactionRecord, str]:
        """One steering writer call; code keeps only what quotes support.
        No repair call: a claim that fails its quote is dropped."""
        chunk_text = render_turns(chunk)
        budget = max(MIN_UPDATE_TOKENS, int(count_tokens(chunk_text) * settings.target_ratio))
        user = STEERING_USER.format(
            record=steering_view(record, carried),
            recaps=_recap_block(recaps),
            span=f"{chunk[0].label}–{chunk[-1].label}",
            turns=chunk_text,
            budget=budget,
            guidance=settings.guidance,
        )
        max_tokens = min(16_000, 2_000 + 3 * budget)
        output = await self._call_json(STEERING_SYSTEM, user, max_tokens, result)
        timestamps = {t.label: t.timestamp for t in chunk}
        chunk_turns_text = {t.label: turn_text[t.label] for t in chunk}
        trial = record.copy()
        result.dropped += merge_steering(trial, output, chunk_turns_text, turn_text, timestamps)
        apply_stated_changes(trial, chunk_turns_text, timestamps)
        _say_unlabelled(trial, chunk_turns_text)
        return trial, str(output.get("chunk_recap") or "").strip()

    def _evaluate(self, record, output, chunk_turns_text, turn_text, timestamps, chunk_tokens, before, settings,
                  roles=None):
        problems: List[Problem] = check_quotes(output, turn_text)
        trial = record.copy()
        notes = merge_output(trial, output, timestamps)
        problems += check_structure(notes)
        problems += check_literals(trial, chunk_turns_text)
        problems += check_coverage(trial, inventory_items(output))
        problems += check_promises(trial, chunk_turns_text)
        problems += check_suggestions(trial, chunk_turns_text, roles or {})
        problems += check_responses(inventory_items(output), chunk_turns_text, roles or {})
        growth = budgeted_tokens(trial) - before
        problems += check_size(growth, chunk_tokens, settings.target_ratio)
        return trial, problems

    def _settle(self, record, output, problems, chunk_turns_text, turn_text, timestamps, result,
                roles=None, settings: Optional[EngineSettings] = None) -> CompactionRecord:
        """After the one repair: drop what still fails, add missing literals."""
        failing = {p.ref.get("action_index") for p in problems if p.kind == "quote" and "action_index" in p.ref}
        bad_values = {p.ref.get("value") for p in problems if p.kind == "quote" and "value" in p.ref}
        kept_actions = []
        for i, action in enumerate(output.get("agreed_actions") or []):
            if i in failing:
                result.dropped.append(f"action: {action.get('who', '')}: {action.get('what', '')}")
                continue
            kept_actions.append(action)
        kept_literals = []
        for op in output.get("literals") or []:
            if isinstance(op, dict) and op.get("value") in bad_values:
                result.dropped.append(f"literal: {op.get('label', '')}: {op.get('value')}")
                continue
            kept_literals.append(op)
        cleaned = {**output, "agreed_actions": kept_actions, "literals": kept_literals}
        # An advice item's response is the other side's next turn, verbatim.
        items = inventory_items(output)
        for p in problems:
            if p.kind != "response":
                continue
            item = items[p.ref["index"]]
            parts = []
            if p.ref["reason"]:
                # The suggester's own words, joined as they were said: code
                # adding "because" would assert a cause (P2).
                parts.append(p.ref["reason"].strip().rstrip("."))
            if p.ref["reply"]:
                parts.append(f'{(roles or {}).get(p.ref["reply_m"], "the other side")} replied: "{p.ref["reply"]}"')
            items[p.ref["index"]] = {**item, "why": "; ".join(parts)}
            if not p.ref["reply"]:
                items[p.ref["index"]]["why_is_reason"] = True  # no reply: not joined as "response:"
        cleaned.update({"turns": [], "inventory": items})
        trial = record.copy()
        merge_output(trial, cleaned, timestamps)
        missing = [p.ref for p in check_literals(trial, chunk_turns_text)]
        net = settings.value_net if settings is not None else "full"
        for ref in missing:
            value, m = ref["value"], ref["m"]
            if net != "full":
                # Pasted logs, code and tables stay searchable in the raw
                # turns; a value nobody said in their own words is left there.
                kind = value_kind(value)
                if only_in_bulk(value, chunk_turns_text.values()) or (
                        net == "kinds" and settings.value_kinds and kind and kind not in settings.value_kinds
                        and kind not in ALWAYS_KEPT_KINDS):
                    result.skipped_values.append(value)
                    continue
            context = _context(value, chunk_turns_text.get(m, ""), TRIM_CONTEXT_WORDS if net != "full" else CONTEXT_WORDS)
            key = value_key(value, chunk_turns_text.get(m, "")) if settings is not None and settings.link_values else ""
            if key:
                earlier = linked_entry(trial.ledger, value, key, m, chunk_turns_text.get(m, ""))
                if earlier is not None:
                    earlier.history.append(LedgerChange(value=earlier.value, m=earlier.m, at=earlier.at))
                    earlier.value, earlier.m, earlier.at = value, m, timestamps.get(m, "")
                    earlier.label, earlier.key = context or earlier.label, key
                    result.linked_values.append(value)
                    continue
            # The record already says what the value was said in. The value
            # alone somewhere in the prose is not enough: "15:12" in another
            # line does not say a release was deployed at 15:12.
            if said_with_context(trial, value, context):
                continue
            # "45,150" when "$45,150" is kept too: one is enough.
            if any(o is not ref and o["m"] == m and normalise(value) in normalise(o["value"])
                   and len(o["value"]) > len(value) for o in missing):
                continue
            trial.ledger.append(LedgerEntry(label=context or f"value from {m}",
                                            value=value, m=m, at=timestamps.get(m, ""), auto=True, key=key))
            result.auto_literals.append(value)
        # Items the writer listed in its own inventory and still left out
        # after the repair: kept as the writer worded them, with the turn.
        missing = [p.ref for p in check_coverage(trial, items)]
        if missing:
            section = trial.section_by_header(ALSO_NOTED_HEADER) or _add_section(trial, ALSO_NOTED_HEADER, "", MergeNotes())
            lines = [section.text] if section.text else []
            lines += [f"{ref['item'].rstrip('.')}. [{ref['m']}]" if ref["m"] else f"{ref['item'].rstrip('.')}."
                      for ref in missing]
            section.text = "\n".join(lines)
            result.auto_items += [ref["item"] for ref in missing]
        suggestions = check_suggestions(trial, chunk_turns_text, roles or {})
        if suggestions:
            header = "Proposals and responses"
            section = trial.section_by_header(header) or _add_section(trial, header, "", MergeNotes())
            lines = [section.text] if section.text else []
            for p in suggestions:
                line = f'- the assistant suggested: "{p.ref["sentence"]}"'
                if p.ref["reply"]:
                    line += f'; {(roles or {}).get(p.ref["reply_m"], "the user")} replied: "{p.ref["reply"]}"'
                lines.append(line + f' [{p.ref["m"]}]')
            section.text = "\n".join(lines)
            result.auto_items += [p.ref["sentence"] for p in suggestions]
        # A promise the writer left out: kept verbatim, credited by the turn's
        # role. A quote cannot invent anything.
        promises = check_promises(trial, chunk_turns_text)
        if promises:
            section = trial.section_by_header(SAID_HEADER) or _add_section(trial, SAID_HEADER, "", MergeNotes())
            lines = [section.text] if section.text else []
            lines += [f'{(roles or {}).get(p.ref["m"], "someone")} said: "{p.ref["sentence"]}" [{p.ref["m"]}]' for p in promises]
            section.text = "\n".join(lines)
            result.auto_items += [p.ref["sentence"] for p in promises]
        result.problems_after_repair += [p.message for p in problems]
        for line in result.dropped:
            logger.info("compaction: dropped after repair: %s", line)
        return trial

    async def _roll_up(self, record: CompactionRecord, settings: EngineSettings,
                       result: UpdateResult) -> CompactionRecord:
        sections_text = "\n\n".join(f"[{s.id}] {s.header}\n{s.text}" for s in record.sections)
        current = count_tokens(sections_text)
        other = count_tokens(render_document(record)) - count_tokens(
            "\n\n".join(f"### {s.header}\n{s.text}" for s in record.sections))
        # The ledger and the actions are never condensed; if they alone fill
        # the ceiling, the ledger wins and the sections halve.
        target = max(settings.record_ceiling - other - 200, current // 2)
        if target >= current:
            return record
        try:
            output = await self._call_json(
                ROLLUP_SYSTEM, ROLLUP_USER.format(current=current, target=target, sections=sections_text),
                max_tokens=min(16_000, target * 2 + 2_000), result=result,
            )
        except Exception as exc:
            logger.warning("compaction roll-up failed, keeping the record over its ceiling: %r", exc)
            return record
        rolled = record.copy()
        for change in output.get("changed_sections") or []:
            section = rolled.section_by_id(str((change or {}).get("id") or ""))
            if section is not None and str(change.get("text") or "").strip():
                section.text = str(change["text"]).strip()
        if count_tokens(render_document(rolled)) >= count_tokens(render_document(record)):
            return record
        # A roll-up far below its target dropped current material with the
        # settled (seen: 5,100 tokens asked down to 4,000, written as 1,300,
        # and half the record's answers gone). Keep the record over its
        # ceiling instead: the ceiling is a soft one.
        after = count_tokens("\n\n".join(f"[{s.id}] {s.header}\n{s.text}" for s in rolled.sections))
        if after < ROLLUP_FLOOR * target:
            logger.warning("compaction roll-up over-condensed (%d tokens for a %d target), keeping the record",
                           after, target)
            return record
        result.rollup = True
        return rolled


# =============================================================================
# helpers
# =============================================================================

def _fallback_plan(record: CompactionRecord) -> None:
    """The writer returned no plan: use the fixed per-use-case table."""
    from ..passes.section_plans import get_section_plan

    plan = get_section_plan(record.use_case)
    notes = MergeNotes()
    for header in plan["factual_sections"] + plan["conversational_sections"]:
        _add_section(record, header, "", notes)
    logger.info("compaction: writer returned no plan; using the %s fallback headers", record.use_case)


def _count(value: Any) -> int:
    """A provider's token count, or 0 when it reported none."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _parse_json(content: str) -> Any:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text.strip())


CONTEXT_WORDS = 24
TRIM_CONTEXT_WORDS = 8


def _table_caption(value: str, text: str) -> str:
    """For a value in a table row, the line just above the table ("Schedule B
    - Fees (Rev. 3)"), cut to ten words: a row copied from a revised document
    then reads as what that document says, not as the agreed term (20 Oct:
    a wrong 16,000 minimum from Corvane's revised schedule sat in the record
    next to the agreed 15,000 with nothing to tell them apart)."""
    lines = (text or "").splitlines()
    at = next((i for i, line in enumerate(lines) if value in line and line.strip().startswith("|")), None)
    if at is None:
        return ""
    i = at
    while i > 0 and lines[i - 1].strip().startswith("|"):
        i -= 1
    above = next((lines[j].strip() for j in range(i - 1, -1, -1) if lines[j].strip()), "")
    words = above.strip("#*: ").split()
    return " ".join(words[:10]) + (" ..." if len(words) > 10 else "") if words else ""


def _context(value: str, text: str, size: int = CONTEXT_WORDS) -> str:
    """The sentence a value was said in, verbatim and cut to about ``size``
    words around it: a value code adds keeps what it was. A table row's value
    is prefixed with the table's caption, in brackets."""
    caption = _table_caption(value, text)
    found = _sentence_context(value, text, size)
    return f"[{caption}] {found}" if caption and found else found


def _sentence_context(value: str, text: str, size: int) -> str:
    for sentence in split_sentences(text):
        if value not in sentence:
            continue
        words = sentence.split()
        if len(words) <= size:
            return sentence.strip()
        at = next((i for i, w in enumerate(words) if value.split()[0] in w), 0)
        start = max(0, min(at - size // 2, len(words) - size))
        return ("..." if start else "") + " ".join(words[start:start + size]) + "..."
    return ""


_SUFFIX = r"(?:[MmKkBb]n?|bn)\b"


def _complete_suffixes(record: CompactionRecord, turns: Dict[str, str]) -> None:
    """An amount cut before its unit ("$2" or "$2.01" where the turn says
    "$2.4M" or "$2.01M") gets the rest back from its turn: without it the
    record says two dollars."""
    for entry in record.ledger:
        text = turns.get(entry.m, "")
        if not text or not re.match(r"^[$€£¥₹]?\d[\d,.]*$", entry.value):
            continue
        whole = re.search(re.escape(entry.value) + r"(?:[.,]\d+)?\s?" + _SUFFIX, text)
        if whole and not re.search(re.escape(entry.value) + r"(?![\d.,]*\s?" + _SUFFIX + ")", text):
            entry.value = whole.group(0)


def _say_unlabelled(record: CompactionRecord, turns: Dict[str, str]) -> None:
    """A value kept without a label gets the sentence it was said in."""
    for entry in record.ledger:
        if entry.auto and entry.label.startswith(UNLABELLED) and entry.m in turns:
            context = _context(entry.value, turns[entry.m])
            if context:
                entry.label = context


def _record_view(record: CompactionRecord) -> str:
    if not record.planned and not record.ledger and not record.actions:
        return "(empty: this is the first compaction, so choose the plan)"
    view = {
        "use_case": record.use_case,
        "sections": [{"id": s.id, "header": s.header, "text": s.text} for s in record.sections],
        "ledger": [
            {"label": e.label, "value": e.value, "m": e.m,
             **({"was": [h.value for h in e.history]} if e.history else {})}
            for e in record.ledger
        ],
        "agreed_actions": [
            {"id": a.id, "who": a.who, "what": a.what, "status": a.status} for a in record.actions
        ],
    }
    return json.dumps(view, ensure_ascii=False, indent=1)


def _recap_block(recaps: List[str]) -> str:
    if not recaps:
        return ""
    lines = "\n".join(f"- {r}" for r in recaps)
    return f"EARLIER CHUNKS OF THIS UPDATE (already folded into the record; context only)\n{lines}\n\n"


def _trim_recaps(recaps: List[str]) -> List[str]:
    """Keep the recap block near 400 tokens by merging the oldest lines."""
    while len(recaps) > 1 and count_tokens("\n".join(recaps)) > RECAP_TOKEN_LIMIT:
        merged = "; ".join(recaps[:2])
        words = merged.split()
        recaps = [" ".join(words[:60])] + recaps[2:]
    return recaps


def _density_guidance(rates: Dict[str, float]) -> str:
    low = max(rates.get("low_density", 0.10), 0.01)
    return DENSITY_GUIDANCE.format(
        high=rates.get("high_density", 0.35) / low, medium=rates.get("medium_density", 0.25) / low
    )


def build_guidance(config) -> str:
    """MACA priorities and the use-case template's rules, as prompt text (I-18)."""
    lines = []
    if getattr(config, "instance_category", None):
        lines.append(f"Agent category: {config.instance_category}")
    if getattr(config, "preservation_priorities", None):
        lines.append("Always keep: " + ", ".join(p.replace("_", " ") for p in config.preservation_priorities))
    if getattr(config, "summarization_candidates", None):
        lines.append("Condense hard: " + ", ".join(p.replace("_", " ") for p in config.summarization_candidates))
    template = getattr(config, "template", None)
    if template is not None:
        never = (template.preservation_rules or {}).get("never_compress") or []
        if never:
            lines.append("Never condense: " + ", ".join(n.replace("_", " ") for n in never))
        guidelines = template.summarization_guidelines or {}
        for key, title in (("focus", "Focus on"), ("preserve", "Preserve"), ("compress", "Compress")):
            if guidelines.get(key):
                lines.append(f"{title}: " + "; ".join(guidelines[key]))
        if template.special_handling:
            lines.append("Also: " + "; ".join(template.special_handling))
    return "\n".join(lines) or "(none)"
