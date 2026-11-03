"""What the agent receives: one server-rendered block.

REST, retrieval and the SDK all get the same text, so the framing lives here
and not in a formatter on one side of the wire. The block is printed once;
the raw turns after the compaction boundary follow it as normal turns.

Message citations ("[m12]") stay in the stored record, where the checks use
them, and are stripped here: the agent cannot resolve a position label.
Turn numbers in the header and the "changed at" notes are positions in the
conversation, which an agent can read.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .prompts import steering as stg
from .prompts.framing import FRAMING
from .record import (
    AGREED_ACTIONS_HEADER,
    KEY_VALUES_HEADER,
    STATUS_HEADERS,
    STEERING,
    CompactionRecord,
    normalise,
    status_key,
)

_CITATION = re.compile(r"\s*\[(?:m\d+(?:\s*[-–,]\s*m?\d+)*)(?:\s*;\s*m\d+(?:\s*[-–,]\s*m?\d+)*)*\]")


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text or "")


def _day(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return f"{dt.day} {dt.strftime('%b %Y')}"


def _span(start: str, end: str) -> str:
    try:
        a = datetime.fromisoformat(start).astimezone(timezone.utc)
        b = datetime.fromisoformat(end).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return ""
    if a.date() == b.date():
        return f"{a:%Y-%m-%d %H:%M}–{b:%H:%M} UTC"
    return f"{a:%Y-%m-%d %H:%M} – {b:%Y-%m-%d %H:%M} UTC"


def _turn(label: str) -> str:
    return f"turn {label[1:]}" if label.startswith("m") and label[1:].isdigit() else label


def _label_words(label: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z0-9]{3,}", normalise(label))
            if w not in {"the", "and", "for", "with", "from", "that", "this", "value", "user"}}


def _said_with_label(entry, prose_lines) -> bool:
    """One prose line holds the value and most words of its label."""
    value, words = normalise(entry.value), _label_words(entry.label)
    for line in prose_lines:
        if value in line:
            held = {w[:5] for w in re.findall(r"[a-z0-9]{3,}", line)}
            if not words or len(words & held) / len(words) >= 0.6:
                return True
    return False


def render_document(record: CompactionRecord) -> str:
    """The record as text, without the framing (``document``)."""
    if record.mode == STEERING:
        from .steering import render_steering

        return render_steering(record)
    parts = []
    end = _turn(str((record.covers or {}).get("end_m") or ""))
    retired: List[str] = []
    if record.status_block:
        parts.append(f"### {STATUS_BLOCK_HEADER}\n" + "\n".join(f"- {line}" for line in record.status_block))
    for section in record.sections:
        source = section.text
        if record.status_mode in ("restate", "delta"):
            source, gone = _with_line_states(source, record.line_states)
            retired += gone
        elif section.header in STATUS_HEADERS and record.statuses:
            source = "\n".join(_with_status(line, record.statuses, end) for line in source.splitlines())
        text = strip_citations(source).strip()
        if text:
            parts.append(f"### {section.header}\n{text}")
    prose_lines = [normalise(line) for section in record.sections for line in section.text.splitlines()]
    ledger = ledger_lines(record, prose_lines)
    if record.status_mode in ("restate", "delta"):
        retired += [_retired_value(e) for e in record.ledger if e.stale.get("state") in ("replaced", "declined", "dropped")]
    if retired:
        parts.append(f"### {NO_LONGER_CURRENT_HEADER}\n" + strip_citations("\n".join(retired)))
    if ledger:
        parts.append(f"### {KEY_VALUES_HEADER}\n" + "\n".join(ledger))
    parts.append(f"### {AGREED_ACTIONS_HEADER}\n" + "\n".join(action_lines(record) or ["None recorded."]))
    return "\n\n".join(parts)


STATUS_BLOCK_HEADER = "Where things stand now (as of the last turn covered; this overrides anything older below)"
NO_LONGER_CURRENT_HEADER = "No longer current (a later turn replaced, declined or dropped these; history only)"


def _with_line_states(text: str, states: Dict[str, Dict[str, str]]) -> tuple:
    """Status modes: done lines say so with the turn's words; replaced,
    declined and dropped lines leave their section for the history list."""
    kept, gone = [], []
    for line in text.splitlines():
        state = states.get(status_key(line)) if line.strip() else None
        if not state:
            kept.append(line)
            continue
        cite = re.search(r"\s*\[m\d+[^\]]*\]\s*$", line)
        body = (line[:cite.start()] if cite else line).rstrip().rstrip(".")
        turn = _turn(state.get("m", ""))
        if state["state"] == "done":
            kept.append(f'{body} (done, {turn}: "{state.get("quote", "")}").')
        elif state["state"] == "replaced":
            gone.append(f"{body} → now: {state.get('now', '')} ({turn}).")
        else:
            gone.append(f"{body} ({state['state']}, {turn}).")
    return "\n".join(kept), gone


def _retired_value(entry) -> str:
    stale = entry.stale
    after = f" → now: {stale['now']}" if stale.get("now") else f" ({stale['state']})"
    return f"- {entry.label}: {entry.value}{after} ({_turn(stale.get('m', ''))})."


def _with_status(line: str, statuses: Dict[str, Dict[str, str]], end: str) -> str:
    """The line with what the turns say of it since: done (with the words that
    say so) or, up to the last turn the record covers, not reported done."""
    entry = statuses.get(status_key(line)) if line.strip() else None
    if not entry:
        return line
    cite = re.search(r"\s*\[m\d+[^\]]*\]\s*$", line)
    body, tail = (line[:cite.start()], line[cite.start():]) if cite else (line, "")
    body = body.rstrip().rstrip(".")
    if entry.get("done_m"):
        return f'{body} (done, {_turn(entry["done_m"])}: "{entry["quote"]}").{tail}'
    # Only when two independent reads agree: the writer, which read every
    # later turn of its chunk, and code, which found no turn saying it
    # happened. Either alone was wrong on 19 Oct records ("rule out the
    # database first" was settled in words no lexical match links).
    if entry.get("writer") == "not_done" and end:
        return f"{body} (not reported done by {end}).{tail}"
    return line


def ledger_lines(record: CompactionRecord, prose_lines: List[str]) -> List[str]:
    """The ledger as lines: labelled values with their trail, then values
    code added with the sentence they were said in."""
    labelled = [e for e in record.ledger if not e.auto and e.stale.get("state") not in ("replaced", "declined", "dropped")]
    if not labelled and not any(e.auto for e in record.ledger):
        return []
    lines = []

    def shown(value: str, resolved: str) -> str:
        return f"{value} ({resolved})" if resolved else value

    for entry in labelled:
        # A value one line of prose already states, with what it is, is
        # not shown twice; it stays in the ledger and shows again if a
        # roll-up drops that line.
        if not entry.history and not entry.resolved and _said_with_label(entry, prose_lines):
            continue
        line = f"- {entry.label}: {shown(entry.value, entry.resolved)}"
        if entry.history:
            was = "; ".join(shown(h.value, h.resolved) for h in reversed(entry.history))
            when = ", ".join(x for x in (_turn(entry.m), _day(entry.at)) if x)
            line += f" (was {was}; changed at {when})" if when else f" (was {was})"
        lines.append(line)
    unlabelled = [e for e in record.ledger if e.auto]
    # A value code added keeps the sentence it was said in, verbatim.
    bare = [e for e in unlabelled if e.label.startswith("value from ")]
    said: Dict[tuple, List[str]] = {}
    for e in unlabelled:
        if e.history:
            # Linked: the same thing's values in order, the last one current.
            chain = " → ".join([h.value for h in e.history] + [e.value])
            lines.append(f'- {chain} (now {e.value}): "{e.label}"' + (f" ({_turn(e.m)})" if e.m else ""))
        elif e not in bare:
            said.setdefault((e.label, e.m), []).append(e.value)
    for (sentence, m), values in said.items():
        lines.append(f'- {", ".join(values)}: "{sentence}"' + (f" ({_turn(m)})" if m else ""))
    if bare:
        lines.append("- Also mentioned: " + ", ".join(
            f"{e.value} ({_turn(e.m)})" if e.m else e.value for e in bare))
    return lines


def action_lines(record: CompactionRecord) -> List[str]:
    lines = []
    for action in record.actions:
        quote = " / ".join(f"\"{q.text}\"" for q in action.quotes)
        line = f"- {action.who}: {action.what} ({action.status})"
        if quote:
            line += f": {quote}"
        lines.append(line)
    return lines


def budgeted_tokens(record: CompactionRecord) -> int:
    """Tokens the size check counts: sections and agreed actions, not the
    ledger (steering: everything but the ledger)."""
    from synap.cloud.shared.tokens import count_tokens

    if record.mode == STEERING:
        from .steering import render_steering

        return count_tokens(render_steering(record, with_ledger=False))

    sections = "\n\n".join(f"### {s.header}\n{strip_citations(s.text)}" for s in record.sections)
    actions = "\n".join(f"- {a.who}: {a.what} ({a.status}): " + " / ".join(q.text for q in a.quotes)
                         for a in record.actions)
    return count_tokens(sections) + count_tokens(actions)


def render_context_block(record: CompactionRecord, document: Optional[str] = None) -> str:
    covers = record.covers or {}
    count = covers.get("message_count") or 0
    start, end = covers.get("start_m", ""), covers.get("end_m", "")
    turns = f"turns {start[1:]}–{end[1:]}" if start and end else f"{count} turns"
    span = _span(covers.get("start_timestamp", ""), covers.get("end_timestamp", ""))
    header = f"## Conversation so far (compacted: {turns}" + (f", {span}" if span else "") + ")"
    body = document if document is not None else render_document(record)
    framing = FRAMING + (" " + stg.STEERING_FRAMING if record.mode == STEERING else "")
    return f"{header}\n{framing}\n\n{body}"
