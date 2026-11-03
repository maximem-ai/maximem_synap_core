"""Statuses carried from chunk to chunk by the writer (20 Oct 2026 experiment).

A line written from an early chunk ("the initial term is five years") stays
in the record after a later chunk changes it, unless the writer rewrites the
whole section, which it rarely does: on 20 Oct, 54 of 74 stale lines the
claims judge found were writer lines never corrected. Code finding the
change itself (apply_stated_changes, track_statuses) catches only the
phrasings it knows.

Here every line and value of the current record gets an id in the writer's
view (L1, L2, ... and V1, V2, ...), and the writer returns a status for them:

  restate  every id, changed or not, so each chunk restates the whole set
           and the last chunk's set is the one kept
  delta    only the ids the new turns change

A status other than "current" needs the turn that says so and a quote
copied from it; code checks the quote and ignores an update it cannot find.
The writer's statuses are the only ones: in these modes code does not write
statuses of its own (no apply_stated_changes, track_statuses or "later
changed to" notes).

Ids live for one writer call only. What is kept is the status, keyed by the
line's text (record.line_states) or on the ledger entry (entry.stale).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .prompts import statuses as stp
from .record import CompactionRecord, apply_ledger_op, normalise, status_key

RESTATE = "restate"
DELTA = "delta"
BLOCK = "block"
MODES = ("off", RESTATE, DELTA, BLOCK)
LINE_MODES = (RESTATE, DELTA)  # statuses on the record's own lines, by id
STATES = ("current", "done", "replaced", "declined", "dropped")
RETIRED = ("replaced", "declined", "dropped")
MIN_QUOTE_WORDS = 3

# (kind, where): ("line", status key) or ("value", index into record.ledger)
Ids = Dict[str, Tuple[str, Any]]


def status_view(record: CompactionRecord) -> Tuple[str, Ids]:
    """The current record for the writer, every live line and value with an
    id and its status so far. Retired lines are listed without an id."""
    if not record.planned and not record.ledger and not record.actions:
        return "(empty: this is the first compaction, so choose the plan)", {}
    ids: Ids = {}
    sections, retired = [], []
    n = 0
    for section in record.sections:
        lines = []
        for line in section.text.splitlines():
            if not line.strip():
                continue
            key = status_key(line)
            state = record.line_states.get(key, {})
            if state.get("state") in RETIRED:
                retired.append({"text": line.strip(), "status": state["state"], "now": state.get("now", "")})
                continue
            n += 1
            ids[f"L{n}"] = ("line", key)
            entry = {"id": f"L{n}", "text": line.strip()}
            if state.get("state") == "done":
                entry["status"] = "done"
            lines.append(entry)
        sections.append({"id": section.id, "header": section.header, "lines": lines})
    ledger = []
    for i, e in enumerate(record.ledger):
        if e.stale.get("state") in RETIRED:
            retired.append({"text": f"{e.label}: {e.value}", "status": e.stale["state"], "now": e.stale.get("now", "")})
            continue
        ids[f"V{i + 1}"] = ("value", i)
        ledger.append({"id": f"V{i + 1}", "label": e.label, "value": e.value, "m": e.m,
                       **({"was": [h.value for h in e.history]} if e.history else {}),
                       **({"status": e.stale["state"]} if e.stale.get("state") else {})})
    view = {
        "use_case": record.use_case,
        "sections": sections,
        "ledger": ledger,
        "agreed_actions": [{"id": a.id, "who": a.who, "what": a.what, "status": a.status} for a in record.actions],
        **({"no_longer_current": retired} if retired else {}),
    }
    return json.dumps(view, ensure_ascii=False, indent=1), ids


def writer_system(base: str, mode: str) -> str:
    """The writer's system prompt with the status step: the rules go just
    before the output shape and "statuses" is the shape's first field.
    Appended after the shape instead (first try, 20 Oct), gpt-4o left the
    field out of 23 of 24 outputs."""
    assert base.count(stp.SHAPE) == 1, "the writer prompt's output shape moved"
    step = f"{stp.STEP_PREFIX}{stp.RULES[mode]}\n{stp.STATUS_DEFINITIONS}\n\n"
    return base.replace(stp.SHAPE, step + stp.SHAPE + stp.OUTPUT_FIELD + "\n")


def apply_statuses(record: CompactionRecord, output: Dict[str, Any], ids: Ids, chunk_turns: Dict[str, str],
                   timestamps: Dict[str, str]) -> Dict[str, int]:
    """Apply the writer's statuses to ``record`` in place. Returns counts:
    applied, ignored (unknown id, no quote found, a turn outside this chunk)."""
    counts = {"applied": 0, "ignored": 0, "listed": 0}
    retire: List[int] = []
    for raw in output.get("statuses") or []:
        if not isinstance(raw, dict):
            continue
        counts["listed"] += 1
        sid = str(raw.get("id") or "").strip()
        state = str(raw.get("status") or "").strip().lower()
        if sid not in ids or state not in STATES or state == "current":
            continue
        m = str(raw.get("m") or "").strip()
        quote = " ".join(str(raw.get("quote") or "").split())
        now = " ".join(str(raw.get("now") or "").split())
        kind, where = ids[sid]
        if kind == "line" and record.line_states.get(where, {}).get("state") == state:
            continue  # restated, already held
        if kind == "value" and record.ledger[where].stale.get("state") == state:
            continue
        if (m not in chunk_turns or len(quote.split()) < MIN_QUOTE_WORDS
                or normalise(quote) not in normalise(chunk_turns[m]) or (state == "replaced" and not now)):
            counts["ignored"] += 1
            continue
        counts["applied"] += 1
        if kind == "line":
            record.line_states[where] = {"state": state, "now": now, "m": m, "quote": quote}
            continue
        entry = record.ledger[where]
        if entry.auto and state in RETIRED:
            retire.append(where)  # a value code kept; the raw turn stays searchable
        elif state == "replaced" and normalise(now) in normalise(chunk_turns[m]) and len(now.split()) <= 8:
            apply_ledger_op(record, {"label": entry.label, "value": now, "m": m, "replaces": entry.value},
                            timestamps)
        else:
            entry.stale = {"state": state, "now": now, "m": m, "quote": quote}
    for i in sorted(retire, reverse=True):
        del record.ledger[i]
    # A line a correction rewrote takes its status with it.
    live = {status_key(line) for s in record.sections for line in s.text.splitlines() if line.strip()}
    record.line_states = {k: v for k, v in record.line_states.items() if k in live}
    return counts


# -- the "current status" block (BLOCK mode) ---------------------------------
# One block of where things stand, handed to the writer with
# each chunk and returned whole, updated; no ids, no turn numbers. The record's
# lines and code's own statuses are untouched: the block is added on top.

def block_system(base: str) -> str:
    assert base.count(stp.SHAPE) == 1, "the writer prompt's output shape moved"
    return base.replace(stp.SHAPE, stp.BLOCK_RULES + "\n\n" + stp.SHAPE + stp.BLOCK_FIELD + "\n")


def block_view(view: str, record: CompactionRecord) -> str:
    lines = "\n".join(f"- {line}" for line in record.status_block) or "(none yet)"
    return f"{view}\n\nCURRENT STATUS\n{lines}"


def apply_block(record: CompactionRecord, output: Dict[str, Any]) -> int:
    """The writer's updated block replaces the old one; an empty answer keeps
    the old one (a dropped field must not erase it). Returns the line count."""
    raw = output.get("current_status")
    lines = [" ".join(str(x).split()).lstrip("-• ").strip() for x in raw] if isinstance(raw, list) else []
    lines = [x for x in lines if x]
    if lines:
        record.status_block = lines
    return len(lines)
