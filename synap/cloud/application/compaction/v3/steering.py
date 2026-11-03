"""Steering mode (COMPACTION_HYBRID_PLAN.md P3): state, not detail.

A record in steering mode keeps what an agent needs to continue the
conversation correctly:

    Where things stand     3-6 lines, rewritten whole by each update
    Open threads           open / planned / maybe, and recently settled ones
    Decisions in force     with their reasons
    Current values         the ledger, with the trail of changed values
    Agreed actions earlier verbatim quotes, as in fidelity mode
    In the history         topic names only

and leaves the rest to a search of the raw turns, which the SDK keeps in its
local store. Fidelity mode keeps everything and cannot fit 0.25-0.35 of a
long span (measured 19 Oct: item text fell from 30% of the input at 3K
tokens to 15% at 15K, so long spans lost detail anyway, unevenly).

Every status, decision and action must carry the verbatim words of the turn
that shows it, checked here by code; a claim no turn's words show is dropped,
not written. No LLM call besides the one writer call per chunk: switching
modes and the roll-up are mechanical.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from synap.cloud.shared.tokens import count_tokens

from .prompts import steering as stg
from .record import (
    AGREED_ACTIONS_HEADER,
    CLOSED_STATUSES,
    FIDELITY,
    STEERING,
    THREAD_STATUSES,
    CompactionRecord,
    Decision,
    StatusChange,
    Thread,
    normalise,
    split_sentences,
)

logger = logging.getLogger(__name__)

# Record + new turns at or above this switch a conversation to steering for
# good. Below it the fidelity writer keeps nearly every item (1K: 91% of
# probes judged correct); above it, item coverage fell with span length.
STEERING_FROM_TOKENS = 6_000
# One writer call writes about 2K tokens whatever its input (measured 19 Oct:
# a 15K chunk gave 1.2K), so steering folds in smaller chunks to reach
# 0.25-0.35 of the span. Accuracy before call count.
STEERING_CHUNK_TOKENS = 4_000
STANDING_LINES = 8
# Settled threads stay visible for this many turns after they close, so
# "has X been done?" is answered; older ones are left to search.
SETTLED_VISIBLE_TURNS = 120
MAX_TOPICS = 40

# Words that claim a status or an act. A standing line that uses one must
# share most of its content words with the turn it cites.
_CLAIM = re.compile(
    r"\b(?:agreed|accepted|declined|refused|rejected|decided|chose|chosen|booked|signed|sent|paid|confirmed|"
    r"approved|cancel\w*|completed|done|finished|scheduled|submitted|filed|delivered|received|will|won't|"
    r"promised|committed|settled|resolved)\b",
    re.IGNORECASE,
)
_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "was", "are", "be", "it",
         "that", "this", "at", "by", "as", "from", "user", "assistant", "has", "have", "had", "not", "their",
         "they", "them", "its", "his", "her", "so", "but", "if", "then", "now", "still"}


def choose_mode(record: Optional[CompactionRecord], span_tokens: int, enabled: bool = True,
                steering_from: int = STEERING_FROM_TOKENS) -> str:
    """Fidelity until record + new turns reach ``steering_from``; steering
    from then on, never back (a record that keeps flipping would lose detail
    on each switch and confuse a reader)."""
    if record is not None and record.mode == STEERING:
        return STEERING
    if not enabled:
        return FIDELITY
    from .render import budgeted_tokens

    held = budgeted_tokens(record) if record is not None else 0
    return STEERING if held + span_tokens >= steering_from else FIDELITY


_CITED = re.compile(r"\[(m\d+)[^\]]*\]\s*$")


def _line_and_m(line: str) -> tuple:
    text = line.strip().lstrip("-• ").strip()
    found = _CITED.search(text)
    m = found.group(1) if found else ""
    text = _CITED.sub("", text).strip().rstrip(".")
    return text, m


def switch_to_steering(record: CompactionRecord) -> str:
    """Turn a fidelity record into a steering one, mechanically: decisions,
    open items, plans and conditions become decisions and threads; the
    ledger and agreed actions carry over unchanged. Returns the fidelity
    prose, which the next writer call sees once (for the standing lines)
    and which is then dropped: those turns stay searchable."""
    carried = "\n\n".join(f"### {s.header}\n{s.text}" for s in record.sections if s.text.strip())
    for section in record.sections:
        header = section.header.lower()
        for raw in section.text.splitlines():
            text, m = _line_and_m(raw)
            if not text:
                continue
            if header.startswith("decisions"):
                record.decisions.append(Decision(id=record.next_id("d", record.decisions), text=text, m=m))
            elif header.startswith("open items") or header.startswith("conditions"):
                record.threads.append(Thread(id=record.next_id("t", record.threads), item=text, m=m))
            elif header.startswith("stated plans"):
                record.threads.append(Thread(id=record.next_id("t", record.threads), item=text, status="planned", m=m))
    record.sections = []
    record.mode = STEERING
    return carried


def steering_view(record: CompactionRecord, carried: str = "") -> str:
    import json

    if not (record.standing or record.threads or record.decisions or record.ledger or record.actions or carried):
        return "(empty: this is the first compaction)"
    view: Dict[str, Any] = {
        "standing": record.standing,
        "threads": [{"id": t.id, "item": t.item, "status": t.status, "owner": t.owner, "m": t.m}
                    for t in record.threads if t.status not in CLOSED_STATUSES],
        "decisions": [{"id": d.id, "text": d.text, "reason": d.reason, "m": d.m} for d in record.decisions],
        "values": [{"label": e.label, "value": e.value, "m": e.m} for e in record.ledger if not e.auto],
        "agreed_actions": [{"id": a.id, "who": a.who, "what": a.what, "status": a.status} for a in record.actions],
        "topics": record.topics,
    }
    text = json.dumps(view, ensure_ascii=False, indent=1)
    if carried:
        text += stg.CARRIED_HEADER + carried
    return text


def _content(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", normalise(text)) if w not in _STOP and len(w) > 2}


def _quote_turn(quote: str, cited: str, turns: Dict[str, str]) -> str:
    """The turn holding ``quote`` verbatim (the cited one first), or ''."""
    q = normalise(quote)
    if len(q) < 4:
        return ""
    if cited in turns and q in normalise(turns[cited]):
        return cited
    return next((label for label, text in turns.items() if q in normalise(text)), "")


# Words that show a closed status in a turn: a "done" or "declined" needs
# them in its quote, recovered or not.
_SHOWS = {
    "done": re.compile(r"\b(?:done|booked|sent|paid|signed|delivered|confirmed|finished|completed|submitted|"
                       r"filed|received|went through|sorted|arrived|approved|accepted|issued|transferred|sold|"
                       r"moved|enrolled|registered|bought|ordered|cancel\w*|closed|settled|decided|"
                       r"(?:have|has|had|just|already)\s+\w+ed)\b", re.IGNORECASE),
    "declined": re.compile(r"\b(?:no|not|declin\w*|won't|will not|rather not|against|pass|skip|refus\w*|"
                           r"don't|do not|instead|turn\w* down|rul\w* out)\b", re.IGNORECASE),
    "replaced": re.compile(r"\b(?:instead|replac\w*|switch\w*|chang\w*|mov\w*|now|new|rather|"
                           r"reschedul\w*|swap\w*)\b", re.IGNORECASE),
}
RECOVER_SHARE = 0.5
# A standing line that claims an act: share of its content words found in
# the turns it cites (and two either side). "The user signed the Harbor Loft
# contract" next to turns that only name the Loft holds 2 of 4.
STANDING_SHARE = 0.6
# A thread that is only open or planned claims less than one that says it is
# done: the turn that raised it is evidence enough when it holds a third of
# the item's words in one sentence.
RAISED_SHARE = 0.34


def recover_quote(claim: str, status: str, cited: str, turns: Dict[str, str],
                  share: float = RECOVER_SHARE) -> tuple:
    """(m, sentence) of the cited turn that best carries ``claim``, when the
    writer's quote was not verbatim (it often paraphrases). The sentence
    must hold at least half the claim's content words and, for a closed
    status, words that show it. Code copies the sentence, so it cannot be
    invented; ('', '') when no sentence qualifies."""
    if cited not in turns:
        return "", ""
    words = _content(claim)
    if not words:
        return "", ""
    best, top = "", 0.0
    for sentence in split_sentences(turns[cited]):
        got = len(words & _content(sentence)) / len(words)
        if got > top and (status not in _SHOWS or _SHOWS[status].search(sentence)):
            best, top = sentence, got
    return (cited, best.strip()) if top >= share else ("", "")


def _supported(line: str, cited: List[str], turns: Dict[str, str]) -> bool:
    """A standing line that claims an act shares most of its content words
    with the turns it cites and the two turns either side of each."""
    words = _content(line)
    if not words:
        return False
    source = set()
    for m in cited:
        n = int(m[1:]) if m[1:].isdigit() else -10
        for k in range(n - 2, n + 3):
            source |= _content(turns.get(f"m{k}", ""))
    return len(words & source) / len(words) >= STANDING_SHARE


def merge_steering(record: CompactionRecord, output: Dict[str, Any], turns: Dict[str, str],
                   known_turns: Dict[str, str], timestamps: Dict[str, str]) -> List[str]:
    """Fold one steering output into ``record``. Returns what was dropped
    and why. ``turns``: this chunk's turns; ``known_turns``: all turns seen,
    for a quote of an earlier turn."""
    from .record import apply_action, apply_ledger_op

    dropped: List[str] = []
    every = {**known_turns, **turns}

    # Where things stand: rewritten whole; a line that claims an act must
    # match the turn it cites.
    lines = []
    for entry in (output.get("standing") or [])[:STANDING_LINES]:
        line = str(entry.get("line") if isinstance(entry, dict) else entry or "").strip()
        cited = entry.get("m") if isinstance(entry, dict) else []
        cited = [str(x) for x in (cited if isinstance(cited, list) else [cited]) if x]
        if not line:
            continue
        if _CLAIM.search(line) and not _supported(line, cited, every):
            dropped.append(f"standing (claim not in {', '.join(cited) or 'any turn'}): {line}")
            continue
        lines.append(line)
    if lines:
        record.standing = "\n".join(lines)

    # Threads: every new thread and every status change carries a quote.
    events = []
    for raw in list(output.get("events") or []) + list(output.get("threads") or []):
        if isinstance(raw, dict):
            thread = str(raw.get("thread") or raw.get("id") or "").strip()
            events.append({**raw, "id": "" if thread.lower() in ("", "new") else thread})
    for raw in events:
        status = str(raw.get("status") or "open").lower().strip()
        status = status if status in THREAD_STATUSES else "open"
        quote, cited = str(raw.get("quote") or "").strip(), str(raw.get("m") or "").strip()
        m = _quote_turn(quote, cited, every)
        existing = record.thread_by_id(str(raw.get("id") or ""))
        if not m:
            m, quote = recover_quote(f"{raw.get('item') or (existing.item if existing else '')} {quote}",
                                     status, cited, every,
                                     share=RAISED_SHARE if status in ("open", "planned", "maybe") else RECOVER_SHARE)
        if not m:
            dropped.append(f"thread (quote not found): {raw.get('item') or raw.get('id')}: {raw.get('quote')}")
            continue
        if existing is not None:
            if existing.status != status:
                existing.history.append(StatusChange(status=existing.status, m=existing.m, quote=existing.quote))
                existing.status, existing.m, existing.quote = status, m, quote
            if raw.get("item"):
                existing.item = str(raw["item"]).strip()
            if raw.get("owner"):
                existing.owner = str(raw["owner"]).strip()
            continue
        item = str(raw.get("item") or "").strip()
        if not item:
            continue
        same = next((t for t in record.threads if normalise(t.item) == normalise(item)), None)
        if same is not None:
            continue
        record.threads.append(Thread(id=record.next_id("t", record.threads), item=item, status=status,
                                     owner=str(raw.get("owner") or "").strip(), m=m, quote=quote))

    # Decisions in force.
    for raw in output.get("decisions") or []:
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            continue
        quote = str(raw.get("quote") or "").strip()
        m = _quote_turn(quote, str(raw.get("m") or ""), every)
        if not m:
            m, quote = recover_quote(f"{raw.get('text')} {quote}", "", str(raw.get("m") or ""), every)
        if not m:
            dropped.append(f"decision (quote not found): {raw.get('text')}: {raw.get('quote')}")
            continue
        replaces = str(raw.get("replaces") or "").strip()
        if replaces:
            record.decisions = [d for d in record.decisions if d.id != replaces]
        text = str(raw["text"]).strip()
        if any(normalise(d.text) == normalise(text) for d in record.decisions):
            continue
        record.decisions.append(Decision(id=record.next_id("d", record.decisions), text=text,
                                         reason=str(raw.get("reason") or "").strip(), m=m, quote=quote))

    # Values and agreed actions: the fidelity checks and merge, unchanged.
    from .checks import check_quotes

    problems = check_quotes(output, every)
    bad_actions = {p.ref.get("action_index") for p in problems if "action_index" in p.ref}
    bad_values = {p.ref.get("value") for p in problems if "value" in p.ref}
    written: set = set()
    for op in output.get("literals") or []:
        if isinstance(op, dict) and op.get("value") in bad_values:
            dropped.append(f"value (not in its turn): {op.get('label')}: {op.get('value')}")
            continue
        apply_ledger_op(record, op, timestamps, written)
    for i, action in enumerate(output.get("agreed_actions") or []):
        if i in bad_actions:
            dropped.append(f"action (quote not found): {action.get('who')}: {action.get('what')}")
            continue
        apply_action(record, action, timestamps)

    for topic in output.get("topics") or []:
        topic = " ".join(str(topic).split())
        if topic and all(normalise(topic) != normalise(t) for t in record.topics):
            record.topics.append(topic)
    return dropped


def _turn_no(m: str) -> int:
    return int(m[1:]) if m[1:].isdigit() else 0


def roll_up_steering(record: CompactionRecord, ceiling: int, last_turn: int) -> bool:
    """Mechanical: drop settled threads that closed long ago, then the
    oldest topics, until the record fits. Open threads, decisions in force,
    the ledger and agreed actions are never dropped here (the ledger wins
    over the ceiling, as in fidelity mode). Returns whether anything went."""
    before = len(record.threads), len(record.topics)
    record.threads = [t for t in record.threads
                      if t.status not in CLOSED_STATUSES or last_turn - _turn_no(t.m) <= SETTLED_VISIBLE_TURNS]
    record.topics = record.topics[-MAX_TOPICS:]
    while count_tokens(render_steering(record)) > ceiling:
        closed = [t for t in record.threads if t.status in CLOSED_STATUSES]
        if closed:
            record.threads.remove(min(closed, key=lambda t: _turn_no(t.m)))
        elif len(record.topics) > 10:
            record.topics = record.topics[len(record.topics) // 4:]
        else:
            break
    return (len(record.threads), len(record.topics)) != before


def _turn(m: str) -> str:
    return f"turn {m[1:]}" if m[1:].isdigit() else m


def render_steering(record: CompactionRecord, with_ledger: bool = True) -> str:
    from .render import KEY_VALUES_HEADER, action_lines, ledger_lines

    parts = []
    if record.standing.strip():
        parts.append("### Where things stand\n" + "\n".join(
            f"- {line.strip().lstrip('-• ')}" for line in record.standing.splitlines() if line.strip()))
    live = [t for t in record.threads if t.status not in CLOSED_STATUSES]
    settled = [t for t in record.threads if t.status in CLOSED_STATUSES]

    def thread_line(t: Thread) -> str:
        who = f", {t.owner}" if t.owner else ""
        when = f" ({_turn(t.m)})" if t.m else ""
        return f"- {t.item} — {t.status}{who}{when}"

    if live:
        parts.append("### Open threads\n" + "\n".join(thread_line(t) for t in live))
    if settled:
        parts.append("### Settled recently\n" + "\n".join(thread_line(t) for t in settled))
    if record.decisions:
        parts.append("### Decisions in force\n" + "\n".join(
            f"- {d.text}" + (f" — {d.reason}" if d.reason else "") + (f" ({_turn(d.m)})" if d.m else "")
            for d in record.decisions))
    if with_ledger:
        ledger = ledger_lines(record, [])
        if ledger:
            parts.append(f"### {KEY_VALUES_HEADER}\n" + "\n".join(ledger))
    parts.append(f"### {AGREED_ACTIONS_HEADER}\n" + "\n".join(action_lines(record) or ["None recorded."]))
    if record.topics:
        parts.append("### In the earlier turns\n" + "; ".join(record.topics))
    return "\n\n".join(parts)

