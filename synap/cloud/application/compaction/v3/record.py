"""The stored compaction record and the code that merges into it.

One record per conversation, updated in place by each compaction:

  plan       use case + section headers, chosen once and then frozen
  sections   prose per header, with stable ids; the model rewrites only the
             ones that changed and code copies the rest byte for byte
  ledger     literal values (IDs, amounts, dates, names...). Additive: an
             entry is never deleted, and a replaced value stays next to its
             replacement
  actions    "Agreed actions earlier": things someone stated or agreed to,
             credited to whoever made them, with verbatim quotes
  density    one score per covered stretch of turns
  covers     which turns the record covers

Messages are referred to by position, "m1" for the first turn of the
conversation. Positions are stable while the history is unchanged, and the
sequence hash catches any change (an edit, a deletion, a backfill), which
triggers a rebuild that renumbers from scratch.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

RECORD_VERSION = 3

AGREED_ACTIONS_HEADER = "Agreed actions earlier"
# Label of a value kept without a label of its own, until the engine gives
# it the sentence it was said in.
UNLABELLED = "value from "
KEY_VALUES_HEADER = "Key values"

# Headers that describe what should happen rather than what happened (I-1).
# Matched against the header with any leading '#' and trailing punctuation
# removed, case-insensitively.
_BANNED_HEADER = re.compile(
    r"^(?:"
    r"next\b.*"                                  # Next Steps, Next Beats, Next Meeting Items
    r"|what'?s next\b.*|what comes next\b.*"
    r"|(?:open\s+|pending\s+)?action\s*items?\b.*"
    r"|action\s*plan\b.*|plan of action\b.*"
    r"|to-?\s?dos?\b.*"
    r"|follow[\s-]?ups?\b.*"
    r"|recommend\w*\b.*|suggest\w*\b.*|advice\b.*"
    r"|upcoming\s+(?:steps|actions)\b.*"
    r")$",
    re.IGNORECASE,
)


def clean_header(header: str) -> str:
    return re.sub(r"^#+\s*", "", (header or "").strip()).strip().rstrip(":").strip()


# The same phrases anywhere in a header: "Open questions and next steps" is
# a next-step header too.
_BANNED_PHRASE = re.compile(
    r"\b(?:next\s+steps?|next\s+actions?|what'?s\s+next|action\s*items?|to-?\s?dos?|follow[\s-]?ups?"
    r"|recommendations?)\b",
    re.IGNORECASE,
)


def is_banned_header(header: str) -> bool:
    name = clean_header(header)
    return bool(_BANNED_HEADER.match(name) or _BANNED_PHRASE.search(name))


def is_reserved_header(header: str) -> bool:
    name = clean_header(header).lower()
    return name in (AGREED_ACTIONS_HEADER.lower(), KEY_VALUES_HEADER.lower())


@dataclass
class Section:
    id: str
    header: str
    text: str = ""


@dataclass
class LedgerChange:
    value: str
    m: str  # message label the value was stated in, "m12"
    at: str = ""  # ISO timestamp of that message
    resolved: str = ""


@dataclass
class LedgerEntry:
    label: str
    value: str
    m: str
    at: str = ""
    # The absolute form of a relative or partial value ("the 30th" ->
    # "2026-10-30"), worked out by the writer. ``value`` stays verbatim so the
    # quote check can find it in the turn.
    resolved: str = ""
    # Earlier values, oldest first. Never trimmed: this is the one-line trail
    # for replaced values.
    history: List[LedgerChange] = field(default_factory=list)
    # Added by code because the writer left a literal out twice; shown apart
    # from the labelled values.
    auto: bool = False
    # Status modes (statuses.py): what a later turn said of this value when
    # it was not a plain replacement: {"state", "now", "m", "quote"}.
    stale: Dict[str, str] = field(default_factory=dict)
    # Value linking (engine link_values): what the value belongs to (a
    # table row's name, the words before it), so a later value of the same
    # thing replaces it.
    key: str = ""


@dataclass
class Quote:
    text: str
    m: str


@dataclass
class AgreedAction:
    id: str
    who: str
    what: str
    quotes: List[Quote] = field(default_factory=list)
    status: str = "open"  # open | done
    m: str = ""
    at: str = ""


@dataclass
class DensityMark:
    start: str  # first message label of the stretch
    end: str
    level: str  # high | medium | low


# Record modes (COMPACTION_HYBRID_PLAN.md P3). "fidelity" keeps every item of
# the turns it covers; "steering" keeps what an agent needs to steer (where
# things stand, open threads, decisions in force, current values, agreed
# actions) and leaves detail to a search of the raw turns, which the SDK
# keeps locally. A conversation moves from fidelity to steering once and
# never back.
FIDELITY = "fidelity"
STEERING = "steering"
THREAD_STATUSES = ("open", "planned", "maybe", "done", "declined", "replaced")
CLOSED_STATUSES = ("done", "declined", "replaced")


@dataclass
class StatusChange:
    status: str
    m: str
    quote: str = ""


@dataclass
class Thread:
    """Something still in play: a question, a pending task, a plan, a
    proposal awaiting an answer, a condition with a deadline. A status other
    than open/planned needs the verbatim words of the turn that set it."""
    id: str
    item: str
    status: str = "open"
    owner: str = ""
    m: str = ""
    quote: str = ""
    history: List[StatusChange] = field(default_factory=list)


@dataclass
class Decision:
    id: str
    text: str
    reason: str = ""
    m: str = ""
    quote: str = ""


@dataclass
class CompactionRecord:
    use_case: str = "general"
    mode: str = FIDELITY
    sections: List[Section] = field(default_factory=list)
    ledger: List[LedgerEntry] = field(default_factory=list)
    actions: List[AgreedAction] = field(default_factory=list)
    density: List[DensityMark] = field(default_factory=list)
    # {"start_m", "end_m", "message_count", "start_timestamp", "end_timestamp"}
    covers: Dict[str, Any] = field(default_factory=dict)
    version: int = RECORD_VERSION
    # Fidelity: the status of each plan, promise and open item line, keyed by
    # status_key(line): {"m": raised, "done_m": "", "quote": ""}. Code sets it
    # (track_statuses); the line itself is never rewritten.
    statuses: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Status modes (statuses.py, "restate" or "delta"; "" = off): the writer
    # carries every line's status from chunk to chunk, keyed by
    # status_key(line): {"state", "now", "m", "quote"}. Code then writes no
    # statuses of its own.
    status_mode: str = ""
    line_states: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # "block" mode: where things stand, rewritten whole by each writer call.
    status_block: List[str] = field(default_factory=list)
    # Steering mode only.
    standing: str = ""  # where things stand: rewritten whole by each update
    threads: List[Thread] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)  # what the history holds, names only

    # -- plan ---------------------------------------------------------------
    @property
    def planned(self) -> bool:
        return bool(self.sections)

    def section_by_id(self, section_id: str) -> Optional[Section]:
        return next((s for s in self.sections if s.id == section_id), None)

    def section_by_header(self, header: str) -> Optional[Section]:
        key = clean_header(header).lower()
        return next((s for s in self.sections if s.header.lower() == key), None)

    def next_section_id(self) -> str:
        used = {int(s.id[1:]) for s in self.sections if s.id[1:].isdigit()}
        return f"s{max(used, default=0) + 1}"

    def next_id(self, prefix: str, existing) -> str:
        used = {x.id for x in existing}
        n = len(used) + 1
        while f"{prefix}{n}" in used:
            n += 1
        return f"{prefix}{n}"

    def thread_by_id(self, thread_id: str) -> Optional[Thread]:
        return next((t for t in self.threads if t.id == thread_id), None)

    def next_action_id(self) -> str:
        used = {int(a.id[1:]) for a in self.actions if a.id[1:].isdigit()}
        return f"a{max(used, default=0) + 1}"

    def ledger_by_label(self, label: str) -> Optional[LedgerEntry]:
        key = _label_key(label)
        return next((e for e in self.ledger if _label_key(e.label) == key), None)

    def ledger_values(self) -> List[str]:
        values = []
        for entry in self.ledger:
            values.append(entry.value)
            values.extend(change.value for change in entry.history)
        return values

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["CompactionRecord"]:
        if not data or int(data.get("version", 0)) != RECORD_VERSION:
            return None
        return cls(
            use_case=data.get("use_case", "general"),
            sections=[Section(**s) for s in data.get("sections", [])],
            ledger=[
                LedgerEntry(
                    label=e["label"], value=e["value"], m=e.get("m", ""), at=e.get("at", ""),
                    resolved=e.get("resolved", ""),
                    history=[LedgerChange(**h) for h in e.get("history", [])],
                    auto=e.get("auto", False),
                    stale=dict(e.get("stale", {})),
                    key=e.get("key", ""),
                )
                for e in data.get("ledger", [])
            ],
            actions=[
                AgreedAction(
                    id=a["id"], who=a.get("who", ""), what=a.get("what", ""),
                    quotes=[Quote(**q) for q in a.get("quotes", [])],
                    status=a.get("status", "open"), m=a.get("m", ""), at=a.get("at", ""),
                )
                for a in data.get("actions", [])
            ],
            density=[DensityMark(**d) for d in data.get("density", [])],
            covers=data.get("covers", {}),
            mode=data.get("mode", FIDELITY),
            standing=data.get("standing", ""),
            threads=[
                Thread(**{**t, "history": [StatusChange(**h) for h in t.get("history", [])]})
                for t in data.get("threads", [])
            ],
            decisions=[Decision(**d) for d in data.get("decisions", [])],
            topics=list(data.get("topics", [])),
            statuses=dict(data.get("statuses", {})),
            status_mode=data.get("status_mode", ""),
            line_states=dict(data.get("line_states", {})),
            status_block=list(data.get("status_block", [])),
        )

    def copy(self) -> "CompactionRecord":
        return CompactionRecord.from_dict(self.to_dict())  # type: ignore[return-value]


def _label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()


# =============================================================================
# Merging one writer output into the record
# =============================================================================

@dataclass
class MergeNotes:
    """What the merge did that the checks and the logs should know about."""
    new_sections: List[str] = field(default_factory=list)
    rejected_headers: List[str] = field(default_factory=list)
    unknown_section_ids: List[str] = field(default_factory=list)


def merge_output(
    record: CompactionRecord,
    output: Dict[str, Any],
    timestamps: Dict[str, str],
) -> MergeNotes:
    """Apply a writer output to ``record`` in place.

    Sections the output does not name are untouched. Ledger and actions only
    grow or change status; nothing is removed. ``timestamps`` maps message
    labels to ISO timestamps.
    """
    notes = MergeNotes()

    if not record.planned:
        plan = output.get("plan") or {}
        record.use_case = str(plan.get("use_case") or output.get("use_case") or record.use_case or "general")
        for header in plan.get("headers") or []:
            _add_section(record, header, "", notes)

    for change in output.get("changed_sections") or []:
        if not isinstance(change, dict):
            continue
        text = str(change.get("text") or "").strip()
        section = None
        if change.get("id"):
            section = record.section_by_id(str(change["id"]))
            if section is None and not change.get("header"):
                notes.unknown_section_ids.append(str(change["id"]))
                continue
        if section is None and change.get("header"):
            section = record.section_by_header(str(change["header"]))
        if section is None:
            # A header the plan does not have yet: later updates may add one,
            # never rename or drop one.
            section = _add_section(record, str(change.get("header") or ""), text, notes)
            continue
        section.text = text

    add_inventory(record, inventory_items(output), notes)

    written: set = set()
    for op in list(output.get("literals") or []) + _superseded_as_ops(output.get("superseded")):
        apply_ledger_op(record, op, timestamps, written)

    for item in output.get("agreed_actions") or []:
        apply_action(record, item, timestamps)

    for mark in output.get("segment_density") or []:
        if isinstance(mark, dict) and mark.get("start") and mark.get("level"):
            level = str(mark["level"]).lower()
            record.density.append(DensityMark(
                start=str(mark["start"]), end=str(mark.get("end") or mark["start"]),
                level=level if level in ("high", "medium", "low") else "medium",
            ))
    return notes


def _add_section(record: CompactionRecord, header: str, text: str, notes: MergeNotes) -> Optional[Section]:
    name = clean_header(header)
    if not name:
        return None
    if is_banned_header(name) or is_reserved_header(name):
        notes.rejected_headers.append(name)
        return None
    if record.section_by_header(name):
        return record.section_by_header(name)
    section = Section(id=record.next_section_id(), header=name, text=text)
    record.sections.append(section)
    notes.new_sections.append(name)
    return section


# Where each inventory type is written. Values go to the ledger and
# commitments to the agreed actions, which check them; neither is repeated.
INVENTORY_HEADERS = {
    "fact": "Facts",
    "decision": "Decisions and reasons",
    "rejected": "Decisions and reasons",
    "condition": "Conditions and deadlines",
    "advice": "Proposals and responses",
    "suggestion": "Proposals and responses",
    "plan": "Stated plans",
    "open": "Open items",
    "feeling": "How the user feels",
}
_INVENTORY_ORDER = ["Facts", "Decisions and reasons", "Conditions and deadlines", "Proposals and responses",
                    "Stated plans", "Open items", "How the user feels"]
_ASKED = re.compile(r"\?\s*$|^(?:the\s+)?(?:user|assistant)\s+ask(?:s|ed)\b", re.IGNORECASE)
# How an item's "why" joins it, by type: a reason, a response, a consequence.
# A reason without its own connective is joined with a dash: code writing
# "because" asserted a cause the turn did not state ("stressed about the
# neighbor, because more than the money").
_WHY_JOIN = {"advice": "response:", "suggestion": "response:"}
# A rejected item says it was turned down; written bare under "Decisions and
# reasons" it read as the decision taken.
_REJECTION = re.compile(r"\b(?:reject\w*|turn\w* down|rul\w* out|declin\w*|decided against|chose not|not to|"
                        r"won'?t|will not|instead|dropp\w*|refus\w*|no to|passed on|skip\w*|against|"
                        r"not (?:go|take|use|do|pursue|buy|book|sign)\w*)\b", re.IGNORECASE)
_CONNECTIVE = re.compile(r"^(?:because|since|as|so|to|due|but|and|response|then|if|or|unless|in case)\b", re.IGNORECASE)


def inventory_items(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The writer's items as one flat list. The writer groups them by turn
    ({"m", "items"}), so a skipped turn is visible; an older output has a
    flat "inventory"."""
    items: List[Dict[str, Any]] = []
    for turn in output.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        for item in turn.get("items") or []:
            if isinstance(item, dict):
                items.append({**item, "m": str(item.get("m") or turn.get("m") or "")})
    items += [i for i in output.get("inventory") or [] if isinstance(i, dict)]
    return items


def item_text(item: Dict[str, Any]) -> str:
    """An item's line: its text, then its reason or response when it has one
    the text does not already say."""
    text = " ".join(str(item.get("item") or "").split()).rstrip(".")
    why = " ".join(str(item.get("why") or "").split()).rstrip(".")
    kind = str(item.get("type") or "").lower().strip()
    if text and kind == "rejected" and not _REJECTION.search(text):
        text = f"Turned down: {text}"
    if not text or not why or normalise(why) in normalise(text):
        return text
    if _CONNECTIVE.match(why):
        return f"{text}, {why}"
    join = None if item.get("why_is_reason") else _WHY_JOIN.get(kind)
    return f"{text}, {join} {why}" if join else f"{text} — {why}"


def add_inventory(record: CompactionRecord, inventory: Any, notes: MergeNotes) -> None:
    """Write the writer's inventory items into the record, one line each,
    under the header for their type. The inventory is the record of the new
    turns: nothing listed is lost in a rewrite."""
    if not isinstance(inventory, list):
        return
    for header in _INVENTORY_ORDER:
        lines = []
        for item in inventory:
            if not isinstance(item, dict):
                continue
            if INVENTORY_HEADERS.get(str(item.get("type") or "").lower().strip()) != header:
                continue
            text = item_text(item)
            if not text or _ASKED.search(text):
                continue  # a question asked: its answer is its own item
            m = str(item.get("m") or "").strip()
            lines.append((text, f"- {text}. [{m}]" if m else f"- {text}."))
            status = str(item.get("status") or "").lower().strip()
            if header in STATUS_HEADERS and status in ("done", "not_done", "unclear"):
                record.statuses[status_key(text)] = {"m": m, "done_m": "", "quote": "", "writer": status}
        # Anywhere in the record: an earlier record may say it under another header.
        held = normalise(" ".join(s.text for s in record.sections))
        new = [line for text, line in lines if normalise(text) not in held]
        if not new:
            continue
        section = record.section_by_header(header) or _add_section(record, header, "", notes)
        section.text = "\n".join(([section.text] if section.text else []) + new)


def _superseded_as_ops(items: Any) -> List[Dict[str, Any]]:
    """``superseded`` entries are ledger replacements under another name."""
    ops = []
    for item in items or []:
        if isinstance(item, dict) and item.get("label") and item.get("new"):
            ops.append({
                "label": item["label"], "value": item["new"],
                "m": item.get("m") or item.get("m_id"), "replaces": item.get("old"),
            })
    return ops


def apply_ledger_op(
    record: CompactionRecord,
    op: Dict[str, Any],
    timestamps: Dict[str, str],
    written: Optional[set] = None,
) -> None:
    if not isinstance(op, dict):
        return
    label = str(op.get("label") or "").strip()
    value = str(op.get("value") or "").strip()
    m = str(op.get("m") or op.get("m_id") or "").strip()
    if not label or not value:
        return
    if normalise(label) == normalise(value):
        # "This Agreement: This Agreement": a capitalised term the repair asked
        # for, labelled with itself; it says nothing (20 Oct, 37 such lines in
        # one documents run, every one counted as an invented claim).
        return
    at = timestamps.get(m, "")
    replaces = str(op.get("replaces") or "").strip()
    resolved = str(op.get("resolved") or "").strip()
    if _same_value(resolved, value) or not resolved_date_ok(value, resolved, at):
        resolved = ""

    entry = record.ledger_by_label(label)
    if entry is None and not replaces:
        # The same thing under another name, from an earlier turn: "Visa
        # appointment" then "VFS appointment time". Left alone, the old value
        # would read as current.
        entry = _renamed_entry(record, label, value, m)
        if entry is not None:
            label = entry.label
    if (entry is not None and written is not None and label in written and not replaces
            and not _same_value(entry.value, value)):
        # The same output already set this label, and this value does not say
        # it replaces anything: two items that got one name ("Commit" for
        # three commits in a release), not a change. Chaining them would
        # report the first as superseded by the second, a wrong fact.
        a, b = normalise(entry.value), normalise(value)
        if a in b or b in a:
            return  # "$3,050" and "3,050": one value, written twice
        # A numbered label ("Rental coverage per day (2): $61") states the
        # second value is the same kind of thing, which it was not (seen:
        # $61 was the rental's price, not the coverage). The value is kept
        # unlabelled; the engine gives it the sentence it was said in.
        if not any(_same_value(e.value, value) for e in record.ledger):
            record.ledger.append(LedgerEntry(label=f"{UNLABELLED}{m}", value=value, m=m, at=at, auto=True))
        return
    if written is not None:
        written.add(label)
    if entry is None and replaces:
        entry = next((e for e in record.ledger if _same_value(e.value, replaces)), None)
    if entry is None:
        # A value that replaced one stated earlier in the same turns (the
        # first compaction, or a value the ledger never held): the old value
        # is still the trail, with no turn of its own to point at.
        history = [LedgerChange(value=replaces, m="")] if replaces and not _same_value(replaces, value) else []
        record.ledger.append(LedgerEntry(label=label, value=value, m=m, at=at, resolved=resolved, history=history))
        return
    if _same_value(entry.value, value):
        if resolved and not entry.resolved:
            entry.resolved = resolved
        return
    if any(_same_value(h.value, value) for h in entry.history) and not replaces:
        # The writer re-reported an old value without saying it came back.
        return
    entry.history.append(LedgerChange(value=entry.value, m=entry.m, at=entry.at, resolved=entry.resolved))
    old_value = entry.value
    entry.value, entry.m, entry.at, entry.auto, entry.resolved = value, m, at, False, resolved
    _mark_changed_in_prose(record, entry.label, old_value, value, m)


_LABEL_STOP = {"the", "and", "for", "with", "from", "that", "this", "new", "old", "current", "updated", "revised",
               "original", "initial", "first", "final", "value", "user", "user's", "date", "time", "amount",
               "number", "cost", "price"}


def _label_words(label: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z0-9']{3,}", normalise(label)) if w not in _LABEL_STOP}


def _value_kind(value: str) -> str:
    v = value.strip()
    if re.search(r"\b\d{1,2}:\d{2}\b", v):
        return "time"
    if re.search(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{4}-\d{2}-\d{2}\b", v):
        return "date"
    if re.search(r"(?i)[$€£]|\b(?:cad|eur|usd|gbp)\b", v):
        return "money"
    if re.search(r"%", v):
        return "percent"
    if re.search(r"(?=.*[A-Za-z])(?=.*\d)^[A-Za-z0-9-]{4,}$", v):
        return "code"
    if re.fullmatch(r"[\d,.\s]+", v):
        return "number"
    return ""


def _turn_number(m: str) -> int:
    return int(m[1:]) if m[1:].isdigit() else -1


def _renamed_entry(record: CompactionRecord, label: str, value: str, m: str) -> Optional[LedgerEntry]:
    """An entry from an earlier turn whose label names the same thing (most
    of the shorter label's words) and whose value is the same kind."""
    kind, words = _value_kind(value), _label_words(label)
    if not kind or not words or _turn_number(m) < 0:
        return None
    best, best_share = None, 0.0
    for entry in record.ledger:
        if entry.auto or _same_value(entry.value, value) or _value_kind(entry.value) != kind:
            continue
        if not 0 <= _turn_number(entry.m) < _turn_number(m):
            continue
        other = _label_words(entry.label)
        if not other:
            continue
        share = len(words & other) / min(len(words), len(other))
        if share >= 0.75 and len(words & other) >= 2 and share > best_share:
            best, best_share = entry, share
    return best


_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bSt\.)(?<!\bNo\.)"
                           r"(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)\s+")


def split_sentences(text: str) -> List[str]:
    """Sentences, not split after "Dr.", "Mrs.", "St." and the like."""
    return [x.strip() for x in _SENTENCE_END.split(text or "") if x.strip()]


_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_AT_TIME = r"(?:,?\s+at\s+\d{1,2}:\d{2})?"
_VALUE_PATTERNS = {
    "time": rf"\b\d{{1,2}}\s+{_MONTHS}(?:\s+\d{{4}})?\s+at\s+\d{{1,2}}:\d{{2}}|\b{_MONTHS}\s+\d{{1,2}}\s+at\s+\d{{1,2}}:\d{{2}}|\b\d{{1,2}}:\d{{2}}\b",
    "date": rf"\b\d{{1,2}}\s+{_MONTHS}(?:\s+\d{{4}})?{_AT_TIME}|\b{_MONTHS}\s+\d{{1,2}}(?:,?\s+\d{{4}})?{_AT_TIME}",
    "money": r"(?:\b(?:CAD|EUR|USD|GBP)\s?|[$€£])\d[\d,]*(?:\.\d+)?",
    "percent": r"\b\d+(?:\.\d+)?\s?%",
    "code": r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{3,}\b",
}
_CHANGE = re.compile(r"\b(?:mov(?:ed|es|ing)|chang(?:ed|es)|reschedul\w*|push(?:ed|es)|switch(?:ed|es)|"
                     r"brought forward)\b", re.IGNORECASE)
_PAST_LABEL = re.compile(r"\b(?:original|initial|first|old|previous|earlier|former|prior)\b", re.IGNORECASE)
CHANGE_WINDOW = 12  # words between the change word, the label and the new value


def _words_at(text: str) -> List[tuple]:
    return [(m.start(), m.group(0)) for m in re.finditer(r"\S+", text)]


def apply_stated_changes(record: CompactionRecord, turns: Dict[str, str], timestamps: Dict[str, str]) -> None:
    """A later turn that states a value changed ("moved from 18 November at
    10:15 to 25 November at 09:40", "the exam moves to 21 January at 10:00")
    updates the ledger entry the writer left at the old value. The new value
    is copied from the turn; the old one stays in the entry's history.

    Either the turn says "from <old value>", or a movement word ("moved",
    "rescheduled") sits within CHANGE_WINDOW words of the entry's label words
    and the new value within CHANGE_WINDOW words after them. Summary words
    ("new", "updated") are not enough: "the new balance" is a different
    quantity, and "everything updated: ..." restates many."""
    for label_m, text in turns.items():
        turn = _turn_number(label_m)
        for sentence in split_sentences(text):
            change = _CHANGE.search(sentence)
            plain = normalise(sentence)
            words = _words_at(sentence)
            index = lambda pos: sum(1 for p, _ in words if p < pos)  # noqa: E731
            for entry in list(record.ledger):
                kind = _value_kind(entry.value)
                if entry.auto or kind not in _VALUE_PATTERNS or not 0 <= _turn_number(entry.m) < turn:
                    continue
                if _PAST_LABEL.search(entry.label):
                    continue  # "original budget" records the old value on purpose
                short = re.sub(r"(?:,?\s+\d{4}|,?\s+at\s+\d{1,2}:\d{2})+$", "", entry.value.strip())
                after = -1
                for old in {entry.value.strip(), short}:
                    hit = re.search(r"\bfrom\s+" + re.escape(old), sentence, re.IGNORECASE)
                    if hit:
                        after = max(after, hit.end())
                strict = after < 0
                if strict:
                    # "stays 15 January even though the start moved": the
                    # sentence restates this entry's value, so it is not
                    # this entry that changed.
                    if not change or any(normalise(v) in plain for v in {entry.value, short} if normalise(v)):
                        continue
                    label_words = _label_words(entry.label)
                    found_at = [i for i, (_, w) in enumerate(words) if normalise(w)[:5].strip(".,;:'") in label_words]
                    held = {normalise(words[i][1])[:5].strip(".,;:'") for i in found_at}
                    if len(held) < 2 or len(held) / max(len(label_words), 1) < 0.6:
                        continue
                    verb = index(change.start())
                    if min(abs(verb - i) for i in found_at) > CHANGE_WINDOW:
                        continue
                    after = words[max(found_at)][0]
                limit = index(after) + CHANGE_WINDOW
                found = ""
                for match in re.finditer(_VALUE_PATTERNS[kind], sentence):
                    if match.start() < after or index(match.start()) > limit:
                        continue
                    if _same_value(match.group(0), entry.value) or _same_value(match.group(0), short):
                        continue
                    if strict:
                        # The movement word between the label and the value
                        # ("the exam moves to 21 January"), or the value right
                        # after "to" ("the endorsement in Mississauga to 22
                        # January"). Not a value further along a list.
                        label_end = index(after)
                        at = index(match.start())
                        verb_between = any(label_end <= index(m.start()) < at for m in _CHANGE.finditer(sentence))
                        to_before = sentence[:match.start()].rstrip().lower().endswith(" to")
                        if not (verb_between or (to_before and at - label_end <= 5)):
                            break
                    found = match.group(0).strip(" ,")
                    break
                if not found or any(_same_value(h.value, found) for h in entry.history):
                    continue
                apply_ledger_op(record, {"label": entry.label, "value": found, "m": label_m,
                                         "replaces": entry.value}, timestamps)


def _mark_changed_in_prose(record: CompactionRecord, label: str, old: str, new: str, m: str) -> None:
    """A prose line that states the replaced value, with what it is, says it
    changed: the sections are never rewritten by code otherwise. Not in the
    status modes, where the writer says what changed."""
    if record.status_mode in ("restate", "delta"):
        return
    old_n, words = normalise(old), _label_words(label)
    note = f" (later changed to {new}"
    for section in record.sections:
        lines = section.text.splitlines()
        for i, line in enumerate(lines):
            text = normalise(line)
            if old_n not in text or normalise(new) in text or "later changed to" in text:
                continue
            held = {w[:5] for w in re.findall(r"[a-z0-9']{3,}", text)}
            if words and len(words & held) / len(words) < 0.5:
                continue
            body, cite = (line.rsplit(" [", 1) + [""])[:2] if line.rstrip().endswith("]") else (line, "")
            body = body.rstrip().rstrip(".")
            lines[i] = f"{body}{note}, turn {_turn_number(m)}).{' [' + cite if cite else ''}"
        section.text = "\n".join(lines)


def apply_action(record: CompactionRecord, item: Dict[str, Any], timestamps: Dict[str, str]) -> None:
    if not isinstance(item, dict):
        return
    quotes = []
    for q in (item.get("quotes") or [])[:2]:
        if isinstance(q, dict) and q.get("text"):
            quotes.append(Quote(text=str(q["text"]).strip(), m=str(q.get("m") or q.get("m_id") or "").strip()))
    status = "done" if str(item.get("status", "open")).lower() == "done" else "open"

    existing = next((a for a in record.actions if item.get("id") and a.id == item.get("id")), None)
    if existing is not None:
        existing.status = status
        if item.get("what"):
            existing.what = str(item["what"]).strip()
        for q in quotes:
            if all(q.text != old.text for old in existing.quotes) and len(existing.quotes) < 2:
                existing.quotes.append(q)
        return
    if not item.get("who") or not item.get("what"):
        return
    m = quotes[0].m if quotes else ""
    record.actions.append(AgreedAction(
        id=record.next_action_id(), who=str(item["who"]).strip(), what=str(item["what"]).strip(),
        quotes=quotes, status=status, m=m, at=timestamps.get(m, ""),
    ))


def _same_value(a: str, b: str) -> bool:
    return normalise(a) == normalise(b)


def normalise(text: str) -> str:
    """Fold case, whitespace, quote marks and dashes for comparisons."""
    import unicodedata

    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[‘’‚‛′`]", "'", text)
    text = re.sub(r"[“”„‟″]", '"', text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[  -​  ]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def resolved_date_ok(value: str, resolved: str, at: str) -> bool:
    """The writer's absolute date agrees with what the value says: the same
    weekday, the same day and month, and for a bare weekday within a week of
    the turn. Unparseable dates pass: the check removes only a resolved date
    the value contradicts."""
    from datetime import date, datetime

    found = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", resolved or "")
    if not found:
        return True
    try:
        day = date(int(found.group(1)), int(found.group(2)), int(found.group(3)))
    except ValueError:
        return False
    text = value.lower()
    weekday = next((i for i, w in enumerate(_WEEKDAYS) if re.search(rf"\b{w[:3]}(?:{w[3:]})?\b", text)), None)
    if weekday is not None and day.weekday() != weekday:
        return False
    month = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\b", text)
    number = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", text)
    if month and day.month != _MONTH_NUM[month.group(1)]:
        return False
    if number and (month or "the " + number.group(0) in text) and day.day != int(number.group(1)):
        return False
    if weekday is not None and not month and not number and at:
        try:
            said = datetime.fromisoformat(at).date()
        except ValueError:
            return True
        if abs((day - said).days) > 7:
            return False
    return True


# =============================================================================
# Statuses of plans, promises and open items (fidelity)
# =============================================================================

# Sections whose lines say something will or may happen. A reader asked "has
# it happened yet?" needs the record to say, and a literal reader would not
# infer "not yet" from "will sign on Thursday" (19 Oct: 15 of the 18 1K misses).
# Open items are left out: "not done" says nothing about a question.
STATUS_HEADERS = ("Stated plans", "Promises, as said")
# A sentence that says something happened.
_DONE = re.compile(
    r"\b(?:done|sent|signed|booked|uploaded|registered|paid|bought|filed|submitted|received|arrived|confirmed|"
    r"approved|completed|finished|sorted|ordered|delivered|enrolled|opened|called|emailed|got|picked up|went through|"
    r"handed|transferred|cancel(?:l)?ed|scheduled|renewed|returned|collected|settled|took care)\b",
    re.IGNORECASE)
_STATUS_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "was", "be", "it", "that",
                "this", "at", "by", "as", "from", "user", "assistant", "will", "would", "plans", "plan", "planned",
                "going", "said", "says", "she", "he", "they", "them", "their", "his", "her", "said:", "i'll", "we'll",
                "i", "we", "my", "our", "you", "your", "today", "tomorrow", "this", "week", "then", "also", "still"}
DONE_SHARE = 0.5


def status_key(line: str) -> str:
    text = re.sub(r"\s*\[m\d+[^\]]*\]\s*$", "", (line or "").strip().lstrip("-• ").strip())
    return normalise(text).rstrip(".")


def _status_words(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-z0-9']+", normalise(text)) if w not in _STATUS_STOP and len(w) > 2}


def _line_turn(line: str) -> str:
    found = re.search(r"\[(m\d+)", line)
    return found.group(1) if found else ""


def track_statuses(record: CompactionRecord, turns: Dict[str, str]) -> None:
    """Find, for every plan and promise line, a later turn whose sentence says
    it happened and names most of what the line names (kept as the quote).
    Runs on each chunk's turns, so every turn after a line is checked once.
    What is shown (render._with_status): done with that quote; "not reported
    done" only when the writer also judged it not done; else nothing."""
    lines = [line for section in record.sections if section.header in STATUS_HEADERS
             for line in section.text.splitlines() if line.strip()]
    live = set()
    for line in lines:
        key = status_key(line)
        if not key:
            continue
        live.add(key)
        entry = record.statuses.setdefault(key, {"m": _line_turn(line), "done_m": "", "quote": "", "writer": ""})
        if entry["done_m"]:
            continue
        words = _status_words(key)
        if len(words) < 2:
            continue
        raised = _turn_number(entry["m"])
        for label, text in turns.items():
            if _turn_number(label) <= raised:
                continue
            for sentence in split_sentences(text):
                if sentence.rstrip().endswith("?") or not _DONE.search(sentence):
                    continue
                if len(words & _status_words(sentence)) / len(words) >= DONE_SHARE:
                    entry["done_m"], entry["quote"] = label, sentence.strip()
                    break
            if entry["done_m"]:
                break
    # Lines a correction rewrote or a roll-up dropped take their status with them.
    record.statuses = {k: v for k, v in record.statuses.items() if k in live}
