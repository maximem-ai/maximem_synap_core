"""Code checks on one writer output. No model, every profile.

  literal check    every literal in the new turns (IDs, emails, phones,
                   amounts, ISO dates, versions) is in the ledger
  quote check      every agreed-action quote and every ledger value appears
                   in the turn it cites, after folding whitespace, quote
                   marks, dashes and case
  structure check  no banned or reserved header was asked for
  size check       record growth is within target_ratio ±5 points of the
                   new turns' tokens, with a ~400-token floor. Growth counts
                   the sections and the agreed actions; the literal ledger is
                   outside the ratio, because it is never condensed and wins
                   over size, so a repair could not shrink it

A citation that names the wrong turn, when the quoted text is in another
turn the writer saw, is corrected here rather than sent back: the claim is
grounded, only the pointer was off.

Each problem is phrased as the instruction the repair call receives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .literals import literal_candidates as _jev_literals
from .prompts import checks as msg

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b")
# Beyond the Jev check's patterns (IDs, emails, phones, money, ISO dates,
# versions), values a probe asked for and the ledger let slip: file paths,
# comma-grouped numbers, lower-case commit hashes, percentages, clock times.
_EXTRA_LITERALS = (
    r"(?<![\w/.:])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,5}\b",   # jobs/orders/conf.yaml
    r"[$€£¥₹]\d+(?:[.,]\d+)?\s?(?:[MmKkBb]n?|bn)\b",                    # $2.4M, €150k
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",                   # 1,204,551
    r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b",   # 7be21d0
    r"\b\d+(?:\.\d+)?\s?%",                                 # 22%
    r"\b\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?\b",                  # 15:30, 4:00 PM
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH,                    # 24 October
    _MONTH + r"\s+\d{1,2}(?:st|nd|rd|th)?\b(?!:)",              # December 1
    r"\bv\d+(?:\.\d+)+\b",                                       # v4.18.3
    r"\(\d{2,4}\)\s?\d{3,4}[\s.-]\d{3,4}\b",                       # (775) 555-0148
    r"\b\d{3}[.-]\d{3}[.-]\d{4}\b",                                 # 775-555-0148
    r"\b(?:Mr|Mrs|Ms|Dr)\.\s[A-Z][a-z]+",                          # Mrs. Halvorsen
    # Two or more capitalised words: a person, company or place ("Marisol
    # Ekwueme", "Hotel Ribeira Oeste"), not a sentence opener ("Okay That").
    r"\b(?!(?:I|The|A|An|And|But|So|Okay|Ok|Yes|No|Hi|Thanks|With|In|On|At|For|From|To|Is|Are|Should|Then|"
    r"Can|Could|Would|Will|Did|Do|Does|Has|Have|Let|Please|Also|Plus|Because|If|When|After|Before|Ask|Tell|"
    r"Send|Call|Book|Keep|Check|Use|Try|Get|Got|Make|Our|My|Your|Their|His|Her|What|Why|How|Who|Where)\b)"
    r"(?![A-Z][a-z]+ing\b)[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",
)


# Kept whatever the memory design's list says: how to reach someone is never
# safely dropped (20 Oct: a contract agent's list left out phone numbers, and
# "the best number to reach Renata" was lost).
ALWAYS_KEPT_KINDS = ("email", "phone", "url")

# Value kinds a memory design (MACA) can name as the ones its agent needs
# kept when the writer leaves them out (EngineSettings.value_kinds).
VALUE_KINDS = ("email", "url", "phone", "money", "percent", "time", "date", "version", "commit_hash",
               "file_path", "number", "code_or_id", "name")
_KIND_PATTERNS = (
    ("email", r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$"),
    ("url", r"^(?:https?://|www\.)\S+$"),
    ("date", r"^\d{4}-\d{2}-\d{2}"),
    ("phone", r"^\+?\(?\d[\d\s().-]{7,}\d$"),
    ("money", r"(?i)[$€£¥₹]|\b(?:usd|eur|gbp|cad|inr|aud)\b"),
    ("percent", r"%$"),
    ("time", r"^\d{1,2}:\d{2}(?:\s?[ap]m)?$"),
    ("date", r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|^\d{4}-\d{2}-\d{2}"),
    ("version", r"^v?\d+\.\d+(?:\.\d+)+(?:[-+.][\w.]+)?$|^v\d+(?:\.\d+)*$"),
    ("commit_hash", r"^(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}$"),
    ("number", r"^[\d.,\s]+$"),
    ("file_path", r"/|\\|^[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,4}$"),
    ("code_or_id", r"^(?=.*\d)[\w-]{3,}$|^[A-Z][A-Z0-9_-]{2,}$"),
    ("name", r"^[A-Z][\w'.-]*(?:\s[A-Z][\w'.-]*)+$"),
)


def value_kind(value: str) -> str:
    """The first kind in _KIND_PATTERNS a value matches, or ""."""
    v = (value or "").strip()
    for kind, pattern in _KIND_PATTERNS:
        if re.search(pattern, v):
            return kind
    return ""


_FENCE = re.compile(r"^\s*(?:```|~~~)")
# Lines that are machine output or pasted structure, not someone's words:
# table rows, indented code, log lines, stack frames, shell prompts.
_BULK_LINE = re.compile(
    r"^\s*\|"
    r"|^(?: {4}|\t)"
    r"|^\s*(?:\d{4}-\d{2}-\d{2}[T ]\d|\[?\d{2}:\d{2}:\d{2}|\[\w+\]\s"
    r"|(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL|CRITICAL)\b|Traceback|File \"|at \S+\("
    r"|\S+\.(?:py|js|ts|go|java|rb|rs|cs|php|kt|swift):\d+|[$#>] \S|[+-]{3} |@@ )"
)


def plain_words(text: str) -> str:
    """``text`` without its pasted bulk: fenced blocks, table rows, indented
    code, log and stack lines. What is left is what someone wrote."""
    out, fenced = [], False
    for line in (text or "").splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced or _BULK_LINE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def only_in_bulk(value: str, texts) -> bool:
    """True when ``value`` appears in these turns only inside pasted bulk,
    never in anyone's own words."""
    seen = False
    for text in texts:
        if value in plain_words(text):
            return False
        seen = seen or value in (text or "")
    return seen


_KEY_STOP = frozenset("""
the a an and or of to in on at for by with from per is are was were be it its this that these those as
now new old total's vs via into than then also only just about over under up down out we our you your
""".split())


def _key_words(text: str) -> List[str]:
    return [w[:6] for w in re.findall(r"[a-z][a-z0-9]+", text.lower()) if w not in _KEY_STOP and len(w) > 2]


def value_key(value: str, text: str) -> str:
    """What a value belongs to, as a few word stems: in a table row, the
    row's words that are not values ("| Corvane | Storage | 2,100 pallets x
    10 months | $18.50 | $388,500 |" -> "corvan storag pallet months"); in
    a sentence, up to four words just before it ("the storage total is
    $388,500" -> "storag total"). "" when nothing names it."""
    for line in (text or "").splitlines():
        if value not in line:
            continue
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            words = [w for c in cells if c and not re.search(r"\d", c) for w in _key_words(c)]
            if not words:  # "| 2,100 pallets x 10 months |": the words of cells with numbers too
                words = [w for c in cells if c and value not in c for w in _key_words(re.sub(r"[\d$€£,.%]+", " ", c))]
            return " ".join(dict.fromkeys(words))
        before = re.split(r"[.;:!?\n(]", line[:line.index(value)])[-1]
        return " ".join(dict.fromkeys(_key_words(before)[-4:]))
    return ""


def linked_entry(ledger, value: str, key: str, m: str, text: str = ""):
    """The one code-added entry from an earlier turn that names the same
    thing as ``value`` (same kind of value, and one key's words all in the
    other's), or None: no match, or more than one (ambiguous, so no link).
    Among several, the one whose value is on the same line of ``text`` as
    the new value wins ("| Storage | $388,500 | $357,000 |", "from $40 to
    $45"), then the one with exactly the same key."""
    words = set(key.split())
    if not words:
        return None
    kind = value_kind(value)
    turn = int(m[1:]) if m[1:].isdigit() else -1
    found = []
    for entry in ledger:
        if not entry.auto or not entry.key or value_kind(entry.value) != kind:
            continue
        if not (entry.m[1:].isdigit() and int(entry.m[1:]) < turn):
            continue
        other = set(entry.key.split())
        if (words <= other or other <= words) and entry.value != value:
            found.append(entry)
    line = next((ln for ln in (text or "").splitlines() if value in ln), "")
    named = [e for e in found if line and e.value in line]
    if len(named) == 1:
        return named[0]
    exact = [e for e in found if set(e.key.split()) == words]
    if len(exact) == 1:
        return exact[0]
    return found[0] if len(found) == 1 else None


def literal_candidates(texts):
    found = dict.fromkeys(_jev_literals(texts))
    for text in texts:
        for pattern in _EXTRA_LITERALS:
            for match in re.finditer(pattern, text):
                found.setdefault(match.group(0).strip(), None)
    # "$2" from "$2.4M", "$150" from "$150k": a value that never stands on its
    # own in the text, only as the start of a longer one, is that one.
    joined = "\n".join(texts)
    return [v for v in found if re.search(re.escape(v) + r"(?![\w]|[.,]\d)", joined)]
from .record import CompactionRecord, MergeNotes, item_text, normalise, split_sentences

TOLERANCE = 0.05
MIN_UPDATE_TOKENS = 400


@dataclass
class Problem:
    kind: str  # literal | quote | structure | size_over | size_under
    message: str
    ref: Dict[str, Any] = field(default_factory=dict)


def _significant(value: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9.,:/@+$€£%-]*", normalise(value)) if len(t) >= 3 or t.isdigit()]


def _grounded(value: str, text: str) -> bool:
    """The value appears in the text, or every significant token of it does
    (a ledger value may join words the turn spread over a phrase, such as
    "Aetna HSA" from "the Aetna HSA plan")."""
    v, t = normalise(value), normalise(text)
    if not v:
        return True
    if v in t:
        return True
    tokens = _significant(value)
    return bool(tokens) and all(tok in t for tok in tokens)


def _find_turn(value: str, turns: Dict[str, str], prefer: str) -> str:
    if prefer in turns and _grounded(value, turns[prefer]):
        return prefer
    for label, text in turns.items():
        if _grounded(value, text):
            return label
    return ""


def check_quotes(output: Dict[str, Any], turns: Dict[str, str]) -> List[Problem]:
    """Quote check, fixing mis-cited turns in ``output`` in place."""
    problems: List[Problem] = []
    for i, action in enumerate(output.get("agreed_actions") or []):
        if not isinstance(action, dict):
            continue
        for quote in action.get("quotes") or []:
            if not isinstance(quote, dict):
                continue
            text, cited = str(quote.get("text") or ""), str(quote.get("m") or "")
            found = _find_turn(text, turns, cited) if normalise(text) else ""
            # For a quote, the whole text must appear, not just its tokens.
            if found and normalise(text) not in normalise(turns[found]):
                found = next((lbl for lbl, t in turns.items() if normalise(text) in normalise(t)), "")
            if found:
                quote["m"] = found
                continue
            if cited and cited not in turns:
                # An older turn the writer did not see in this call: it was
                # checked when it was new.
                continue
            problems.append(Problem(
                "quote",
                msg.action_quote_missing(action.get("what", ""), action.get("who", ""), text, cited),
                {"action_index": i},
            ))
        if not action.get("id") and not action.get("quotes"):
            problems.append(Problem(
                "quote",
                msg.action_no_quote(action.get("what", "")),
                {"action_index": i},
            ))
    for op in output.get("literals") or []:
        if not isinstance(op, dict) or not op.get("value"):
            continue
        cited = str(op.get("m") or "")
        found = _find_turn(str(op["value"]), turns, cited)
        if found:
            op["m"] = found
        elif not cited or cited in turns:
            problems.append(Problem(
                "quote",
                msg.ledger_value_missing(op["value"], op.get("label", ""), cited),
                {"value": op["value"]},
            ))
    return problems


def check_literals(record: CompactionRecord, new_turns: Dict[str, str]) -> List[Problem]:
    """Literals in the new turns that the ledger (current or past values) lacks."""
    held = [normalise(v) for v in record.ledger_values()] + [normalise(e.label) for e in record.ledger]
    problems = []
    seen = set()
    for label, text in new_turns.items():
        for literal in literal_candidates([text]):
            literal = literal.rstrip(".,;:")
            key = normalise(literal)
            if key in seen:
                continue
            seen.add(key)
            if key and not any(key in value for value in held):
                problems.append(Problem(
                    "literal",
                    msg.literal_missing(literal, label),
                    {"value": literal, "m": label},
                ))
    return problems


# Words that carry no content of an inventory item: who said it, how.
_FILLER = frozenset("""
the and that this with from they them their there then than what when which who whom whose will would
could should have has had been being were was are for not but nor its it's into onto over under about
user users assistant said says stated asked asks suggested suggests suggestion noted mentioned told
also only just very more most some such because since while after before again
you your might want may consider worth
""".split())
COVERAGE_SHARE = 0.6
# Types the ledger and the agreed actions already check.
_COVERED_ELSEWHERE = {"value", "commitment"}


def _content_words(text: str) -> List[str]:
    text = re.sub(r"[\u00a0\u2009\u202f]", " ", text or "")
    words = re.findall(r"[a-z0-9$€£%][a-z0-9$€£%.,'-]*", normalise(text))
    return [w.strip(".,'-")[:5] for w in words if len(w.strip(".,'-")) >= 3 and w.strip(".,'-") not in _FILLER]


def _record_units(record: CompactionRecord) -> List[set]:
    """Content words of each section and each agreed action. A section, not
    a sentence: terse records split one item over clauses."""
    texts = [s.text for s in record.sections] + [f"{a.who} {a.what}" for a in record.actions]
    return [words for words in (set(_content_words(t)) for t in texts) if words]


def said_with_context(record: CompactionRecord, value: str, text: str) -> bool:
    """One line of the record holds ``value`` and most content words of
    ``text``, the sentence it was said in."""
    words, key = set(_content_words(text)), normalise(value)
    lines = [line for s in record.sections for line in s.text.splitlines()]
    lines += [f"{a.who} {a.what}" for a in record.actions]
    for line in lines:
        if key in normalise(line):
            held = set(_content_words(line))
            if not words or len(words & held) / len(words) >= COVERAGE_SHARE:
                return True
    return False


_QUESTION = re.compile(r"\?\s*$|^(?:the\s+)?(?:user|assistant)\s+ask(?:s|ed)\b", re.IGNORECASE)


def check_coverage(record: CompactionRecord, inventory: Any) -> List[Problem]:
    """Inventory items (the writer's own list) that the record does not carry:
    most of an item's content words, other than ledger values, must appear in
    one section or agreed action."""
    if not isinstance(inventory, list):
        return []
    units = _record_units(record)
    in_ledger = set()
    for entry in record.ledger:
        in_ledger |= set(_content_words(entry.value))
    problems = []
    for item in inventory:
        if not isinstance(item, dict) or str(item.get("type") or "").lower() in _COVERED_ELSEWHERE:
            continue
        text = item_text(item)
        if _QUESTION.search(text):
            continue  # a question asked; its answer is its own item
        # Values are the ledger's job; what is left is what the prose owes.
        words = set(_content_words(text)) - in_ledger
        if len(words) < 2:
            continue
        best = max((len(words & u) / len(words) for u in units), default=0.0)
        if best < COVERAGE_SHARE:
            m = str(item.get("m") or "")
            problems.append(Problem(
                "coverage",
                msg.inventory_not_in_record(text, m),
                {"item": text, "m": m, "type": str(item.get("type") or "")},
            ))
    return problems


# A first-person promise: "I'll", "I will", "we'll", "I'm going to". Not "I'd".
_PROMISE = re.compile(r"\b(?:I'll|I will|we'll|we will|I'm going to|we're going to)\b(?!\s+(?:think|see|consider|check whether)\b)",
                      re.IGNORECASE)


def check_promises(record: CompactionRecord, new_turns: Dict[str, str]) -> List[Problem]:
    """First-person promises in the new turns that no agreed action quotes
    and no line of the record carries."""
    quoted = {normalise(q.text) for a in record.actions for q in a.quotes}
    units = _record_units(record)
    problems = []
    for label, text in new_turns.items():
        for sentence in split_sentences(text):
            if not _PROMISE.search(sentence) or sentence.rstrip().endswith("?"):
                continue
            key = normalise(sentence)
            if any(key in q or q in key for q in quoted if q):
                continue
            words = set(_content_words(sentence))
            if len(words) >= 2 and max((len(words & u) / len(words) for u in units), default=0.0) >= COVERAGE_SHARE:
                continue
            problems.append(Problem("promise", msg.promise_not_carried(label, sentence.strip()),
                                    {"sentence": sentence.strip(), "m": label}))
    return problems


# An assistant suggestion: the item the writer most often drops, when the
# turn opens by repeating values back.
_SUGGESTION = re.compile(
    r"\b(?:you (?:might|could|may) (?:also )?(?:want to|consider|try|ask|add|check|get)|you could also|"
    r"it(?:'s| is| would be) worth|I'd (?:suggest|recommend)|I (?:suggest|recommend))\b", re.IGNORECASE)
REPLY_WORDS = 30


def _sentences(text: str) -> List[str]:
    return split_sentences(text)


def _clip(text: str, words: int) -> str:
    """The first ``words`` words, verbatim, with "..." when cut."""
    parts = text.split()
    return text.strip() if len(parts) <= words else " ".join(parts[:words]) + "..."


def reply_after(label: str, turns: Dict[str, str], roles: Dict[str, str]) -> tuple[str, str]:
    """(label, first sentences of) the next turn by the other side."""
    labels = list(turns)
    if label not in labels:
        return "", ""
    for later in labels[labels.index(label) + 1:]:
        if roles.get(later) and roles.get(later) != roles.get(label):
            return later, _clip(" ".join(_sentences(turns[later])), REPLY_WORDS // 2)
    return "", ""


_DIRECT_ANSWER = re.compile(
    r"^\W*(?:yes|yeah|yep|no|nope|nah|maybe|perhaps|sure|ok(?:ay)?|fine|agreed|deal|definitely|absolutely|"
    r"not (?:now|yet|really|sure)|i'?d rather|i'?ll (?:think|pass|skip)|let'?s (?:not|do)|good idea|"
    r"sounds good|probably|no thanks|thanks,? but)\b", re.IGNORECASE)
DIRECT_ANSWER_WORDS = 15
_POINTS_BACK = re.compile(r"^\W*(?:hm+|oh|ah|well|honestly|actually)\b|\b(?:it|that|this|those|them|one)\b",
                          re.IGNORECASE)


def answers(about: str, reply: str) -> bool:
    """The reply is about the suggestion: it shares a content word with it,
    or it is a short direct answer ("maybe", "I'd rather not"). "What's the
    offer?" after "keep the tire receipt" answers nothing, and neither does
    a statement on another subject."""
    if not reply:
        return False
    if set(_content_words(about)) & set(_content_words(reply)):
        return True
    first = (_sentences(reply) or [""])[0]
    if len(first.split()) > DIRECT_ANSWER_WORDS or first.rstrip().endswith("?"):
        return False
    # "Hm, I already threw it away.": a short statement that points back at
    # the suggestion answers it without repeating its words.
    return bool(_DIRECT_ANSWER.match(first) or _POINTS_BACK.search(first))


def check_suggestions(record: CompactionRecord, new_turns: Dict[str, str], roles: Dict[str, str]) -> List[Problem]:
    """Assistant suggestions in the new turns that no line of the record carries."""
    problems = []
    for label, text in new_turns.items():
        if roles.get(label) != "the assistant":
            continue
        for sentence in _sentences(text):
            if not _SUGGESTION.search(sentence) or sentence.endswith("?"):
                continue
            words = set(_content_words(sentence))
            if len(words) < 2 or max((len(words & u) / len(words) for u in _record_units(record)),
                                     default=0.0) >= COVERAGE_SHARE:
                continue
            reply_m, reply = reply_after(label, new_turns, roles)
            if not answers(sentence, reply):
                reply_m, reply = "", ""
            problems.append(Problem("suggestion", msg.suggestion_not_carried(label, sentence),
                                    {"sentence": sentence, "m": label, "reply": reply, "reply_m": reply_m}))
    return problems


def check_responses(items: Sequence[Dict[str, Any]], new_turns: Dict[str, str],
                    roles: Dict[str, str]) -> List[Problem]:
    """Advice items whose "why" is not what the other side said back: the
    writer fills it with the suggestion's own reason."""
    problems = []
    for i, item in enumerate(items):
        if str(item.get("type") or "").lower() not in ("advice", "suggestion"):
            continue
        m = str(item.get("m") or "")
        reply_m, reply = reply_after(m, new_turns, roles)
        if not reply:
            continue
        why_text = str(item.get("why") or "")
        why = set(_content_words(why_text))
        if why and len(why & set(_content_words(new_turns[reply_m]))) / len(why) >= 0.5:
            continue
        # The suggester's own reason, written as the reply, stays as a
        # reason; one grounded nowhere ("no reply") is dropped. The next turn
        # is the reply only when it is about the suggestion: "What's the
        # offer?" after "keep the tire receipt" answers nothing.
        own = bool(why) and len(why & set(_content_words(new_turns.get(m, "")))) / len(why) >= 0.5
        relevant = answers(f'{item.get("item", "")} {why_text}', reply)
        if own and not relevant and re.match(r"\s*(?:because|since|as|in case)\b", why_text, re.IGNORECASE):
            continue  # already written as a reason, and no reply to add
        problems.append(Problem("response", msg.response_mismatch(item.get("item", ""), reply_m),
                                {"index": i, "reply": reply if relevant else "", "reply_m": reply_m,
                                 "reason": why_text if own else ""}))
    return problems


def check_structure(notes: MergeNotes) -> List[Problem]:
    problems = []
    for header in notes.rejected_headers:
        problems.append(Problem(
            "structure",
            msg.header_not_allowed(header),
            {"header": header},
        ))
    for section_id in notes.unknown_section_ids:
        problems.append(Problem(
            "structure", msg.no_such_section(section_id),
            {"id": section_id},
        ))
    return problems


def size_bounds(new_tokens: int, target_ratio: float) -> tuple[int, int]:
    """(low, high) record growth allowed for ``new_tokens`` of new turns."""
    low = int(new_tokens * max(0.0, target_ratio - TOLERANCE))
    high = max(int(new_tokens * (target_ratio + TOLERANCE)), MIN_UPDATE_TOKENS)
    return low, high


def check_size(growth: int, new_tokens: int, target_ratio: float) -> List[Problem]:
    low, high = size_bounds(new_tokens, target_ratio)
    if growth > high:
        return [Problem(
            "size_over",
            msg.size_over(growth, high),
            {"growth": growth, "high": high},
        )]
    # Below the floor a span is too short to call a small update a loss:
    # "Thanks!" rightly adds nothing.
    if growth < low and new_tokens * target_ratio >= MIN_UPDATE_TOKENS:
        return [Problem(
            "size_under",
            msg.size_under(growth, new_tokens, low),
            {"growth": growth, "low": low},
        )]
    return []


def format_problems(problems: Sequence[Problem]) -> str:
    return "\n".join(f"{i}. {p.message}" for i, p in enumerate(problems, 1))
