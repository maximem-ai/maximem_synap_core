"""Who "he" is, when the document does not say.

Entity and relationship extraction read one document. A conversation sent as
one transcript carries its own names, and there a pronoun resolves inside the
document. These arrive with a pronoun and no name:

  - a conversation sent turn by turn or in batches (listening agent, compaction
    delta): "He is joining Fennimore Labs next month."
  - a short memory written on its own: "He called again about the rate card."
  - a new session that picks a subject back up: "Did she finish the review?"

Today the graph gets nothing from them. This finds the one entity the pronoun
means among what the same user said recently, or nothing.

How, cheapest first:
  1. Gate, in code. Only a document with a third-person pronoun and no entity
     of its own that the pronoun could mean. Every other document is untouched.
  2. Search, in code. The user's most recent memories: always the last few
     documents, plus older ones that share words with this document.
  3. Candidates, in code. Entities on file whose name appears in those memories.
  4. Judge, Jev: one yes/no per candidate. A reference is returned only when
     exactly one candidate is a clear yes and no other is close.

It only ever adds a name the user already used. It never creates an entity,
and when two candidates fit it returns nothing. Off unless
SYNAP_STAGE3_EARLIER_REFERENCES=true; user scope only (it needs a user id).
Any failure returns nothing.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from synap.cloud.agents.ingestion.prompts.earlier_reference import (
    QUESTION,
    CRITERIA,
    PRONOUN_QUESTION,
    PRONOUN_CRITERIA,
)

logger = logging.getLogger(__name__)

_FLAG_ENV = "SYNAP_STAGE3_EARLIER_REFERENCES"

# Set 13 Oct 2026 on harnesses/entity_resolution/fixtures/notes_earlier.txt: the
# 9 right candidates scored 0.64 to 0.87, the wrong ones 0.03 to 0.14, and the
# two that both fitted one pronoun 0.89 and 0.93. A small sample: re-measure
# with harnesses/entity_resolution/earlier.py before changing either.
# On the second fixture (notes_earlier_heldout.txt) two wrong candidates scored
# 0.51 and 0.54, so ACCEPT_AT sits between those and the lowest right one. That
# is a thin margin chosen with both files in view; notes_earlier_check.txt is
# the one nothing was set on.
ACCEPT_AT = float(os.environ.get("SYNAP_STAGE3_EARLIER_REFERENCES_ACCEPT_AT", "0.6"))
RIVAL_AT = float(os.environ.get("SYNAP_STAGE3_EARLIER_REFERENCES_RIVAL_AT", "0.3"))
# A second, narrower question per candidate: did the earlier notes call them
# "he" where this note says "she"? A shared topic can outweigh that in the main
# question (the 0.51 above was a man offered for "she").
WRONG_PRONOUN_AT = 0.5

RECENT_DOCUMENTS = 3      # always searched, whatever they are about
MAX_MEMORIES = 8          # sent on to the candidate step
MAX_CANDIDATES = 6
FETCH_LIMIT = 60          # memories read per document, most recent first

PRONOUNS: Dict[str, Tuple[str, ...]] = {
    "he": ("he", "him", "his", "himself"),
    "she": ("she", "her", "hers", "herself"),
    "they": ("they", "them", "their", "theirs"),
}
_PERSON_TYPES = {"person"}
_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”')
_WORD = re.compile(r"[a-z][a-z'\-]{3,}")
_STOP = {
    "about", "again", "also", "been", "before", "called", "could", "does", "from", "have",
    "into", "just", "more", "next", "said", "says", "some", "than", "that", "their", "them",
    "then", "there", "they", "this", "today", "told", "tomorrow", "want", "wants", "week",
    "were", "what", "when", "will", "with", "would", "yesterday", "your",
}


def enabled(instance_id: Any = None) -> bool:
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_FLAG_ENV, instance_id)


@dataclass
class Candidate:
    name: str
    entity_type: str                 # as extraction names it: person, organization, ...
    about: Optional[str] = None      # the registry's description, when there is one
    notes: List[str] = field(default_factory=list)   # what earlier notes say, newest first
    documents_ago: int = 0           # 1 = the document just before this one


def named_in_text(text: str, references: Sequence["EarlierReference"],
                  document: Optional[str] = None) -> str:
    """*text* with each resolved pronoun's entity named in it.

    A memory extracted from "He called again about the damage claim" keeps the
    "he": the memory prompts have no name to give it. Resolving the pronoun for
    the relationship alone then changes nothing a reader can use, because a
    memory is tied to an entity by the entity's name appearing in its text.

    The subject form is replaced ("He called" -> "Ignatius Bello called"),
    which is always the same word for word. An object or possessive form is
    left as written and the name is added after the text, because "her" is
    both and rewriting it wrongly would change what the memory says.

    A memory can also lose the pronoun altogether: "She wants the signage
    proofs approved by Wednesday" comes back as "Signage proofs approval
    deadline by Wednesday". Given the *document*, such a text gets "[about
    <name>]" when the document itself uses the pronoun. The resolver only
    answers for a document that names no person at all (no entity at all, for
    "they"), so what such a document says is about whoever the pronoun means.

    A text that already has the name is returned as is.
    """
    out = text
    for ref in references or []:
        forms = PRONOUNS.get(ref.pronoun)
        if not forms or re.search(r"\b" + re.escape(ref.name) + r"\b", out, re.I):
            continue
        # Quoted speech is someone else's "he"; the gate ignores it and so does this.
        spoken = [m.span() for m in re.finditer(r'"[^"]*"', out)]
        used = [m for m in re.finditer(r"\b(?:" + "|".join(forms) + r")\b", out, re.I)
                if not any(a <= m.start() < b for a, b in spoken)]
        if not used:
            if document and re.search(
                    r"\b(?:" + "|".join(forms) + r")\b", re.sub(r'"[^"]*"', " ", document), re.I):
                out = f"{out.rstrip()} [about {ref.name}]"
            continue
        subject = next((m for m in used if m.group(0).lower() == forms[0]), None)
        if subject is not None:
            out = out[:subject.start()] + ref.name + out[subject.end():]
        else:
            out = f"{out.rstrip()} [{used[0].group(0).lower()} = {ref.name}]"
    return out


@dataclass
class EarlierReference:
    pronoun: str                     # "he" | "she" | "they"
    name: str
    entity_type: str
    score: float


def unresolved_pronouns(content: str, entity_types: Sequence[str]) -> List[str]:
    """The pronoun groups this document uses with nothing of its own to mean.

    "he"/"she": the document names no person. "they": it names nothing at all.
    A document that names a person keeps its own reading of "he", right or
    wrong, so this can never change a document that already resolves.
    """
    text = _QUOTED.sub(" ", content or "").lower()
    types = {str(t).lower().strip() for t in entity_types}
    found = []
    for group, words in PRONOUNS.items():
        if not re.search(r"\b(?:" + "|".join(words) + r")\b", text):
            continue
        if group == "they":
            if not types:
                found.append(group)
        elif not (types & _PERSON_TYPES):
            found.append(group)
    return found


def _words(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def select_memories(content: str, memories: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The memories worth reading for this document, newest first.

    *memories* are newest first, each with ``content`` and ``document_id``.
    Kept: everything from the last RECENT_DOCUMENTS documents (a reply in a
    running conversation shares no words with what it answers), then older
    memories by how many words they share with this document.
    """
    order: List[Any] = []
    for m in memories:
        if m.get("document_id") not in order:
            order.append(m.get("document_id"))
    ago = {doc: i + 1 for i, doc in enumerate(order)}
    mine = _words(content)
    recent, older = [], []
    for position, m in enumerate(memories):
        row = dict(m, documents_ago=ago[m.get("document_id")])
        if row["documents_ago"] <= RECENT_DOCUMENTS:
            recent.append(row)
        else:
            shared = len(mine & _words(m.get("content") or ""))
            if shared:
                older.append((-shared, position, row))
    kept = recent[:MAX_MEMORIES]
    kept += [row for _, _, row in sorted(older, key=lambda t: t[:2])[:MAX_MEMORIES - len(kept)]]
    return sorted(kept, key=lambda r: r["documents_ago"])


def candidates_for(pronoun: str, memories: Sequence[Dict[str, Any]],
                   names: Sequence[Dict[str, Any]]) -> List[Candidate]:
    """Entities on file that the selected memories name. *names*: dicts with
    ``name``, ``entity_type`` (person, organization, ...) and optional ``about``."""
    found: List[Candidate] = []
    for entry in names:
        name = str(entry.get("name") or "").strip()
        kind = str(entry.get("entity_type") or "").lower().strip()
        if not name or (pronoun != "they" and kind not in _PERSON_TYPES):
            continue
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        said = [m for m in memories if pattern.search(m.get("content") or "")]
        if not said:
            continue
        found.append(Candidate(
            name=name, entity_type=kind, about=entry.get("about") or None,
            notes=[m["content"] for m in said[:3]],
            documents_ago=min(m["documents_ago"] for m in said),
        ))
    found.sort(key=lambda c: c.documents_ago)
    return found[:MAX_CANDIDATES]


def _how_recent(documents_ago: int) -> str:
    if documents_ago <= 1:
        return "the note written just before this one"
    if documents_ago <= RECENT_DOCUMENTS:
        return f"{documents_ago} notes before this one"
    return f"{documents_ago} notes before this one, with other notes in between"


class JevReferenceJudge:
    """One Jev yes/no per candidate; each is asked without sight of the others."""

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0

    async def scores(self, content: str, pronoun: str,
                     candidates: Sequence[Candidate]) -> Optional[List[float]]:
        from synap.cloud.agents.ingestion.resolution.jev_settings import (
            JEV_MODEL, JEV_TIMEOUT_S,
        )
        from synap.cloud.shared.jev_client import ask

        questions = {
            f"c{i}": {
                "type": "noul",
                "instructions": {
                    "pronoun": pronoun,
                    "candidate": c.name + (f" ({c.about})" if c.about else ""),
                    "candidate_is_a": c.entity_type,
                    "earlier_notes": c.notes,
                    "how_recent": _how_recent(c.documents_ago),
                    "question": QUESTION,
                },
                "criteria": CRITERIA,
            }
            for i, c in enumerate(candidates)
        }
        if pronoun in ("he", "she"):
            for i, c in enumerate(candidates):
                questions[f"p{i}"] = {
                    "type": "noul",
                    "instructions": {"pronoun": pronoun, "candidate": c.name,
                                     "earlier_notes": c.notes, "question": PRONOUN_QUESTION},
                    "criteria": PRONOUN_CRITERIA,
                }
        self.calls += 1
        try:
            reply = await ask({"note": content}, questions, model=JEV_MODEL,
                              timeout=JEV_TIMEOUT_S, retries=1)
        except Exception:
            self.failures += 1
            logger.warning("Earlier-reference judge failed; nothing grounded", exc_info=True)
            return None
        scores = []
        for i in range(len(candidates)):
            score = float((reply.get(f"c{i}") or {}).get("noul") or 0.0)
            wrong_pronoun = float((reply.get(f"p{i}") or {}).get("noul") or 0.0)
            # Not the one meant, whatever else fits. Zero, so it is no rival either.
            scores.append(0.0 if wrong_pronoun >= WRONG_PRONOUN_AT else score)
        return scores


def decide(pronoun: str, candidates: Sequence[Candidate],
           scores: Sequence[float]) -> Optional[EarlierReference]:
    """Exactly one clear yes and no rival, or nothing."""
    ranked = sorted(zip(scores, candidates), key=lambda t: -t[0])
    if not ranked or ranked[0][0] < ACCEPT_AT:
        return None
    if len(ranked) > 1 and ranked[1][0] >= RIVAL_AT:
        return None
    score, best = ranked[0]
    return EarlierReference(pronoun=pronoun, name=best.name, entity_type=best.entity_type,
                            score=round(score, 3))


# async fn(scope, document_id) -> (memories newest first, names on file)
Fetch = Callable[[Any, Optional[str]], Awaitable[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]]


class EarlierReferenceResolver:
    def __init__(self, fetch: Fetch, judge: Optional[Any] = None) -> None:
        self._fetch = fetch
        self._judge = judge or JevReferenceJudge()
        self.log: List[dict] = []      # this process, for the harness

    async def resolve(self, content: str, entity_types: Sequence[str], scope: Any,
                      document_id: Optional[str] = None) -> List[EarlierReference]:
        pronouns = unresolved_pronouns(content, entity_types)
        if not pronouns or not getattr(scope, "user_id", None):
            return []
        try:
            memories, names = await self._fetch(scope, document_id)
        except Exception:  # noqa: BLE001 — an enhancement, never a gate
            logger.warning("Earlier-reference fetch failed; nothing grounded", exc_info=True)
            return []
        selected = select_memories(content, memories)
        found: List[EarlierReference] = []
        for pronoun in pronouns:
            candidates = candidates_for(pronoun, selected, names)
            entry = {"pronoun": pronoun, "candidates": [c.name for c in candidates],
                     "scores": [], "chosen": None}
            self.log.append(entry)
            if not candidates:
                continue
            scores = await self._judge.scores(content, pronoun, candidates)
            if scores is None:
                continue
            entry["scores"] = [round(s, 3) for s in scores]
            reference = decide(pronoun, candidates, scores)
            if reference and reference.name not in [r.name for r in found]:
                entry["chosen"] = reference.name
                found.append(reference)
        if found:
            logger.info("earlier_references user_id=%s grounded=%s", getattr(scope, "user_id", None),
                        [(r.pronoun, r.name, r.score) for r in found])
        return found


# Registry types (synap-er's four) back to what extraction calls them. The type
# Synap extracted is kept in metadata.entity_subtype and wins when present.
_REGISTRY_TYPES = {"PERSON": "person", "ORG": "organization"}


def build_fetcher(session_factory) -> Fetch:
    """Reads the user's recent memories, and the customer's registry entries
    those memories name. Two reads per gated document, none otherwise."""
    from sqlalchemy import text

    async def fetch(scope, document_id: Optional[str] = None):
        user_id = getattr(scope, "user_id", None)
        if session_factory is None or not user_id:
            return [], []
        # Built conditionally, not `(:doc IS NULL OR ...)`: asyncpg cannot type
        # a bind parameter that stands alone beside IS NULL (see prior_context).
        sql = ("SELECT content, source_document_id FROM memories "
               "WHERE user_id = CAST(:uid AS uuid) "
               "  AND superseded_by IS NULL AND deprecated_at IS NULL AND deleted_at IS NULL ")
        params: Dict[str, Any] = {"uid": str(user_id), "lim": FETCH_LIMIT}
        if document_id:
            sql += "  AND source_document_id IS DISTINCT FROM CAST(:doc AS text) "
            params["doc"] = str(document_id)
        sql += "ORDER BY document_created_at DESC NULLS LAST, created_at DESC LIMIT :lim"
        async with session_factory() as session:
            rows = (await session.execute(text(sql), params)).fetchall()
            memories = [{"content": r[0], "document_id": r[1]} for r in rows if r[0]]
            if not memories:
                return [], []
            blob = "\n".join(m["content"] for m in memories).lower()
            entries = (await session.execute(text(
                "SELECT canonical_name, entity_type, metadata FROM registry_entries "
                # The customer's entries, and this user's own: where a name from
                # their notes is registered under SYNAP_ER_SCOPED_REGISTRY.
                "WHERE ((scope = 'CUSTOMER' AND scope_id = :cid) OR (scope = 'USER' AND scope_id = :uid)) "
                "  AND is_deleted = FALSE "
                "  AND position(lower(canonical_name) in :blob) > 0 "
                "LIMIT 200"), {"cid": str(getattr(scope, "customer_id", None)),
                               "uid": str(getattr(scope, "user_id", None)), "blob": blob})).fetchall()
        from synap.cloud.agents.ingestion.resolution.descriptions import render
        names = []
        for name, registry_type, metadata in entries:
            metadata = metadata if isinstance(metadata, dict) else {}
            names.append({
                "name": name,
                "entity_type": metadata.get("entity_subtype") or _REGISTRY_TYPES.get(str(registry_type), "other"),
                "about": render(metadata),
            })
        return memories, names

    return fetch
