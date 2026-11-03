"""Reading the reply of the second graph prompt (prompts/extraction/graph_v2).

With ``SYNAP_GRAPH_EXTRACTION_V2`` set, Stage 3 extracts entities and
relationships with that prompt, in ``hybrid_graph`` and in ``baseline`` alike.

Nothing the model wrote is dropped for a reason the code can repair:
  - an end of a relationship that is not in "entities" becomes an entity, of
    the type the relationship gives it;
  - an entity type the code has never seen is kept;
  - an end is found by any of an entity's aliases, not only its name.
What is still dropped (an end with no name, a relationship from an entity to
itself, a repeat) is counted in ``Parsed.dropped``, and the scenario test
asserts that the first two stay at zero.

A relationship carries, in its properties:
  inverse_type  the same fact read from target to source ("EMPLOYS")
  basis         "stated" or "implied"
  confidence    0..1
  evidence      the words of the content it rests on

``basis`` is checked, not trusted: "stated" with evidence that is not in the
content is "implied". ``confidence`` is Jev's reading of whether the content
supports the relationship when Jev is reachable (``support``), else the
model's own number. A stated relationship is kept at STATED_MIN and above, an
implied one at IMPLIED_MIN and above.

Off by default.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from synap.cloud.agents.ingestion.models import ExtractedEntity, ExtractedRelationship
from synap.cloud.agents.ingestion.services import relationship_type
from synap.cloud.agents.ingestion.prompts.graph_v2_judge import (
    FOR_THE_JUDGE as _FOR_THE_JUDGE,
    SUPPORT_QUESTION,
    SUPPORT_CRITERIA,
    BELONGS_QUESTION,
    BELONGS_CRITERIA,
)

logger = logging.getLogger(__name__)

_FLAG_ENV = "SYNAP_GRAPH_EXTRACTION_V2"
_EVENTS_ENV = "SYNAP_GRAPH_EVENT_NODES"
_TYPE_EXAMPLES_ENV = "SYNAP_GRAPH_TYPE_EXAMPLES"       # "off" hides the configured relationship types
_SUPPORT_ENV = "SYNAP_GRAPH_SUPPORT_CHECK"
_STATEMENTS_ENV = "SYNAP_GRAPH_STATEMENTS"             # "off" skips the Jev check
_LONE_ENV = "SYNAP_GRAPH_DROP_LONE_ENTITIES"

STATED_MIN = float(os.environ.get("SYNAP_GRAPH_STATED_MIN", "0.5"))
IMPLIED_MIN = float(os.environ.get("SYNAP_GRAPH_IMPLIED_MIN", "0.7"))

STATED, IMPLIED = "stated", "implied"
DEFAULT_ENTITY_TYPE = "thing"


def enabled(instance_id: Optional[Any] = None) -> bool:
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_FLAG_ENV, instance_id)


def events_enabled(instance_id: Optional[Any] = None) -> bool:
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_EVENTS_ENV, instance_id)


def statements_enabled(instance_id: Optional[Any] = None) -> bool:
    """SYNAP_GRAPH_STATEMENTS: the call first writes every fact of the note as
    a plain sentence, then a relationship for each that joins two entities."""
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_STATEMENTS_ENV, instance_id)


def drop_lone_enabled(instance_id: Optional[Any] = None) -> bool:
    """SYNAP_GRAPH_DROP_LONE_ENTITIES: an entity no relationship of the note uses is not stored."""
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_LONE_ENV, instance_id)


def without_lone(entities: Sequence[ExtractedEntity], relationships: Sequence[Any]) -> Tuple[List[ExtractedEntity], List[ExtractedEntity]]:
    """(entities a relationship uses, the others). Run it last: after the support check has
    removed relationships, the ends it leaves behind have none, and a node with no edge says
    nothing the note's memories do not already say."""
    used = {i for r in relationships for i in (r.source_entity_id, r.target_entity_id)}
    kept = [e for e in entities if e.entity_id in used]
    return kept, [e for e in entities if e.entity_id not in used]


def type_examples() -> bool:
    return os.environ.get(_TYPE_EXAMPLES_ENV, "on").strip().lower() not in ("off", "false", "0", "no")


def support_check() -> bool:
    return os.environ.get(_SUPPORT_ENV, "on").strip().lower() not in ("off", "false", "0", "no")


def entity_type(raw: Any) -> str:
    """The model's word for a kind of entity, as a graph label can hold it."""
    text = re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")
    if not text:
        return DEFAULT_ENTITY_TYPE
    return text if text[0].isalpha() else f"t_{text}"


def _plain(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def in_content(evidence: Any, content: str) -> bool:
    """True when *evidence* is words of *content* (case, spacing and punctuation aside)."""
    words = _plain(evidence)
    return bool(words) and words in _plain(content)


@dataclass
class Parsed:
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    dropped: List[Tuple[str, str]] = field(default_factory=list)      # (reason, what)
    added_ends: int = 0                                               # entities made from a relationship's end
    statements: List[str] = field(default_factory=list)               # the facts, when the call wrote them
    uncovered: List[str] = field(default_factory=list)                # statements no relationship came from
    lone: List[str] = field(default_factory=list)                     # entities left out because no relationship uses them
    response_chars: int = 0
    added_belongs: int = 0                                            # joined to the writer's organization by Jev


def _confidence(raw: Any, basis: str) -> float:
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0 if basis == STATED else 0.6


def _fold(data: dict, given: Any, label: Optional[str]) -> Optional[str]:
    """Rewrite *data* so that the entity the content calls *given* is the one
    called *label*. Returns the name folded in, to keep as an alias."""
    given = str(given or "").strip()
    if not label or not given or given.lower() == label.lower():
        return None
    low, hit = given.lower(), False
    for raw in data.get("entities") or []:
        if isinstance(raw, dict) and str(raw.get("name") or "").strip().lower() == low:
            raw["name"], hit = label, True
    for raw in data.get("relationships") or []:
        if isinstance(raw, dict):
            for end in ("source_entity", "target_entity"):
                if str(raw.get(end) or "").strip().lower() == low:
                    raw[end], hit = label, True
    return given if hit else None


def parse(data: Any, content: str, writer: Optional[str] = None,
          customer: Optional[str] = None) -> Parsed:
    """Entities and relationships from the model's JSON object.

    *writer*, *customer*: the names the prompt gave the writer and the
    writer's organization. When the content also gives them a name of their own
    ("I'm Imre Menon, at Corliss Freight") the model reports it under
    "writer_name" and "writer_organization", and that entity is the same one:
    the name becomes an alias.
    """
    out = Parsed()
    if not isinstance(data, dict):
        return out
    folded = [(label, _fold(data, data.get(key), label))
              for key, label in (("writer_name", writer), ("writer_organization", customer))]
    by_name: Dict[str, ExtractedEntity] = {}

    def add(name: Any, kind: Any, raw: Optional[dict] = None) -> Optional[ExtractedEntity]:
        name = str(name or "").strip()
        if not name:
            return None
        found = by_name.get(name.lower())
        if found is not None:
            return found
        raw = raw or {}
        aliases = [str(a).strip() for a in raw.get("aliases") or [] if isinstance(a, str) and a.strip()]
        entity = ExtractedEntity(
            entity_id=str(uuid4()), entity_type=entity_type(kind), name=name,
            aliases=[a for a in aliases if a.lower() != name.lower()],
            about=(str(raw["about"]).strip() or None) if isinstance(raw.get("about"), str) else None,
            source_span=raw.get("source_span") if isinstance(raw.get("source_span"), str) else None,
            properties=_model_properties(raw.get("properties")),
        )
        out.entities.append(entity)
        by_name[name.lower()] = entity
        for alias in entity.aliases:
            by_name.setdefault(alias.lower(), entity)
        return entity

    for raw in data.get("entities") or []:
        if isinstance(raw, dict):
            add(raw.get("name"), raw.get("entity_type"), raw)

    for label, alias in folded:
        entity = by_name.get((label or "").lower())
        if entity is not None and alias and alias not in entity.aliases:
            entity.aliases.append(alias)
            by_name.setdefault(alias.lower(), entity)

    out.statements = [str(t).strip() for t in data.get("statements") or [] if isinstance(t, (str, int, float))]
    cited = set()
    seen = set()
    for raw in data.get("relationships") or []:
        if not isinstance(raw, dict):
            continue
        said = f"{raw.get('source_entity')} {raw.get('relationship_type')} {raw.get('target_entity')}"
        if not str(raw.get("source_entity") or "").strip() or not str(raw.get("target_entity") or "").strip():
            out.dropped.append(("an end has no name", said))
            continue
        before = len(out.entities)
        source = add(raw.get("source_entity"), raw.get("source_type"))
        target = add(raw.get("target_entity"), raw.get("target_type"))
        out.added_ends += len(out.entities) - before
        if source is target:
            out.dropped.append(("both ends are the same entity", said))
            continue
        forward = relationship_type.storable(raw.get("relationship_type"))
        backward = relationship_type.storable(raw.get("inverse_type") or raw.get("relationship_type"))
        if (source.entity_id, target.entity_id, forward) in seen or \
                (target.entity_id, source.entity_id, backward) in seen:
            out.dropped.append(("said twice", said))
            continue
        seen.add((source.entity_id, target.entity_id, forward))
        basis = IMPLIED if str(raw.get("basis") or "").strip().lower() == IMPLIED else STATED
        evidence = raw.get("evidence") if isinstance(raw.get("evidence"), str) else ""
        if basis == STATED and not in_content(evidence, content):
            basis = IMPLIED                     # no words of the content say it
        properties = dict(raw.get("properties")) if isinstance(raw.get("properties"), dict) else {}
        properties.update({
            "inverse_type": backward, "basis": basis,
            "confidence": _confidence(raw.get("confidence"), basis), "evidence": evidence.strip(),
        })
        index = raw.get("statement")
        if isinstance(index, int) and 0 <= index < len(out.statements):
            properties["statement"] = out.statements[index]
            cited.add(index)
        out.relationships.append(ExtractedRelationship(
            source_entity_id=source.entity_id, target_entity_id=target.entity_id,
            relationship_type=forward, properties=properties))
    out.uncovered = [t for i, t in enumerate(out.statements) if i not in cited] if cited else []
    return out


_VERB_START = {"will", "would", "can", "could", "must", "may", "might", "should", "shall", "has", "have",
               "had", "is", "was", "were", "are", "does", "did", "do"}


def sentence(source: str, forward: str, target: str) -> str:
    """A relationship as a plain sentence: "Corbin takes Veltrin", "Corbin is
    allergic to penicillin" (a type that does not start with a verb gets "is")."""
    words = relationship_type.readable(forward)
    first, _, rest = words.partition(" ")
    passive = rest.split(" ", 1)[0] == "by"          # "led by", "owned by"
    if passive or not (first in _VERB_START or first.endswith("s") or first.endswith("ed")):
        words = f"is {words}"
    return f"{source} {words} {target}"


# What "belongs" is called between a thing and the writer's organization.
_BELONGS = {"person": ("MEMBER_OF", "HAS_MEMBER"), "team": ("PART_OF", "HAS_PART"),
            "department": ("PART_OF", "HAS_PART")}
BELONGS_AT = float(os.environ.get("SYNAP_GRAPH_BELONGS_AT", "0.8"))


async def support(content: str, claims: Sequence[str], reversed_claims: Sequence[str] = (),
                  things: Sequence[str] = ()):
    """One Jev request about *content*, every question in it:
      - each claim, and each claim read the other way round: (forward, reversed)
      - each of *things*: does it belong to the writer's organization
    Returns (pairs, belongs), or None when Jev is not configured, switched
    off or failed. The note is sent once and the questions are answered side
    by side, so asking more of them costs no extra call."""
    from synap.cloud.shared.jev_client import ask, jev_configured
    if not (claims or things) or not support_check() or not jev_configured():
        return None
    from synap.cloud.agents.ingestion.resolution.jev_settings import JEV_MODEL, JEV_TIMEOUT_S
    questions = {}
    for key, texts in (("r", claims), ("v", reversed_claims)):
        for i, claim in enumerate(texts):
            questions[f"{key}{i}"] = {
                "type": "noul", "instructions": {"claim": claim, "question": SUPPORT_QUESTION},
                "criteria": SUPPORT_CRITERIA}
    for i, thing in enumerate(things):
        questions[f"b{i}"] = {"type": "noul", "instructions": {"thing": thing, "question": BELONGS_QUESTION},
                              "criteria": BELONGS_CRITERIA}
    try:
        reply = await ask({"note": content}, questions, model=JEV_MODEL,
                          timeout=max(JEV_TIMEOUT_S, 6.0), retries=1)
    except Exception:  # noqa: BLE001 - an enhancement, never a gate
        logger.warning("Graph support check failed; model confidence kept", exc_info=True)
        return None

    def score(key: str) -> Optional[float]:
        value = (reply.get(key) or {}).get("noul")
        return float(value) if value is not None else None

    try:
        return ([(score(f"r{i}"), score(f"v{i}")) for i in range(len(claims))],
                [score(f"b{i}") for i in range(len(things))])
    except (AttributeError, TypeError, ValueError):
        logger.warning("Graph support check reply unreadable; model confidence kept", exc_info=True)
        return None


# A relationship the content does not support as written, but clearly supports
# with its two ends exchanged, was written the wrong way round.
REVERSED_AT = float(os.environ.get("SYNAP_GRAPH_REVERSED_AT", "0.8"))


async def confident(parsed: Parsed, content: str, shown: Dict[str, str], notes: str = "",
                    customer_id: Optional[str] = None) -> Parsed:
    """*parsed* with each relationship's confidence settled and the ones under
    their bar removed. *shown* maps an entity id to the name a reader is shown.
    *notes*: what the reader was told that the content does not say (who a
    pronoun means), so the judge reads the content the way the model did.

    *customer_id*: the entity of the writer's organization, when "we" means it.
    Every other entity not yet joined to it is asked about in the same request:
    does the note place it inside the organization? The prompt's rule for
    "our design team" is not followed, whatever its wording; this asks of the
    note itself, not of the word "our"."""
    # With no judge (Core, or Cloud without a Jev key) there is nothing to word the claims for, and the
    # judge's prompt module does not exist in Core: its wording is not read, every relationship keeps the
    # model's own confidence, and no "belongs to the company" question is asked.
    from synap.cloud.shared.jev_client import jev_configured
    judge_on = support_check() and jev_configured()
    words = _FOR_THE_JUDGE if judge_on else {}

    def claim(rel, flip=False):
        a, b = shown.get(rel.source_entity_id, "?"), shown.get(rel.target_entity_id, "?")
        a, b = words.get(a.lower(), a), words.get(b.lower(), b)
        return sentence(b, rel.relationship_type, a) if flip else sentence(a, rel.relationship_type, b)

    joined = {end for r in parsed.relationships for end in (r.source_entity_id, r.target_entity_id)
              if customer_id in (r.source_entity_id, r.target_entity_id)}
    loose = [e for e in parsed.entities if judge_on and customer_id and e.entity_id != customer_id
             and e.entity_id not in joined and shown.get(e.entity_id, "").lower() not in words]
    claims = [claim(r) for r in parsed.relationships]
    answer = await support(content + (f"\n\n({notes})" if notes else ""), claims,
                           [claim(r, flip=True) for r in parsed.relationships],
                           [shown.get(e.entity_id, e.name) for e in loose])
    scores, belongs = answer if answer else (None, None)
    kept = []
    for i, rel in enumerate(parsed.relationships):
        forward, backward = scores[i] if scores else (None, None)
        bar = STATED_MIN if rel.properties["basis"] == STATED else IMPLIED_MIN
        if forward is not None:
            rel.properties["model_confidence"] = rel.properties["confidence"]
            if forward < bar and backward is not None and backward >= REVERSED_AT:
                rel.source_entity_id, rel.target_entity_id = rel.target_entity_id, rel.source_entity_id
                rel.properties["direction_corrected"] = True
                claims[i], forward = claim(rel), backward
            rel.properties["confidence"] = round(forward, 3)
        if rel.properties["confidence"] >= bar:
            kept.append(rel)
        else:
            reversed_score = scores[i][1] if scores else None
            parsed.dropped.append((f"{rel.properties['basis']}, confidence under {bar}",
                                   f"{claims[i]} ({rel.properties['confidence']}"
                                   + (f"; read the other way round {round(reversed_score, 2)}"
                                      if reversed_score is not None else "") + ")"))
    for entity, score in zip(loose, belongs or []):
        if score is None or score < BELONGS_AT:
            continue
        forward, backward = _BELONGS.get(entity.entity_type, ("BELONGS_TO", "OWNS"))
        kept.append(ExtractedRelationship(
            source_entity_id=entity.entity_id, target_entity_id=customer_id, relationship_type=forward,
            properties={"inverse_type": backward, "basis": IMPLIED, "confidence": round(score, 3),
                        "evidence": "", "added_by": "belongs_check"}))
        parsed.added_belongs += 1
    parsed.relationships = kept
    return parsed


def _model_properties(value):
    from synap.cloud.agents.ingestion.services.writer_entity import model_properties
    return model_properties(value)
