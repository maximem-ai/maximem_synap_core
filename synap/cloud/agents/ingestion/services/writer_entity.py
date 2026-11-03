"""The writer of a document as an entity of their own.

"I report to Leopold" states a relationship whose source is the writer. The
entity prompts leave the writer out, so the relationship prompt, which may only
connect entities it is given, puts that relationship between the two entities
it does have. With ``SYNAP_WRITER_ENTITY`` set, Stage 3 hands the writer in as
one more entity whenever the document speaks in the first person.

One node per user, never resolved by name:
  - ``name`` is "the user", the words the memory prompts already use, so a
    memory that says "The user reports to Leopold" is linked to this entity
    like any other (Stage 3 matches entity names in memory text).
  - ``canonical_name`` is the user id: unique, never renamed, and not personal
    data. ``display_name`` is what a reader is shown, "the user" until a real
    name is known and allowed.
  - ``er_node_key`` is the user id, so the graph writer keeps one node for the
    user whatever it is called, and entity resolution leaves it alone: matching
    "the user" by name would merge every user of the customer into one.

Retrieval cannot find this node the way it finds others (by the entity's name
appearing in a memory it retrieved), so ``anchor_for`` names it directly when
the question itself is in the first person.

Off by default. A document with no user id, or with no "I", is untouched.

The customer the writer belongs to is an entity in the same way, for an
instance whose users are members of the customer (a company's staff talking to
a billing agent: "our plan", "we were charged twice"). With
``SYNAP_CUSTOMER_ENTITY`` set, "we" and "our" mean that node: one per customer,
keyed on the customer id. It is a separate flag because it is wrong for an
instance whose users are a customer's consumers, where "we" is a household.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence
from uuid import uuid4

from synap.cloud.agents.ingestion.models import ExtractedEntity
from synap.cloud.agents.ingestion.prompts.extraction.relationships import writer_speaks
from synap.cloud.agents.ingestion.resolution.node_key import NODE_KEY_PROPERTY

_FLAG_ENV = "SYNAP_WRITER_ENTITY"

LABEL = "the user"
WRITER_PROPERTY = "is_writer"

_CUSTOMER_FLAG_ENV = "SYNAP_CUSTOMER_ENTITY"
CUSTOMER_LABEL = "the user's company"
CUSTOMER_PROPERTY = "is_customer"
_FIRST_PLURAL = re.compile(r"\b(?i:we|us|our|ours|ourselves)\b")
_QUOTED = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d')


def enabled(instance_id: Optional[Any] = None) -> bool:
    """SYNAP_WRITER_ENTITY, or the second graph prompt, which always has the writer."""
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_FLAG_ENV, instance_id) or enabled_for("SYNAP_GRAPH_EXTRACTION_V2", instance_id)


def node_name(user_id: Any) -> str:
    """The canonical name of this user's node: the user id itself."""
    return str(user_id)


def shown_name(properties: Any, fallback: str) -> str:
    """What a reader, or an embedding, should see for a node: the display
    name for the writer's or the customer's node (its canonical name is an
    id), else *fallback*."""
    properties = properties or {}
    if properties.get(WRITER_PROPERTY):
        return properties.get("display_name") or LABEL
    if properties.get(CUSTOMER_PROPERTY):
        return properties.get("display_name") or CUSTOMER_LABEL
    return fallback


def label_for(content: str, scope: Any) -> Optional[str]:
    """The name to give the prompts for the writer of *content*, or None."""
    if not enabled(getattr(scope, "instance_id", None)) or not getattr(scope, "user_id", None):
        return None
    return LABEL if writer_speaks(content) else None


def model_properties(value: Any) -> dict:
    """The properties a model wrote for an entity, without the keys that mark
    the writer's or the customer's node. Those are set by code (_mark,
    _mark_customer) and by nothing else: a model that writes ``is_customer`` on
    "Acme" must not turn Acme into the user's company."""
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items()
            if k not in (WRITER_PROPERTY, CUSTOMER_PROPERTY, NODE_KEY_PROPERTY)}


def is_writer(entity: Any) -> bool:
    return bool((getattr(entity, "properties", None) or {}).get(WRITER_PROPERTY))


def is_customer(entity: Any) -> bool:
    return bool((getattr(entity, "properties", None) or {}).get(CUSTOMER_PROPERTY))


def is_fixed(entity: Any) -> bool:
    """The writer or the writer's customer: a node keyed on an id, which entity
    resolution leaves alone and which carries no memory references."""
    return is_writer(entity) or is_customer(entity)


def customer_enabled(instance_id: Optional[Any] = None) -> bool:
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for(_CUSTOMER_FLAG_ENV, instance_id)


def says_we(content: str) -> bool:
    """True when the content says "we", "us" or "our" outside a quotation."""
    return bool(_FIRST_PLURAL.search(_QUOTED.sub(" ", content or "")))


def customer_label_for(content: str, scope: Any) -> Optional[str]:
    """The name to give the prompt for the writer's customer, or None."""
    if not customer_enabled(getattr(scope, "instance_id", None)):
        return None
    if not getattr(scope, "customer_id", None) or not getattr(scope, "user_id", None):
        return None
    return CUSTOMER_LABEL if says_we(content) else None


def _mark_customer(entity: ExtractedEntity, scope: Any) -> ExtractedEntity:
    entity.entity_type = "organization"
    entity.canonical_name = str(scope.customer_id)
    entity.properties.update({CUSTOMER_PROPERTY: True, "display_name": CUSTOMER_LABEL,
                              NODE_KEY_PROPERTY: f"customer:{scope.customer_id}"})
    return entity


def customer_stamped(entities: Sequence[ExtractedEntity], scope: Any) -> List[ExtractedEntity]:
    """*entities* with the one the model returned under the customer's name
    marked, or a new one when it returned none: the note says "we", and what
    belongs to "us" needs a node to belong to. ``without_unused`` drops it
    again if nothing ends up joined to it."""
    out = list(entities)
    for e in out:
        if e.name.strip().lower() == CUSTOMER_LABEL:
            e.name = CUSTOMER_LABEL
            _mark_customer(e, scope)
            return out
    return out + [_mark_customer(ExtractedEntity(entity_id=str(uuid4()), entity_type="organization",
                                                 name=CUSTOMER_LABEL), scope)]


def customer_anchor_for(queries: Sequence[str], scope: Any, entity_names: Sequence[str]) -> Optional[str]:
    """The asker's customer node as a place to start the graph walk, when a
    query says "we" or "our" and the node is in reach."""
    customer_id = getattr(scope, "customer_id", None)
    if not customer_enabled(getattr(scope, "instance_id", None)) or not customer_id:
        return None
    name = str(customer_id)
    if name not in entity_names:
        return None
    return name if any(says_we(q or "") for q in queries) else None


def _mark(entity: ExtractedEntity, scope: Any) -> ExtractedEntity:
    entity.entity_type = "person"
    entity.canonical_name = node_name(scope.user_id)
    entity.properties.update({WRITER_PROPERTY: True, "display_name": LABEL,
                              NODE_KEY_PROPERTY: f"user:{scope.user_id}"})
    return entity


def entity(scope: Any) -> ExtractedEntity:
    return _mark(ExtractedEntity(entity_id=str(uuid4()), entity_type="person", name=LABEL), scope)


def stamped(entities: Sequence[ExtractedEntity], scope: Any) -> List[ExtractedEntity]:
    """*entities* with the writer among them: the one the model returned under
    the writer's name, marked, or a new one when it returned none."""
    out = list(entities)
    for e in out:
        if e.name.strip().lower() == LABEL:
            e.name = LABEL
            _mark(e, scope)
            return out
    return out + [entity(scope)]


def without_unused(entities: Sequence[ExtractedEntity], relationships: Iterable[Any]) -> List[ExtractedEntity]:
    """Drop the writer when no relationship uses it: a node with no edge says nothing."""
    used = {i for r in relationships for i in (r.source_entity_id, r.target_entity_id)}
    return [e for e in entities if not is_fixed(e) or e.entity_id in used]


def anchor_for(queries: Sequence[str], scope: Any, entity_names: Sequence[str]) -> Optional[str]:
    """The asker's own node as a place to start the graph walk: when a query is
    in the first person and the node exists among the entities in reach."""
    user_id = getattr(scope, "user_id", None)
    if not enabled(getattr(scope, "instance_id", None)) or not user_id:
        return None
    name = node_name(user_id)
    if name not in entity_names:
        return None
    return name if any(writer_speaks(q or "") for q in queries) else None
