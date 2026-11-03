"""One edge for one relation between two entities, however it was worded.

"A works at B" in one note and "B employs A" in the next are one fact. The
second graph prompt gives every relationship both readings of its type
(services/graph_v2), so before a relationship is written it is compared with
what is already stored between the same two nodes:

  1. same direction, same type            the store's own MERGE lands on it
  2. opposite direction, and one's type
     is the other's inverse               written as the stored edge
  3. a different type either way          Jev is asked whether the two say the
                                          same thing; at SAME_AT and above the
                                          stored edge is used

Step 3 is a judgement because the names of two types being close says little:
LIKES and DISLIKES are close. Only edges between the same two nodes are ever
compared, so nothing is renamed across the graph. Jev unreachable: step 3 is
skipped and the relationship is written as it came.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from synap.cloud.agents.ingestion.services import graph_v2, relationship_type
from synap.cloud.agents.ingestion.prompts.edge_identity import (
    QUESTION,
    CRITERIA,
)

logger = logging.getLogger(__name__)

SAME_AT = float(os.environ.get("SYNAP_GRAPH_SAME_RELATION_AT", "0.8"))


def _adopt(rel: Any, edge: Any, opposite: bool) -> None:
    """Write *rel* as the stored *edge*."""
    forward, backward = rel.relationship_type, rel.properties.get("inverse_type")
    if opposite:
        rel.source_entity_id, rel.target_entity_id = rel.target_entity_id, rel.source_entity_id
    rel.relationship_type = edge.edge_type
    rel.properties["inverse_type"] = (edge.properties or {}).get("inverse_type") or (
        forward if opposite else backward)


async def _same(first: Sequence[str], second: str) -> Optional[List[Optional[float]]]:
    from synap.cloud.shared.jev_client import ask, jev_configured
    if not first or not graph_v2.support_check() or not jev_configured():
        return None
    from synap.cloud.agents.ingestion.resolution.jev_settings import JEV_MODEL, JEV_TIMEOUT_S
    questions = {
        f"e{i}": {"type": "noul", "instructions": {"first": f, "second": second, "question": QUESTION},
                  "criteria": CRITERIA}
        for i, f in enumerate(first)
    }
    try:
        reply = await ask({"statements": {"second": second}}, questions, model=JEV_MODEL,
                          timeout=JEV_TIMEOUT_S, retries=1)
    except Exception:  # noqa: BLE001 - an enhancement, never a gate
        logger.warning("Same-relation check failed; relationship written as it came", exc_info=True)
        return None
    return [(reply.get(f"e{i}") or {}).get("noul") for i in range(len(first))]


async def settle(rel: Any, existing: Sequence[Any], shown: Dict[str, str]) -> Optional[str]:
    """Rewrite *rel* in place when an edge in *existing* already states it.
    *shown* maps a node id to its name. Returns how it was matched, or None."""
    if not existing or not rel.properties.get("inverse_type"):
        return None
    forward, backward = rel.relationship_type, rel.properties["inverse_type"]
    others = []
    for edge in existing:
        opposite = edge.from_id != rel.source_entity_id
        inverse = (edge.properties or {}).get("inverse_type")
        if not opposite and edge.edge_type == forward:
            return "same"
        if opposite and (edge.edge_type == backward or (inverse and inverse == forward)):
            _adopt(rel, edge, True)
            return "inverse"
        others.append((edge, opposite))
    said = graph_v2.sentence(shown.get(rel.source_entity_id, "A"), forward,
                             shown.get(rel.target_entity_id, "B"))
    scores = await _same(
        [graph_v2.sentence(shown.get(e.from_id, "A"), e.edge_type, shown.get(e.to_id, "B"))
         for e, _ in others], said)
    if not scores:
        return None
    best = max(range(len(others)), key=lambda i: scores[i] or 0.0)
    if (scores[best] or 0.0) >= SAME_AT:
        edge, opposite = others[best]
        logger.info("edge_identity: %r written as stored %s (%.2f)", said, edge.edge_type, scores[best])
        _adopt(rel, edge, opposite)
        return "judged"
    return None


__all__ = ["settle", "SAME_AT", "relationship_type"]
