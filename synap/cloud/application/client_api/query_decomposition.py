"""
Compound query decomposition using spaCy dependency parsing.

Detects coordinating conjunctions ("or", "and") in queries and splits
compound predicates into independent subqueries for better retrieval recall.

Uses structure-aware reconstruction that handles:
- Chain coordination (buy, assemble, sell, or fix → 4 subqueries)
- Shared pre-modifiers for noun coordination (different museums or galleries)
- Shared trailing PPs when only the last branch has one (return from a store)
- xcomp parent inclusion (learned to cook or tried out)
- Conditional aux sharing (did I buy/sell vs have I led / am I leading)

Falls back to semantic pattern decomposition when no conjunction is found:
- Comparison: "X compared to Y" / "X versus Y" / "X vs Y"
- Inclusion: ", including X" / ", especially X"
- Temporal: "before/after/when/while [clause]"

Example:
    "How much did I spend in Hawaii compared to Tokyo?"
    → ["How much did I spend in Hawaii?",
       "How much did I spend in Tokyo?"]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

import spacy

logger = logging.getLogger(__name__)

# Load once at module level — key to sub-10ms latency.
# Disable NER and lemmatizer since we only need dependency parsing.
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

_TEMPORAL_MARKERS = frozenset(("before", "after", "when", "while", "since", "until", "once"))


@dataclass
class DecomposeResult:
    is_compound: bool
    subqueries: List[str] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _subtree_ids(token) -> Set[int]:
    return {t.i for t in token.subtree}


def _tokens_to_text(doc, indices: Set[int]) -> str:
    return " ".join(doc[i].text for i in sorted(indices)).strip()


# ── Conjunction decomposition ──────────────────────────────────────────────────


def _find_coord_root(head):
    """Trace up conj chain to the root of the coordination."""
    while head.dep_ == "conj":
        head = head.head
    return head


def _collect_branches(root):
    """BFS to collect all branches in a coordination chain."""
    branches = [root]
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in node.children:
            if child.dep_ == "conj":
                branches.append(child)
                queue.append(child)
    branches.sort(key=lambda t: t.i)
    return branches


def _is_clausal(root, conj_tokens):
    """
    Check if the coordination joins full clauses rather than verb/noun phrases.

    "I'm planning to visit AND I was wondering" → clausal (both have subjects) → skip
    "pick up or return" → verb phrase coordination → decompose
    """
    root_has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
    for conj in conj_tokens:
        if any(c.dep_ in ("nsubj", "nsubjpass") for c in conj.children) and root_has_subj:
            return True
    return False


def _get_direct_tokens(branch):
    """Get tokens directly owned by a branch: the branch token + non-conj/cc/comma children subtrees."""
    own = {branch.i}
    for child in branch.children:
        if child.dep_ in ("conj", "cc"):
            continue
        if child.dep_ == "punct" and child.text == ",":
            continue
        for t in child.subtree:
            own.add(t.i)
    return own


def _build_subqueries(doc, branches, coord_root):
    """
    Build N subqueries from a coordinated structure using structure-aware reconstruction.

    Strategy:
    1. Compute direct tokens for each branch (excluding conj/cc/comma children)
    2. For verb coordination: promote structural deps (nsubj, dobj, aux) to shared
    3. For noun coordination: share head's pre-modifiers when conjuncts lack their own
    4. Share trailing PPs when only the last branch has one
    5. Include xcomp parent in head's branch to prevent dangling ancestors
    6. Everything unassigned becomes shared context for all subqueries
    """
    # ── Collect delimiters ──
    cc_ids = set()
    comma_ids = set()
    for b in branches:
        for child in b.children:
            if child.dep_ == "cc":
                cc_ids.add(child.i)
            if child.dep_ == "punct" and child.text == ",":
                comma_ids.add(child.i)

    # ── Direct tokens per branch ──
    branch_groups = [_get_direct_tokens(b) for b in branches]
    for g in branch_groups:
        g -= cc_ids | comma_ids

    shared_extra: Set[int] = set()

    # ── Verb coordination: move structural deps to shared ──
    if coord_root.pos_ in ("VERB", "ADJ"):
        for child in coord_root.children:
            if child.dep_ in ("nsubj", "nsubjpass", "dobj", "attr", "npadvmod"):
                sub_ids = _subtree_ids(child)
                shared_extra |= sub_ids
                branch_groups[0] -= sub_ids
            if child.dep_ == "punct" and child.text in ("?", ".", "!"):
                shared_extra.add(child.i)
                branch_groups[0].discard(child.i)

        # Share aux only if no conjunct has own aux AND root is not xcomp.
        # When a conjunct has its own aux (e.g. "have led or am leading"),
        # the root's aux is tense-specific and must stay with its branch.
        # When root is xcomp, its aux connects to the parent ("to" in "need to pick up").
        conj_has_aux = any(
            any(c.dep_ in ("aux", "auxpass") for c in b.children)
            for b in branches[1:]
        )
        is_xcomp = coord_root.dep_ == "xcomp"
        if not conj_has_aux and not is_xcomp:
            for child in coord_root.children:
                if child.dep_ in ("aux", "auxpass"):
                    sub_ids = _subtree_ids(child)
                    shared_extra |= sub_ids
                    branch_groups[0] -= sub_ids

    # ── Noun coordination: share head's pre-modifiers ──
    # Heuristic: if conjuncts lack their own pre-modifiers, head's modifiers
    # apply to all branches (e.g. "different museums or galleries").
    # If conjuncts have own modifiers, they're branch-specific
    # (e.g. "Marvel... movies and Star Wars films").
    if coord_root.pos_ in ("NOUN", "PROPN"):
        premods = set()
        for child in coord_root.children:
            if child.i < coord_root.i and child.dep_ in (
                "amod", "det", "advmod", "nummod", "predet", "compound",
            ):
                for t in child.subtree:
                    premods.add(t.i)

        conj_has_mods = any(
            any(
                c.i < b.i and c.dep_ in ("amod", "det", "compound", "nummod")
                for c in b.children
            )
            for b in branches[1:]
        )
        if premods and not conj_has_mods:
            shared_extra |= premods
            branch_groups[0] -= premods

    # ── Share trailing PP from last branch if unique ──
    # When only the last branch has a prep phrase (e.g. "from a store",
    # "in the past few months"), it's likely a shared modifier, not branch-specific.
    last_pps = [c for c in branches[-1].children if c.dep_ == "prep"]
    other_has_pp = any(
        any(c.dep_ == "prep" for c in b.children) for b in branches[:-1]
    )
    if last_pps and not other_has_pp:
        pp_ids = set()
        for pp in last_pps:
            for t in pp.subtree:
                pp_ids.add(t.i)
        shared_extra |= pp_ids
        branch_groups[-1] -= pp_ids

    # ── xcomp parent: include in head's branch ──
    # When the coord root is xcomp of a parent verb (e.g. "learned to cook or tried out"),
    # include the parent so it stays with the head branch, preventing
    # broken output like "learned tried out".
    if coord_root.dep_ == "xcomp":
        branch_groups[0].add(coord_root.head.i)

    # ── Shared = explicitly promoted + anything unassigned ──
    all_assigned = set().union(*branch_groups) | cc_ids | comma_ids
    shared = {t.i for t in doc if t.i not in all_assigned}
    shared |= shared_extra

    # ── Build subqueries ──
    subqueries = []
    for group in branch_groups:
        keep = shared | group
        sq = " ".join(doc[i].text for i in sorted(keep)).strip()
        subqueries.append(sq)

    return subqueries


# ── Semantic pattern decomposition ─────────────────────────────────────────────


def _try_comparison(doc) -> Optional[List[str]]:
    """
    Detect "X compared to Y" / "X versus Y" / "X vs Y".

    Produces two queries: one with the base entity, one with the compared entity.
    """
    for t in doc:
        if t.text.lower() not in ("compared", "versus", "vs"):
            continue
        if t.dep_ not in ("prep", "advcl", "amod"):
            continue

        comp_subtree = _subtree_ids(t)
        pobjs = [c for c in t.subtree if c.dep_ == "pobj" and c.pos_ in ("PROPN", "NOUN")]
        if not pobjs:
            continue
        compared_entity = pobjs[-1]

        parent = t.head

        # Find the base entity
        base_entity = None
        # Case A: comparison token is child of the base entity itself
        # e.g. "at Costco vs Walmart" → vs is prep of Costco
        if parent.pos_ in ("PROPN", "NOUN") and parent.dep_ == "pobj":
            base_entity = parent
        else:
            # Case B: base entity is in a sibling prep phrase
            # e.g. "spend ... in Hawaii compared to Tokyo"
            for sibling in parent.children:
                if sibling.dep_ == "prep" and sibling.i != t.i:
                    for pobj in sibling.children:
                        if pobj.dep_ == "pobj" and pobj.pos_ in ("PROPN", "NOUN"):
                            base_entity = pobj

        if not base_entity:
            continue

        # Q1: original minus comparison subtree (and preceding comma)
        remove_ids = set(comp_subtree)
        if t.i > 0 and doc[t.i - 1].text == ",":
            remove_ids.add(t.i - 1)
        q1_ids = {tok.i for tok in doc} - remove_ids
        q1 = _tokens_to_text(doc, q1_ids)

        # Q2: Q1 with base entity replaced by compared entity
        base_ids = _subtree_ids(base_entity) - comp_subtree
        q2_tokens = []
        for i in sorted(q1_ids):
            if i == base_entity.i:
                q2_tokens.append(compared_entity.text)
            elif i not in base_ids:
                q2_tokens.append(doc[i].text)
        q2 = " ".join(q2_tokens).strip()

        if q1 and q2 and q1 != q2 and len(q1.split()) > 3 and len(q2.split()) > 3:
            return [q1, q2]

    return None


def _try_inclusion(doc) -> Optional[List[str]]:
    """
    Detect ", including X" / ", especially X" / ", particularly X".

    Produces: main clause without inclusion = Q1, included object = Q2.
    """
    for t in doc:
        if t.text.lower() not in ("including", "especially", "particularly"):
            continue
        if t.dep_ not in ("prep", "advcl"):
            continue

        incl_subtree = _subtree_ids(t)

        # Q1: remove inclusion subtree + preceding comma
        remove_ids = set(incl_subtree)
        if t.i > 0 and doc[t.i - 1].text == ",":
            remove_ids.add(t.i - 1)
        q1_ids = {tok.i for tok in doc} - remove_ids
        q1 = _tokens_to_text(doc, q1_ids)

        # Q2: the included object, with topic noun for context if short
        pobjs = [c for c in t.children if c.dep_ == "pobj"]
        if pobjs:
            q2_content = _tokens_to_text(doc, _subtree_ids(pobjs[0]))
        else:
            q2_content = _tokens_to_text(doc, incl_subtree - {t.i})

        if len(q2_content.split()) <= 3:
            for child in t.head.children:
                if child.dep_ in ("dobj", "nsubj") and child.i not in remove_ids:
                    q2_content = child.text + " " + q2_content
                    break

        q2 = q2_content.strip()

        if q1 and q2 and len(q1.split()) > 3 and len(q2.split()) > 3:
            return [q1, q2]

    return None


def _try_temporal(doc) -> Optional[List[str]]:
    """
    Detect temporal clauses: "before/after/when/while I did X".

    Handles both advcl with temporal mark/advmod and prep with pcomp (gerund).
    Produces: main clause = Q1, temporal anchor event = Q2.
    """
    for t in doc:
        # Case 1: advcl with temporal marker (before/after/when/while + full clause)
        if t.dep_ == "advcl":
            temporal_kids = [
                c for c in t.children
                if c.text.lower() in _TEMPORAL_MARKERS
                and c.dep_ in ("mark", "advmod")
            ]
            if temporal_kids:
                advcl_subtree = _subtree_ids(t)
                remove_ids = set(advcl_subtree)

                # Remove orphaned prep phrases connecting to the temporal clause
                # e.g. "on the day" in "go to bed on the day before I had..."
                if t.head.dep_ == "ROOT":
                    for child in t.head.children:
                        if child.dep_ == "prep" and child.i < t.i:
                            prep_end = max(c.i for c in child.subtree)
                            if prep_end + 1 == min(advcl_subtree) or prep_end + 1 in advcl_subtree:
                                remove_ids |= _subtree_ids(child)

                q1_ids = {tok.i for tok in doc} - remove_ids
                q1 = _tokens_to_text(doc, q1_ids)

                # Q2: temporal event content (minus marker and subject pronouns)
                marker = temporal_kids[0]
                event_ids = advcl_subtree - {marker.i}
                event_ids = {i for i in event_ids if not (doc[i].dep_ == "nsubj" and doc[i].pos_ == "PRON")}
                q2 = _tokens_to_text(doc, event_ids)

                if q1 and q2 and len(q1.split()) > 3 and len(q2.split()) > 3:
                    return [q1, q2]

        # Case 2: prep (before/after) with pcomp (gerund phrase)
        if t.text.lower() in ("before", "after") and t.dep_ == "prep":
            pcomps = [c for c in t.children if c.dep_ == "pcomp"]
            if pcomps:
                prep_subtree = _subtree_ids(t)
                remove_ids = set(prep_subtree)

                q1_ids = {tok.i for tok in doc} - remove_ids
                q1 = _tokens_to_text(doc, q1_ids)

                # Q2: gerund clause content (minus before/after)
                event_ids = prep_subtree - {t.i}
                q2 = _tokens_to_text(doc, event_ids)

                if q1 and q2 and len(q1.split()) > 3 and len(q2.split()) > 3:
                    return [q1, q2]

    return None


# ── Public API ─────────────────────────────────────────────────────────────────


def decompose(text: str) -> DecomposeResult:
    """
    Decompose a compound query into independent subqueries.

    Tries conjunction-based decomposition first, then falls back to
    semantic patterns (comparison, inclusion, temporal clauses).
    Returns the original text unchanged if no decomposable structure is found.
    """
    if not text or not text.strip():
        return DecomposeResult(is_compound=False, subqueries=[text] if text else [])

    doc = nlp(text)

    # ── Pass 1: conjunction decomposition ──
    seen_roots = set()
    for token in doc:
        if token.dep_ != "cc" or token.text.lower() not in ("or", "and"):
            continue

        coord_root = _find_coord_root(token.head)
        if coord_root.i in seen_roots:
            continue
        seen_roots.add(coord_root.i)

        branches = _collect_branches(coord_root)
        if len(branches) < 2:
            continue

        if _is_clausal(coord_root, branches[1:]):
            continue

        subqueries = _build_subqueries(doc, branches, coord_root)
        subqueries = [sq for sq in subqueries if sq and len(sq.split()) > 3]
        if len(subqueries) > 1:
            return DecomposeResult(is_compound=True, subqueries=subqueries)

    # ── Pass 2: semantic pattern decomposition ──
    for pattern_fn in (_try_comparison, _try_inclusion, _try_temporal):
        result = pattern_fn(doc)
        if result:
            return DecomposeResult(is_compound=True, subqueries=result)

    return DecomposeResult(is_compound=False, subqueries=[text])


def decompose_queries(queries: List[str]) -> List[str]:
    """
    Decompose a list of queries, expanding any compound queries.

    Returns a flat list of all subqueries. Non-compound queries pass through unchanged.
    """
    start = time.perf_counter()
    expanded = []
    for q in queries:
        result = decompose(q)
        expanded.extend(result.subqueries)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if len(expanded) != len(queries):
        logger.debug(f"query_decomposition: {len(queries)} queries -> {len(expanded)} subqueries in {elapsed_ms:.1f}ms")
    return expanded
