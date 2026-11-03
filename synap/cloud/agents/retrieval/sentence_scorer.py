"""Chunk-level evidence scoring for retrieval enrichment.

Given a memory's source chunks and a search query, selects the most
relevant CHUNK (not sentence) to attach as evidence. The full chunk
is sent to the downstream LLM, preserving tables, lists, and
structured content that sentence-level extraction would destroy.

Why chunk-level, not sentence-level:
- Tables like "Admon: 8am-4pm (Day Shift)" are meaningless as isolated rows
- Sentence splitting breaks structured content
- BM25 on sentences picks keyword-rich preambles over data-rich table rows
- The downstream LLM is better at finding specific answers within a 500-token
  chunk than we are at selecting the right 25-token sentence

Token cost: ~300-500 tokens per enriched memory (1 chunk).
For 3 enriched memories: ~900-1500 extra tokens per request.
"""

import os
import re
from collections import Counter
from typing import List, Optional, Set


# ── Support: does the chunk actually say what the fact says? ─────────────────
#
# `score_chunks` below ranks a candidate chunk on query coverage AND on
# novelty, where novelty is the share of the chunk's words that are NOT in the
# memory. 40% of the ranking therefore prefers a chunk that says something
# OTHER than the fact it is about to be attached to. That was deliberate
# ("information gain"), and it has a cost nobody had measured: the chunk we
# attach is the raw passage that most resembles the QUESTION while sharing the
# least with the FACT.
#
# Measured over the 6,688 enriched items of one 500-question benchmark run: the
# attached chunk shares a median 50% of its own fact's content words, 17% share
# under a quarter, and 5% share nothing at all. One item's fact read "the user
# started as a Marketing Coordinator and worked their way up to Senior
# Marketing Specialist over a period of 2 years and 4 months" and the chunk
# attached to it was about a colleague on the sales team; the career sentence
# itself had been attached to a different fact about CRM segmentation.
#
# The reader then answers from the raw passage rather than from the fact. On
# three replicates of 500 questions, refusing to attach a chunk that shares
# less than 25% of its fact's content words scored 441.3 against 436.7 with
# every chunk kept, and the two arms did not overlap (439-443 against 436-438).
# The whole gain was on the questions whose context actually changed (310.3 ->
# 316.3 on 362 questions); the 138 whose context was untouched moved -1.3,
# which is the measurement noise.
#
# OFF by default. Turning it on changes what every caller receives, so it is a
# decision, not a default: SYNAP_EVIDENCE_MIN_SUPPORT=25.
_SUPPORT_STOPWORDS: frozenset = frozenset(
    "the a an of and or to in on at for with is was were be been has have had "
    "this that these those user assistant i my me you your it its they their "
    "them he she his her as by from not no but if so about also just now then "
    "very can could would should will more most some any all what which who "
    "when where how do does did".split())


def _content_words(text: str) -> Set[str]:
    """Words that carry meaning: no stopwords, nothing shorter than 3 letters."""
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
            if w not in _SUPPORT_STOPWORDS and len(w) > 2}


def evidence_min_support() -> float:
    """Percent of a fact's content words a chunk must share to be attached.

    0 keeps every chunk, which is the behaviour before this existed.
    """
    try:
        return max(0.0, min(100.0, float(
            os.environ.get("SYNAP_EVIDENCE_MIN_SUPPORT", "0"))))
    except (TypeError, ValueError):
        return 0.0


def support_pct(chunk: str, memory_content: str) -> float:
    """How much of the fact this chunk actually repeats, 0-100.

    A memory with no content words at all returns 100: there is nothing to
    support, so the chunk is not the thing at fault and is left alone.
    """
    mw = _content_words(memory_content)
    if not mw:
        return 100.0
    return 100.0 * len(mw & _content_words(chunk)) / len(mw)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r'\b\w+\b', text.lower())


def score_chunks(
    chunks: List[str],
    query: str,
    memory_content: str,
) -> List[tuple]:
    """Score full chunks by combined query + novelty coverage.

    Each chunk gets two scores:
    1. Query coverage: fraction of query terms found in chunk
    2. Novelty: fraction of chunk's content-words NOT in the memory
       (weighted by word rarity across chunks — less common = more novel)

    Final score = query_coverage * 0.6 + novelty * 0.4

    This balances relevance (does the chunk address the question?)
    with information gain (does it add details the memory lost?).

    Args:
        chunks: List of chunk text strings
        query: User's search query
        memory_content: The memory's content text

    Returns:
        List of (chunk_text, score) sorted descending by score
    """
    if not chunks or not query:
        return []

    q_tokens = set(_tokenize(query))
    m_tokens = set(_tokenize(memory_content))

    # IDF-like weighting: tokens that appear in fewer chunks are more novel
    chunk_token_sets = [set(_tokenize(c)) for c in chunks]
    doc_freq: Counter = Counter()
    for ts in chunk_token_sets:
        for t in ts:
            doc_freq[t] += 1

    scored = []
    for chunk_text, c_tokens in zip(chunks, chunk_token_sets):
        if not c_tokens:
            continue

        # Query coverage
        q_overlap = len(q_tokens & c_tokens) / len(q_tokens) if q_tokens else 0

        # Novelty: content tokens in chunk but NOT in memory, weighted by rarity
        novel_tokens = c_tokens - m_tokens
        content_tokens = {t for t in novel_tokens if len(t) > 3}
        if content_tokens and doc_freq:
            novelty = sum(1.0 / doc_freq[t] for t in content_tokens) / len(content_tokens)
        else:
            novelty = 0

        score = q_overlap * 0.6 + novelty * 0.4
        scored.append((chunk_text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def extract_evidence(
    chunk_texts: List[str],
    query: str,
    memory_content: str,
    max_chunks: int = 1,
    min_score: float = 0.1,
    seen_chunks: Optional[set] = None,
    min_support: Optional[float] = None,
) -> Optional[List[str]]:
    """Extract the best evidence chunk(s) for a memory.

    Scores all chunks, deduplicates against already-selected chunks
    (from sibling memories), and returns the top chunk(s).

    Args:
        chunk_texts: List of chunk text strings from the memory's source document
        query: User's search query
        memory_content: The memory's content text
        max_chunks: Maximum chunks to return per memory (default: 1)
        min_score: Minimum score threshold for inclusion
        seen_chunks: Set of dedup keys (first 100 chars) already used.
                     Will be mutated to add newly selected chunks.
        min_support: Percent of the memory's content words a chunk must repeat
                     to be eligible. None reads SYNAP_EVIDENCE_MIN_SUPPORT,
                     which is 0 (every chunk eligible) unless set.

    Returns:
        List of chunk texts, or None if no qualifying chunks found

    The support floor is applied BEFORE scoring, not after, so a document that
    holds both a supporting passage and a better-scoring unrelated one yields
    the supporting passage instead of nothing. Dropping the evidence is the
    fallback when the document has no supporting passage at all, not the
    mechanism.
    """
    if not chunk_texts:
        return None

    floor = evidence_min_support() if min_support is None else min_support
    if floor > 0:
        eligible = [c for c in chunk_texts
                    if support_pct(c, memory_content) >= floor]
        if not eligible:
            return None
        chunk_texts = eligible

    scored = score_chunks(chunk_texts, query, memory_content)

    evidence = []
    for text, score in scored:
        if score < min_score:
            break
        dedup_key = text[:100]
        if seen_chunks is not None and dedup_key in seen_chunks:
            continue
        evidence.append(text)
        if seen_chunks is not None:
            seen_chunks.add(dedup_key)
        if len(evidence) >= max_chunks:
            break

    return evidence if evidence else None
