"""Lightweight BM25 scorer for retrieval re-ranking and novelty detection.

Pure Python, zero external dependencies. Designed for scoring queries against
small candidate sets (< 100 docs) at retrieval time.

Reuses the stemming+stop-word tokenizer from synap.cloud.shared.tokenizer.
"""

import math
from typing import List

from synap.cloud.shared.tokenizer import tokenize


class BM25:
    """Okapi BM25 scorer over a small in-memory corpus.

    With tiny corpora (1-5 docs), standard IDF can be very low.
    We use a floored IDF of max(idf, 0.5) so that matching terms
    always contribute meaningful signal.
    """

    IDF_FLOOR = 0.5

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avgdl = (
            sum(len(doc) for doc in corpus) / self.doc_count
            if self.doc_count
            else 1.0
        )
        self.doc_lens = [len(doc) for doc in corpus]

        # Precompute IDF for all terms in corpus
        self.idf = {}
        all_terms = set(t for doc in corpus for t in doc)
        for term in all_terms:
            df = sum(1 for doc in corpus if term in set(doc))
            raw_idf = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1.0
            )
            self.idf[term] = max(raw_idf, self.IDF_FLOOR)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """BM25 score for a single document against query tokens."""
        doc = self.corpus[doc_idx]
        dl = self.doc_lens[doc_idx]
        s = 0.0

        tf_map: dict = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term in query_tokens:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            idf = self.idf.get(term, self.IDF_FLOOR)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * numerator / denominator

        return s

    def scores(self, query_tokens: List[str]) -> List[float]:
        """BM25 scores for all documents in corpus against query tokens."""
        return [self.score(query_tokens, i) for i in range(self.doc_count)]
