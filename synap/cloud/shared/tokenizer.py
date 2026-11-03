"""Lightweight tokenizer for BM25 and overlap detection.

Mirrors the tokenizer in synap/sdk/python/maximem_synap/cache/bm25.py
to avoid cross-package imports. Keep in sync if the SDK version changes.
"""

import re
from typing import List

# Ordered longest-first. Only strip suffixes that are common inflections
# and unlikely to change the root meaning.
_SUFFIXES = (
    "ences", "ances",
    "ation", "tion", "sion",
    "ence", "ance",
    "ness", "ment", "able", "ible",
    "ical", "ally",
    "ies", "ive", "ful", "ous",
    "ing", "ary",
    "ly", "ed", "es",
    "s",
)

_MIN_STEM_LEN = 4


def _stem(word: str) -> str:
    """Basic suffix-stripping normalizer."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM_LEN:
            return word[: -len(suffix)]
    return word


# Common English stop words to ignore
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can",
    "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "nor", "so", "if", "then", "than",
    "about", "into", "over", "after", "before",
})


# An alias label, as it appears in text: `[[PERSON_PHONE_h2n7v5cx8m0d]]`.
#
# Matched here so it can be kept whole. Left to the ordinary tokenizer it
# becomes three tokens, `person`, `phone` and the random suffix, and the first
# two are boilerplate shared by every label of that field type. Two memories
# about two different people's phone numbers would then agree on most of their
# tokens and the deduplicator would treat them as the same memory, dropping
# one. That is invariant I6 broken by a tokenizer: aliasing on its own is
# supposed to be invisible to the client, and this made it delete their data.
#
# Kept whole, the label contributes exactly one token that differs between two
# values, which is what the real value contributed before aliasing.
_ALIAS_TOKEN = re.compile(r"\[\[([A-Za-z][A-Za-z0-9_]*_[a-z0-9]{12})\]\]")


def tokenize(text: str) -> List[str]:
    """Tokenize, lowercase, remove stop words, and stem.

    Alias labels survive as single tokens. See `_ALIAS_TOKEN`.
    """
    if not text:
        return []

    labels: List[str] = []

    def _take(match: "re.Match") -> str:
        labels.append(match.group(1).lower())
        return " "

    remainder = _ALIAS_TOKEN.sub(_take, text)
    tokens = re.findall(r"[a-z0-9]+", remainder.lower())
    return labels + [
        _stem(t) for t in tokens if len(t) >= 2 and t not in _STOP_WORDS
    ]
