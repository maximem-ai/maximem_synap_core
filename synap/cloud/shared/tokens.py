"""
Token counting utilities for budget enforcement.

Uses tiktoken for accurate counting. This is the one token counter for the
server: retrieval budgets, ingestion and compaction triggers all count with
it, so a "15K" in one place is the same 15K in another.

The cl100k_base encoding ships in ``data/`` because tiktoken otherwise
downloads it on first use, and an air-gapped install has nowhere to
download it from. The file is checked against tiktoken's published hash.
"""

import base64
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency in lightweight envs
    tiktoken = None

logger = logging.getLogger(__name__)

_BUNDLED_CL100K = Path(__file__).resolve().parent / "data" / "cl100k_base.tiktoken"
_CL100K_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
# Copied from tiktoken_ext.openai_public.cl100k_base.
_CL100K_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
_CL100K_SPECIAL = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}
_bundled_encoding = None


def _read_bundled_ranks() -> Dict[bytes, int]:
    """Parse the bundled .tiktoken file. tiktoken's own loader would do
    this too, but it insists on a writable cache directory even for a local
    file, which a read-only container does not have."""
    data = _BUNDLED_CL100K.read_bytes()
    if hashlib.sha256(data).hexdigest() != _CL100K_SHA256:
        raise ValueError(f"{_BUNDLED_CL100K} does not match the published cl100k_base hash")
    ranks: Dict[bytes, int] = {}
    for line in data.splitlines():
        if line:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


def _cl100k():
    """cl100k_base from the bundled file, without touching the network.

    Returns None when neither the bundled file nor tiktoken's own copy can be
    loaded; the counter then falls back to a word count.
    """
    global _bundled_encoding
    if _bundled_encoding is None and tiktoken is not None:
        try:
            _bundled_encoding = tiktoken.Encoding(
                name="cl100k_base",
                pat_str=_CL100K_PATTERN,
                mergeable_ranks=_read_bundled_ranks(),
                special_tokens=_CL100K_SPECIAL,
            )
        except Exception as exc:  # a broken install must not stop counting
            logger.warning("Bundled cl100k_base encoding unavailable (%s); trying tiktoken", exc)
            try:
                _bundled_encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as exc2:
                logger.warning("tiktoken cl100k_base unavailable too (%s); counting words", exc2)
    return _bundled_encoding


class TokenCounter:
    """
    Count tokens for budget enforcement.

    Default model: gpt-4 (cl100k_base encoding)
    """

    def __init__(self, model: str = "gpt-4"):
        """Initialize with model's tokenizer."""
        self.encoding = None
        if tiktoken is not None:
            try:
                name = tiktoken.encoding_name_for_model(model)
            except KeyError:
                name = "cl100k_base"
            if name == "cl100k_base":
                self.encoding = _cl100k()
            else:
                try:
                    self.encoding = tiktoken.get_encoding(name)
                except Exception:
                    # Not cached and no network: count with the bundled one.
                    self.encoding = _cl100k()

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self.encoding is None:
            # Lightweight fallback when tiktoken is unavailable.
            return len(text.split())
        return len(self.encoding.encode(text))

    def count_items(self, items: List[Dict[str, Any]], content_key: str = "content") -> int:
        """Count total tokens across items."""
        total = 0
        for item in items:
            content = item.get(content_key, "")
            if content:
                total += self.count(content)
        return total

    def truncate_to_budget(
        self,
        items: List[Dict[str, Any]],
        budget: int,
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """
        Truncate items list to fit within token budget.

        Items are assumed to be sorted by relevance (most relevant first).
        Returns items that fit within budget.
        """
        result = []
        total_tokens = 0

        for item in items:
            content = item.get(content_key, "")
            item_tokens = self.count(content) if content else 0

            if total_tokens + item_tokens <= budget:
                result.append(item)
                total_tokens += item_tokens
            else:
                # Budget exceeded, stop adding items
                break

        return result


# Global instance for convenience
_default_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get or create default token counter."""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter()
    return _default_counter


def count_tokens(text: str) -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count(text)
