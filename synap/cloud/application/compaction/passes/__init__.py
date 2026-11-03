"""The v2 compaction passes.

The product no longer runs these: compaction v3 (``../v3``) replaced them on
17 Oct 2026 with one bounded writer call. They are kept for two reasons only:
``base.py`` holds the compaction profiles every caller resolves, and the
model-swap harness runs passes 1-4 in its ``--pipeline v2`` mode so v3 can
be measured against what it replaced. Delete passes 1-4 and their prompts
when that comparison is no longer wanted.
"""

import importlib

from .base import BaseCompactionPass

# The four v2 passes are loaded on first use: Core does not ship them (or their
# prompts), and nothing in the product runs them. Cloud gets the same names, the
# same objects, through this lookup.
_LAZY = {
    "Pass1Classification": ".pass1_classify",
    "Pass2Extraction": ".pass2_extract",
    "Pass3Summarization": ".pass3_summarize",
    "Pass4Validation": ".pass4_validate",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "BaseCompactionPass",
    "Pass1Classification",
    "Pass2Extraction",
    "Pass3Summarization",
    "Pass4Validation",
]
