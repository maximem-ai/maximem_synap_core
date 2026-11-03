"""Compaction v3: one bounded writer call that updates a stored record.

See COMPACTION_V3_PLAN.md at the repo root for the design and the issues
each part answers.
"""
from .chunker import Turn, chunk_turns, label_turns, render_turns
from .engine import CompactionWriter, EngineSettings, UpdateResult, build_guidance
from .record import AGREED_ACTIONS_HEADER, CompactionRecord, Section, is_banned_header
from .render import FRAMING, render_context_block, render_document, strip_citations

__all__ = [
    "AGREED_ACTIONS_HEADER", "CompactionRecord", "CompactionWriter", "EngineSettings", "FRAMING",
    "Section", "Turn", "UpdateResult", "build_guidance", "chunk_turns", "is_banned_header",
    "label_turns", "render_context_block", "render_document", "render_turns", "strip_citations",
]
