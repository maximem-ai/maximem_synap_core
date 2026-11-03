"""Core's preamble above a compacted record. Read from ``core_prompts/compaction_v3``."""

from synap.cloud.shared.prompt_switch import pack_text

FRAMING = pack_text("compaction_v3", "preamble.txt").strip()
