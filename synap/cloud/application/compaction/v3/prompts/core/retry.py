"""Core's line the v3 writer call adds when the answer was not valid JSON."""

from synap.cloud.shared.prompt_switch import pack_text

JSON_RETRY_SUFFIX = "\n\n" + pack_text("compaction_v3", "json_retry.txt").strip()
