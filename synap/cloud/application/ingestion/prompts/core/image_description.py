"""Core's prompt for the vision model. Read from ``core_prompts/image_description``."""

from synap.cloud.shared.prompt_switch import pack_text

VISION_PROMPT = pack_text("image_description", "system.txt").strip()
