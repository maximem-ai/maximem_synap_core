"""Core's Jev questions for finding the memories a forget request names.

Read from the Core prompt pack (``core_prompts/forget_judge``). Same names as the Cloud module.
"""

from typing import List

from synap.cloud.shared.prompt_switch import fill, pack_text

# Ordered; position IS the score, starting at zero.
RUBRIC: List[str] = [pack_text("forget_judge", f"rubric_{n}.txt").strip() for n in range(4)]

PURPOSE = pack_text("forget_judge", "purpose.txt").strip()


def rate_instruction(i):
    """The instruction of score question ``s{i}``."""
    return fill(pack_text("forget_judge", "question.txt").strip(), index=i)
