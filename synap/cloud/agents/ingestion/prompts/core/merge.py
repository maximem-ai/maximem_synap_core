"""Core's text for merging memories that state the same fact.

The system prompt is read from the Core prompt pack (``core_prompts/coalesce_merge``).
"""

from synap.cloud.shared.prompt_switch import pack_text

MERGE_SYSTEM_PROMPT = pack_text("coalesce_merge", "merge.txt")


def build_merge_user_prompt(memory_contents: "list[str]", memory_types: "list[str]" = None) -> str:
    """The numbered list of memories to write one memory for."""
    types = memory_types or [""] * len(memory_contents)
    lines = ["Memories that describe the same fact:"]
    for i, (content, mtype) in enumerate(zip(memory_contents, types), 1):
        suffix = f" [{mtype}]" if mtype else ""
        lines.append(f"{i}.{suffix} {content}")
    return "\n".join(lines)
