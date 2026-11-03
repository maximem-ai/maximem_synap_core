"""Core's text for grouping memories that say the same thing.

The system prompt is read from the Core prompt pack (``core_prompts/coalesce_merge``).
"""

from synap.cloud.shared.prompt_switch import pack_text

COALESCE_SYSTEM_PROMPT = pack_text("coalesce_merge", "coalesce.txt")


def build_coalesce_user_prompt(contents: "list[str]", memory_types: "list[str]") -> str:
    """The numbered list the model groups over. Numbering starts at 1, as the system prompt says."""
    lines = ["Memories extracted from one conversation:"]
    for i, (content, mtype) in enumerate(zip(contents, memory_types), 1):
        suffix = f" [{mtype}]" if mtype else ""
        lines.append(f"{i}.{suffix} {content}")
    return "\n".join(lines)
