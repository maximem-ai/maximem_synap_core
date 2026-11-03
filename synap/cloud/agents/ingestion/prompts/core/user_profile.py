"""Core's text for the user-profile merge.

The prompt is read from the Core prompt pack (``core_prompts/user_profile``). The caller builds the
attribute spec, the current view and the evidence lines; these functions put them into the text.
"""

import json

from synap.cloud.shared.prompt_switch import fill, pack_text


class _OverviewRule(str):
    """What ``render_overview_rule`` gives back: the engine passes it on to ``build_merge_prompt_text``."""

    enabled: bool
    max_tokens: int


def render_overview_rule(user_profile_cfg):
    """Whether the overview is kept up to date, and how long it may be, from the profile config."""
    rule = _OverviewRule("overview on" if user_profile_cfg.overview_enabled else "overview off")
    rule.enabled = bool(user_profile_cfg.overview_enabled)
    rule.max_tokens = int(user_profile_cfg.max_profile_tokens)
    return rule


def build_merge_prompt_text(*, attrs_spec, current_view, memory_lines, analysis,
                            conversation_id, overview_rule):
    return fill(
        pack_text("user_profile", "system.txt"),
        attribute_specs_json=json.dumps(attrs_spec, indent=2),
        current_profile_json=json.dumps(current_view, indent=2),
        memory_lines_json=json.dumps(memory_lines, indent=2),
        analysis_json=json.dumps(analysis, indent=2, default=str),
        conversation_id_json=json.dumps(conversation_id),
        overview_enabled="yes" if overview_rule.enabled else "no",
        max_profile_tokens=overview_rule.max_tokens,
    )
