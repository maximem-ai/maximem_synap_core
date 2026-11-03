"""Neutral interface for the prompt module graph_v2_judge.

The text is in ``cloud/graph_v2_judge.py`` (Cloud) and ``core/graph_v2_judge.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
