"""Neutral interface for the prompt module merged_graph.

The text is in ``cloud/merged_graph.py`` (Cloud) and ``core/merged_graph.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
