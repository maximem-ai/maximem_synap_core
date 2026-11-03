"""Neutral interface for the prompt module edge_identity.

The text is in ``cloud/edge_identity.py`` (Cloud) and ``core/edge_identity.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
