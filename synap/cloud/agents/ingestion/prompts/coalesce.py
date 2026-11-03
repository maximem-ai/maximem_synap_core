"""Neutral interface for the prompt module coalesce.

The text is in ``cloud/coalesce.py`` (Cloud) and ``core/coalesce.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
