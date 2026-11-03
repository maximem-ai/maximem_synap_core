"""Neutral interface for the prompt module relevance_filter.

The text is in ``cloud/relevance_filter.py`` (Cloud) and ``core/relevance_filter.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
