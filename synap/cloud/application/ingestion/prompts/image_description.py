"""Neutral interface for the prompt module image_description.

The text is in ``cloud/image_description.py`` (Cloud) and ``core/image_description.py``
(Core); the edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
