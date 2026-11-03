"""Core's header above earlier memories given to the extractor.

Read from ``core_prompts/extraction_blocks/prior_context.txt``. ``HEADER.format(context=...)`` as the
engine calls it.
"""

from synap.cloud.shared.prompt_switch import PackTemplate, pack_text

HEADER = PackTemplate(pack_text("extraction_blocks", "prior_context.txt"))
