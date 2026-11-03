"""Core's blocks that stage 3 adds to the extraction prompts.

* ``CUSTOMER_WE_SECTION``: the "the user's company" wording for memories, from
  ``core_prompts/person_and_company/customer_memory.txt``.
* ``ANALYSIS_HINTS_HEADER``: the line above the caller's analysis, from ``core_prompts/extraction_blocks``.
* ``SPEAKER_ATTRIBUTION_ADDENDUM``: speaker attribution is off in Core, so it has no Core text.
"""

from synap.cloud.shared.prompt_switch import PackTemplate, core_gap, pack_text

_ME = "synap.cloud.agents.ingestion.prompts.stage3_extraction"


class _CustomerWe(PackTemplate):
    """Answers ``.format(customer=..., customer_cap=...)`` as the engine calls it."""

    __slots__ = ()

    def format(self, *args, customer, customer_cap, **_ignored):  # type: ignore[override]
        # The memories call the writer "the user" (writer_entity.LABEL).
        return "\n\n" + super().format(customer_name=customer, customer_name_cap=customer_cap,
                                       writer_name="the user")


SPEAKER_ATTRIBUTION_ADDENDUM = core_gap(_ME, "SPEAKER_ATTRIBUTION_ADDENDUM")
CUSTOMER_WE_SECTION = _CustomerWe(pack_text("person_and_company", "customer_memory.txt").rstrip("\n"))
ANALYSIS_HINTS_HEADER = "\n\n" + pack_text("extraction_blocks", "analysis_hints_header.txt").rstrip("\n") + "\n"
