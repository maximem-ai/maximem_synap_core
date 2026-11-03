"""Per-use-case section plans for the compacted memory document.

Compaction v2 produces two complementary paragraphs:
  - factual paragraph (Pass 2): what we KNOW (state, identity, decisions)
  - conversational paragraph (Pass 3): what is HAPPENING (flow, momentum, mood)

Each paragraph is section-headed; the headers come from this dict,
keyed by the classifier's ``primary_category``. Pass 1 picks the
section plan and threads it into Pass 2 and Pass 3's user prompts.

Stage 1 uses a hard-coded Python dict for the per-category mapping;
Stage 2 will migrate this into the YAML compaction templates so
non-developers can tune per-customer section emphasis.

If a category isn't listed here, the "default" plan is used.

Compaction v3 uses this table only as a fallback, when the writer returns no
plan of its own. No header here names what should happen next (I-1): the
old "Next Steps" / "What's Next" entries were renamed to what has happened.
"""

from typing import Dict, List


SectionPlan = Dict[str, List[str]]  # {"factual_sections": [...], "conversational_sections": [...]}


SECTION_PLANS_BY_CATEGORY: Dict[str, SectionPlan] = {
    "customer_support": {
        "factual_sections": [
            "## Customer Profile",
            "## Issue Summary",
            "## Steps Tried",
            "## Stated Preferences",
            "## Constraints",
        ],
        "conversational_sections": [
            "## Conversation Timeline",
            "## Open Issues",
            "## Customer Mood",
            "## Resolved So Far",
        ],
    },
    "technical_debugging": {
        "factual_sections": [
            "## Problem Statement",
            "## Environment",
            "## Hypotheses Tried",
            "## Working Configuration",
            "## Failed Attempts",
        ],
        "conversational_sections": [
            "## Debug Timeline",
            "## Open Hypotheses",
            "## Confidence Signals",
            "## Fixes Confirmed",
        ],
    },
    "planning_coordination": {
        "factual_sections": [
            "## Stakeholders",
            "## Goals",
            "## Decisions Made",
            "## Resource Constraints",
            "## Dependencies",
        ],
        "conversational_sections": [
            "## Discussion Timeline",
            "## Unresolved",
            "## Tone and Alignment",
            "## Decisions Pending",
        ],
    },
    "analytical_research": {
        "factual_sections": [
            "## Research Question",
            "## Findings",
            "## Sources",
            "## Methodology Decisions",
            "## Open Questions",
        ],
        "conversational_sections": [
            "## Research Timeline",
            "## Pending Investigation",
            "## Confidence Signals",
            "## Resolved So Far",
        ],
    },
    "learning_educational": {
        "factual_sections": [
            "## Learner Profile",
            "## Topics Covered",
            "## Concepts Mastered",
            "## Misconceptions Identified",
            "## Learning Preferences",
        ],
        "conversational_sections": [
            "## Lesson Timeline",
            "## Open Questions",
            "## Engagement Level",
            "## Topics Covered So Far",
        ],
    },
    "creative_writing": {
        "factual_sections": [
            "## Project Premise",
            "## Characters and Setting",
            "## Plot Points Decided",
            "## Stylistic Choices",
            "## Constraints",
        ],
        "conversational_sections": [
            "## Drafting Timeline",
            "## Open Threads",
            "## Creative Direction",
            "## Beats Drafted",
        ],
    },
    "transactional": {
        "factual_sections": [
            "## Customer Profile",
            "## Order or Request",
            "## Decisions Made",
            "## Constraints",
            "## Outcome",
        ],
        "conversational_sections": [
            "## Transaction Timeline",
            "## Open Items",
            "## Customer Disposition",
            "## Resolved So Far",
        ],
    },
    "general": {
        "factual_sections": [
            "## User Profile",
            "## Current State",
            "## Decisions Made",
            "## Preferences Expressed",
            "## Constraints",
        ],
        "conversational_sections": [
            "## Conversation Timeline",
            "## Open Threads",
            "## Emotional Context",
            "## Resolved So Far",
        ],
    },
}


def get_section_plan(primary_category: str) -> SectionPlan:
    """Look up the section plan for a primary category.

    Falls back to the ``general`` plan if the category isn't registered.
    Pass 1's LLM can return any string for ``primary_category``; we tolerate
    that here so an unknown category doesn't break compaction.
    """
    key = (primary_category or "").strip().lower().replace(" ", "_")
    return SECTION_PLANS_BY_CATEGORY.get(key, SECTION_PLANS_BY_CATEGORY["general"])
