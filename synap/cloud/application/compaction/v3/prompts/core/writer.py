"""Core's compaction v3 texts: writer, density guidance, repair, roll-up.

Read from ``core_prompts/compaction_v3``. The ``*_USER`` and ``DENSITY_GUIDANCE`` values answer
``.format(...)`` the way the engine calls them. Steering is a Cloud-only mode (``prompt_switch``
lists ``statuses`` and ``steering`` as Cloud only), so its two texts have no Core version.
"""

from synap.cloud.shared.prompt_switch import PackTemplate, core_gap, pack_text

_ME = "synap.cloud.application.compaction.v3.prompts.writer"

WRITER_SYSTEM = pack_text("compaction_v3", "writer_system.txt").rstrip("\n")
WRITER_USER = PackTemplate(pack_text("compaction_v3", "writer_user.txt").rstrip("\n"))
DENSITY_GUIDANCE = PackTemplate(pack_text("compaction_v3", "density_guidance.txt").rstrip("\n"))
REPAIR_USER = PackTemplate(pack_text("compaction_v3", "repair_user.txt").rstrip("\n"))
ROLLUP_SYSTEM = pack_text("compaction_v3", "rollup_system.txt").rstrip("\n")
ROLLUP_USER = PackTemplate(pack_text("compaction_v3", "rollup_user.txt").rstrip("\n"))

STEERING_SYSTEM = core_gap(_ME, "STEERING_SYSTEM")
STEERING_USER = core_gap(_ME, "STEERING_USER")
