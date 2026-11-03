"""Which entity-resolution path Stage 4 runs."""

import os


def adjudicated_mode() -> bool:
    """Whether Stage 4 uses the three-way, adjudicated resolution path.

    Default is the legacy path. ``SYNAP_ER_RESOLUTION_MODE=adjudicated`` turns
    the new one on; it needs synap-er >= 0.2.0 and is ignored on older wheels.
    """
    return os.environ.get("SYNAP_ER_RESOLUTION_MODE", "legacy").strip().lower() == "adjudicated"


def node_keys_enabled() -> bool:
    """Whether adjudicated resolution also keys graph nodes by entity.

    On by default inside adjudicated mode. ``SYNAP_ER_NODE_KEYS=false`` keeps
    the three-way decisions but leaves graph nodes keyed by name alone.
    """
    return os.environ.get("SYNAP_ER_NODE_KEYS", "true").strip().lower() in ("true", "1", "yes")
