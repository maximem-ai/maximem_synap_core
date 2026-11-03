"""Memory Architecture Setup Manager.

Service for managing memory architecture configuration lifecycle.

The names below load on first use (PEP 562), not when the package is imported.
Importing a submodule such as ``configurators.storage`` runs this file first, and
most of these names live in files that need Cloud-only folders (the admin API,
the S3 config store). Core ships only ``configurators`` and ``models``, so an
eager import here would stop the embedding client and the ingestion workers from
loading. Cloud sees the same names, imported the first time they are asked for.
"""

from importlib import import_module
from typing import Any

_LAZY = {
    "MemoryArchitectureSetupManager": ".setup_manager",
    "MemoryArchitectureRepository": ".repository",
    "ConfigStore": ".config_store",
    "InMemoryConfigStore": ".config_store",
    "ConflictDetector": ".conflict_detector",
    "BaseConfigurator": ".configurators",
    "StorageConfigurator": ".configurators",
    "IngestionConfigurator": ".configurators",
    "RetrievalConfigurator": ".configurators",
    "ApplyMode": ".models",
    "ReviewDecision": ".models",
    "ConflictSeverity": ".models",
    "Conflict": ".models",
    "ConflictResolution": ".models",
    "ConflictDetectionResult": ".models",
    "InitResult": ".models",
    "UpdateResult": ".models",
    "ApplyResult": ".models",
    "RollbackResult": ".models",
    "ReviewResult": ".models",
    "ConfigVersion": ".models",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target, __name__), name)
    globals()[name] = value
    return value
