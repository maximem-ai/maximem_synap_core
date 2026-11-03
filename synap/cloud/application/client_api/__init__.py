"""Client-facing API managers (SDK-accessible)."""

# Lazy-loaded to avoid circular imports.
# Import directly from submodules:
#   from synap.cloud.application.client_api.retrieval_manager import RetrievalManager

_LAZY_IMPORTS = {
    "RetrievalManager": ".retrieval_manager",
    "IngestionManager": ".ingestion_manager",
    "ClientConfigManager": ".client_config_manager",
}


def __getattr__(name):
    """Lazy-load manager classes on first access."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RetrievalManager",
    "IngestionManager",
    "ClientConfigManager",
]
