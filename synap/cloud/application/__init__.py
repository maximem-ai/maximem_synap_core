"""Application Layer - Business logic and orchestration."""

# Error Models (lightweight, no circular risk)
from .models import (
    ApplicationError,
    ValidationError,
    NotFoundError,
    PermissionDeniedError,
    ConfigurationError,
    IngestionError,
    RetrievalError,
)

# Managers are lazy-loaded to avoid circular imports.
# Import directly from submodules:
#   from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
#   from synap.cloud.application.admin_api import SetupManager

_LAZY_IMPORTS = {
    # Client API
    "RetrievalManager": ".client_api.retrieval_manager",
    "IngestionManager": ".client_api.ingestion_manager",
    "ClientConfigManager": ".client_api.client_config_manager",
    # Admin API
    "DashboardManager": ".admin_api.dashboard_manager",
    "SetupManager": ".admin_api.setup_manager",
    "TenantManager": ".admin_api.tenant_manager",
    "CustomerUserManager": ".admin_api.customer_user_manager",
    "MemoryArchitectureManager": ".admin_api.memory_architecture_manager",
    "ConsolidationManager": ".admin_api.consolidation_manager",
    "ReportingManager": ".admin_api.reporting_manager",
    # Memory Architecture Setup Manager
    "MemoryArchitectureSetupManager": ".memory_architecture",
}


def __getattr__(name):
    """Lazy-load manager classes on first access."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Client API
    "RetrievalManager",
    "IngestionManager",
    "ClientConfigManager",
    # Admin API
    "DashboardManager",
    "SetupManager",
    "TenantManager",
    "CustomerUserManager",
    "MemoryArchitectureManager",
    "ConsolidationManager",
    "ReportingManager",
    # Memory Architecture Setup Manager
    "MemoryArchitectureSetupManager",
    # Errors
    "ApplicationError",
    "ValidationError",
    "NotFoundError",
    "PermissionDeniedError",
    "ConfigurationError",
    "IngestionError",
    "RetrievalError",
]
