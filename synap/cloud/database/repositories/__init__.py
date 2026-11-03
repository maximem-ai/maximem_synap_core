"""Repository module for database operations.

Provides the repository pattern for data access.

The repositories for tables Core ships are imported here as before. The ones for
tables only Cloud has (dashboard users, admin users, impersonation log,
credentials) are loaded on first use, so this package imports in a tree where
those files are absent. ``from synap.cloud.database.repositories import
DashboardUserRepository`` still works in Cloud and gives the same class.
"""

from importlib import import_module as _import_module

from synap.cloud.database.repositories.base import BaseRepository
from synap.cloud.database.repositories.client_repo import ClientRepository
from synap.cloud.database.repositories.instance_repo import InstanceRepository
from synap.cloud.database.repositories.client_config_repo import ClientConfigRepository

# name -> module that defines it. Cloud only: these files do not ship in Core.
_LAZY = {
    "DashboardUserRepository": "synap.cloud.database.repositories.dashboard_user_repo",
    "SynapAdminUserRepository": "synap.cloud.database.repositories.synap_admin_user_repo",
    "ImpersonationLogRepository": "synap.cloud.database.repositories.impersonation_log_repo",
    "CredentialRepository": "synap.cloud.database.repositories.credential_repo",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "BaseRepository",
    "DashboardUserRepository",
    "SynapAdminUserRepository",
    "ImpersonationLogRepository",
    "ClientRepository",
    "InstanceRepository",
    "CredentialRepository",
    "ClientConfigRepository",
]
