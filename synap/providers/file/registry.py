"""File store provider registry."""

import importlib
import importlib.util
import yaml
import os
from typing import Optional
from synap.cloud.shared import edition
from .interface import FileStoreInterface

# Which adapter builds the store: SYNAP_FILE_STORE if set, else the edition's choice (the edition module
# holds it; Cloud keeps S3, Core uses the local folder). Each adapter is imported only when it is chosen,
# and a name whose module is not in this build is not offered, so a build without the S3 adapter never
# needs boto3.
_ADAPTERS = {
    "s3": ("synap.providers.file.adapters.s3", "S3Adapter"),
    "local": ("synap.providers.file.adapters.local", "LocalDiskAdapter"),
}


def _available() -> list:
    return sorted(name for name, (module, _) in _ADAPTERS.items() if importlib.util.find_spec(module) is not None)


def _adapter_class(name: str):
    module, cls = _ADAPTERS[name]
    return getattr(importlib.import_module(module), cls)


_instance: Optional[FileStoreInterface] = None


def _load_config() -> dict:
    """Load file store config from db_registry.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "db_registry.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    store_name = (os.environ.get("SYNAP_FILE_STORE") or edition.file_store or config["defaults"]["file"]).strip().lower()
    if store_name not in _available():
        raise ValueError(
            f"The file store {store_name!r} is not available; choose one of: {', '.join(_available())}")
    store_config = config["stores"][store_name]

    # Resolve environment variables
    resolved = {"adapter": store_name}
    for key, value in store_config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved[key] = os.environ.get(env_var, "")
        else:
            resolved[key] = value

    return resolved


def get_file_store(config: dict = None) -> FileStoreInterface:
    """Get configured file store adapter (singleton per process)."""
    global _instance
    if _instance is None:
        cfg = config or _load_config()
        store = _adapter_class(cfg.get("adapter", "s3"))(cfg)
        store.connect()          # raises with the reason; a store that failed to connect is not kept
        _instance = store
    return _instance


def check_file_store_at_startup() -> None:
    """The local folder for the copies must be writable, or the server stops at start with the reason
    not at the first ingest. Does nothing for any other store (Cloud's S3 reports as before)."""
    if _load_config()["adapter"] == "local":
        get_file_store()


def reset_file_store() -> None:
    """Reset singleton (for testing)."""
    global _instance
    if _instance:
        _instance.disconnect()
    _instance = None
