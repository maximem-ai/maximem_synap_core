"""Vector store provider registry."""

import yaml
import os
from typing import Optional
from .interface import VectorStoreInterface
from .adapters.chromadb import ChromaDBAdapter

_instance: Optional[VectorStoreInterface] = None


def _load_config() -> dict:
    """Load config from db_registry.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "db_registry.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    store_name = config["defaults"]["vector"]
    store_config = config["stores"][store_name]

    # Resolve environment variables
    resolved = {}
    for key, value in store_config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved[key] = os.environ.get(env_var, "")
        else:
            resolved[key] = value

    return resolved


def get_vector_store(config: dict = None) -> VectorStoreInterface:
    """Get configured vector store adapter (singleton per process)."""
    global _instance
    if _instance is None:
        cfg = config or _load_config()
        _instance = ChromaDBAdapter(cfg)
        _instance.connect()
    return _instance


def reset_vector_store() -> None:
    """Reset singleton (for testing)."""
    global _instance
    if _instance:
        _instance.disconnect()
    _instance = None
