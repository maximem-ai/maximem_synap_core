"""Graph store provider registry."""

import yaml
import os
from typing import Optional
from .interface import GraphStoreInterface
from .adapters.neo4j import Neo4jAdapter

_instance: Optional[GraphStoreInterface] = None


def _load_config() -> dict:
    """Load config from db_registry.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "db_registry.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    store_name = config["defaults"]["graph"]
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


def get_graph_store(config: dict = None) -> GraphStoreInterface:
    """Get configured graph store adapter (singleton per process)."""
    global _instance
    if _instance is None:
        cfg = config or _load_config()
        _instance = Neo4jAdapter(cfg)
        _instance.connect()
    return _instance


def reset_graph_store() -> None:
    """Reset singleton (for testing)."""
    global _instance
    if _instance:
        _instance.disconnect()
    _instance = None
