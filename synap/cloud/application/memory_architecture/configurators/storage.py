"""Storage system configurator.

Configures vector stores, graph stores, and scoping.
"""

import asyncio
import logging
from typing import Any, Optional

from synap.cloud.memory_config.config_schema import (
    MemoryArchitectureConfig,
    StorageArchitectureConfig,
)
from .base import BaseConfigurator

logger = logging.getLogger(__name__)

# Mapping from embedding model name to vector dimension.
# Default 1024 matches Snowflake/snowflake-arctic-embed-l-v2.0 (self-hosted, multilingual).
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "default": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "Snowflake/snowflake-arctic-embed-m-v1.5": 768,
    "Snowflake/snowflake-arctic-embed-l-v2.0": 1024,
}



class StorageConfigurator(BaseConfigurator):
    """
    Configures storage systems based on YAML config.

    Responsibilities:
    - Create/update vector collections
    - Create/update graph schemas
    - Configure scoping (user/customer partitioning)
    - Set retention policies
    - Configure indexes
    """

    def __init__(
        self,
        vector_store_client: Any,
        graph_store_client: Any,
        config_store: Optional[Any] = None,
    ):
        self.vector_client = vector_store_client
        self.graph_client = graph_store_client
        self.config_store = config_store

    async def apply(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> dict[str, Any]:
        """Apply storage configuration.

        Orchestrates vector store, graph store, scoping, and retention setup.
        """
        results: dict[str, Any] = {}
        storage = config.storage_architecture

        if storage.vector_config.enabled:
            results["vector"] = await self._configure_vector_store(instance_id, storage)
        else:
            await self._disable_vector_store(instance_id)
            results["vector"] = {"status": "disabled"}

        if storage.graph_config.enabled:
            results["graph"] = await self._configure_graph_store(instance_id, storage)
        else:
            await self._disable_graph_store(instance_id, storage)
            results["graph"] = {"status": "disabled"}

        results["scoping"] = await self._configure_scoping(instance_id, storage.scoping)
        results["retention"] = await self._configure_retention(instance_id, config)

        logger.info(f"Storage configuration applied for instance {instance_id}")
        return {"success": True, **results}

    async def validate(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> list[str]:
        """Validate storage config (schema + business rules, no external probing)."""
        errors: list[str] = []
        storage = config.storage_architecture

        if storage.vector_config.enabled:
            if not storage.vector_config.embedding_model:
                errors.append("Vector store enabled but embedding_model is empty")

        if storage.graph_config.enabled:
            if not storage.graph_config.graph_type:
                errors.append("Graph store enabled but graph_type is empty")

        for gs in storage.graph_schemas:
            if gs.enabled and not gs.schema_name:
                errors.append("Graph schema enabled but schema_name is empty")

        retention = config.guardrails.retention
        if retention.max_memory_age_days <= 0:
            errors.append("max_memory_age_days must be > 0")
        if retention.episode_retention_days <= 0:
            errors.append("episode_retention_days must be > 0")
        if retention.episode_retention_days > retention.max_memory_age_days:
            errors.append(
                "episode_retention_days cannot exceed max_memory_age_days"
            )

        return errors

    async def rollback(
        self,
        instance_id: str,
        from_config: MemoryArchitectureConfig,
        to_config: MemoryArchitectureConfig
    ) -> bool:
        """Rollback storage configuration by re-applying the previous config."""
        result = await self.apply(instance_id, to_config)
        return result.get("success", False)

    # ---- Private helpers ----

    async def _configure_vector_store(
        self,
        instance_id: str,
        storage_config: StorageArchitectureConfig
    ) -> dict:
        """Configure vector store collection.

        Uses asyncio.to_thread because VectorStoreInterface is sync.
        """
        model = storage_config.vector_config.embedding_model
        dim = EMBEDDING_DIMENSIONS.get(model, 1536)
        await asyncio.to_thread(
            self.vector_client.create_namespace, instance_id, dim
        )
        return {"status": "configured", "namespace": instance_id, "dimension": dim}

    async def _configure_graph_store(
        self,
        instance_id: str,
        storage_config: StorageArchitectureConfig
    ) -> dict:
        """Configure graph store schema(s).

        Uses asyncio.to_thread because GraphStoreInterface is sync.
        Supports multi-graph: when graph_schemas is populated, creates a
        separate namespace per enabled schema.
        """
        if storage_config.graph_schemas:
            results = {}
            for schema in storage_config.graph_schemas:
                if not schema.enabled:
                    continue
                namespace = f"{instance_id}__{schema.schema_name}"
                await asyncio.to_thread(
                    self.graph_client.create_namespace, namespace
                )
                results[schema.schema_name] = {
                    "status": "configured",
                    "namespace": namespace,
                    "schema_type": schema.schema_type,
                }
            return {"status": "configured", "mode": "multi_graph", "schemas": results}
        else:
            await asyncio.to_thread(
                self.graph_client.create_namespace, instance_id
            )
            return {"status": "configured", "namespace": instance_id}

    async def _configure_scoping(
        self,
        instance_id: str,
        scoping_config: Any
    ) -> dict:
        """Configure memory scoping."""
        if self.config_store:
            await self.config_store.put(
                instance_id, "storage/scoping", scoping_config.model_dump()
            )
        return {"status": "configured"}

    async def _configure_retention(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> dict:
        """Configure retention policies.

        Stores config only; enforcement is handled by a future retention worker.
        """
        retention = config.guardrails.retention
        if self.config_store:
            await self.config_store.put(
                instance_id, "storage/retention", retention.model_dump()
            )
        return {
            "status": "configured",
            "max_memory_age_days": retention.max_memory_age_days,
        }

    async def _disable_vector_store(self, instance_id: str):
        """Disable vector store for instance.

        Soft-flag only — data stays intact, no delete_namespace call.
        """
        if self.config_store:
            await self.config_store.put(
                instance_id, "storage/vector_enabled", {"enabled": False}
            )

    async def _disable_graph_store(
        self, instance_id: str, storage_config: StorageArchitectureConfig | None = None
    ):
        """Disable graph store for instance.

        Soft-flag only — data stays intact, no delete_namespace call.
        Handles multi-graph namespaces when graph_schemas is populated.
        """
        if self.config_store:
            if storage_config and storage_config.graph_schemas:
                for schema in storage_config.graph_schemas:
                    namespace = f"{instance_id}__{schema.schema_name}"
                    await self.config_store.put(
                        namespace, "storage/graph_enabled", {"enabled": False}
                    )
            else:
                await self.config_store.put(
                    instance_id, "storage/graph_enabled", {"enabled": False}
                )

    async def _can_create_vector_collection(self, instance_id: str) -> bool:
        """Check if vector collection can be created (connectivity + existence)."""
        try:
            await asyncio.to_thread(self.vector_client.connect)
            await asyncio.to_thread(
                self.vector_client.create_namespace, instance_id, EMBEDDING_DIMENSIONS.get("default", 1024)
            )
            return True
        except Exception:
            return False

    async def _can_create_graph_schema(self, instance_id: str) -> bool:
        """Check if graph schema can be created (connectivity + existence)."""
        try:
            await asyncio.to_thread(self.graph_client.connect)
            await asyncio.to_thread(
                self.graph_client.create_namespace, instance_id
            )
            return True
        except Exception:
            return False

