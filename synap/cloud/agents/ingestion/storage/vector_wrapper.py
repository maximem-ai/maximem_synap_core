"""Async wrapper for Vector provider."""

import asyncio
import re
import concurrent.futures
import hashlib
import logging
import os
from typing import List, Optional, Dict, Any
from uuid import UUID

from synap.providers.vector.replication_context import (  # noqa: F401
    applying_replication,
    is_replication_apply,
)

from synap.cloud.shared import edition

logger = logging.getLogger(__name__)


# Dedicated thread pool for synchronous Chroma RPCs.
#
# Background: the Chroma Python client is synchronous and we adapt it to
# asyncio by submitting blocking calls to a thread pool. Previously we used
# ``asyncio.to_thread`` which runs on Python's default executor (8 workers
# on a 4-vCPU box). That pool is shared with HTTP serving, DB calls,
# embedding model loads, etc. — under load a fetch_context fans out to
# ~16 parallel Chroma calls and competes with everything else for those
# 8 slots, queueing calls behind unrelated work.
#
# This dedicated pool isolates Chroma from the rest of the app and scales
# independently. Workers spend almost all their time blocked on socket
# reads from the Chroma server (I/O-bound), so oversubscribing CPU cores
# is fine and necessary. Sized to comfortably handle ~16 concurrent
# fetch_context calls × 8 batch_searches each = 128 worst case, but in
# practice 32 is enough since Chroma itself becomes the next bottleneck
# beyond that.
#
# Configurable via SYNAP_CHROMA_THREAD_POOL_SIZE for ops to retune
# without a code change.
_CHROMA_THREAD_POOL_SIZE = int(os.environ.get("SYNAP_CHROMA_THREAD_POOL_SIZE", "32"))
_CHROMA_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_chroma_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazy-init the Chroma thread pool. Idempotent."""
    global _CHROMA_EXECUTOR
    if _CHROMA_EXECUTOR is None:
        _CHROMA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=_CHROMA_THREAD_POOL_SIZE,
            thread_name_prefix="chroma",
        )
        logger.info(
            "Chroma dedicated thread pool initialized: max_workers=%d",
            _CHROMA_THREAD_POOL_SIZE,
        )
    return _CHROMA_EXECUTOR


_MISSING_COLLECTION = re.compile(r"^\s*collection\b.*\bdoes not exist", re.IGNORECASE | re.DOTALL)


def _is_missing_collection(exc: BaseException) -> bool:
    """True only for Chroma's "Collection <name> does not exist." error.

    Chroma raises `NotFoundError` (older versions `ValueError`) with exactly that
    message when the collection is absent (chromadb `api/segment.py`
    `get_collection`). A missing database or tenant is also a `NotFoundError`, and
    so is anything else Chroma cannot find: those are failures, so the message must
    name a COLLECTION and the type must be one of Chroma's, not any exception.
    """
    if type(exc).__name__ not in ("NotFoundError", "InvalidCollectionException", "ValueError"):
        return False
    return bool(_MISSING_COLLECTION.search(str(exc)))


async def _run_chroma(fn, /, *args, **kwargs):
    """Run a blocking Chroma callable on the dedicated thread pool.

    Drop-in replacement for ``asyncio.to_thread(fn, *args, **kwargs)``
    that uses our dedicated pool instead of the default executor.

    Every call this wrapper makes into the provider, upsert and delete
    included, passes through here on its way to the thread pool, which
    makes this the one place to feed the vector write-ahead log
    (``synap.cloud.replication.vector_log``, migration 138). Chroma has no
    WAL of its own, so a write that fails to reach whatever ships it
    onward is otherwise gone with no record it was ever supposed to
    happen. Reads and namespace management pass through here too; only an
    actual upsert or delete gets logged.
    """
    loop = asyncio.get_running_loop()
    op = getattr(fn, "__name__", None)
    write_args = args if op in ("upsert", "delete") else None
    if kwargs:
        # functools.partial avoids the limitation that run_in_executor
        # doesn't accept kwargs directly.
        import functools
        fn = functools.partial(fn, *args, **kwargs)
        args = ()
    result = await loop.run_in_executor(_get_chroma_executor(), fn, *args)
    if write_args is not None and not is_replication_apply():
        await _log_vector_write(op, write_args)
    return result


async def _log_vector_write(op: str, write_args) -> None:
    """Feed the write-ahead log after a write has actually landed.

    ``write_args`` is ``(namespace, documents)`` for an upsert or
    ``(namespace, ids)`` for a delete: the shape every upsert/delete call
    site in this file already passes to the provider.

    The log is the ``replication_log_write`` hook (migration 138). With none
    registered (Core) nothing is written.
    """
    await edition.acall("replication_log_write", op, write_args)


def shutdown_chroma_executor() -> None:
    """Shut down the Chroma thread pool. Call on app shutdown to allow
    Python's interpreter to exit cleanly. Idempotent."""
    global _CHROMA_EXECUTOR
    if _CHROMA_EXECUTOR is not None:
        _CHROMA_EXECUTOR.shutdown(wait=False)
        _CHROMA_EXECUTOR = None
        logger.info("Chroma dedicated thread pool shut down")

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    ExtractedEntity,
    ExtractedRelationship,
    Scope,
)
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL
from synap.providers.vector import VectorStoreInterface
from synap.providers.vector.models import VectorDocument, VectorQuery


def _effective_scope_level(scope: "Scope", override: Optional[str] = None) -> str:
    """Pick the scope level to use for namespace + metadata building.

    Precedence: explicit ``override`` → ``scope.scope_level`` → "user".
    "user" is the legacy default for old callers that never set scope_level.
    """
    return override or getattr(scope, "scope_level", None) or "user"


def _scope_metadata(scope: "Scope", scope_level: Optional[str] = None) -> Dict[str, Any]:
    """Build the scope-related metadata keys for a ChromaDB document.

    ChromaDB silently drops None metadata values, so we NEVER set a key to
    None — we either set a real string or omit the key. The shape of the
    metadata encodes the level:

    - level="user":     {client_id, instance_id, customer_id, user_id=<uuid>}
    - level="customer": {client_id, instance_id, customer_id, user_id=<SENTINEL>}
    - level="client":   {client_id, instance_id} only (tenant-wide collection)
    """
    level = _effective_scope_level(scope, scope_level)
    meta: Dict[str, Any] = {
        "client_id": str(scope.client_id),
        "instance_id": str(scope.instance_id),
    }
    if level != "client":
        if scope.customer_id:
            meta["customer_id"] = str(scope.customer_id)
        if level == "customer":
            meta["user_id"] = CUSTOMER_SCOPE_SENTINEL
        elif scope.user_id:
            meta["user_id"] = str(scope.user_id)
    meta.update(_nested_scope_metadata(scope, level))
    return meta


# Legacy level names in terms of the roles they have always meant. The write
# level is a role, not a name: a client who calls their tenant level "hospital"
# still writes shared material at the tenant of record.
_LEVEL_TO_ROLE = {"client": "root", "customer": "tenant_of_record", "user": "identity"}


def _rung_has_role(rung, name: str) -> bool:
    """Whether a rung holds `name`, across both chain shapes.

    Rungs arrive either as `ScopeRung` models or as the plain dicts
    `chain_to_rungs` emits, depending on the caller, so this reads both rather
    than assuming one.
    """
    if hasattr(rung, "has_role"):
        return rung.has_role(name)
    roles = getattr(rung, "roles", None)
    if roles is None and isinstance(rung, dict):
        roles = rung.get("roles")
    if roles is None:
        single = getattr(rung, "role", None)
        if single is None and isinstance(rung, dict):
            single = rung.get("role")
        roles = [single] if single else []
    return name in roles


def _nested_scope_metadata(scope: "Scope", level: str) -> Dict[str, Any]:
    """S-03: node_id plus one ``anc_<level_key>`` per level at or above it.

    Empty for every caller that has not resolved a chain, which today is all of
    them. That is deliberate: the keys are additive, so this can ship and run in
    production long before anything reads them, and a partially backfilled
    collection is incomplete rather than wrong.

    Level-keyed and not positional (``anc_2``). Positional keys would change
    what an existing key means the moment a level is inserted mid-ladder, so
    every record written before the insert would silently start answering a
    different question. Level keys only ever gain a key.

    No path string is written here, and S-06 is the reason: the path lives once
    in the Postgres registry. Putting it on records would mean a shape change
    rewrites millions of vector documents instead of thousands of rows.
    """
    chain = getattr(scope, "chain", None)
    if not chain:
        return {}

    want_role = _LEVEL_TO_ROLE.get(level)
    cut = None
    for i, rung in enumerate(chain):
        # An exact level match wins, so a deep ladder can write at "team"
        # without teaching this function the client's level names.
        if rung.level_key == level:
            cut = i
            break
        # Role membership, not equality. In B2C the tenant rung also holds the
        # identity role, and comparing `rung.role == "identity"` finds nothing,
        # so `cut` stayed None and this function returned {} for every B2C
        # write. Nested scoping was switched on and no record got a node_id.
        if want_role and _rung_has_role(rung, want_role):
            cut = i
    if cut is None:
        # A chain that does not reach the level being written is a resolution
        # bug, not something to paper over: stamping the leaf would file the
        # record at a narrower scope than the caller asked for.
        logger.warning(
            "scope chain %s does not contain level %r; writing no node_id",
            [r.level_key for r in chain], level,
        )
        return {}

    out: Dict[str, Any] = {"node_id": str(chain[cut].node_id)}
    for rung in chain[: cut + 1]:
        out[f"anc_{rung.level_key}"] = str(rung.node_id)
    return out


def _shown_name(properties, fallback):
    from synap.cloud.agents.ingestion.services.writer_entity import shown_name
    return shown_name(properties, fallback)


def entity_id_scope(scope: "Scope") -> str:
    """The scope id an entity vector's id is hashed with.

    The customer (the client, for client scope) until SYNAP_GRAPH_PER_SCOPE:
    one entity vector per name per customer, which two users overwrote in turn.
    With the flag, the scope the document is written at, so the entity vector
    exists once per scope like its graph node (shared/scope_key).
    """
    from synap.cloud.shared.scope_key import per_scope_enabled, write_scope_id
    if per_scope_enabled(getattr(scope, "instance_id", None)):
        return write_scope_id(scope)
    return str(scope.customer_id) if scope.customer_id else str(scope.client_id)


def deterministic_entity_id(canonical_name: str, entity_type: str, client_id: str, customer_id: str) -> str:
    """Compute deterministic entity ID for ChromaDB dedup.

    Same (canonical_name, entity_type, client_id, customer_id) always produces the same ID,
    so ChromaDB upsert overwrites instead of creating duplicates.
    customer_id is already client-scoped (uuid5(client_uuid, external_customer_id)),
    so including it alone is sufficient for uniqueness.
    """
    key = f"{canonical_name.lower().strip()}|{entity_type.lower().strip()}|{customer_id}"
    return hashlib.blake2b(key.encode(), digest_size=16).hexdigest()


def entity_id_name(entity: Any) -> str:
    """The name an entity's vector id is hashed from.

    Its display name, unless entity resolution renamed the entity to a fuller
    name since its records were first written: then the name they were first
    written under, so the id stays the same (see resolution/naming).
    """
    return entity.properties.get("er_vector_name") or entity.canonical_name or entity.name


def _looks_like_metadata_rejection(exc: BaseException) -> bool:
    """Is this the vector server refusing a metadata value (an older server that
    cannot store a list), as opposed to any other failure?"""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(word in text for word in (
        "metadata", "invalidargument", "invalid argument", "validation", "unsupported",
        "expected a str", "must be a str", "not a valid", "type error"))


class AsyncVectorWrapper:
    """
    Async wrapper around sync VectorStoreInterface.

    Uses asyncio.to_thread() to make blocking calls non-blocking.
    Converts ingestion domain models to VectorDocument format.

    Collections per customer (customer_id is already client-scoped):
    - synap_{customer_id}           : memories (facts, preferences, episodes, etc.)
    - synap_{customer_id}_entities  : entity nodes with embeddings
    - synap_{customer_id}_triples   : Entity A <Relationship> Entity B
    - synap_{customer_id}_chunks    : raw content chunks (~500 tokens each)
    - synap_{customer_id}_sentences : sentence-level embeddings

    Client-level collections (for client-scope shared data):
    - synap_{client_id}           : client-scope memories
    - synap_{client_id}_entities  : client-scope entities
    """

    def __init__(self, provider: VectorStoreInterface):
        self._provider = provider

    # --- Namespace helpers ---
    # When scope_level is not passed, fall back to scope.scope_level then "user".
    # Customer-scope shared memories land in the same synap_{customer_id}
    # collection as user-scope memories (distinguished by user_id=sentinel),
    # so only "client" routes to a different namespace.

    def _get_namespace(self, scope: Scope, scope_level: Optional[str] = None) -> str:
        """Derive namespace from scope for memories collection."""
        level = _effective_scope_level(scope, scope_level)
        if level == "client":
            return f"synap_{scope.client_id}_v2"
        return f"synap_{scope.customer_id}_v2"

    def _get_entity_namespace(self, scope: Scope, scope_level: Optional[str] = None) -> str:
        """Derive namespace for entity collection."""
        level = _effective_scope_level(scope, scope_level)
        if level == "client":
            return f"synap_{scope.client_id}_entities_v2"
        return f"synap_{scope.customer_id}_entities_v2"

    def _get_triples_namespace(self, scope: Scope, scope_level: Optional[str] = None) -> str:
        """Derive namespace for triples collection."""
        level = _effective_scope_level(scope, scope_level)
        if level == "client":
            return f"synap_{scope.client_id}_triples_v2"
        return f"synap_{scope.customer_id}_triples_v2"

    def _get_chunks_namespace(self, scope: Scope, scope_level: Optional[str] = None) -> str:
        """Derive namespace for raw content chunks collection."""
        level = _effective_scope_level(scope, scope_level)
        if level == "client":
            return f"synap_{scope.client_id}_chunks_v2"
        return f"synap_{scope.customer_id}_chunks_v2"

    def _get_sentences_namespace(self, scope: Scope, scope_level: Optional[str] = None) -> str:
        """Derive namespace for sentence-level embeddings collection."""
        level = _effective_scope_level(scope, scope_level)
        if level == "client":
            return f"synap_{scope.client_id}_sentences_v2"
        return f"synap_{scope.customer_id}_sentences_v2"

    def _get_questions_namespace(self, scope: Scope, scope_level: Optional[str] = None) -> str:
        """Derive namespace for hypothetical questions collection."""
        level = _effective_scope_level(scope, scope_level)
        if level == "client":
            return f"synap_{scope.client_id}_questions_v2"
        return f"synap_{scope.customer_id}_questions_v2"

    async def get_all_scope_memories(self, scope: Scope, limit: int = 5000) -> List[dict]:
        """Load all memories for a scope as relink-ready dicts.

        Returns ``[{memory_id, content, memory_type, confidence, embedding}]`` from the
        scope's memory collection. Used by the scope-level cross-document relink (which
        needs every memory's content, not just one document's).
        """
        namespace = self._get_namespace(scope)
        docs = await asyncio.to_thread(self._provider.get_all, namespace, None, limit)
        out: List[dict] = []
        for d in docs or []:
            content = getattr(d, "content", None)
            if not content:
                continue
            md = getattr(d, "metadata", None) or {}
            emb = getattr(d, "embedding", None)
            # Provider may return a numpy array; coerce to a plain list so the memory
            # (incl. its embedding) is JSON-serializable when written onto a graph edge.
            if emb is not None and hasattr(emb, "tolist"):
                emb = emb.tolist()
            record = {
                "memory_id": d.id,
                "content": content,
                "memory_type": md.get("memory_type", "facts"),
                "confidence": md.get("confidence", 0.7),
                "embedding": emb,
            }
            # Which instance wrote it. These dicts are written onto graph
            # relationships (the relink pass), and the reader refuses a memory
            # that names no instance when a client's policy is restrictive.
            if md.get("instance_id"):
                record["instance_id"] = str(md["instance_id"])
            out.append(record)
        return out

    async def upsert_memory(
        self,
        memory: ExtractedMemory,
        embedding: List[float],
        scope: Scope,
        ere_context: Optional[str] = None,
    ) -> str:
        """Upsert a memory with its embedding."""
        namespace = self._get_namespace(scope)

        # Build full content with ERE context if available
        full_content = memory.content
        if ere_context:
            full_content = f"{memory.content}\n\n[Context: {ere_context}]"

        # Build metadata for filtering
        metadata = {
            **_scope_metadata(scope),
            "memory_type": memory.memory_type.value,
            "category": memory.category,
            "subcategory": memory.subcategory,
            "confidence": memory.confidence,
            "created_at": memory.extracted_at.isoformat(),
            "source_document_id": memory.source_document_id,
            "entity_type": "user",  # Default entity type
            "entities_mentioned": ",".join(memory.entities_mentioned) if memory.entities_mentioned else "",
            "temporal_category": memory.temporal_category,
            "temporal_confidence": memory.temporal_confidence,
            "is_episode": str(memory.is_episode).lower(),
        }
        # ChromaDB rejects None values — only add temporal dates if present
        if memory.valid_from:
            metadata["valid_from"] = memory.valid_from.isoformat()
        if memory.valid_until:
            metadata["valid_until"] = memory.valid_until.isoformat()
        if memory.document_created_at:
            metadata["document_created_at"] = memory.document_created_at.isoformat()
        if memory.assertion_source:
            metadata["assertion_source"] = memory.assertion_source
        from synap.cloud.agents.ingestion.services.event_window import event_window_metadata
        metadata.update(event_window_metadata(
            getattr(memory, "event_start", None), getattr(memory, "event_end", None),
            getattr(memory, "event_precision", None)))
        # Which types claimed this fact, when coalescing merged several rows into
        # one. Empty, and so byte-identical to before, for every unmerged row.
        from synap.cloud.agents.ingestion.services.coalesce import (
            type_labels_metadata,
        )
        metadata.update(type_labels_metadata(getattr(memory, "type_labels", None)))

        # Create VectorDocument
        doc = VectorDocument(
            id=str(memory.memory_id),
            embedding=embedding,
            metadata=metadata,
            content=full_content,
        )
        
        # Upsert to vector store
        await _run_chroma(
            self._provider.upsert,
            namespace,
            [doc]
        )
        
        return str(memory.memory_id)

    async def upsert_memories_batch(
        self,
        memories: List[ExtractedMemory],
        embeddings: List[List[float]],
        scope: Scope,
        ere_contexts: Optional[List[Optional[str]]] = None,
        remapped_entities: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """Batch upsert multiple memories with embeddings.

        Args:
            remapped_entities: If provided, use these deterministic entity IDs
                in ChromaDB metadata instead of memory.entities_mentioned (which
                uses extraction-time UUIDs). This ensures _link_memory_edges can
                match memory references to entity nodes in the adjacency graph.
        """
        if len(memories) != len(embeddings):
            raise ValueError("Number of memories must match number of embeddings")

        namespace = self._get_namespace(scope)

        if ere_contexts is None:
            ere_contexts = [None] * len(memories)

        # Build VectorDocuments
        documents = []
        vector_ids = []

        for idx, (memory, embedding, ere_context) in enumerate(zip(memories, embeddings, ere_contexts)):
            # Build full content
            full_content = memory.content
            if ere_context:
                full_content = f"{memory.content}\n\n[Context: {ere_context}]"

            # Use remapped entity IDs if provided, otherwise fall back to original
            entity_ids_for_meta = (
                remapped_entities[idx] if remapped_entities and idx < len(remapped_entities)
                else memory.entities_mentioned
            )

            # Build metadata
            metadata = {
                **_scope_metadata(scope),
                "memory_type": memory.memory_type.value,
                "category": memory.category,
                "subcategory": memory.subcategory,
                "confidence": memory.confidence,
                "created_at": memory.extracted_at.isoformat(),
                "source_document_id": memory.source_document_id,
                "entity_type": "user",
                "entities_mentioned": ",".join(entity_ids_for_meta) if entity_ids_for_meta else "",
                "temporal_category": memory.temporal_category,
                "temporal_confidence": memory.temporal_confidence,
                "is_episode": str(memory.is_episode).lower(),
            }
            # ChromaDB rejects None values — only add temporal dates if present
            if memory.valid_from:
                metadata["valid_from"] = memory.valid_from.isoformat()
            if memory.valid_until:
                metadata["valid_until"] = memory.valid_until.isoformat()
            if memory.document_created_at:
                metadata["document_created_at"] = memory.document_created_at.isoformat()
            if memory.assertion_source:
                metadata["assertion_source"] = memory.assertion_source
            from synap.cloud.agents.ingestion.services.event_window import event_window_metadata
            metadata.update(event_window_metadata(
                getattr(memory, "event_start", None), getattr(memory, "event_end", None),
                getattr(memory, "event_precision", None)))
            # Same as the single-memory path above. Both builders, because this is
            # the batch one ingestion actually uses and the other is still live.
            from synap.cloud.agents.ingestion.services.coalesce import (
                type_labels_metadata,
            )
            metadata.update(
                type_labels_metadata(getattr(memory, "type_labels", None)))

            doc = VectorDocument(
                id=str(memory.memory_id),
                embedding=embedding,
                metadata=metadata,
                content=full_content,
            )
            
            documents.append(doc)
            vector_ids.append(str(memory.memory_id))
        
        # Batch upsert
        await _run_chroma(
            self._provider.upsert,
            namespace,
            documents
        )
        
        return vector_ids

    async def ensure_namespace(
        self,
        scope: Scope,
        dimension: int = 1024,  # Default snowflake-arctic-embed-l-v2.0 embedding dimension
    ) -> None:
        """Ensure namespace exists for the given scope."""
        namespace = self._get_namespace(scope)
        
        # Try to create namespace (idempotent operation)
        try:
            await _run_chroma(
                self._provider.create_namespace,
                namespace,
                dimension
            )
        except Exception as e:
            logger.debug("Namespace creation skipped (likely exists): %s", e)

    async def ensure_entity_namespace(
        self,
        scope: Scope,
        dimension: int = 1024,
    ) -> None:
        """Ensure entity collection exists for the given scope."""
        namespace = self._get_entity_namespace(scope)
        try:
            await _run_chroma(
                self._provider.create_namespace,
                namespace,
                dimension
            )
        except Exception as e:
            logger.debug("Entity namespace creation skipped (likely exists): %s", e)

    async def ensure_triples_namespace(
        self,
        scope: Scope,
        dimension: int = 1024,
    ) -> None:
        """Ensure triples collection exists for the given scope."""
        namespace = self._get_triples_namespace(scope)
        try:
            await _run_chroma(
                self._provider.create_namespace,
                namespace,
                dimension
            )
        except Exception as e:
            logger.debug("Triples namespace creation skipped (likely exists): %s", e)

    async def _stamp_instance_lists(self, namespace: str, documents: List[VectorDocument],
                                    scope: Scope) -> None:
        """Give each entity/triple document the LIST of instances that wrote it.

        These documents are keyed by name and customer, not by instance, so two
        instances of one customer write the SAME document and the scalar
        `instance_id` in its metadata is whichever wrote last. The list is what
        the instance policy is read from (shared/graph_boundary): the existing
        document's list is kept, a legacy document's last-writer scalar is
        treated as an owner too, and this writer is added.

        Chroma has no atomic "add to a list", so this reads and then rewrites:
        two instances writing the SAME entity in the same instant can drop one
        entry. That costs the loser an anchor until it writes again and can never
        show a viewer anything, and Neo4j (which does it atomically) is where the
        records are served from. Best effort: a failure here must not fail the
        ingestion.
        """
        from synap.cloud.shared import graph_boundary
        if not documents or not graph_boundary.stamping_enabled():
            return
        token = graph_boundary.instance_token(scope)
        if not token:
            return
        prior: Dict[str, Dict[str, Any]] = {}
        try:
            ids = [d.id for d in documents]
            reader = getattr(self._provider, "get_metadata", None)
            if reader is not None:
                prior = await _run_chroma(reader, namespace, ids)
            else:
                prior = {d.id: (d.metadata or {}) for d in await _run_chroma(self._provider.get, namespace, ids)}
        except Exception as exc:  # noqa: BLE001
            # A collection that does not exist yet has no prior owners. Anything
            # else means the other owners are unknown, so this write cannot add to
            # them: say so, because the upsert below replaces the metadata.
            if "does not exist" not in str(exc) and "NotFoundError" not in type(exc).__name__:
                logger.warning(
                    "entity_vector_owner_lookup_failed namespace=%s docs=%d error=%s: other instances' "
                    "ownership of these documents may be dropped by this write",
                    namespace, len(documents), exc)
            prior = {}
        for doc in documents:
            before = prior.get(doc.id, {})
            owners = [str(i) for i in (before.get(graph_boundary.INSTANCE_IDS) or [])]
            legacy = before.get(graph_boundary.INSTANCE_ID)
            if legacy and str(legacy) not in owners:
                owners.append(str(legacy))
            if token not in owners:
                owners.append(token)
            doc.metadata[graph_boundary.INSTANCE_IDS] = owners[-graph_boundary.MAX_INSTANCES_PER_RECORD:]

    async def _upsert_with_instance_lists(self, namespace: str, documents: List[VectorDocument],
                                          scope: Scope) -> None:
        """Upsert entity/triple documents carrying their instance list.

        A vector server that rejects list-valued metadata (an older Chroma) must
        cost the list, never the ingestion: on that failure the same documents are
        written again without it, which is exactly what was written before lists
        existed.
        """
        from synap.cloud.shared import graph_boundary
        await self._stamp_instance_lists(namespace, documents, scope)
        try:
            await _run_chroma(self._provider.upsert, namespace, documents)
        except Exception as exc:  # noqa: BLE001
            # Only a server that REJECTS the list may cost the list. Any other
            # failure (a dropped connection, a timeout) must surface as it always
            # did: retrying without the list would overwrite the document and drop
            # every other instance's ownership, for a failure that had nothing to
            # do with lists.
            if (not any(graph_boundary.INSTANCE_IDS in (d.metadata or {}) for d in documents)
                    or not _looks_like_metadata_rejection(exc)):
                raise
            logger.warning(
                "entity_vector_list_metadata_rejected namespace=%s error=%s: writing without the "
                "instance list", namespace, exc)
            for d in documents:
                d.metadata.pop(graph_boundary.INSTANCE_IDS, None)
            await _run_chroma(self._provider.upsert, namespace, documents)

    async def upsert_entities_batch(
        self,
        entities: List[ExtractedEntity],
        embeddings: List[List[float]],
        scope: Scope,
    ) -> List[str]:
        """Batch upsert entities with embeddings to entity collection.

        Uses deterministic IDs based on (canonical_name, entity_type, customer)
        so that re-ingesting the same entity overwrites instead of duplicating.
        The instance is NOT in the id: instances of one customer share the
        document, and `instance_ids` in its metadata lists every writer.
        """
        if len(entities) != len(embeddings):
            raise ValueError("Number of entities must match number of embeddings")

        namespace = self._get_entity_namespace(scope)
        client_id = str(scope.client_id)
        # For client-scope ingestion customer_id is absent; fall back to
        # client_id so deterministic_entity_id still produces stable hashes.
        customer_id_for_hash = entity_id_scope(scope)

        # Deduplicate within this batch: same canonical_name → last one wins
        seen: Dict[str, int] = {}
        documents = []
        entity_ids = []

        for idx, (entity, embedding) in enumerate(zip(entities, embeddings)):
            display_name = entity.canonical_name or entity.name
            det_id = deterministic_entity_id(
                entity_id_name(entity), entity.entity_type, client_id, customer_id_for_hash)

            metadata = {
                **_scope_metadata(scope),
                "entity_type": entity.entity_type,
                "name": entity.name,
                "canonical_name": display_name,
            }

            doc = VectorDocument(
                id=det_id,
                embedding=embedding,
                metadata=metadata,
                # The writer's node is named by the user id; show "the user".
                content=f"{_shown_name(entity.properties, display_name)} ({entity.entity_type})",
            )

            if det_id in seen:
                # Overwrite earlier duplicate in the batch
                documents[seen[det_id]] = doc
            else:
                seen[det_id] = len(documents)
                documents.append(doc)

            entity_ids.append(det_id)

        if documents:
            await self._upsert_with_instance_lists(namespace, documents, scope)

        return entity_ids

    async def upsert_triples_batch(
        self,
        triples: List[Dict[str, Any]],
        embeddings: List[List[float]],
        scope: Scope,
    ) -> List[str]:
        """
        Batch upsert triples with embeddings to triples collection.

        Each triple dict should have:
          id, content, source_entity_id, target_entity_id,
          relationship_type, source_entity_name, target_entity_name
        """
        if len(triples) != len(embeddings):
            raise ValueError("Number of triples must match number of embeddings")

        namespace = self._get_triples_namespace(scope)
        # Deduplicate within this batch: same triple ID → last one wins
        seen: Dict[str, int] = {}
        documents = []
        triple_ids = []

        for triple, embedding in zip(triples, embeddings):
            metadata = {
                **_scope_metadata(scope),
                "source_entity_id": triple["source_entity_id"],
                "target_entity_id": triple["target_entity_id"],
                "relationship_type": triple["relationship_type"],
                "source_entity_name": triple["source_entity_name"],
                "target_entity_name": triple["target_entity_name"],
            }

            doc = VectorDocument(
                id=triple["id"],
                embedding=embedding,
                metadata=metadata,
                content=triple["content"],
            )

            if triple["id"] in seen:
                documents[seen[triple["id"]]] = doc
            else:
                seen[triple["id"]] = len(documents)
                documents.append(doc)

            triple_ids.append(triple["id"])

        if documents:
            await self._upsert_with_instance_lists(namespace, documents, scope)

        return triple_ids

    async def ensure_chunks_namespace(
        self,
        scope: Scope,
        dimension: int = 1024,
    ) -> None:
        """Ensure raw content chunks collection exists for the given scope."""
        namespace = self._get_chunks_namespace(scope)
        try:
            await _run_chroma(
                self._provider.create_namespace,
                namespace,
                dimension
            )
        except Exception as e:
            logger.debug("Chunks namespace creation skipped (likely exists): %s", e)

    async def ensure_sentences_namespace(
        self,
        scope: Scope,
        dimension: int = 1024,
    ) -> None:
        """Ensure sentence-level embeddings collection exists for the given scope."""
        namespace = self._get_sentences_namespace(scope)
        try:
            await _run_chroma(
                self._provider.create_namespace,
                namespace,
                dimension
            )
        except Exception as e:
            logger.debug("Sentences namespace creation skipped (likely exists): %s", e)

    async def ensure_questions_namespace(
        self,
        scope: Scope,
        dimension: int = 1024,
    ) -> None:
        """Ensure hypothetical questions collection exists for the given scope."""
        namespace = self._get_questions_namespace(scope)
        try:
            await _run_chroma(
                self._provider.create_namespace,
                namespace,
                dimension
            )
        except Exception as e:
            logger.debug("Questions namespace creation skipped (likely exists): %s", e)

    async def upsert_chunks_batch(
        self,
        chunks: List[str],
        chunk_metadata: List[Dict[str, Any]],
        embeddings: List,
        scope: Scope,
    ) -> List[str]:
        """Batch upsert raw content chunks with embeddings to chunks collection.

        Uses deterministic IDs based on (source_document_id, chunk_index) so
        re-ingesting the same document overwrites instead of duplicating.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        namespace = self._get_chunks_namespace(scope)
        documents = []
        chunk_ids = []

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            meta = chunk_metadata[i] if i < len(chunk_metadata) else {}
            source_doc_id = meta.get("source_document_id", "unknown")
            chunk_index = meta.get("chunk_index", i)

            # Deterministic ID: same source doc + chunk index always produces same ID.
            # Include sub_chunk_id if present (Stage 0 sub-chunks) to prevent
            # different sub-chunks of the same document from overwriting each other.
            sub_chunk_id = meta.get("sub_chunk_id", "")
            det_key = f"{source_doc_id}|{sub_chunk_id}|{chunk_index}" if sub_chunk_id else f"{source_doc_id}|{chunk_index}"
            det_id = hashlib.blake2b(det_key.encode(), digest_size=16).hexdigest()

            metadata = {
                **_scope_metadata(scope),
                "source_document_id": source_doc_id,
                "chunk_index": chunk_index,
                "total_chunks": meta.get("total_chunks", len(chunks)),
                "created_at": meta.get("created_at", ""),
            }

            doc = VectorDocument(
                id=det_id,
                embedding=embedding,
                metadata=metadata,
                content=chunk_text,
            )
            documents.append(doc)
            chunk_ids.append(det_id)

        if documents:
            await _run_chroma(
                self._provider.upsert,
                namespace,
                documents
            )

        return chunk_ids

    async def upsert_questions_batch(
        self,
        questions: List[Dict[str, Any]],
        embeddings: List[List[float]],
        scope: Scope,
    ) -> List[str]:
        """Batch upsert hypothetical questions with embeddings to questions collection.

        Each question dict should have:
          content (question text), memory_id, memory_content, memory_type,
          source_document_id, category, subcategory, confidence, question_index

        Uses deterministic IDs based on (memory_id, question_index) so
        re-ingesting the same memory overwrites instead of duplicating.
        """
        if len(questions) != len(embeddings):
            raise ValueError("Number of questions must match number of embeddings")

        namespace = self._get_questions_namespace(scope)
        documents = []
        question_ids = []

        for question, embedding in zip(questions, embeddings):
            # Deterministic ID: same memory + question index = same ID
            det_key = f"{question['memory_id']}|{question.get('question_index', 0)}"
            det_id = hashlib.blake2b(det_key.encode(), digest_size=16).hexdigest()

            metadata = {
                **_scope_metadata(scope),
                "memory_id": question["memory_id"],
                "memory_type": question.get("memory_type", ""),
                "memory_content": question.get("memory_content", "")[:500],
                "source_document_id": question.get("source_document_id", ""),
                "category": question.get("category", ""),
                "subcategory": question.get("subcategory", ""),
                "confidence": question.get("confidence", 0.5),
                "temporal_category": question.get("temporal_category") or "",
                "temporal_confidence": question.get("temporal_confidence", 0.0),
                "is_episode": str(question.get("is_episode", False)).lower(),
            }
            # Temporal dates — ChromaDB rejects None values
            if question.get("extracted_at"):
                metadata["created_at"] = question["extracted_at"]
            if question.get("document_created_at"):
                metadata["document_created_at"] = question["document_created_at"]
            if question.get("valid_from"):
                metadata["valid_from"] = question["valid_from"]
            if question.get("valid_until"):
                metadata["valid_until"] = question["valid_until"]
            for key in ("event_start", "event_end", "event_precision"):
                if question.get(key):
                    metadata[key] = question[key]

            doc = VectorDocument(
                id=det_id,
                embedding=embedding,
                metadata=metadata,
                content=question["content"],
            )
            documents.append(doc)
            question_ids.append(det_id)

        if documents:
            await _run_chroma(
                self._provider.upsert,
                namespace,
                documents
            )

        return question_ids

    async def batch_search_questions(
        self,
        embeddings: List[List[float]],
        scope: Scope,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scope_level: str = "customer",
    ) -> List[List[Dict[str, Any]]]:
        """Batch search hypothetical questions — single ChromaDB call with multiple embeddings."""
        namespace = self._get_questions_namespace(scope, scope_level=scope_level)
        queries = [
            VectorQuery(embedding=emb, top_k=top_k, filters=filters, namespace=namespace)
            for emb in embeddings
        ]

        results = await _run_chroma(self._provider.batch_search, queries)

        return [
            [
                {"id": r.id, "score": r.score, "metadata": r.metadata, "content": r.content}
                for r in result.results
            ]
            for result in results
        ]

    async def search_collection(
        self,
        embedding: List[float],
        scope: Scope,
        collection_type: str = "memories",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search a specific collection type.

        Args:
            collection_type: "memories", "entities", or "triples"
        """
        if collection_type == "entities":
            namespace = self._get_entity_namespace(scope)
        elif collection_type == "triples":
            namespace = self._get_triples_namespace(scope)
        elif collection_type == "chunks":
            namespace = self._get_chunks_namespace(scope)
        else:
            namespace = self._get_namespace(scope)

        query = VectorQuery(
            embedding=embedding,
            top_k=top_k,
            filters=filters,
            namespace=namespace,
        )

        result = await _run_chroma(
            self._provider.search,
            query
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "metadata": r.metadata,
                "content": r.content,
            }
            for r in result.results
        ]

    async def search(
        self,
        embedding: List[float],
        scope: Scope,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scope_level: str = "customer",
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        namespace = self._get_namespace(scope, scope_level=scope_level)
        
        query = VectorQuery(
            embedding=embedding,
            top_k=top_k,
            filters=filters,
            namespace=namespace,
        )
        
        result = await _run_chroma(
            self._provider.search,
            query
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "metadata": r.metadata,
                "content": r.content,
            }
            for r in result.results
        ]

    async def batch_search(
        self,
        embeddings: List[List[float]],
        scope: Scope,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        scope_level: str = "customer",
    ) -> List[List[Dict[str, Any]]]:
        """Batch search — single ChromaDB call with multiple embeddings."""
        namespace = self._get_namespace(scope, scope_level=scope_level)
        queries = [
            VectorQuery(embedding=emb, top_k=top_k, filters=filters, namespace=namespace)
            for emb in embeddings
        ]

        results = await _run_chroma(self._provider.batch_search, queries)

        return [
            [
                {"id": r.id, "score": r.score, "metadata": r.metadata, "content": r.content}
                for r in result.results
            ]
            for result in results
        ]

    async def batch_search_chunks(
        self,
        embeddings: List[List[float]],
        scope: Scope,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Batch search raw content chunks — single ChromaDB call with multiple embeddings."""
        namespace = self._get_chunks_namespace(scope)
        queries = [
            VectorQuery(embedding=emb, top_k=top_k, filters=filters, namespace=namespace)
            for emb in embeddings
        ]

        results = await _run_chroma(self._provider.batch_search, queries)

        return [
            [
                {"id": r.id, "score": r.score, "metadata": r.metadata, "content": r.content}
                for r in result.results
            ]
            for result in results
        ]

    async def batch_search_sentences(
        self,
        embeddings: List[List[float]],
        scope: Scope,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None,
        scope_level: str = "customer",
    ) -> List[List[Dict[str, Any]]]:
        """Batch search sentence-level embeddings — single ChromaDB call with multiple embeddings."""
        namespace = self._get_sentences_namespace(scope, scope_level=scope_level)
        queries = [
            VectorQuery(embedding=emb, top_k=top_k, filters=filters, namespace=namespace)
            for emb in embeddings
        ]

        results = await _run_chroma(self._provider.batch_search, queries)

        return [
            [
                {"id": r.id, "score": r.score, "metadata": r.metadata, "content": r.content}
                for r in result.results
            ]
            for result in results
        ]

    async def get_all_from_collection(
        self,
        scope: Scope,
        collection_type: str = "memories",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 500,
        scope_level: str = "customer",
    ) -> List[Dict[str, Any]]:
        """
        Get all documents (with embeddings) from a collection.

        Used by adjacency builder to load entities and triples.
        """
        if collection_type == "entities":
            namespace = self._get_entity_namespace(scope, scope_level=scope_level)
        elif collection_type == "triples":
            namespace = self._get_triples_namespace(scope, scope_level=scope_level)
        elif collection_type == "chunks":
            namespace = self._get_chunks_namespace(scope, scope_level=scope_level)
        elif collection_type == "questions":
            namespace = self._get_questions_namespace(scope, scope_level=scope_level)
        else:
            namespace = self._get_namespace(scope, scope_level=scope_level)

        documents = await _run_chroma(
            self._provider.get_all,
            namespace,
            filters,
            limit,
        )

        return [
            {
                "id": doc.id,
                "embedding": doc.embedding,
                "metadata": doc.metadata,
                "content": doc.content,
            }
            for doc in documents
        ]

    async def delete(
        self,
        ids: List[str],
        scope: Scope,
    ) -> int:
        """Delete vectors by ID."""
        namespace = self._get_namespace(scope)
        
        return await _run_chroma(
            self._provider.delete,
            namespace,
            ids
        )

    # The collections `search_collection` and `delete` do not reach, addressed by
    # name. Used by forget-by-description, which has to find text by what it says
    # and not by a memory id, because chunks, sentences, entities and triples are
    # keyed by their own ids.
    def _namespace_for(self, scope: Scope, collection: str) -> str:
        getters = {
            "memories": self._get_namespace,
            "questions": self._get_questions_namespace,
            "chunks": self._get_chunks_namespace,
            "sentences": self._get_sentences_namespace,
            "entities": self._get_entity_namespace,
            "triples": self._get_triples_namespace,
        }
        if collection not in getters:
            raise ValueError(f"unknown collection {collection!r}")
        return getters[collection](scope)

    async def search_in(
        self,
        embedding: List[float],
        scope: Scope,
        collection: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Similarity search in any one collection, by name."""
        query = VectorQuery(embedding=embedding, top_k=top_k, filters=filters,
                            namespace=self._namespace_for(scope, collection))
        try:
            result = await _run_chroma(self._provider.search, query)
        except Exception as exc:  # noqa: BLE001
            # A collection nothing was ever written to is empty, not broken: there is
            # nothing in it to clear. Any other error still propagates.
            if "does not exist" in str(exc).lower():
                return []
            raise
        return [{"id": r.id, "score": r.score, "metadata": r.metadata,
                 "content": r.content} for r in result.results]

    async def delete_in(self, scope: Scope, collection: str, ids: List[str]) -> int:
        """Delete vectors by id from any one collection, by name. Logged for
        replication like every other delete that passes through `_run_chroma`."""
        if not ids:
            return 0
        return await _run_chroma(self._provider.delete,
                                 self._namespace_for(scope, collection), ids)

    async def delete_memory_vectors(
        self,
        memory_ids: List[str],
        scope: Scope,
    ) -> int:
        """Delete memory vectors AND their associated HQE question vectors.

        When a memory is superseded, both its embedding in the memories
        collection AND the hypothetical questions in the questions collection
        must be removed. Orphaned questions contain stale memory_content in
        their metadata, which gets served to the downstream LLM and causes
        it to use outdated values.
        """
        deleted = 0

        # 1. Delete from memories collection
        namespace = self._get_namespace(scope)
        try:
            count = await _run_chroma(
                self._provider.delete,
                namespace,
                memory_ids,
            )
            deleted += count if isinstance(count, int) else len(memory_ids)
        except Exception as e:
            logger.warning("Failed to delete from memories collection: %s", e)

        # 2. Delete associated HQE questions from questions collection.
        # Question IDs are deterministic: blake2b(f"{memory_id}|{question_index}").
        # Max 5 questions per memory (HQE_MAX_QUESTIONS_PER_MEMORY).
        questions_namespace = self._get_questions_namespace(scope)
        question_ids_to_delete = []
        for memory_id in memory_ids:
            for q_idx in range(5):  # Max 5 questions per memory
                det_key = f"{memory_id}|{q_idx}"
                det_id = hashlib.blake2b(det_key.encode(), digest_size=16).hexdigest()
                question_ids_to_delete.append(det_id)

        if question_ids_to_delete:
            try:
                await _run_chroma(
                    self._provider.delete,
                    questions_namespace,
                    question_ids_to_delete,
                )
                logger.debug(
                    "Deleted %d potential HQE question vectors for %d superseded memories",
                    len(question_ids_to_delete), len(memory_ids),
                )
            except Exception as e:
                # Questions collection may not exist — non-fatal
                logger.debug("Failed to delete HQE questions (collection may not exist): %s", e)

        return deleted

    async def delete_person_vectors(
        self,
        scope: Scope,
        page: int = 500,
        instance_id: Optional[str] = None,
        keep_if_referenced=None,
    ) -> Dict[str, int]:
        """Remove every vector this person owns, from all six collections.

        `delete_memory_vectors` clears two of them, memories and questions,
        because it deletes by memory id and only those two are keyed that way.
        The other four are real stores of the same text: `chunks` holds the raw
        source, `entity` and `triples` hold names lifted out of it, `sentences`
        holds it split up. An erasure that stops at two leaves the person's
        name in four places and reports itself done.

        Those four are keyed by their own ids, so this finds the ids by scope
        first and then deletes them. It returns what each collection actually
        gave up rather than what it was asked to remove: the count this
        function chose for itself is exactly how the graph clear hid the fact
        it was deleting nothing.
        """
        filters: Dict[str, Any] = {"client_id": str(scope.client_id)}
        if getattr(scope, "user_id", None):
            filters["user_id"] = str(scope.user_id)
        elif getattr(scope, "customer_id", None):
            filters["customer_id"] = str(scope.customer_id)
        # Optional: a customer-called delete is bounded to ITS instance. Every
        # document is written with `instance_id` in its metadata
        # (`_scope_metadata`). Omitted, this is exactly what it always was, and
        # the staff erase route, which is client-wide on purpose, omits it.
        if instance_id:
            filters["instance_id"] = str(instance_id)

        namespaces = {
            "memories": self._get_namespace(scope),
            "questions": self._get_questions_namespace(scope),
            "chunks": self._get_chunks_namespace(scope),
            "entity": self._get_entity_namespace(scope),
            "triples": self._get_triples_namespace(scope),
            "sentences": self._get_sentences_namespace(scope),
        }

        out: Dict[str, int] = {}
        for label, namespace in namespaces.items():
            removed = 0
            kept_ids: set = set()
            try:
                while True:
                    limit = page + len(kept_ids)
                    docs = await _run_chroma(
                        self._provider.get_all, namespace, filters, limit
                    )
                    cand = [d for d in (docs or [])
                            if getattr(d, "id", None) and d.id not in kept_ids]
                    if label == "chunks" and keep_if_referenced is not None and cand:
                        # A chunk is shared source text: the same document backs
                        # other users' memories. It goes only when nothing that
                        # remains references it (the rule memories.delete uses).
                        by_doc = {d.id: (getattr(d, "metadata", None) or {}).get(
                            "source_document_id") for d in cand}
                        keep = await keep_if_referenced(
                            sorted({v for v in by_doc.values() if v}))
                        for cid, doc in by_doc.items():
                            if doc and doc in keep:
                                kept_ids.add(cid)
                        cand = [d for d in cand if d.id not in kept_ids]
                    ids = [d.id for d in cand]
                    if not ids:
                        break
                    removed += await _run_chroma(
                        self._provider.delete, namespace, ids
                    ) or len(ids)
                    if len(docs or []) < limit:
                        break
            except Exception as exc:
                # A collection that does not exist for this scope is normal: an
                # instance that never wrote to `sentences` has no such
                # collection, so there is nothing in it to delete (0). Before,
                # this branch said that in a comment and then returned -1 for it
                # too, so every delete on such an instance ended 'partial' with
                # "search index" still present. Anything else is recorded
                # against that collection so a partial clearance is visible.
                if _is_missing_collection(exc):
                    logger.info(
                        "person_vector_clear_collection_absent collection=%s", label)
                    out[label] = removed
                    continue
                logger.warning(
                    "person_vector_clear_failed collection=%s: %s", label, exc
                )
                out[label] = -1
                continue
            out[label] = removed
            if label == "chunks" and keep_if_referenced is not None:
                out["chunks_kept_shared"] = len(kept_ids)
        return out

    async def get_chunks_by_document_id(
        self,
        scope: Scope,
        source_document_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch all chunks for a given source_document_id from the chunks collection.

        Uses ChromaDB's get_all() with where filter — no embedding needed.
        Returns chunks ordered by chunk_index for coherent reading.

        Args:
            scope: Tenant scope (used to derive chunks collection namespace)
            source_document_id: The document ID to fetch chunks for
            limit: Maximum number of chunks to return

        Returns:
            List of dicts with keys: id, content, metadata.
            Empty list if collection doesn't exist or no chunks found.
        """
        namespace = self._get_chunks_namespace(scope)

        try:
            documents = await _run_chroma(
                self._provider.get_all,
                namespace,
                {"source_document_id": source_document_id},
                limit,
            )
        except Exception as e:
            # Collection may not exist if SKIP_CHUNK_EMBEDDING was always true
            logger.debug(
                "get_chunks_by_document_id failed (collection may not exist): %s", e
            )
            return []

        # Sort by chunk_index for coherent ordering
        results = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in documents
            if doc.content
        ]
        results.sort(key=lambda r: r.get("metadata", {}).get("chunk_index", 0))

        return results

    async def delete_source_document_chunks(
        self,
        scope: Scope,
        source_document_id: str,
    ) -> int:
        """Remove the raw source text a document contributed, and say how much.

        Deleting a memory used to leave the sentences it came from sitting in
        the `chunks` collection, still searchable, still holding the original
        wording. A client deleting an allergy for a privacy reason had the
        allergy removed and the sentence that stated it left behind.

        ⚠ THE CALLER MUST CHECK FIRST THAT NOTHING ELSE NEEDS THIS DOCUMENT.
        One source document backs many memories: every memory carries a
        `source_document_id` and a busy document can back hundreds of them.
        Purging chunks because ONE of those memories was deleted would strip the
        provenance from all the others, which is a far worse outcome than the
        problem being fixed. This function does not make that judgement,
        because it cannot see the other memories; `delete_memory` does.

        Returns the number of chunks removed, so the caller can tell a client
        what actually happened rather than assuming.
        """
        namespace = self._get_chunks_namespace(scope)
        try:
            documents = await _run_chroma(
                self._provider.get_all,
                namespace,
                {"source_document_id": source_document_id},
                1000,
            )
        except Exception as e:
            # The collection legitimately may not exist, when chunk embedding
            # was switched off for this client. Nothing to remove is not a
            # failure, but it must not be reported as a successful purge.
            logger.debug("no chunks collection for %s: %s", namespace, e)
            return 0

        ids = [d.id for d in (documents or []) if getattr(d, "id", None)]
        if not ids:
            return 0
        await _run_chroma(self._provider.delete, namespace, ids)
        logger.info(
            "source_chunks_purged document=%s namespace=%s chunks=%d",
            source_document_id, namespace, len(ids),
        )
        return len(ids)

    async def get_chunks_by_document_ids(
        self,
        scope: Scope,
        source_document_ids: List[str],
        limit_per_document: int = 50,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """The same as `get_chunks_by_document_id`, for many documents at once.

        Every document lives in the SAME chunks collection for a given scope, so
        fetching them one at a time is one round trip per document against a
        store that serialises them. For ten documents, one `$in` query is
        roughly 8x faster than ten concurrent single fetches, returning
        byte-identical rows, for a change that alters nothing
        about what comes back.

        The caller's per-document limit is preserved by asking for
        `limit_per_document * len(ids)` and truncating each document's chunks
        after grouping.

        ⚠ **The truncation guard is the point.** A single global limit can
        starve a document: one document with many chunks could consume the whole
        allowance and leave another with none, which the per-document shape can
        never do. So when the batched read comes back at exactly the limit, and
        truncation is therefore possible, this falls back to the per-document
        path rather than returning a quietly incomplete answer.
        """
        if not source_document_ids:
            return {}

        namespace = self._get_chunks_namespace(scope)
        want = limit_per_document * len(source_document_ids)

        try:
            documents = await _run_chroma(
                self._provider.get_all,
                namespace,
                {"source_document_id": {"$in": list(source_document_ids)}},
                want,
            )
        except Exception as e:
            logger.debug("get_chunks_by_document_ids batched read failed: %s", e)
            documents = None

        if documents is not None and len(documents) >= want:
            # The batched read came back full, so truncation is possible and is
            # invisible in the rows. Falling back is correct; falling back
            # SILENTLY is not, because the caller then pays the per-document
            # cost while the log still says the batch path is in use.
            logger.info(
                "chunk_batch_full_falling_back: docs=%d limit_per_document=%d "
                "rows=%d; re-reading per document to avoid starving one",
                len(source_document_ids), limit_per_document, len(documents),
            )

        if documents is not None and len(documents) < want:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for doc in documents:
                if not doc.content:
                    continue
                meta = doc.metadata or {}
                key = meta.get("source_document_id")
                if key is None:
                    continue
                grouped.setdefault(key, []).append(
                    {"id": doc.id, "content": doc.content, "metadata": meta}
                )
            for key, rows in grouped.items():
                rows.sort(key=lambda r: r.get("metadata", {}).get("chunk_index", 0))
                del rows[limit_per_document:]
            return {d: grouped.get(d, []) for d in source_document_ids}

        # Either the batched read failed, or it came back full and may have been
        # truncated. Fall back to the shape whose limit is per document.
        out: Dict[str, List[Dict[str, Any]]] = {}
        for doc_id in source_document_ids:
            out[doc_id] = await self.get_chunks_by_document_id(
                scope=scope, source_document_id=doc_id, limit=limit_per_document
            )
        return out

    async def get_by_ids(
        self,
        ids: List[str],
        scope: Scope,
    ) -> List[VectorDocument]:
        """Get vector documents (with embeddings) by ID."""
        namespace = self._get_namespace(scope)
        
        return await _run_chroma(
            self._provider.get,
            namespace,
            ids
        )
