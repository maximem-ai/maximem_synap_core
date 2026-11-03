"""Storage coordinator for orchestrating persistence across all storage systems."""

import copy
import logging
import os
import re
from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, List, Optional
from uuid import UUID

from synap.cloud.agents.ingestion.storage.vector_wrapper import (
    deterministic_entity_id,
    entity_id_name,
)

from synap.cloud.agents.ingestion.interfaces import EmbeddingClient

logger = logging.getLogger(__name__)
from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    ExtractedEntity,
    ExtractedRelationship,
    MemoryDestination,
    Scope,
)
from .rdbms_protocol import RDBMSStorageProtocol
from .graph_wrapper import AsyncGraphWrapper
from .vector_wrapper import AsyncVectorWrapper


#: Longest a subject label may be. Long enough to carry the words that tell one
#: coffee table from another ("wooden coffee table with metal legs from West
#: Elm"), short enough that repeating it on every served item is not the reason
#: a bundle runs out of room.
SUBJECT_LABEL_MAX = 160


def _specificity(text: Optional[str]) -> tuple:
    """How specifically a sentence names the thing it is about.

    Ranked by the count of tokens that must survive a merge: named entities,
    quoted titles, quantities, dates. Those are exactly the words that tell one
    of the person's things from another, and the same function already guards
    every merge in the pipeline. Length breaks ties, because between two equally
    specific wordings the fuller one reads better.
    """
    body = (text or "").strip()
    if not body:
        return (0, 0)
    try:
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            extract_must_preserve_tokens,
        )
        return (len(extract_must_preserve_tokens(body)), len(body))
    except Exception:  # noqa: BLE001 — a label never fails an ingest
        return (0, len(body))


def _group_label(texts: List[str]) -> str:
    """One label for a whole group, from the member that names the thing best.

    Computed over EVERY member, not over the two sides of one link. The pairwise
    version shipped first and it was visibly wrong in a live payload: one
    subject_id came back carrying three different labels, because each link
    relabelled only its own two rows. The reader never saw that, since the
    served marker is an ordinal, but it made a wrong grouping unreadable to us,
    which is the one job the label has.

    The person's own words, never a word of ours: a label we invent is a second
    chance to be wrong about the thing, and a broader category word ("furniture"
    for a coffee table) would quietly put two real things under one name.
    """
    best = max((t or "" for t in texts), key=_specificity, default="").strip()
    if len(best) <= SUBJECT_LABEL_MAX:
        return best
    cut = best[:SUBJECT_LABEL_MAX - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > SUBJECT_LABEL_MAX // 2 else cut).rstrip(" ,;:") + "\u2026"


@dataclass
class RDBMSPersistResult:
    """Result of RDBMS persistence."""
    memory_ids: List[UUID]
    candidate_ids: List[UUID]
    entity_ids: List[str]
    # Ids stored in customer_knowledge / client_knowledge. Kept apart from memory_ids on
    # purpose: usage metering and the profile updater read memory_ids and expect rows in the
    # memories table. Only the ingestion status reports these.
    knowledge_ids: List[UUID] = field(default_factory=list)


@dataclass
class GraphPersistResult:
    """Result of Graph persistence."""
    entity_ids: List[str]
    relationship_ids: List[str]


@dataclass
class VectorPersistResult:
    """Result of Vector persistence."""
    vector_ids: List[str]


@dataclass
class EntityVectorPersistResult:
    """Result of entity vector persistence."""
    entity_vector_ids: List[str]


@dataclass
class TripleVectorPersistResult:
    """Result of triple vector persistence."""
    triple_vector_ids: List[str]


@dataclass
class ChunkVectorPersistResult:
    """Result of raw content chunk vector persistence."""
    chunk_vector_ids: List[str]
    chunk_count: int


@dataclass
class QuestionVectorPersistResult:
    """Result of hypothetical question vector persistence."""
    question_vector_ids: List[str]
    question_count: int


# Bound on the retraction sweep v2 class prefilter fetch. Loose by design (it
# is an ILIKE keyword net); every fetched row is re-confirmed against the real
# attribute-class regexes before it can be retired, and at most
# max_rows_per_sweep of them survive that check.
RETRACTION_CLASS_FETCH_LIMIT = 200


class StorageCoordinator:
    """
    Coordinates persistence to all storage providers.
    
    Execution order:
    1. RDBMS first (sync) - generates authoritative IDs
    2. Graph and Vector in parallel (async)
    """

    def __init__(
        self,
        rdbms: RDBMSStorageProtocol,
        graph: AsyncGraphWrapper,
        vector: AsyncVectorWrapper,
        embedding_client: EmbeddingClient,
        graph_config: Optional[Dict[str, Any]] = None,
    ):
        self._rdbms = rdbms
        self._graph = graph
        self._vector = vector
        self._embedding = embedding_client
        self._graph_config = graph_config

    async def persist_rdbms(
        self,
        memories: List[ExtractedMemory],
        candidates: List[ExtractedMemory],
        entities: List[ExtractedEntity],
        scope: Scope,
        document_id: str,
        client_knowledge: Optional[List[ExtractedMemory]] = None,
        customer_knowledge: Optional[List[ExtractedMemory]] = None,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> RDBMSPersistResult:
        """
        Persist to RDBMS (sync, first).

        Tables:
        - memories: Qualified memories (above threshold)
        - candidates: Below threshold (pending review)
        - client_knowledge: Client-level knowledge
        - customer_knowledge: Customer-level knowledge
        - memory_entity_links: Entity references for audit
        - memory_actions: Per-request changelog (when http_correlation_id is provided)
        """
        memory_ids = []
        candidate_ids = []
        knowledge_ids = []

        # Persist memories
        for memory in memories:
            memory_id = await self._rdbms.insert_memory(
                memory=memory,
                scope=scope,
                document_id=document_id,
            )
            memory_ids.append(memory_id)

            # KU v2: write the M:N source-document attribution from day one
            # so retrieval can fan out chunk-fetches without a
            # backfill gap on memories ingested before the attribution existed.
            # Idempotent on (memory_id, source_document_id).
            if document_id:
                await self._rdbms.insert_memory_source_documents(
                    memory_id=str(memory_id),
                    source_document_ids=[document_id],
                )

            # Changelog: one CREATE row per memory successfully inserted.
            if http_correlation_id:
                await self._rdbms.insert_memory_action(
                    correlation_id=http_correlation_id,
                    ingestion_id=ingestion_id,
                    scope=scope,
                    action="CREATE",
                    memory_id=str(memory_id),
                    content=memory.content,
                    previous_content=None,
                    memory_type=memory.memory_type.value,
                )

                # And a COALESCE row when this memory used to be several. The
                # rows it replaced were folded away in stage 3 and never reached
                # the database, so this entry is the only record that they
                # existed or why they were judged one fact. See migration 156.
                #
                # Wrapped, and deliberately: a database without 156 rejects the
                # action value, and an audit row must never be the reason a
                # memory fails to store. One warning, then carry on.
                provenance = getattr(memory, "coalesced_from", None)
                if provenance:
                    try:
                        await self._rdbms.insert_memory_action(
                            correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                            scope=scope,
                            action="COALESCE",
                            memory_id=str(memory_id),
                            content=memory.content,
                            previous_content="\n".join(
                                f"[{t}] {c}" for t, c in zip(
                                    provenance.get("types") or [],
                                    provenance.get("contributors") or [])),
                            memory_type=memory.memory_type.value,
                            reason=provenance.get("reason") or "",
                        )
                    except Exception:  # noqa: BLE001 — audit never blocks a write
                        logger.warning(
                            "coalesce_audit_row_failed memory_id=%s "
                            "(migration 156 applied?)", memory_id, exc_info=True)

            # Create entity references
            for entity_id in memory.entities_mentioned:
                await self._rdbms.insert_entity_reference(
                    memory_id=memory_id,
                    entity_id=entity_id,
                    scope=scope,
                )

        # Persist candidates
        for memory in candidates:
            candidate_id = await self._rdbms.insert_candidate(
                memory=memory,
                scope=scope,
                document_id=document_id,
            )
            candidate_ids.append(candidate_id)
            
            # Create entity references for candidates too
            for entity_id in memory.entities_mentioned:
                await self._rdbms.insert_entity_reference(
                    memory_id=candidate_id,
                    entity_id=entity_id,
                    scope=scope,
                )

        # Persist client knowledge if provided
        if client_knowledge:
            for memory in client_knowledge:
                knowledge_ids.append(await self._rdbms.insert_client_knowledge(
                    memory=memory,
                    scope=scope,
                    document_id=document_id,
                ))

        # Persist customer knowledge if provided
        if customer_knowledge:
            for memory in customer_knowledge:
                knowledge_ids.append(await self._rdbms.insert_customer_knowledge(
                    memory=memory,
                    scope=scope,
                    document_id=document_id,
                ))

        return RDBMSPersistResult(
            memory_ids=memory_ids,
            candidate_ids=candidate_ids,
            entity_ids=[e.entity_id for e in entities],
            knowledge_ids=knowledge_ids,
        )

    async def persist_graph(
        self,
        entities: List[ExtractedEntity],
        relationships: List[ExtractedRelationship],
        memories: List[ExtractedMemory],
        scope: Scope,
        entity_embeddings: Optional[List] = None,
        triple_embeddings: Optional[List] = None,
        memory_embeddings: Optional[List] = None,
    ) -> GraphPersistResult:
        """
        Persist entities and relationships to graph store.

        Uses existing Graph Provider interface with namespace isolation.
        After creating relationship edges, attaches memory content as a JSON list
        property on edges whose entity pairs match the memory's entities_mentioned.
        """
        if self._graph is None:
            logger.debug("Graph store unavailable — skipping graph persistence")
            return GraphPersistResult(entity_ids=[], relationship_ids=[])

        # Deep-copy to avoid mutating shared objects used concurrently by persist_triple_vectors
        relationships = copy.deepcopy(relationships)

        # Ensure namespace exists
        await self._graph.ensure_namespace(scope)

        entity_ids = []
        relationship_ids = []

        # Determine merge setting from graph config
        merge_similar = True
        if self._graph_config:
            canon_config = self._graph_config.get("entity_canonicalization", {})
            merge_similar = canon_config.get("merge_similar_entities", True)

        # Create/update entity nodes (with pre-computed embeddings if available)
        for i, entity in enumerate(entities):
            emb = entity_embeddings[i] if entity_embeddings and i < len(entity_embeddings) else None
            entity_id = await self._graph.upsert_entity_node(
                entity=entity,
                scope=scope,
                merge_similar=merge_similar,
                embedding=emb,
            )
            entity_ids.append(entity_id)

        # Build mapping: extractor entity_id → neo4j node_id
        # Neo4j add_node generates its own UUID, so we need to remap
        entity_id_to_graph_id = {}
        for entity, graph_node_id in zip(entities, entity_ids):
            entity_id_to_graph_id[entity.entity_id] = graph_node_id

        logger.info(f"Graph persist: entity ID remapping built entity_count={len(entity_id_to_graph_id)} mappings={dict(list(entity_id_to_graph_id.items())[:10])}")

        # Remap relationship entity IDs to Neo4j node IDs
        # Skip relationships where either side is unmapped (prevents phantom edges)
        original_for_memories = []
        remapped_rels = []
        remapped_embeds = []
        skipped_count = 0

        # An entity merged on a second look carries a merge id. Every edge it
        # takes part in gets the id too, so an undo can tell these edges from
        # the ones the entity already had (see resolution/merge_log).
        merge_ids = {
            e.entity_id: e.properties["er_merge_id"]
            for e in entities if e.properties.get("er_merge_id")
        }

        for i, rel in enumerate(relationships):
            mapped_source = entity_id_to_graph_id.get(rel.source_entity_id)
            mapped_target = entity_id_to_graph_id.get(rel.target_entity_id)
            stamped = [
                merge_ids[end] for end in (rel.source_entity_id, rel.target_entity_id)
                if end in merge_ids
            ]
            if stamped:
                rel.properties["er_merge_ids"] = stamped
            if mapped_source and mapped_target and mapped_source == mapped_target:
                # Two entities of this note were resolved to one node. The
                # relationship between them now says nothing.
                skipped_count += 1
                continue
            if mapped_source and mapped_target:
                original_for_memories.append(copy.deepcopy(rel))
                rel.source_entity_id = mapped_source
                rel.target_entity_id = mapped_target
                remapped_rels.append(rel)
                if triple_embeddings and i < len(triple_embeddings):
                    remapped_embeds.append(triple_embeddings[i])
            else:
                skipped_count += 1
                logger.warning(f"Graph persist: skipping relationship with unmapped entity source_id={rel.source_entity_id} target_id={rel.target_entity_id} source_mapped={mapped_source is not None} target_mapped={mapped_target is not None}")

        logger.info(f"Graph persist: relationship IDs remapped remapped_count={len(remapped_rels)} skipped_count={skipped_count} total_relationships={len(relationships)}")

        # Create relationships between entities (handle None returns from failed edge creation)
        # written_for_memories stays in step with relationship_ids: an edge
        # that failed to write is in neither, so a memory cannot be attached to
        # the edge of a different relationship.
        written_for_memories = []
        # One edge for one relation between two nodes, whichever way and in
        # whatever words it was said this time (services/edge_identity).
        from synap.cloud.agents.ingestion.services import edge_identity, graph_v2
        from synap.cloud.agents.ingestion.services.writer_entity import shown_name
        settle_edges = graph_v2.enabled(getattr(scope, "instance_id", None))
        node_names = {
            entity_id_to_graph_id[e.entity_id]: shown_name(e.properties, e.canonical_name or e.name)
            for e in entities if e.entity_id in entity_id_to_graph_id
        }
        for i, rel in enumerate(remapped_rels):
            if settle_edges and rel.properties.get("inverse_type"):
                try:
                    existing = await self._graph.edges_between(
                        scope, rel.source_entity_id, rel.target_entity_id)
                    await edge_identity.settle(rel, existing, node_names)
                except Exception as e:  # noqa: BLE001 - never fail ingestion
                    logger.warning(f"Graph persist: edge identity check failed (non-fatal): {e}")
            emb = remapped_embeds[i] if i < len(remapped_embeds) else None
            rel_id = await self._graph.create_relationship(
                relationship=rel,
                scope=scope,
                embedding=emb,
            )
            if rel_id is not None:
                relationship_ids.append(rel_id)
                written_for_memories.append(original_for_memories[i])
                logger.info(f"Graph persist: relationship created rel_id={rel_id} source_id={rel.source_entity_id} target_id={rel.target_entity_id} type={rel.relationship_type}")
            else:
                logger.warning(f"Graph persist: edge creation failed (nodes not found) source_id={rel.source_entity_id} target_id={rel.target_entity_id} type={rel.relationship_type}")

        # Attach memory content to real relationship edges
        # Uses original_for_memories (extraction-time IDs) for pair lookup
        edges_updated = await self._graph.attach_memories_to_relationships(
            memories=memories,
            relationships=written_for_memories,
            relationship_edge_ids=relationship_ids,
            scope=scope,
            memory_embeddings=memory_embeddings,
        )

        # A memory that names one entity is on no relationship; its node keeps
        # a reference to it instead (SYNAP_GRAPH_NODE_MEMORIES). Best-effort.
        from synap.cloud.agents.ingestion.services import node_memories
        if node_memories.enabled(getattr(scope, "instance_id", None)):
            try:
                await self._graph.add_memory_references(
                    scope, node_memories.references(memories, entities, entity_id_to_graph_id))
            except Exception as e:  # noqa: BLE001 - never fail ingestion
                logger.warning(f"Graph persist: node memory references failed (non-fatal): {e}")

        logger.info(f"Graph persist: all graph operations complete entity_nodes_created={len(entity_ids)} relationships_created={len(relationship_ids)} edges_with_memories={edges_updated}")

        # Cross-document relink (flag-gated, best-effort, SCOPE-LEVEL and order-independent).
        # attach_memories_to_relationships above only links within this document (per-doc
        # entity ids + same-doc entities), so cross-document / secondary-entity facts
        # (e.g. a test report's "current blocker on STATUS-247", where STATUS-247 is
        # under-extracted in the test-report doc) never land on an edge. relink_scope
        # matches EVERY scope memory against the full canonical graph — a per-document
        # pass misses cross-doc edges created after a doc's memories were processed.
        # Never breaks ingestion. For large scopes prefer an admin/cron backfill.
        if os.environ.get("ENABLE_GRAPH_RELINK", "false").lower() in ("true", "1", "yes"):
            try:
                relinked = await self.relink_scope(scope)
                logger.info(f"Graph persist: scope relink complete edges_relinked={relinked}")
            except Exception as e:  # noqa: BLE001 - relink is best-effort, never fail ingestion
                logger.warning(f"Graph persist: scope relink failed (non-fatal): {e}")

        return GraphPersistResult(
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
        )

    async def relink_scope(self, scope: Scope, limit: int = 5000) -> int:
        """Scope-level cross-document memory->edge relink (order-independent).

        Loads ALL of the scope's memories and the full canonical graph, then attaches
        each memory to canonical edges whose endpoints it names (see
        AsyncGraphWrapper.relink_memories_to_canonical_edges). This is the reliable
        form — a per-document pass misses cross-doc edges created later. Intended as a
        post-batch / cold-start / admin-backfill operation. Returns edges updated;
        best-effort, returns 0 if graph/vector unavailable or empty.
        """
        if self._graph is None or self._vector is None:
            return 0
        node_id_to_name, edges = await self._graph.load_namespace_canonical_graph(scope)
        if not node_id_to_name or not edges:
            return 0
        memories = await self._vector.get_all_scope_memories(scope, limit=limit)
        if not memories:
            return 0
        return await self._graph.relink_memories_to_canonical_edges(
            scope, memories, node_id_to_name, edges,
        )

    async def persist_vector(
        self,
        memories: List[ExtractedMemory],
        entities: List[ExtractedEntity],
        scope: Scope,
        embeddings: Optional[List] = None,
    ) -> VectorPersistResult:
        """
        Persist memories to vector store with embeddings.

        Includes:
        - Core memory content
        - ERE triples as context
        - Metadata for filtering
        """
        # Ensure namespace exists
        if self._vector is None:
            logger.debug("Vector store unavailable — skipping memory vector persistence")
            return VectorPersistResult(vector_ids=[])
        await self._vector.ensure_namespace(scope, dimension=self._embedding.dimension)

        if not memories:
            return VectorPersistResult(vector_ids=[])

        # Build entity map for ERE context
        entity_map = {e.entity_id: e for e in entities}

        # Build UUID → deterministic ChromaDB ID mapping.
        # Memory objects use extraction-time UUIDs in entities_mentioned,
        # but ChromaDB entity collection uses deterministic hashes.
        # We remap ONLY in the ChromaDB metadata (not on the memory objects)
        # because Neo4j persist_graph runs in parallel and needs the original UUIDs.
        from synap.cloud.agents.ingestion.storage.vector_wrapper import entity_id_scope
        client_id = str(scope.client_id)
        # Hashed with the same scope id as the entity vectors themselves.
        customer_id = entity_id_scope(scope) if scope.customer_id else str(scope.customer_id)
        uuid_to_det: Dict[str, str] = {}
        for e in entities:
            det_id = deterministic_entity_id(entity_id_name(e), e.entity_type, client_id, customer_id)
            uuid_to_det[e.entity_id] = det_id

        # Build per-memory remapped entity lists for ChromaDB metadata only
        remapped_entities_mentioned: List[List[str]] = []
        for memory in memories:
            remapped_entities_mentioned.append([
                uuid_to_det.get(eid, eid) for eid in memory.entities_mentioned
            ])

        # Build content with ERE context for each memory
        contents = []
        ere_contexts = []

        for memory in memories:
            contents.append(memory.content)

            # Build ERE context
            ere_parts = []
            for entity_id in memory.entities_mentioned:
                entity = entity_map.get(entity_id)
                if entity:
                    # The writer's and the customer's nodes are named by an id;
                    # stored text carries what a reader is shown.
                    from synap.cloud.agents.ingestion.services.writer_entity import shown_name
                    ere_parts.append(
                        f"{entity.entity_type}:"
                        f"{shown_name(entity.properties, entity.canonical_name or entity.name)}"
                    )

            ere_contexts.append(", ".join(ere_parts) if ere_parts else None)

        # Use pre-computed embeddings or generate them
        if embeddings is None:
            embeddings = await self._embedding.embed_batch(contents)

        # Batch upsert to vector store (with remapped entity IDs for ChromaDB linkage)
        vector_ids = await self._vector.upsert_memories_batch(
            memories=memories,
            embeddings=embeddings,
            scope=scope,
            ere_contexts=ere_contexts,
            remapped_entities=remapped_entities_mentioned,
        )

        return VectorPersistResult(vector_ids=vector_ids)

    async def persist_entity_vectors(
        self,
        entities: List[ExtractedEntity],
        scope: Scope,
        embeddings: Optional[List] = None,
    ) -> EntityVectorPersistResult:
        """
        Persist entities to entity vector collection with embeddings.

        Used by V2 traversal engine for vectorized anchor entity search.
        Each entity is embedded as "{name} ({entity_type})".
        """
        if self._vector is None:
            logger.debug("Vector store unavailable — skipping entity vector persistence")
            return EntityVectorPersistResult(entity_vector_ids=[])
        await self._vector.ensure_entity_namespace(scope, dimension=self._embedding.dimension)

        if not entities:
            return EntityVectorPersistResult(entity_vector_ids=[])

        # Use pre-computed embeddings or generate them
        if embeddings is None:
            contents = [
                f"{e.canonical_name or e.name} ({e.entity_type})"
                for e in entities
            ]
            embeddings = await self._embedding.embed_batch(contents)

        # Batch upsert to entity collection
        entity_ids = await self._vector.upsert_entities_batch(
            entities=entities,
            embeddings=embeddings,
            scope=scope,
        )

        return EntityVectorPersistResult(entity_vector_ids=entity_ids)

    async def persist_triple_vectors(
        self,
        relationships: List[ExtractedRelationship],
        entities: List[ExtractedEntity],
        scope: Scope,
        embeddings: Optional[List] = None,
    ) -> TripleVectorPersistResult:
        """
        Persist relationship triples to triples vector collection with embeddings.

        Used by V2 traversal engine for building edge vectors in adjacency matrix.
        Each triple is embedded as "EntityA relationship_type EntityB".
        """
        if self._vector is None:
            logger.debug("Vector store unavailable — skipping triple vector persistence")
            return TripleVectorPersistResult(triple_vector_ids=[])
        await self._vector.ensure_triples_namespace(scope, dimension=self._embedding.dimension)

        if not relationships:
            return TripleVectorPersistResult(triple_vector_ids=[])

        # Build entity lookup: extraction_id → (canonical_name, entity_type)
        entity_info = {
            e.entity_id: (e.canonical_name or e.name, e.entity_type)
            for e in entities
        }
        # The name an entity's vector id is made from; differs from its
        # display name only after the entity was renamed (resolution/naming).
        id_names = {e.entity_id: entity_id_name(e) for e in entities}
        client_id_str = str(scope.client_id)
        from synap.cloud.agents.ingestion.storage.vector_wrapper import entity_id_scope
        customer_id_str = entity_id_scope(scope) if scope.customer_id else str(scope.customer_id)

        # Build triple dicts with deterministic entity IDs (matching ChromaDB entity dedup)
        triples = []
        contents = []
        for rel in relationships:
            src_info = entity_info.get(rel.source_entity_id)
            tgt_info = entity_info.get(rel.target_entity_id)
            source_name = src_info[0] if src_info else rel.source_entity_id
            target_name = tgt_info[0] if tgt_info else rel.target_entity_id
            source_type = src_info[1] if src_info else "unknown"
            target_type = tgt_info[1] if tgt_info else "unknown"

            src_det_id = deterministic_entity_id(
                id_names.get(rel.source_entity_id, source_name), source_type, client_id_str, customer_id_str)
            tgt_det_id = deterministic_entity_id(
                id_names.get(rel.target_entity_id, target_name), target_type, client_id_str, customer_id_str)
            content = f"{source_name} {rel.relationship_type} {target_name}"

            triples.append({
                "id": f"{src_det_id}_{rel.relationship_type}_{tgt_det_id}",
                "content": content,
                "source_entity_id": src_det_id,
                "target_entity_id": tgt_det_id,
                "relationship_type": rel.relationship_type,
                "source_entity_name": source_name,
                "target_entity_name": target_name,
            })
            contents.append(content)

        # Use pre-computed embeddings or generate them
        if embeddings is None:
            embeddings = await self._embedding.embed_batch(contents)

        # Batch upsert to triples collection
        triple_ids = await self._vector.upsert_triples_batch(
            triples=triples,
            embeddings=embeddings,
            scope=scope,
        )

        return TripleVectorPersistResult(triple_vector_ids=triple_ids)

    def split_into_windows(
        self,
        content: str,
        max_tokens: int = 500,
        overlap: int = 75,
    ) -> List[str]:
        """Split content into overlapping token windows for raw chunk embedding.

        Splits at sentence boundaries when possible to avoid breaking mid-sentence.
        Falls back to whitespace split if no sentence boundary found near the target.
        """
        from synap.cloud.shared.tokens import get_token_counter
        counter = get_token_counter()

        total_tokens = counter.count(content)
        if total_tokens <= max_tokens:
            return [content]

        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if not sentences:
            return [content]

        windows = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = counter.count(sentence)

            if current_tokens + sent_tokens > max_tokens and current_sentences:
                # Emit current window
                windows.append(" ".join(current_sentences))

                # Compute overlap: keep last sentences up to `overlap` tokens
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = counter.count(s)
                    if overlap_tokens + s_tokens > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Emit remaining
        if current_sentences:
            windows.append(" ".join(current_sentences))

        return windows

    async def persist_chunk_vectors(
        self,
        content: str,
        scope: Scope,
        source_document_id: str,
        embeddings: Optional[List] = None,
        chunk_meta: Optional[Dict] = None,
    ) -> ChunkVectorPersistResult:
        """Re-chunk content into ~500-token windows and persist to chunks collection.

        Args:
            content: Raw document text to chunk and embed
            scope: Tenant scope
            source_document_id: Original document ID for traceability
            embeddings: Pre-computed embeddings for the windows (from stage4 batch)
            chunk_meta: Optional metadata from Stage 0 preprocessing (_chunk dict)
        """
        if self._vector is None:
            logger.debug("Vector store unavailable — skipping chunk vector persistence")
            return ChunkVectorPersistResult(chunk_vector_ids=[], chunk_count=0)

        await self._vector.ensure_chunks_namespace(scope, dimension=self._embedding.dimension)

        if not content or not content.strip():
            return ChunkVectorPersistResult(chunk_vector_ids=[], chunk_count=0)

        # Split into overlapping windows
        windows = self.split_into_windows(content)
        logger.info(f"Chunk vectors: split content into {len(windows)} windows source_doc={source_document_id}")

        # Build per-chunk metadata
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        chunk_metadata_list = []
        for i in range(len(windows)):
            meta = {
                "source_document_id": source_document_id,
                "chunk_index": i,
                "total_chunks": len(windows),
                "created_at": now_iso,
            }
            if chunk_meta:
                meta.update(chunk_meta)
            chunk_metadata_list.append(meta)

        # Use pre-computed embeddings or generate them
        if embeddings is None:
            embeddings = await self._embedding.embed_batch(windows)

        # Persist to chunks collection
        chunk_ids = await self._vector.upsert_chunks_batch(
            chunks=windows,
            chunk_metadata=chunk_metadata_list,
            embeddings=embeddings,
            scope=scope,
        )

        logger.info(f"Chunk vectors: persisted {len(chunk_ids)} chunks source_doc={source_document_id}")

        return ChunkVectorPersistResult(chunk_vector_ids=chunk_ids, chunk_count=len(windows))

    async def persist_question_vectors(
        self,
        memories: List[ExtractedMemory],
        scope: Scope,
        embeddings: Optional[List] = None,
    ) -> QuestionVectorPersistResult:
        """Embed and persist hypothetical questions from extracted memories.

        Collects all hypothetical_questions from memories, embeds them,
        and stores in the questions collection with memory_id linkage.
        """
        if self._vector is None:
            logger.debug("Vector store unavailable — skipping question vector persistence")
            return QuestionVectorPersistResult(question_vector_ids=[], question_count=0)

        # Collect all questions across all memories
        question_dicts = []
        question_texts = []

        for memory in memories:
            if not memory.hypothetical_questions:
                continue
            for qi, q_text in enumerate(memory.hypothetical_questions):
                if not q_text or not q_text.strip():
                    continue
                question_dicts.append({
                    "content": q_text.strip(),
                    "memory_id": str(memory.memory_id),
                    "memory_content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "source_document_id": memory.source_document_id,
                    "category": memory.category or "",
                    "subcategory": memory.subcategory or "",
                    "confidence": memory.confidence,
                    "question_index": qi,
                    # Scope metadata — needed for retrieval-time filtering
                    "customer_id": str(scope.customer_id),
                    "user_id": str(scope.user_id) if scope.user_id else "",
                    # Temporal metadata — needed for temporal reasoning in retrieval
                    "extracted_at": memory.extracted_at.isoformat() if memory.extracted_at else None,
                    "document_created_at": memory.document_created_at.isoformat() if memory.document_created_at else None,
                    "temporal_category": memory.temporal_category,
                    "temporal_confidence": memory.temporal_confidence,
                    "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
                    "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
                    "is_episode": memory.is_episode,
                    "event_start": memory.event_start.isoformat() if getattr(memory, "event_start", None) else None,
                    "event_end": memory.event_end.isoformat() if getattr(memory, "event_end", None) else None,
                    "event_precision": getattr(memory, "event_precision", None),
                })
                question_texts.append(q_text.strip())

        if not question_texts:
            return QuestionVectorPersistResult(question_vector_ids=[], question_count=0)

        await self._vector.ensure_questions_namespace(scope, dimension=self._embedding.dimension)

        # Embed all questions (or use pre-computed)
        if embeddings is None:
            embeddings = await self._embedding.embed_batch(question_texts)

        # Persist to questions collection
        question_ids = await self._vector.upsert_questions_batch(
            questions=question_dicts,
            embeddings=embeddings,
            scope=scope,
        )

        logger.info(f"Question vectors: persisted {len(question_ids)} questions from {sum(1 for m in memories if m.hypothetical_questions)} memories")

        return QuestionVectorPersistResult(
            question_vector_ids=question_ids,
            question_count=len(question_ids),
        )

    async def execute_supersessions(
        self,
        actions: List,  # List[SupersessionAction] — import avoided for circular deps
        scope: Scope,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> int:
        """Execute supersession actions: deprecate old memories in RDBMS + delete from ChromaDB.

        PostgreSQL is authoritative (write-ahead). ChromaDB delete is best-effort.

        Two RDBMS-write paths exist for backward compat during the KU v2
        rollout, gated by ENABLE_KU_DEPRECATE_WITH_LINK (default: false until
        lifecycle-aware retrieval lands):
          - Path A′ (flag ON):  set deprecated_at + insert memory_links row.
                                Legacy `superseded_by` columns also written
                                so historical queries don't break.
          - Legacy   (flag OFF): only `soft_delete_memory` (sets superseded_by)
                                — the earlier behavior, no memory_links row.

        Both paths still hard-delete from ChromaDB so retrieval stays clean
        without lifecycle-aware filtering. Lifecycle-aware retrieval will switch to
        keep-in-ChromaDB + filter-on-query.

        When http_correlation_id is provided, a SUPERSEDE row is written to
        memory_actions per successful supersession for the Requests-page
        changelog, carrying the LLM-generated `reason` so clients can see why.
        Returns count of successfully superseded memories.
        """
        if not actions:
            return 0

        deprecate_with_link = os.environ.get(
            "ENABLE_KU_DEPRECATE_WITH_LINK", "false"
        ).lower() in ("true", "1", "yes")

        # Lazy import — coordinator<->knowledge_update would be circular at
        # module load (same reason execute_deletions avoids the Action import).
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            _retired_value_guard_enabled,
            _value_sibling_sweep_enabled,
        )
        retired_value_guard = _retired_value_guard_enabled()  # default OFF
        sibling_sweep = _value_sibling_sweep_enabled()  # default OFF

        superseded_count = 0

        for action in actions:
            try:
                # 0. Retired-value guard (SYNAP_KU_RETIRED_VALUE_GUARD): refuse
                #    to let a memory that re-asserts an already-RETIRED value
                #    retire the row carrying the current one. The pair's
                #    outcome becomes NONE — no supersession, no lineage, no
                #    changelog row — see _retired_value_blocks_supersession.
                if retired_value_guard and await self._retired_value_blocks_supersession(
                    action, scope,
                ):
                    continue

                # 1. RDBMS deprecation, its lineage and its trail row, in ONE
                #    transaction. The trail row carries old_memory_content,
                #    which is the only copy of the before-state a reversal can
                #    restore this memory from — so it is written with the
                #    retirement, not after it, and never skipped for want of a
                #    correlation id. The legacy superseded_by column is still
                #    stamped for anything that reads it.
                updated = await self._rdbms.retire_memory_with_audit(
                    memory_id=action.old_memory_id,
                    action="SUPERSEDE",
                    scope=scope,
                    superseded_by=action.new_memory_id,
                    replaced_by_memory_id=action.new_memory_id,
                    previous_content=action.old_memory_content,
                    content=action.new_memory_content,
                    memory_type=getattr(action, "memory_type", None),
                    reason=getattr(action, "reason", None),
                    correlation_id=http_correlation_id,
                    ingestion_id=ingestion_id,
                    deprecate_with_link=deprecate_with_link,
                )
                if not updated:
                    logger.warning(
                        "Knowledge-update: RDBMS soft-delete no-op (not found or already superseded) "
                        "old_id=%s new_id=%s",
                        action.old_memory_id, action.new_memory_id,
                    )
                    continue

                # 2. Hard-delete from ChromaDB (best-effort). Path A′ keeps
                # this — retrieval stays clean until lifecycle-aware
                # filtering is enabled on the metadata side.
                if self._vector is not None:
                    try:
                        await self._vector.delete_memory_vectors(
                            memory_ids=[action.old_memory_id],
                            scope=scope,
                        )
                    except Exception as e:
                        logger.warning(
                            "Knowledge-update: ChromaDB delete failed (non-fatal) old_id=%s error=%s",
                            action.old_memory_id, e,
                        )

                superseded_count += 1
                logger.info(
                    "Knowledge-update: superseded old_id=%s new_id=%s reason=%s",
                    action.old_memory_id, action.new_memory_id,
                    getattr(action, "reason", ""),
                )

                # 3. Sibling value sweep (SYNAP_KU_VALUE_SIBLING_SWEEP):
                #    paraphrase rows carrying the value we just retired are
                #    still CURRENT and would be served alongside the new one
                #    ("two emails on file"). Retire them the same way.
                if sibling_sweep:
                    await self._sweep_value_siblings(
                        action=action,
                        scope=scope,
                        deprecate_with_link=deprecate_with_link,
                        http_correlation_id=http_correlation_id,
                        ingestion_id=ingestion_id,
                    )

            except Exception as e:
                logger.error(
                    "Knowledge-update: supersession failed old_id=%s new_id=%s error=%s",
                    action.old_memory_id, action.new_memory_id, e,
                    exc_info=True,
                )

        return superseded_count

    async def _retired_value_blocks_supersession(self, action, scope) -> bool:
        """Retired-value guard (SYNAP_KU_RETIRED_VALUE_GUARD).

        A later extraction that re-asserted the user's OLD notification email
        SUPERSEDED the row holding the NEW one ("updates the domain of the user's notification
        email address") — the email silently rolled back. A value the user
        already retired must not be able to retire its own replacement.

        Fires only on high-entropy values (a full email, a URL, a >=7-digit
        phone) — never on class keywords like the word "email", which repeat
        across unrelated facts. Stands down when the text explicitly signals a
        flip ("switched back to old@x", "actually, use ... instead"), which is
        a legitimate revert the classifier is right about.

        Fail-open: any lookup problem lets the supersession proceed (today's
        behaviour), because a missed block is recoverable and a spurious block
        would strand the user's real update.

        Blocking is where it stops: the stale new memory is left alone rather
        than dropped as a NONE. Dropping it here would soft-delete a row whose
        vector is written LATER in stage 4 (the ordering the echo-resurrection
        guard has to work around), leaving a deprecated row that retrieval
        still finds — strictly worse than an extra current memory.
        """
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            _extract_high_entropy_values,
            _has_value_change_marker,
        )
        if not getattr(scope, "user_id", None):
            return False
        new_content = getattr(action, "new_memory_content", "") or ""
        values = _extract_high_entropy_values(new_content)
        if not values:
            return False
        change_text = " ".join(
            t for t in (
                new_content,
                getattr(action, "new_memory_source_span", None),
            ) if t
        )
        if _has_value_change_marker(change_text):
            return False
        lookup = getattr(self._rdbms, "retired_memories_with_value", None)
        if lookup is None:
            return False  # storage impl predates the guard
        for value in sorted(values)[:3]:  # bound per-action lookups
            try:
                rows = await lookup(
                    scope=scope,
                    value_token=value,
                    exclude_memory_ids=[
                        action.new_memory_id, action.old_memory_id,
                    ],
                    limit=1,
                )
            except Exception as e:
                logger.warning(
                    "KU retired-value guard: lookup failed (non-fatal) "
                    "value=%s error=%s", value, e,
                )
                continue
            if rows:
                logger.info(
                    'KU retired-value guard: blocked SUPERSEDE of %s — '
                    'value "%s" was previously retired',
                    action.old_memory_id, value,
                )
                return True
        return False

    async def _sweep_value_siblings(
        self,
        action,
        scope: Scope,
        deprecate_with_link: bool,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
        limit: int = 50,
    ) -> int:
        """Sibling value sweep (SYNAP_KU_VALUE_SIBLING_SWEEP).

        A supersession retires exactly one row, but the retired value is
        usually spread over paraphrases ("the user's email is old@x", "the
        user prefers to be contacted at old@x", "receipts go to old@x"). Those
        siblings stay CURRENT and retrieval then serves the old value next to
        the new one. This retires each of them the same way the supersession
        retired its target (superseded_by = the new row, optional
        deprecated_at + lineage link, SUPERSEDE changelog row, Chroma vector
        hard-delete).

        Rows that ALSO carry the new value are left alone — they already
        describe the update ("moved from old@x to new@x"). Best-effort per row.
        """
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            _extract_high_entropy_values,
        )
        if not getattr(scope, "user_id", None):
            return 0
        old_values = _extract_high_entropy_values(
            getattr(action, "old_memory_content", "") or ""
        )
        if not old_values:
            return 0
        new_values = _extract_high_entropy_values(
            getattr(action, "new_memory_content", "") or ""
        )
        lookup = getattr(self._rdbms, "current_memories_matching", None)
        if lookup is None:
            return 0
        retired_total = 0
        seen: "set[str]" = set()
        for value in sorted(old_values)[:2]:  # bound per-action lookups
            if value in new_values:
                continue  # this value didn't change; nothing stale about it
            try:
                rows = await lookup(
                    scope=scope,
                    content_tokens=[value],
                    exclude_memory_ids=[
                        action.new_memory_id, action.old_memory_id,
                    ],
                    limit=limit,
                )
            except Exception as e:
                logger.error(
                    "KU sibling value sweep: lookup failed value=%s error=%s",
                    value, e, exc_info=True,
                )
                continue
            retired_for_value = 0
            for row in rows or []:
                mid = str(row["id"])
                if mid in seen:
                    continue
                content = row.get("content") or ""
                lowered = content.lower()
                if any(v in lowered for v in new_values):
                    continue  # already carries the new value
                try:
                    # Retirement + lineage + trail row in one transaction.
                    # A sibling retired here is as undoable as one retired by
                    # the pair verdict above: same before-state, same rules.
                    updated = await self._rdbms.retire_memory_with_audit(
                        memory_id=mid,
                        action="SUPERSEDE",
                        scope=scope,
                        superseded_by=action.new_memory_id,
                        replaced_by_memory_id=action.new_memory_id,
                        previous_content=content,
                        content=getattr(action, "new_memory_content", None),
                        memory_type=row.get("memory_type"),
                        reason=(
                            "value-sibling-sweep: carried the "
                            f'superseded value "{value}"'
                        ),
                        correlation_id=http_correlation_id,
                        ingestion_id=ingestion_id,
                        deprecate_with_link=deprecate_with_link,
                    )
                    if not updated:
                        continue
                    if self._vector is not None:
                        try:
                            await self._vector.delete_memory_vectors(
                                memory_ids=[mid], scope=scope,
                            )
                        except Exception as e:
                            logger.warning(
                                "KU sibling value sweep: vector delete failed "
                                "(non-fatal) id=%s error=%s", mid, e,
                            )
                    seen.add(mid)
                    retired_for_value += 1
                except Exception as e:
                    logger.error(
                        "KU sibling value sweep: failed id=%s error=%s",
                        mid, e, exc_info=True,
                    )
            if retired_for_value:
                retired_total += retired_for_value
                logger.info(
                    'KU sibling value sweep: retired %d sibling rows carrying '
                    'superseded value "%s"', retired_for_value, value,
                )
        return retired_total

    async def execute_deletions(
        self,
        actions: List,  # List[DeletionAction] — import avoided for circular deps
        scope: Scope,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> int:
        """Execute explicit natural-language deletions (action='DELETE', Gap 6).

        Flag-gated by ENABLE_KU_DELETE upstream — this method is only ever
        called with a non-empty list when the flag is on. A deletion is a SOFT
        delete, NOT a hard destroy: the target is deprecated in RDBMS
        (``set_deprecated_at``) and removed from the vector index
        (``delete_memory_vectors``) so it stops being retrieved immediately,
        while its RDBMS row + a DELETE changelog row remain for audit. This
        keeps the system's "deprecate, don't destroy" guarantee even for
        user-requested forgets.

        There is no replacement memory (unlike SUPERSEDE), so no memory_link is
        written. Returns the count of successfully deleted memories.
        """
        if not actions:
            return 0

        deleted_count = 0
        for action in actions:
            try:
                # 1. RDBMS: mark the target deprecated (historical) and write
                #    the DELETE trail row in the same transaction. No
                #    superseded_by — nothing replaces a retracted memory, so
                #    this is a deprecation with no successor. A retraction is
                #    the retirement a user is most likely to want back, so its
                #    before-state is the least optional of all of them.
                #    `require_retire=False`: an already-deprecated row still
                #    gets its trail row, matching the existing behaviour of
                #    falling through to the vector delete below.
                updated = await self._rdbms.retire_memory_with_audit(
                    memory_id=action.old_memory_id,
                    action="DELETE",
                    scope=scope,
                    superseded_by=None,
                    replaced_by_memory_id=None,
                    previous_content=action.old_memory_content,
                    content=None,
                    memory_type=getattr(action, "memory_type", None),
                    reason=getattr(action, "reason", None),
                    correlation_id=http_correlation_id,
                    ingestion_id=ingestion_id,
                    require_retire=False,
                )
                if not updated:
                    logger.warning(
                        "Knowledge-update DELETE: RDBMS deprecate no-op "
                        "(not found or already deprecated) old_id=%s",
                        action.old_memory_id,
                    )
                    # Still attempt the vector delete below — the row may be
                    # already-deprecated but the vector could linger.

                # 2. Remove from ChromaDB so retrieval stops surfacing it now
                #    (retrieval is not yet lifecycle-aware, so the vector delete
                #    is what actually makes the memory "forgotten").
                if self._vector is not None:
                    try:
                        await self._vector.delete_memory_vectors(
                            memory_ids=[action.old_memory_id],
                            scope=scope,
                        )
                    except Exception as e:
                        logger.warning(
                            "Knowledge-update DELETE: ChromaDB delete failed "
                            "(non-fatal) old_id=%s error=%s",
                            action.old_memory_id, e,
                        )

                deleted_count += 1
                logger.info(
                    "Knowledge-update DELETE: forgot old_id=%s reason=%s",
                    action.old_memory_id, getattr(action, "reason", ""),
                )
            except Exception as e:
                logger.error(
                    "Knowledge-update DELETE: failed old_id=%s error=%s",
                    action.old_memory_id, e, exc_info=True,
                )

        return deleted_count

    async def suppress_retired_value_echoes(
        self,
        memories: List,  # List[ExtractedMemory]
        scope: Scope,
        already_skipped_ids: Optional[set] = None,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> "set[str]":
        """Echo resurrection guard (ENABLE_ECHO_GUARD, WS2 4b of the
        echo-poisoning plan): an assistant-sourced memory that re-asserts a
        value the user already RETIRED — the value token appears in
        superseded/deprecated rows and in no other current row — is stored
        pre-deprecated: deprecate in RDBMS + drop the vector + one NONE
        changelog row. The row itself remains for audit ("deprecate, don't
        destroy").

        This is what stops echo ACCUMULATION: the pair-level echo guard in KU
        only fires on SUPERSEDE verdicts, but most echoes are classified
        UNRELATED (different attribute phrasing — "original email", "billed
        to") and would otherwise pile up as current facts. Conservative and
        best-effort: any lookup failure leaves the memory live.
        """
        if not memories:
            return 0
        # Lazy import — coordinator<->knowledge_update would be circular at
        # module load (same reason execute_deletions avoids the Action import).
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            _echo_guard_enabled,
            _extract_value_tokens,
        )
        if not _echo_guard_enabled() or not getattr(scope, "user_id", None):
            return set()
        already_skipped_ids = already_skipped_ids or set()
        batch_ids = [str(m.memory_id) for m in memories]
        # Returns the suppressed memory ids so stage 4 can exclude them from
        # the DOWNSTREAM vector persistence — the memory-vector upsert runs
        # AFTER this sweep, so the delete_memory_vectors call below is only a
        # backstop; the real guarantee is never writing the vector at all
        # (otherwise a swept echo's vector would be
        # re-created by the later persist_vector step and kept being served).
        suppressed: "set[str]" = set()
        for memory in memories:
            mid = str(memory.memory_id)
            if mid in already_skipped_ids:
                continue  # KU already dropped it
            if getattr(memory, "assertion_source", None) != "assistant":
                continue
            tokens = _extract_value_tokens(memory.content)
            if not tokens:
                continue
            try:
                echo = False
                for token in sorted(tokens)[:3]:  # bound per-memory lookups
                    if await self._rdbms.value_retired_for_user(
                        scope=scope,
                        value_token=token,
                        exclude_memory_ids=batch_ids,
                    ):
                        echo = True
                        break
                if not echo:
                    continue

                # previous_content stays None here, deliberately. This memory
                # was born deprecated: it was stored moments ago and never
                # served, so there is no earlier state to restore it to, and
                # un-deprecating it would resurrect exactly the value the user
                # retired. A NULL before-state makes the reversal refuse it,
                # which is the behaviour we want — this is the one retirement
                # that must NOT be undoable.
                await self._rdbms.retire_memory_with_audit(
                    memory_id=mid,
                    action="NONE",
                    scope=scope,
                    superseded_by=None,
                    replaced_by_memory_id=None,
                    previous_content=None,
                    content=memory.content,
                    memory_type=(
                        memory.memory_type.value
                        if memory.memory_type is not None else None
                    ),
                    reason=(
                        "echo-guard: assistant re-asserted a retired "
                        "value; stored pre-deprecated"
                    ),
                    correlation_id=http_correlation_id,
                    ingestion_id=ingestion_id,
                    require_retire=False,
                )
                if self._vector is not None:
                    try:
                        await self._vector.delete_memory_vectors(
                            memory_ids=[mid], scope=scope,
                        )
                    except Exception as e:
                        logger.warning(
                            "Echo resurrection guard: vector delete failed "
                            "(non-fatal) id=%s error=%s", mid, e,
                        )
                suppressed.add(mid)
                logger.info(
                    "Echo resurrection guard: pre-deprecated assistant echo "
                    "id=%s content=%s", mid, memory.content[:80],
                )
            except Exception as e:
                logger.error(
                    "Echo resurrection guard: failed id=%s error=%s",
                    mid, e, exc_info=True,
                )
        return suppressed

    async def execute_retraction_sweeps(
        self,
        sweeps: List,  # List[RetractionSweep]
        scope: Scope,
        exclude_memory_ids: Optional[set] = None,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
        max_rows_per_sweep: int = 25,
    ) -> "set[str]":
        """Subject-directed retraction sweep (ENABLE_KU_DELETE_SUBJECT_GUARD):
        for each confirmed retraction, retire EVERY current memory of this
        user whose content matches the retraction's subject — not just the one
        vector-paired row the classifier happened to see (for example, the paired row was
        "The user lives in Texas"
        while the address fact itself survived as current).

        Same soft-delete shape as execute_deletions: deprecate + drop the
        vector + one DELETE changelog row per swept memory. Never sweeps rows
        that are themselves retraction requests (the intent record stays for
        audit) nor rows in ``exclude_memory_ids`` (the just-persisted batch —
        pair verdicts own those). Best-effort per row; failure leaves the row
        live.
        """
        if not sweeps:
            return set()
        # Lazy import — coordinator<->knowledge_update would be circular at
        # module load (same reason suppress_retired_value_echoes does this).
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            _delete_subject_guard_enabled,
            _is_retraction_request,
            _retraction_sweep_v2_enabled,
        )
        if not _delete_subject_guard_enabled() or not getattr(scope, "user_id", None):
            return set()
        sweep_v2 = _retraction_sweep_v2_enabled()  # default OFF
        exclude = {str(x) for x in (exclude_memory_ids or set())}
        swept: "set[str]" = set()
        for sweep in sweeps:
            # A concrete value token is the most precise match; fall back to
            # requiring ALL subject tokens. detect_updates only emits sweeps
            # with a value token or >=2 subject tokens.
            tokens = list(sweep.value_tokens) or list(sweep.subject_tokens)
            if not tokens:
                continue
            try:
                rows = await self._rdbms.current_memories_matching(
                    scope=scope,
                    content_tokens=tokens,
                    exclude_memory_ids=list(exclude | swept | {sweep.new_memory_id}),
                    limit=max_rows_per_sweep,
                )
            except Exception as e:
                logger.error(
                    "Retraction sweep: lookup failed tokens=%s error=%s",
                    tokens, e, exc_info=True,
                )
                continue
            if sweep_v2:
                # v2: the ALL-token ILIKE above misses any row that words the
                # fact differently ("payment card kept on file" vs "The user's
                # card ends in 4242"). Add rows sharing the retraction's
                # attribute class — value-token/subject-token matches above
                # keep their priority and their ordering.
                rows = list(rows) + await self._retraction_class_candidates(
                    sweep=sweep,
                    scope=scope,
                    exclude_memory_ids=list(
                        exclude | swept | {sweep.new_memory_id}
                    ),
                    already_matched={str(r["id"]) for r in rows},
                    max_rows=max_rows_per_sweep,
                )
            for row in rows:
                mid = str(row["id"])
                if mid in swept:
                    continue
                if _is_retraction_request(row.get("content") or ""):
                    continue  # keep the intent record itself
                try:
                    # This sweep retires memories that were live and serving,
                    # on nothing more than a text match against a retraction.
                    # It is the most speculative retirement in the pipeline, so
                    # its before-state is the least optional: retirement and
                    # trail row go together or not at all.
                    await self._rdbms.retire_memory_with_audit(
                        memory_id=mid,
                        action="DELETE",
                        scope=scope,
                        superseded_by=None,
                        replaced_by_memory_id=None,
                        previous_content=row.get("content"),
                        content=None,
                        memory_type=row.get("memory_type"),
                        reason=(
                            "delete-sweep: user retraction matched "
                            f"this memory ({sweep.reason})"
                        ),
                        correlation_id=http_correlation_id,
                        ingestion_id=ingestion_id,
                        require_retire=False,
                    )
                    if self._vector is not None:
                        try:
                            await self._vector.delete_memory_vectors(
                                memory_ids=[mid], scope=scope,
                            )
                        except Exception as e:
                            logger.warning(
                                "Retraction sweep: vector delete failed "
                                "(non-fatal) id=%s error=%s", mid, e,
                            )
                    swept.add(mid)
                    logger.info(
                        "Retraction sweep: retired current memory id=%s "
                        "tokens=%s content=%s",
                        mid, tokens, (row.get("content") or "")[:80],
                    )
                except Exception as e:
                    logger.error(
                        "Retraction sweep: failed id=%s error=%s",
                        mid, e, exc_info=True,
                    )
        return swept

    async def _retraction_class_candidates(
        self,
        sweep,
        scope: Scope,
        exclude_memory_ids: List[str],
        already_matched: "set[str]",
        max_rows: int,
    ) -> List[Dict[str, Any]]:
        """Retraction sweep v2 (SYNAP_KU_RETRACTION_SWEEP_V2): current rows
        that share the retraction's ATTRIBUTE CLASS.

        Example: "don't keep my card on file after this transaction" is
        stored as "The user prefers not to have their payment card kept on
        file after the current transaction", so the v1 all-token ILIKE looks
        for %payment% AND %card% AND %after% AND %transaction% against a row
        reading "The user's card ends in 4242" — zero matches, the card is
        never retired. Class matching reaches it: both sides are `payment`.

        Two-step so the SQL stays cheap: a bounded OR-ILIKE over a handful of
        per-class keywords, then the real _ATTRIBUTE_CLASS_PATTERNS regexes in
        Python (which also apply the email/address precedence rule, so an
        address retraction never takes an "email address is x@y" row).
        Best-effort: any failure returns no extra candidates.
        """
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            _attribute_classes,
            _class_sql_keywords,
            _retraction_class_match,
        )
        classes = _attribute_classes(
            getattr(sweep, "new_memory_content", "") or ""
        )
        if not classes:
            return []
        keywords = _class_sql_keywords(classes)
        if not keywords:
            return []
        fetch = getattr(self._rdbms, "current_memories_matching_any", None)
        if fetch is None:
            return []  # storage impl predates the class prefilter
        try:
            rows = await fetch(
                scope=scope,
                content_keywords=keywords,
                exclude_memory_ids=exclude_memory_ids,
                limit=RETRACTION_CLASS_FETCH_LIMIT,
            )
        except Exception as e:
            logger.error(
                "Retraction sweep v2: class lookup failed classes=%s error=%s",
                sorted(classes), e, exc_info=True,
            )
            return []
        extra: List[Dict[str, Any]] = []
        for row in rows or []:
            if str(row["id"]) in already_matched:
                continue
            if not _retraction_class_match(classes, row.get("content") or ""):
                continue
            extra.append(row)
            if len(extra) >= max_rows:
                break
        if extra:
            logger.info(
                "Retraction sweep v2: %d additional class-matched candidates "
                "classes=%s", len(extra), sorted(classes),
            )
        return extra

    async def execute_skips(
        self,
        actions: List,  # List[SkipAction]
        scope: Scope,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> int:
        """Drop new memories classified NONE: soft-delete from RDBMS (pointing
        at the existing survivor as `superseded_by`), best-effort ChromaDB
        delete, and one NONE row in memory_actions per drop.

        These memories were inserted by the upstream RDBMS persist step before
        KU ran. NONE means we now know the new memory is a true duplicate of
        an existing one — so we soft-delete it and tag the existing memory as
        the survivor in the deprecation link.
        """
        if not actions:
            return 0

        deprecate_with_link = os.environ.get(
            "ENABLE_KU_DEPRECATE_WITH_LINK", "false"
        ).lower() in ("true", "1", "yes")

        skipped_count = 0
        for action in actions:
            try:
                # Retire the NEW memory (the existing one is the survivor),
                # stamp the lineage, and write the NONE trail row, all in one
                # transaction. A dropped restatement is undone the same way a
                # supersession is, so it needs the same before-state. The
                # DEPRECATES kind matches SUPERSEDE: semantically the
                # existing memory absorbed the new one.
                updated = await self._rdbms.retire_memory_with_audit(
                    memory_id=action.new_memory_id,
                    # "NONE" (a duplicate drop) unless the action says
                    # otherwise. The out-of-order direction guard in knowledge
                    # update reuses this path because the database outcome is
                    # identical -- retire the new row, point it at the existing
                    # survivor -- but labels it SUPERSEDE, because that is what
                    # it is and the changelog is client-visible.
                    action=getattr(action, "action_label", "NONE") or "NONE",
                    scope=scope,
                    superseded_by=action.existing_memory_id,
                    replaced_by_memory_id=action.existing_memory_id,
                    previous_content=action.new_memory_content,
                    content=action.existing_memory_content,
                    memory_type=getattr(action, "memory_type", None),
                    reason=getattr(action, "reason", None),
                    correlation_id=http_correlation_id,
                    ingestion_id=ingestion_id,
                    deprecate_with_link=deprecate_with_link,
                )
                if not updated:
                    logger.warning(
                        "Knowledge-update NONE: RDBMS soft-delete no-op new_id=%s existing_id=%s",
                        action.new_memory_id, action.existing_memory_id,
                    )
                    continue

                # Best-effort: drop the new memory from ChromaDB if it landed there.
                if self._vector is not None:
                    try:
                        await self._vector.delete_memory_vectors(
                            memory_ids=[action.new_memory_id],
                            scope=scope,
                        )
                    except Exception as e:
                        logger.warning(
                            "Knowledge-update NONE: ChromaDB delete failed new_id=%s error=%s",
                            action.new_memory_id, e,
                        )

                skipped_count += 1
                logger.info(
                    "Knowledge-update NONE: dropped new_id=%s existing_id=%s reason=%s",
                    action.new_memory_id, action.existing_memory_id,
                    getattr(action, "reason", ""),
                )
            except Exception as e:
                logger.error(
                    "Knowledge-update NONE: failed new_id=%s existing_id=%s error=%s",
                    action.new_memory_id, action.existing_memory_id, e,
                    exc_info=True,
                )

        return skipped_count

    async def execute_enrichments(
        self,
        actions: List,  # List[EnrichmentAction]
        scope: Scope,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> int:
        """ENRICH handling: log only.

        Both memories stay alive; no link or merge yet. We emit a memory_actions
        row per enrichment so clients see the classifier's reasoning in the
        Requests UI and the merge step has a clean dataset to bootstrap link/merge logic.
        """
        if not actions or not http_correlation_id:
            return 0

        emitted = 0
        for action in actions:
            try:
                await self._rdbms.insert_memory_action(
                    correlation_id=http_correlation_id,
                    ingestion_id=ingestion_id,
                    scope=scope,
                    action="ENRICH",
                    memory_id=action.existing_memory_id,
                    replaced_by_memory_id=action.new_memory_id,
                    previous_content=action.existing_memory_content,
                    content=action.new_memory_content,
                    memory_type=getattr(action, "memory_type", None),
                    reason=getattr(action, "reason", None),
                )
                emitted += 1
                logger.info(
                    "Knowledge-update ENRICH: logged existing_id=%s new_id=%s reason=%s",
                    action.existing_memory_id, action.new_memory_id,
                    getattr(action, "reason", ""),
                )
            except Exception as e:
                logger.warning(
                    "Knowledge-update ENRICH: log failed existing_id=%s new_id=%s error=%s",
                    action.existing_memory_id, action.new_memory_id, e,
                )

        return emitted

    async def execute_subject_links(
        self,
        links: List,  # List[SubjectCandidates]
        scope: Scope,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> int:
        """Decide which memories are about one thing, and record it.

        Both memories always keep serving. Nothing is retired, merged or
        rewritten and no date is touched: a subject carries identity and a name,
        and the dates stay on the memories where they belong.

        Three outcomes per new memory, from one question asked of the judge:

          exactly one group says yes   join it
          several groups say yes       those groups ARE the same thing. Merge
                                       them, then join the survivor.
          no group says yes           mint a subject holding the new memory and
                                       every individual statement that said yes

        The merge case is the whole reason the groups are in the question at all.
        Without it, two memories about one table whose best partners happened to
        be different ungrouped rows each start their own subject and nothing ever
        joins them. Measured before this existed: scope 6d550036 carried EIGHT
        subjects for about four real things, with "considering using Tableau and
        Power BI" and "prefers using Tableau and Power BI" in two of them, and
        "participated in a case competition" and "working on a case competition
        presentation" in two more.

        Returns the number of memories that ended up in a subject.
        """
        if not links:
            return 0
        from synap.cloud.agents.ingestion.services import relatedness

        try:
            groups = await self._rdbms.subjects_in_scope(scope)
        except Exception as e:  # noqa: BLE001 — never fail an ingest over grouping
            logger.warning("subject: could not read existing subjects: %s", e)
            groups = {}

        touched: set = set()
        for link in links:
            try:
                touched |= await self._resolve_one_subject(
                    link, groups, scope, http_correlation_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("subject resolve failed new=%s: %s",
                               getattr(link, "new_memory_id", "?"), e)
        return len(touched)

    async def _resolve_one_subject(self, link, groups: Dict[str, List[str]],
                                   scope: Scope,
                                   http_correlation_id: Optional[str]) -> set:
        """One memory's subject. ``groups`` is mutated so later memories in the
        same batch see the groups this one created or merged."""
        from synap.cloud.agents.ingestion.services import relatedness

        gids = list(groups)
        verdict = await relatedness.ask_subject(
            link.new_memory_content,
            [{"content": c} for _mid, c in link.candidates],
            [{"members": groups[g]} for g in gids],
            correlation_id=http_correlation_id,
        )
        if verdict is None:
            # The judge could not run. Group nothing, which is exactly the
            # behaviour before this feature existed.
            return set()
        yes_c, yes_g = verdict
        yes_groups = [gids[i] for i in sorted(yes_g) if i < len(gids)]
        yes_members = [link.candidates[i] for i in sorted(yes_c)
                       if i < len(link.candidates)]

        if yes_groups:
            survivor = yes_groups[0]
            for rival in yes_groups[1:]:
                # Two groups the judge says are the same thing. This is the
                # convergence step; without it rivals persist forever.
                moved = await self._rdbms.merge_subjects(rival, survivor)
                groups[survivor] = groups.get(survivor, []) + groups.pop(rival, [])
                logger.info("subject merge: %s absorbed %s (%s members)",
                            survivor[:8], rival[:8], moved)
            ids = [link.new_memory_id] + [mid for mid, _ in yes_members]
            label = _group_label(groups.get(survivor, []) + [link.new_memory_content])
            await self._rdbms.assign_subject(ids, survivor, label)
            groups[survivor] = groups.get(survivor, []) + [link.new_memory_content]
            # The result is settled by the write above, so it is computed BEFORE
            # the audit trail. It used to be the last line, and an audit failure
            # therefore lost the count of rows that had already been grouped.
            done = {link.new_memory_id} | {mid for mid, _ in yes_members}
            # In its own try: the rows are already grouped, and an audit failure
            # must not throw that away on the way out. It used to raise through
            # here and the caller counted zero for memories the database had.
            try:
                await self._audit_subject(link, yes_members, survivor,
                                          http_correlation_id, joined_group=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("subject audit failed after the join landed: %s", e)
            logger.info("subject join: %s -> %s (%d yes-groups, %d yes-members)",
                        link.new_memory_id[:8], survivor[:8],
                        len(yes_groups), len(yes_members))
            return done

        if not yes_members:
            return set()

        # Nothing existing matched, so this is a new thing in the world.
        new_id = str(uuid.uuid4())
        ids = [link.new_memory_id] + [mid for mid, _ in yes_members]
        texts = [link.new_memory_content] + [c for _m, c in yes_members]
        label = _group_label(texts)
        await self._rdbms.assign_subject(ids, new_id, label)
        groups[new_id] = texts
        done = set(ids)
        try:
            await self._audit_subject(link, yes_members, new_id, http_correlation_id)
        except Exception as e:  # noqa: BLE001 — see the join path above
            logger.warning("subject audit failed after the mint landed: %s", e)
        logger.info("subject mint: %s with %d member(s) label=%r",
                    new_id[:8], len(ids), label[:60])
        return done

    async def _audit_subject(self, link, yes_members, subject_id: str,
                             http_correlation_id: Optional[str],
                             joined_group: bool = False) -> None:
        """One edge per decided PAIR. A database without migration 159's widened
        CHECK rejects the kind; the grouping already landed on `memories`, which
        is what serving reads, so this costs the audit trail only.

        A memory admitted by a GROUP verdict and by no individual pair gets no
        edge, deliberately. Membership is recorded where it matters, on
        `memories.subject_id`, and that is the record serving and any audit
        reads; an edge would need the group members' ids, which the groups map
        does not carry because it holds their text. A first attempt at filling
        this in looked up `subjects_of([the new memory])`, which can only ever
        return the new memory itself, so every "peer" it found was empty. The
        honest version is this sentence.
        """
        for mid, _ in yes_members:
            try:
                await self._rdbms.insert_memory_link(
                    from_memory_id=link.new_memory_id,
                    to_memory_id=mid,
                    kind="SAME_SUBJECT",
                    reason="same subject %s" % subject_id[:8],
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("subject audit edge not written: %s", e)

    async def execute_merges(
        self,
        merges: List,  # List[MergeAction] — import avoided for circular deps
        scope: Scope,
        embedding_client: Optional[EmbeddingClient] = None,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Persist merged memories + deprecate their contributors.

        Returns metadata for each successfully written merged memory:
        [{
          "memory_id": str, "content": str,
          "memory_type": str, "source_document_ids": list[str],
        }, ...]
        The HQE re-run consumes this to generate hypothetical
        questions for the merged memories.

        For each MergeAction:
          1. Insert a new row in `memories` with `merged_content` (source_document_id NULL).
          2. Insert N rows in `memory_source_documents` (union of contributors' sources).
          3. Embed and upsert the merged memory's vector to ChromaDB.
          4. For each contributor: set deprecated_at, write memory_links(kind='ENRICHED'),
             hard-delete from ChromaDB (Path A′ — keep retrieval clean).
          5. Emit one ENRICH row per contributor in memory_actions, replaced_by =
             merged_id (so the dashboard's lineage view can link them).

        HQE for the merged memory is generated by the HQE re-run (a separate post-merge
        pass on the merged memories). This method only handles persistence.

        Returns the count of merged memories successfully written.
        """
        if not merges:
            return []

        from uuid import uuid4 as _uuid4
        from .vector_wrapper import VectorDocument

        merged_records: List[Dict[str, Any]] = []
        emb_client = embedding_client or self._embedding

        from datetime import datetime, timezone
        from synap.cloud.database.connection import get_session_factory
        from sqlalchemy import text

        for merge in merges:
            try:
                merged_memory_id = str(_uuid4())
                now_utc = datetime.now(timezone.utc)

                # Resolved fields from Stage4Organizer._attach_metadata_to_merges.
                # Defaults match a single-source perpetual fact so we still
                # write a valid row even if a caller forgot to attach.
                primary_type = getattr(merge, "primary_memory_type", None) or "facts"
                resolved_category = getattr(merge, "category", None)
                resolved_subcategory = getattr(merge, "subcategory", None)
                resolved_confidence = float(getattr(merge, "confidence", 0.9) or 0.9)
                resolved_temporal_cat = getattr(merge, "temporal_category", None) or "perpetual"
                resolved_valid_from = getattr(merge, "valid_from", None)
                resolved_valid_until = getattr(merge, "valid_until", None)
                resolved_temporal_conf = float(getattr(merge, "temporal_confidence", 0.0) or 0.0)
                resolved_is_episode = bool(getattr(merge, "is_episode", False))
                resolved_pii = bool(getattr(merge, "pii_obfuscated", False))
                # A merged row is only "checked" if every contributor was.
                # If any contributor predates the ingress gate, the honest
                # answer for the merge is NULL, because part of what this row
                # now says was never looked at (migration 083).
                resolved_pii_detected = getattr(merge, "pii_detected", None)
                resolved_dct = getattr(merge, "document_created_at", None)
                resolved_dlo = getattr(merge, "document_last_observed_at", None) or resolved_dct
                resolved_extracted_at = getattr(merge, "extracted_at", None) or now_utc
                contributor_entities = list(getattr(merge, "contributor_entity_ids", []) or [])

                # 1. Insert merged row with EVERY resolved field. Raw SQL so we
                # can write source_document_id=NULL (the M:N join table is
                # authoritative) and document_last_observed_at (column added
                # in migration 053). Same column shape as wrapper.insert_memory.
                session_factory = get_session_factory()
                from synap.cloud.shared import event_window_schema as ews
                async with session_factory() as session:
                    _merge_sql = """INSERT INTO memories (
                                    id, client_id, instance_id, customer_id, user_id,
                                    memory_type, content, confidence,
                                    category, subcategory, source_document_id,
                                    pii_obfuscated, pii_detected, created_at,
                                    temporal_category, valid_from, valid_until,
                                    temporal_confidence, is_episode,
                                    document_created_at, document_last_observed_at,
                                    event_start, event_end, event_precision
                                ) VALUES (
                                    CAST(:id AS uuid), :client_id, :instance_id, :customer_id, :user_id,
                                    :memory_type, :content, :confidence,
                                    :category, :subcategory, NULL,
                                    :pii_obfuscated, :pii_detected, :created_at,
                                    :temporal_category, :valid_from, :valid_until,
                                    :temporal_confidence, :is_episode,
                                    :document_created_at, :document_last_observed_at,
                                    :event_start, :event_end, :event_precision
                                )"""
                    _merge_params = {
                            "id": merged_memory_id,
                            "client_id": str(scope.client_id),
                            "instance_id": str(scope.instance_id),
                            "customer_id": str(scope.customer_id),
                            "user_id": str(scope.user_id) if scope.user_id else None,
                            "memory_type": primary_type,
                            "content": merge.merged_content,
                            "confidence": resolved_confidence,
                            "category": resolved_category,
                            "subcategory": resolved_subcategory,
                            "pii_obfuscated": resolved_pii,
                            "pii_detected": resolved_pii_detected,
                            # created_at = NOW (the row is new); extracted_at
                            # is "when the system first knew this fact" but
                            # the table doesn't store it as a separate column,
                            # so we lean on memory_actions for that signal.
                            "created_at": resolved_extracted_at,
                            "temporal_category": resolved_temporal_cat,
                            "valid_from": resolved_valid_from,
                            "valid_until": resolved_valid_until,
                            "temporal_confidence": resolved_temporal_conf,
                            "is_episode": resolved_is_episode,
                            "document_created_at": resolved_dct,
                            "document_last_observed_at": resolved_dlo,
                            "event_start": getattr(merge, "event_start", None),
                            "event_end": getattr(merge, "event_end", None),
                            "event_precision": getattr(merge, "event_precision", None),
                        }
                    try:
                        await session.execute(text(ews.adapt(_merge_sql)), _merge_params)
                    except Exception as exc:
                        # Database without migration 154: write the merge without its window.
                        if not (ews.available() and ews.is_missing_column_error(exc)):
                            raise
                        await session.rollback()
                        ews.mark_missing(exc)
                        await session.execute(text(ews.adapt(_merge_sql)), _merge_params)
                    await session.commit()

                # 2. Source attribution: union of contributors' sources.
                if merge.contributor_source_documents:
                    await self._rdbms.insert_memory_source_documents(
                        memory_id=merged_memory_id,
                        source_document_ids=list(set(merge.contributor_source_documents)),
                    )

                # 2b. Entity links union — copy contributors' graph reachability
                # to the merged memory BEFORE deprecating contributors (after
                # which the cascade-delete on memory_entity_links would wipe
                # them). One INSERT per entity; insert_entity_reference is
                # idempotent on (memory_id, entity_id).
                for entity_id in contributor_entities:
                    try:
                        await self._rdbms.insert_entity_reference(
                            memory_id=merged_memory_id,
                            entity_id=entity_id,
                            scope=scope,
                        )
                    except Exception as e:
                        logger.warning(
                            "Merge: entity link copy failed (non-fatal) "
                            "merged_id=%s entity_id=%s error=%s",
                            merged_memory_id, entity_id, e,
                        )

                # 3. Embed + upsert the merged vector with FULL metadata mirror.
                # Matches AsyncVectorWrapper.upsert_memory's metadata shape so
                # retrieval-time filters (temporal_category, valid_from,
                # confidence, etc.) work the same on merged memories as on
                # any other.
                #
                # CRITICAL HISTORY: an earlier version called
                #   await self._vector.upsert(namespace=..., documents=[...])
                # but `upsert` is NOT a method on AsyncVectorWrapper — it lives
                # on the underlying VectorStoreInterface. The AttributeError
                # was caught by the broad except and logged as "non-fatal",
                # so every merge silently dropped the merged vector.
                # Net effect: contributors hard-deleted from
                # ChromaDB, merged survivor never inserted, the cluster's
                # facts vanished from retrieval. Fix is to go through the
                # underlying provider via asyncio.to_thread, the same path
                # AsyncVectorWrapper.upsert_memory uses internally.
                merged_in_chroma = False
                if self._vector is not None and emb_client is not None:
                    try:
                        embeds = await emb_client.embed_batch([merge.merged_content])
                        if embeds:
                            chroma_meta = {
                                "client_id": str(scope.client_id),
                                "instance_id": str(scope.instance_id),
                                "customer_id": str(scope.customer_id),
                                "user_id": str(scope.user_id) if scope.user_id else "",
                                "memory_type": primary_type,
                                "category": resolved_category,
                                "subcategory": resolved_subcategory,
                                "confidence": resolved_confidence,
                                "created_at": resolved_extracted_at.isoformat(),
                                "entities_mentioned": ",".join(contributor_entities) if contributor_entities else "",
                                "temporal_category": resolved_temporal_cat,
                                "temporal_confidence": resolved_temporal_conf,
                                "is_episode": str(resolved_is_episode).lower(),
                            }
                            # ChromaDB rejects None — only attach optional
                            # date fields when populated.
                            if resolved_valid_from:
                                chroma_meta["valid_from"] = resolved_valid_from.isoformat()
                            if resolved_valid_until:
                                chroma_meta["valid_until"] = resolved_valid_until.isoformat()
                            if resolved_dct:
                                chroma_meta["document_created_at"] = resolved_dct.isoformat()
                            if resolved_dlo:
                                chroma_meta["document_last_observed_at"] = resolved_dlo.isoformat()
                            from synap.cloud.agents.ingestion.services.event_window import event_window_metadata
                            chroma_meta.update(event_window_metadata(
                                getattr(merge, "event_start", None), getattr(merge, "event_end", None),
                                getattr(merge, "event_precision", None)))
                            # Strip any None values defensively (above branches
                            # already guard, but category/subcategory may be None).
                            chroma_meta = {k: v for k, v in chroma_meta.items() if v is not None}

                            import asyncio as _asyncio
                            namespace = self._vector._get_namespace(scope)
                            doc = VectorDocument(
                                id=merged_memory_id,
                                embedding=embeds[0].tolist() if hasattr(embeds[0], "tolist") else list(embeds[0]),
                                metadata=chroma_meta,
                                content=merge.merged_content,
                            )
                            await _asyncio.to_thread(
                                self._vector._provider.upsert,
                                namespace,
                                [doc],
                            )
                            merged_in_chroma = True
                    except Exception as e:
                        logger.error(
                            "Merge: ChromaDB upsert of merged memory FAILED (the merged "
                            "memory will be unreachable by vector retrieval) "
                            "merged_id=%s error=%s",
                            merged_memory_id, e, exc_info=True,
                        )

                # If the vector upsert failed, do NOT proceed to deprecate
                # contributors — that would leave the cluster's facts
                # unreachable in vector search (PG still has the merged row,
                # but vector search is the primary retrieval path). Better
                # to log and skip the deprecation step so the contributors
                # remain searchable until the next merge attempt.
                if (
                    self._vector is not None
                    and emb_client is not None
                    and not merged_in_chroma
                ):
                    logger.error(
                        "Merge: skipping contributor deprecation because the "
                        "merged vector failed to land in ChromaDB. "
                        "merged_id=%s contributors=%s",
                        merged_memory_id, merge.contributor_memory_ids,
                    )
                    continue

                # 4 + 5. Deprecate contributors + emit ENRICH changelog rows.
                for contrib_id, contrib_content in zip(
                    merge.contributor_memory_ids, merge.contributor_contents
                ):
                    try:
                        # Retire the contributor, link it to the merged row,
                        # and record the before-state, in one transaction.
                        # A merge is the one retirement whose trail row is
                        # written against the CONTRIBUTOR while the reversal
                        # resurrects it from the PRODUCT's ENRICH row, so the
                        # before-state here is what a reversed merge restores.
                        # `require_retire=False`: a contributor that is already
                        # superseded still needs its ENRICH row, because the
                        # merge really did absorb it.
                        await self._rdbms.retire_memory_with_audit(
                            memory_id=contrib_id,
                            action="ENRICH",
                            scope=scope,
                            superseded_by=merged_memory_id,
                            replaced_by_memory_id=merged_memory_id,
                            previous_content=contrib_content,
                            content=merge.merged_content,
                            memory_type=getattr(merge, "primary_memory_type", None),
                            reason=merge.rationale,
                            correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                            deprecate_with_link=True,
                            link_kind="ENRICHED",
                            require_retire=False,
                        )

                        # ChromaDB: hard-delete contributor + its HQE
                        # (delete_memory_vectors does both).
                        if self._vector is not None:
                            try:
                                await self._vector.delete_memory_vectors(
                                    memory_ids=[contrib_id],
                                    scope=scope,
                                )
                            except Exception as e:
                                logger.warning(
                                    "Merge: ChromaDB delete of contributor failed (non-fatal) "
                                    "contrib_id=%s merged_id=%s error=%s",
                                    contrib_id, merged_memory_id, e,
                                )

                    except Exception as e:
                        logger.warning(
                            "Merge: contributor deprecation failed contrib_id=%s merged_id=%s error=%s",
                            contrib_id, merged_memory_id, e,
                        )

                merged_records.append({
                    "memory_id": merged_memory_id,
                    "content": merge.merged_content,
                    "memory_type": primary_type,
                    "source_document_ids": list(set(merge.contributor_source_documents or [])),
                    # The HQE wrapper carries these forward so the
                    # generated questions inherit the merged memory's full
                    # context (entities for ERE, temporal scope, etc.).
                    "entity_ids": contributor_entities,
                    "extracted_at": resolved_extracted_at,
                    "document_created_at": resolved_dct,
                })
                logger.info(
                    "Knowledge-update MERGE: synthesized merged_id=%s from %d contributors rationale=%s",
                    merged_memory_id, len(merge.contributor_memory_ids), merge.rationale,
                )
            except Exception as e:
                logger.error(
                    "Merge: failed for cluster of %d contributors error=%s",
                    len(merge.contributor_memory_ids), e, exc_info=True,
                )

        return merged_records
