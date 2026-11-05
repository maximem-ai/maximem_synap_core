"""Async wrapper for Graph provider."""

import asyncio
import json
import logging
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Any
from uuid import UUID

logger = logging.getLogger(__name__)

from synap.cloud.agents.ingestion.models import (
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelationship,
    Scope,
)
from synap.cloud.agents.ingestion.resolution.node_key import (
    NODE_KEY_PROPERTY,
    is_provisional,
    may_use_node,
    node_name_candidates,
)
from synap.cloud.agents.ingestion.services import relationship_type
from synap.cloud.shared import graph_boundary
from synap.providers.graph import GraphStoreInterface


_PRIMITIVES = (str, int, float, bool)

# How many times a keyed entity write re-resolves its node name after losing a concurrent create.
_NODE_NAME_RACE_ATTEMPTS = 5


def storable_properties(properties: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """*properties* in a form Neo4j accepts.

    A property is a primitive or a list of primitives of one type. The
    extraction model sometimes returns more, e.g. {"change": {"from": "Monday",
    "to": "Wednesday"}}, and Neo4j then rejects the whole statement: one such
    value cost a document its entire graph write. A value Neo4j cannot hold is
    kept as its JSON text, and a None is left out.
    """
    out: Dict[str, Any] = {}
    for key, value in (properties or {}).items():
        if value is None:
            continue
        if isinstance(value, _PRIMITIVES):
            out[key] = value
        elif (isinstance(value, (list, tuple)) and value
              and all(isinstance(v, _PRIMITIVES) for v in value)
              and len({type(v) for v in value}) == 1):
            out[key] = list(value)
        elif isinstance(value, (list, tuple)) and not value:
            out[key] = []
        else:
            out[key] = json.dumps(value, default=str, sort_keys=True)
    return out


class AsyncGraphWrapper:
    """
    Async wrapper around sync GraphStoreInterface.
    
    Uses asyncio.to_thread() to make blocking calls non-blocking.
    Maps ingestion domain models to graph operations.
    """

    def __init__(self, provider: GraphStoreInterface):
        self._provider = provider

    def _get_namespace(self, scope: Scope) -> str:
        """Derive namespace from scope.

        Customer-level isolation by default; falls back to client-level for
        client-scope requests (scope.scope_level == "client" or when
        customer_id is absent).
        """
        # One graph per scope when SYNAP_GRAPH_PER_SCOPE is set (shared/scope_key):
        # a user-scope document goes to the user's own graph, so two users
        # naming one entity no longer share, and overwrite, a node.
        from synap.cloud.shared.scope_key import write_scope_id
        return write_scope_id(scope)

    def _get_client_namespace(self, scope: Scope) -> str:
        """Derive client-level namespace for client-scope graph data."""
        return str(scope.client_id)

    def _chain_properties(self, scope: Scope) -> Dict[str, Any]:
        """S-03's keys, on graph records, from the same builder the vectors use.

        A vector document has carried `node_id` plus one `anc_<level_key>` per
        ancestor since S-03, and the read filter is built on them. Graph records
        carried neither, so a graph record's address was still the flat
        customer/user pair and a rung between the two had nowhere to live. That
        is what let the read filter and the graph arm answer differently for the
        same request.

        Not reimplemented here. `_nested_scope_metadata` decides which rung a
        write is filed at, including the B2C case where the tenant rung also
        holds the identity role, and a second copy of that decision is precisely
        the kind of near-agreement this file already suffered from.

        Empty for every caller without a resolved chain, which is every request
        on clients running the default three rungs while
        nested scoping is off. The keys are additive, so this can run in
        production long before anything reads it, and a partly stamped graph is
        incomplete rather than wrong: an unstamped record still answers to the
        transitional customer/user arm exactly as it does today.
        """
        from synap.cloud.agents.ingestion.storage.vector_wrapper import (
            _effective_scope_level,
            _nested_scope_metadata,
        )

        try:
            return _nested_scope_metadata(scope, _effective_scope_level(scope, None))
        except Exception as exc:  # noqa: BLE001
            # Never fail a write over an optional key. A missing node_id costs
            # recall on one record; a raised exception costs the whole ingestion.
            logger.warning("graph chain metadata not built: %s", exc)
            return {}

    async def upsert_entity_node(
        self,
        entity: ExtractedEntity,
        scope: Scope,
        merge_similar: bool = False,
        embedding: list = None,
    ) -> str:
        """Create or update an entity node in the graph.

        When merge_similar=True and entity has a canonical_name, look for
        an existing node with the same canonical_name. If found, update its
        properties (merge) instead of creating a duplicate node.
        """
        namespace = self._get_namespace(scope)
        canonical_name = entity.canonical_name or entity.name

        # Entity resolution keyed this entity: the node is the one carrying
        # that key, and the name is only claimed by it (see resolution/node_key).
        node_key = entity.properties.get(NODE_KEY_PROPERTY)
        if node_key:
            return await self._upsert_keyed_entity_node(
                entity, scope, namespace, canonical_name, node_key, embedding
            )

        # Graph merge: look for existing node with same canonical_name
        existing_node_id = None
        if merge_similar and entity.canonical_name:
            existing_node_id = await self._find_entity_by_canonical_name(
                canonical_name=canonical_name,
                namespace=namespace,
            )

        # Build properties (omit absent scope IDs — Neo4j stores None as null,
        # but retrieval uses `IS NULL` checks that rely on the key being absent
        # for legacy rows and a distinct `__CUSTOMER_SCOPE__` value being the
        # only signal of intentional customer-shared data).
        properties = {
            "client_id": str(scope.client_id),
            "entity_type": entity.entity_type,
            "name": entity.name,
            "canonical_name": canonical_name,
            "entity_id": entity.entity_id,  # Original extraction ID for cross-system correlation
            **storable_properties(entity.properties),
        }
        if scope.customer_id:
            properties["customer_id"] = str(scope.customer_id)
        if scope.user_id:
            properties["user_id"] = str(scope.user_id)
        properties.update(self._chain_properties(scope))
        properties.update(graph_boundary.node_edge_properties(scope))
        if embedding is not None:
            properties["embedding"] = embedding

        if existing_node_id:
            # MERGE: update existing node, accumulate raw name as alias
            existing_node = await self.get_node(namespace, existing_node_id)
            if existing_node:
                existing_aliases = existing_node.properties.get("aliases", [])
                if entity.name not in existing_aliases and entity.name != canonical_name:
                    existing_aliases.append(entity.name)
                    properties["aliases"] = existing_aliases

            await asyncio.to_thread(
                self._provider.update_node,
                namespace,
                existing_node_id,
                properties,
            )
            return existing_node_id

        # CREATE: new entity node
        # Handle race condition: if another concurrent pipeline created the
        # same entity between our lookup and this CREATE, catch the constraint
        # violation and fall back to updating the existing node.
        try:
            node_id = await asyncio.to_thread(
                self._provider.add_node,
                namespace,
                entity.entity_type,
                properties,
            )
            return node_id
        except Exception as e:
            if "ConstraintValidationFailed" in str(e) or "ConstraintError" in type(e).__name__:
                logger.info(
                    "Entity node already exists (concurrent create race), "
                    "falling back to update: canonical_name=%s namespace=%s",
                    canonical_name, namespace,
                )
                # Re-lookup and update the existing node
                existing_id = await self._find_entity_by_canonical_name(
                    canonical_name=canonical_name,
                    namespace=namespace,
                )
                if existing_id:
                    await asyncio.to_thread(
                        self._provider.update_node,
                        namespace,
                        existing_id,
                        properties,
                    )
                    return existing_id
            raise

    async def _upsert_keyed_entity_node(
        self,
        entity: ExtractedEntity,
        scope: Scope,
        namespace: str,
        base_name: str,
        node_key: str,
        embedding: list = None,
    ) -> str:
        """Create or update the node that carries *node_key*.

        Walks the candidate names for this entity until it finds its own node
        or a free name. A name held by a different key is left alone, so two
        entities with one spelling never share a node. One retry covers a
        concurrent pipeline taking the free name between lookup and create; with
        many workers writing the same popular entity (every session names the
        user) the retry can lose too, so it is tried a few times with a short wait.
        """
        for attempt in range(_NODE_NAME_RACE_ATTEMPTS):
            chosen, existing = None, None
            for candidate in node_name_candidates(base_name, entity.entity_type, node_key):
                node = await self._find_entity_node_by_canonical_name(candidate, namespace)
                if node is None:
                    chosen = candidate
                    break
                if may_use_node(node_key, node.properties.get(NODE_KEY_PROPERTY)):
                    chosen, existing = candidate, node
                    break
            if chosen is None:
                # Every candidate name is held by another key. The last one
                # embeds this key, so this is not reachable in practice.
                raise RuntimeError(f"no free graph node name for {base_name!r}")
            renamed_from = None
            if existing is None and not is_provisional(node_key):
                # No node under this name. The entity may have been renamed to
                # a fuller name since its node was written: the node is the
                # one that carries its key, whatever it is called.
                existing = await self._find_entity_node_by_key(node_key, namespace)
                if existing is not None:
                    renamed_from = existing.properties.get("canonical_name")

            properties = {
                "client_id": str(scope.client_id),
                "entity_type": entity.entity_type,
                "name": entity.name,
                "canonical_name": chosen,
                "entity_id": entity.entity_id,
                # er_merge_id describes this one mention; it goes on the edges
                # (StorageCoordinator), not on the node every mention shares.
                **storable_properties(
                    {k: v for k, v in entity.properties.items() if k != "er_merge_id"}),
            }
            if chosen != base_name:
                # What a person would call it; canonical_name had to differ.
                properties["display_name"] = base_name
            if scope.customer_id:
                properties["customer_id"] = str(scope.customer_id)
            if scope.user_id:
                properties["user_id"] = str(scope.user_id)
            properties.update(self._chain_properties(scope))
            properties.update(graph_boundary.node_edge_properties(scope))
            if embedding is not None:
                properties["embedding"] = embedding

            # Every other form the text used for the entity, and for the
            # writer's or the customer's node the name it gave itself.
            text_aliases = [a for a in (getattr(entity, "aliases", None) or []) if a and a != chosen]
            if existing is not None:
                aliases = list(existing.properties.get("aliases", []))
                for other in (entity.name, renamed_from, *text_aliases):
                    if other and other not in aliases and other != chosen:
                        aliases.append(other)
                        properties["aliases"] = aliases
                await asyncio.to_thread(
                    self._provider.update_node, namespace, existing.id, properties
                )
                return existing.id

            if text_aliases:
                properties["aliases"] = text_aliases
            try:
                return await asyncio.to_thread(
                    self._provider.add_node, namespace, entity.entity_type, properties
                )
            except Exception as e:
                taken = "ConstraintValidationFailed" in str(e) or "ConstraintError" in type(e).__name__
                if not taken or attempt == _NODE_NAME_RACE_ATTEMPTS - 1:
                    raise
                logger.info(
                    "Entity node name taken concurrently, re-resolving: canonical_name=%s namespace=%s",
                    chosen, namespace,
                )
                await asyncio.sleep(0.05 * (attempt + 1) * (1 + random.random()))
        raise RuntimeError(f"could not write graph node for {base_name!r}")

    async def _find_entity_node_by_key(self, node_key: str, namespace: str):
        """The node that carries this entity-resolution key, or None."""
        nodes = await asyncio.to_thread(
            self._provider.find_nodes_by_properties_any_type,
            namespace,
            {NODE_KEY_PROPERTY: node_key},
            1,
        )
        return nodes[0] if nodes else None

    async def _find_entity_node_by_canonical_name(self, canonical_name: str, namespace: str):
        """The node with this canonical_name in the namespace, or None."""
        nodes = await asyncio.to_thread(
            self._provider.find_nodes_by_properties_any_type,
            namespace,
            {"canonical_name": canonical_name},
            1,
        )
        return nodes[0] if nodes else None

    async def _find_entity_by_canonical_name(
        self,
        canonical_name: str,
        namespace: str,
    ) -> Optional[str]:
        """Find existing entity node by canonical_name in namespace (any entity_type).

        Uses type-agnostic lookup to match the Neo4j UNIQUE constraint scope
        which is on (canonical_name, _namespace) across ALL entity types.
        """
        nodes = await asyncio.to_thread(
            self._provider.find_nodes_by_properties_any_type,
            namespace,
            {"canonical_name": canonical_name},
            1,
        )
        if nodes:
            return nodes[0].id
        return None

    async def create_relationship(
        self,
        relationship: ExtractedRelationship,
        scope: Scope,
        embedding: list = None,
    ) -> Optional[str]:
        """Create a relationship edge between entities. Returns edge ID or None if nodes not found."""
        namespace = self._get_namespace(scope)

        # Build properties (omit absent scope IDs — see upsert_entity_node comment)
        properties = {
            "client_id": str(scope.client_id),
            **storable_properties(relationship.properties),
        }
        if scope.customer_id:
            properties["customer_id"] = str(scope.customer_id)
        if scope.user_id:
            properties["user_id"] = str(scope.user_id)
        properties.update(self._chain_properties(scope))
        properties.update(graph_boundary.node_edge_properties(scope))
        if embedding is not None:
            properties["embedding"] = embedding

        # Create edge
        edge_id = await asyncio.to_thread(
            self._provider.add_edge,
            namespace,
            relationship.source_entity_id,
            relationship.target_entity_id,
            relationship_type.storable(relationship.relationship_type),
            properties
        )
        
        return edge_id

    async def edges_between(self, scope: Scope, a_id: str, b_id: str) -> list:
        """Every relationship already stored between two nodes, either way."""
        return await asyncio.to_thread(
            self._provider.edges_between, self._get_namespace(scope), a_id, b_id)

    async def strip_person_memories(self, client_id, field: str, value: str, instance_ids,
                                    namespaces=None, include_unstamped: bool = False):
        """Remove one person's memories, written by these instances, from every
        relationship (see ``Neo4jAdapter.strip_person_memories``). Used by an erase
        that keeps a node because another instance still uses it.
        Returns (removed, unstamped left)."""
        return await asyncio.to_thread(
            self._provider.strip_person_memories, str(client_id), field, str(value),
            None if instance_ids is None else list(instance_ids),
            list(namespaces) if namespaces else None, include_unstamped)

    async def add_memory_references(self, scope: Scope, references: dict) -> int:
        """Write {node id: [memory id]} onto the nodes (services/node_memories)."""
        from synap.cloud.agents.ingestion.services import node_memories
        namespace = self._get_namespace(scope)
        for node_id, memory_ids in references.items():
            await asyncio.to_thread(
                self._provider.append_to_node_list, namespace, node_id,
                node_memories.PROPERTY, memory_ids, node_memories.MAX_PER_NODE)
        await self._reevaluate_retired(scope, namespace, list(references))
        return len(references)

    async def _reevaluate_retired(self, scope: Scope, namespace: str, ids: List[str]) -> None:
        """A memory was attached to records that may be retired: let the rule
        decide whether they reopen (GraphLifecycle.reevaluate). Never fails the
        write: a record left retired here is fixed by the reconciler."""
        from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle, lifecycle_enabled
        if not ids or not lifecycle_enabled(getattr(scope, "instance_id", None)):
            return
        try:
            await GraphLifecycle(self).reevaluate(namespace, ids)
        except NotImplementedError:
            return
        except Exception:  # noqa: BLE001
            logger.warning("graph_lifecycle reevaluate after attach failed namespace=%s", namespace, exc_info=True)

    async def attach_memories_to_relationships(
        self,
        memories: List[ExtractedMemory],
        relationships: List[ExtractedRelationship],
        relationship_edge_ids: List[str],
        scope: Scope,
        memory_embeddings: Optional[List] = None,
    ) -> int:
        """Attach memory content as a JSON list property on real relationship edges.

        For each memory, finds which relationship edges connect its mentioned entities,
        then stores the memory content on those edges as a `memory_contents` JSON list.

        Args:
            memories: Extracted memories (with entities_mentioned)
            relationships: Original ExtractedRelationship list (extraction-time entity IDs)
            relationship_edge_ids: Neo4j edge IDs returned from create_relationship(), parallel to relationships
            scope: Tenant scope

        Returns:
            Number of edges updated with memory content
        """
        namespace = self._get_namespace(scope)

        if not memories or not relationships or not relationship_edge_ids:
            return 0

        from synap.cloud.agents.ingestion.services import node_memories
        if node_memories.edges_enabled(getattr(scope, "instance_id", None)):
            # References, not copies: the id of each memory that names both
            # ends, on EVERY relationship this note wrote between that pair.
            pair_edges: Dict[frozenset, List[str]] = defaultdict(list)
            for rel, edge_id in zip(relationships, relationship_edge_ids):
                pair_edges[frozenset((rel.source_entity_id, rel.target_entity_id))].append(edge_id)
            edge_ids: Dict[str, List[str]] = defaultdict(list)
            for memory in memories:
                mentioned = memory.entities_mentioned
                for i, a in enumerate(mentioned):
                    for b in mentioned[i + 1:]:
                        for edge_id in pair_edges.get(frozenset((a, b)), ()):
                            edge_ids[edge_id].append(str(memory.memory_id))
            written = 0
            for edge_id, ids in edge_ids.items():
                try:
                    await asyncio.to_thread(
                        self._provider.append_to_edge_list, namespace, edge_id,
                        node_memories.PROPERTY, ids, node_memories.MAX_PER_NODE)
                    written += 1
                except Exception:  # noqa: BLE001 - the edge is written; its memory list is extra
                    logger.warning("Could not attach memories to edge %s", edge_id, exc_info=True)
            await self._reevaluate_retired(scope, namespace, list(edge_ids))
            return written

        # Build memory_id → embedding lookup for per-memory scoring at retrieval time
        embedding_map: Dict[str, List[float]] = {}
        if memory_embeddings:
            for memory, emb in zip(memories, memory_embeddings):
                if emb is not None:
                    embedding_map[str(memory.memory_id)] = (
                        emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    )

        # 1. Build lookup: (source_entity_id, target_entity_id) → edge_id
        #    Uses extraction-time entity IDs (before Neo4j remapping) since
        #    memory.entities_mentioned uses those same IDs.
        pair_to_edge_id: Dict[tuple, str] = {}
        for rel, edge_id in zip(relationships, relationship_edge_ids):
            pair_to_edge_id[(rel.source_entity_id, rel.target_entity_id)] = edge_id
            pair_to_edge_id[(rel.target_entity_id, rel.source_entity_id)] = edge_id

        # 2. For each memory, find matching relationship edges and collect memories per edge
        chain_properties = self._chain_properties(scope)
        edge_memories: Dict[str, List[dict]] = defaultdict(list)
        for memory in memories:
            mentioned = memory.entities_mentioned
            if len(mentioned) < 2:
                continue  # Single-entity or no entities — skip

            # For customer-scope ingestion (scope.user_id is None by design),
            # leave user_id absent — the `OR rel.user_id IS NULL` branch in
            # combined_name_traverse (gated on roll_up_to_customer) is how
            # retrieval surfaces these to user-scope callers.
            mem_dict = {
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "memory_id": str(memory.memory_id),
                "confidence": memory.confidence,
                "customer_id": str(scope.customer_id) if scope.customer_id else None,
                "user_id": str(scope.user_id) if scope.user_id else None,
                "version": 1,
                "created_at": memory.extracted_at.isoformat(),
                "updated_at": memory.extracted_at.isoformat(),
                "is_current": True,
                "is_deleted": False,
                # WHO said it. The graph stored every other field a served
                # item needs and not this one, so anything reached through the
                # graph arrived with no speaker — and on a counting question
                # the graph is a third to a half of what is served. Reading it
                # back out costs nothing; never writing it cost 67% coverage.
                "assertion_source": getattr(memory, "assertion_source", None),
                "temporal_category": memory.temporal_category,
                "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
                "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
                "temporal_confidence": memory.temporal_confidence,
                "is_episode": memory.is_episode,
                "document_created_at": memory.document_created_at.isoformat() if memory.document_created_at else None,
                "event_start": memory.event_start.isoformat() if getattr(memory, "event_start", None) else None,
                "event_end": memory.event_end.isoformat() if getattr(memory, "event_end", None) else None,
                "event_precision": getattr(memory, "event_precision", None),
            }
            # The rung this memory is filed at, so the read filter's chain arm
            # can answer for a memory reached through the graph the same way it
            # answers for the same memory reached through the vector store.
            mem_dict.update(chain_properties)
            # Which instance wrote this memory. A relationship is shared by every
            # instance of the customer, but each memory text on it has exactly one
            # writer, and the reader must be able to refuse the others'.
            mem_dict.update(graph_boundary.memory_properties(scope))
            # Store embedding for per-memory scoring at retrieval time
            emb = embedding_map.get(str(memory.memory_id))
            if emb is not None:
                mem_dict["embedding"] = emb

            # Check all pairs of mentioned entities
            for i, eid_a in enumerate(mentioned):
                for eid_b in mentioned[i + 1:]:
                    edge_id = pair_to_edge_id.get((eid_a, eid_b))
                    if edge_id:
                        edge_memories[edge_id].append(mem_dict)

        # 3. Update each edge with its collected memories (merge with existing)
        updated_count = 0
        for edge_id, mem_list in edge_memories.items():
            try:
                # Read existing memory_contents to avoid overwriting on re-ingestion
                existing_edge = await asyncio.to_thread(
                    self._provider.get_edge, namespace, edge_id
                )
                existing_memories = []
                if existing_edge and existing_edge.properties:
                    raw = existing_edge.properties.get("memory_contents")
                    if raw:
                        try:
                            existing_memories = json.loads(raw) if isinstance(raw, str) else raw
                        except (json.JSONDecodeError, TypeError):
                            existing_memories = []

                # Deduplicate by memory_id (existing wins = idempotent re-ingestion)
                existing_ids = {m.get("memory_id") for m in existing_memories if isinstance(m, dict)}
                merged = existing_memories + [m for m in mem_list if m.get("memory_id") not in existing_ids]

                await asyncio.to_thread(
                    self._provider.update_edge,
                    namespace,
                    edge_id,
                    {"memory_contents": json.dumps(merged)},
                )
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to attach memories to relationship edge edge_id={edge_id} memory_count={len(mem_list)} error={e}")

        logger.info(f"Memories attached to relationship edges edges_updated={updated_count} total_memories={len(memories)} memories_with_2plus_entities={sum(1 for m in memories if len(m.entities_mentioned) >= 2)} edges_with_memories={len(edge_memories)}")

        return updated_count

    @staticmethod
    def _match_memories_to_canonical_edges(
        memories: List[Dict[str, Any]],
        node_id_to_name: Dict[str, str],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Pure: match each memory's CONTENT against canonical entity names and return
        ``{edge_id: [memory dicts]}`` for memories that name BOTH endpoints of an edge.

        This is the cross-document complement to entities_mentioned-based attachment:
        ``attach_memory_content_to_edges`` only links a memory within the document it
        was extracted from (entities get fresh per-doc UUIDs, and secondary entities
        are often under-extracted), so cross-document / secondary-entity facts get
        empty entities_mentioned and never land on an edge. Matching memory text
        against the scope's *canonical* entity names recovers those links.

        Args:
            memories: dicts with at least ``memory_id`` and ``content``.
            node_id_to_name: canonical graph node id -> canonical_name.
            edges: dicts with ``edge_id``, ``from_id``, ``to_id``.
        """
        names = sorted({n for n in node_id_to_name.values() if n}, key=len, reverse=True)
        compiled = {nm: re.compile(r"\b" + re.escape(nm.lower()) + r"\b") for nm in names}
        mem_ents: List[tuple] = []
        for m in memories:
            content_lower = (m.get("content") or "").lower()
            ents = {nm for nm in names if compiled[nm].search(content_lower)}
            if len(ents) >= 2:
                mem_ents.append((m, ents))

        out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in edges:
            fn = node_id_to_name.get(e.get("from_id"))
            tn = node_id_to_name.get(e.get("to_id"))
            if not fn or not tn or fn == tn:
                continue
            for m, ents in mem_ents:
                if fn in ents and tn in ents:
                    out[e["edge_id"]].append(m)
        return out

    async def relink_memories_to_canonical_edges(
        self,
        scope: Scope,
        memories: List[Dict[str, Any]],
        node_id_to_name: Dict[str, str],
        edges: List[Dict[str, Any]],
    ) -> int:
        """Cross-document relink: attach memories to existing canonical edges whose
        BOTH endpoints are name-mentioned in the memory content.

        Complements ``attach_memory_content_to_edges`` (single-document, via
        entities_mentioned). Run as a post-ingestion pass against the scope's
        canonical graph so a fact like a test report's "current blocker on STATUS-247"
        — whose entities_mentioned came out empty because the test-report document
        under-extracted STATUS-247 — still lands on the STATUS-247↔TEST edge and
        becomes retrievable via accurate-mode graph traversal. Additive + idempotent
        (dedups by memory_id). The caller loads ``node_id_to_name``/``edges`` (the
        namespace's canonical nodes/edges).

        Returns the number of edges updated.
        """
        namespace = self._get_namespace(scope)
        edge_to_mems = self._match_memories_to_canonical_edges(memories, node_id_to_name, edges)
        updated = 0
        for edge_id, mem_list in edge_to_mems.items():
            try:
                existing_edge = await asyncio.to_thread(self._provider.get_edge, namespace, edge_id)
                existing: List[Dict[str, Any]] = []
                if existing_edge and existing_edge.properties:
                    raw = existing_edge.properties.get("memory_contents")
                    if raw:
                        try:
                            existing = json.loads(raw) if isinstance(raw, str) else raw
                        except (json.JSONDecodeError, TypeError):
                            existing = []
                existing_ids = {m.get("memory_id") for m in existing if isinstance(m, dict)}
                new_items = [m for m in mem_list if m.get("memory_id") not in existing_ids]
                if not new_items:
                    continue
                await asyncio.to_thread(
                    self._provider.update_edge,
                    namespace,
                    edge_id,
                    {"memory_contents": json.dumps(
                        existing + new_items,
                        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                    )},
                )
                updated += 1
            except Exception as e:  # noqa: BLE001 - best-effort, per-edge
                logger.warning(f"relink: failed to attach memories to edge {edge_id}: {e}")

        logger.info(
            "relink_memories_to_canonical_edges: edges_updated=%d candidate_edges=%d memories=%d canonical_entities=%d",
            updated, len(edge_to_mems), len(memories), len(node_id_to_name),
        )
        return updated

    async def load_namespace_canonical_graph(self, scope: Scope):
        """Load the scope's canonical graph for cross-document relink.

        Returns ``(node_id_to_name, edges)`` where node_id_to_name maps canonical
        node id -> canonical_name and edges is a list of
        ``{edge_id, from_id, to_id}``. Returns empty if the provider can't list a
        namespace subgraph.
        """
        namespace = self._get_namespace(scope)
        fn = getattr(self._provider, "get_namespace_subgraph", None)
        if fn is None:
            logger.debug("provider has no get_namespace_subgraph — skipping relink load")
            return {}, []
        nodes, edges = await asyncio.to_thread(fn, namespace)
        node_id_to_name = {n["id"]: n["name"] for n in nodes if n.get("id") and n.get("name")}
        return node_id_to_name, edges

    async def ensure_namespace(self, scope: Scope) -> None:
        """Ensure namespace exists for the given scope."""
        namespace = self._get_namespace(scope)
        
        # Try to create namespace (idempotent operation)
        try:
            await asyncio.to_thread(
                self._provider.create_namespace,
                namespace
            )
        except Exception:
            # Namespace likely already exists
            pass

        # NOTE: previously created a per-namespace `entity_id_idx` range index
        # here. Removed 2026-07 (PR-1): live PROFILE/audit confirmed nothing
        # queries `n.entity_id` (no reader in the adapter or retrieval_manager),
        # so it was pure write-amplification. Existing indexes are dropped as a
        # deploy step. See Neo4j_Implementation_Improvement_Design_v1.md (T3.3).

    async def get_node(self, namespace: str, node_id: str) -> Optional[Any]:
        """Get a node by ID."""
        return await asyncio.to_thread(
            self._provider.get_node,
            namespace,
            node_id
        )

    async def get_neighbors(
        self,
        namespace: str,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "out"
    ) -> List[Any]:
        """Get neighboring nodes."""
        return await asyncio.to_thread(
            self._provider.get_neighbors,
            namespace,
            node_id,
            edge_type,
            direction
        )

    async def run_query(self, cypher: str, params: dict = None) -> list:
        """Execute raw Cypher query. Async wrapper around adapter._run_query."""
        return await asyncio.to_thread(
            self._provider._run_query, cypher, params or {}
        )

    async def run_write(self, cypher: str, params: dict = None) -> list:
        """Execute a Cypher statement that changes data (DETACH DELETE, SET,
        MERGE) in a write transaction and return its rows.

        `run_query` runs in a read transaction and Neo4j rejects a write there.
        Anything that deletes or updates through raw Cypher uses this.
        """
        return await asyncio.to_thread(
            self._provider._run_write_rows, cypher, params or {}
        )

    async def traverse(
        self,
        namespace: str,
        start_node_id: str,
        pattern: Any,
    ) -> Any:
        """Traverse graph from start node. Async wrapper around adapter.traverse."""
        return await asyncio.to_thread(
            self._provider.traverse, namespace, start_node_id, pattern
        )

    async def batch_traverse(
        self,
        namespace: str,
        start_ids: list,
        pattern: Any,
    ) -> Any:
        """Traverse graph from multiple start nodes in one query. Async wrapper around adapter.batch_traverse."""
        return await asyncio.to_thread(
            self._provider.batch_traverse, namespace, start_ids, pattern
        )

    async def combined_name_traverse(
        self,
        namespace: str,
        anchor_names: List[str],
        max_depth: int = 3,
        per_anchor_limit: int = 50,
        customer_id: Optional[str] = None,
        user_id: Optional[str] = None,
        roll_up_to_customer: bool = True,
        across_customer: bool = False,
        allowed_instances: Optional[List[str]] = None,
    ) -> Any:
        """Combined name resolution + traversal in a single Cypher round-trip.

        ``allowed_instances``: the instances whose records this reader may see
        (the cross-instance visibility policy, resolved). None means default-open
        and the query is exactly what it was before. When given, every node and
        relationship on a path must belong to at least one of them, in the query
        itself: filtering afterwards would let another instance's records use up
        the per-anchor limit and starve the viewer of their own.

        ``across_customer``: walk every graph of ``customer_id`` at once,
        matching on the records' own customer_id instead of one namespace. For
        a reader the instance allows to read across its users, once each user
        has a graph of their own (shared/scope_key).

        Replaces the two-step pattern of _resolve_neo4j_by_name → batch_traverse
        by resolving canonical_name AND traversing in one query, saving ~200ms
        of cross-region latency.

        When ``roll_up_to_customer`` is True (default), a user-scope traversal
        also picks up customer-shared edges (rel.user_id IS NULL). When False,
        the traversal is strict to the requested user_id.

        Returns GraphResult with the same Node/Edge structure as batch_traverse.
        """
        from synap.providers.graph.models import Node, Edge, GraphResult

        if not anchor_names:
            return GraphResult(nodes=[], edges=[], paths=[])

        # user_id rollup branch: include rel.user_id IS NULL only when
        # roll_up_to_customer is on. Always-on: $user_id IS NULL (caller
        # didn't pass a user_id — customer- or client-scope request).
        user_filter = (
            "($user_id IS NULL OR rel.user_id IS NULL OR rel.user_id = $user_id)"
            if roll_up_to_customer
            else "($user_id IS NULL OR rel.user_id = $user_id)"
        )
        # Distinct-return: the only production consumer
        # (_neo4j_graph_discovery) keeps just the distinct node/edge sets —
        # GraphResult.paths has no production reader — but RETURN path
        # re-serialized every node (embeddings included) once per path it
        # appeared in; ~96% of the query's db-hits were ProduceResults.
        # Deduping in Cypher ships each node/edge exactly once. It cuts
        # ProduceResults db-hits by roughly 3 to 5x, server-side time is ~2x lower, and
        # the node-id/edge-id sets are identical (incl. endpoint hydration:
        # every rel's endpoints travel in the same record's nodes list).
        # SYNAP_GRAPH_DISTINCT_RETURN=0 reverts to the legacy per-path
        # query; remove the legacy branch after a release of soak.
        distinct_return = os.environ.get(
            "SYNAP_GRAPH_DISTINCT_RETURN", "true"
        ).lower() in ("true", "1", "yes")

        if across_customer and customer_id:
            # A path never leaves one graph (a relationship joins two nodes of
            # one namespace), so matching on the customer walks each of the
            # customer's graphs separately. One name can be a node in many of
            # them, hence the wider anchor limit.
            anchor_match = "{customer_id: $customer_id, canonical_name: name}"
            node_check = "node.customer_id = $customer_id"
            anchor_limit = 25
        else:
            anchor_match = "{_namespace: $ns, canonical_name: name}"
            node_check = "node._namespace = $ns"
            anchor_limit = 3

        # Cross-instance visibility, pushed into the query. Empty strings when
        # default-open, so the default query text is unchanged.
        anchor_where = ""
        rel_extra = ""
        if allowed_instances is not None:
            _iid = graph_boundary.INSTANCE_IDS
            node_check += f" AND ANY(i IN coalesce(node.{_iid}, []) WHERE i IN $allowed_instances)"
            rel_extra = f" AND ANY(i IN coalesce(rel.{_iid}, []) WHERE i IN $allowed_instances)"
            anchor_where = f" WHERE ANY(i IN coalesce(anchor.{_iid}, []) WHERE i IN $allowed_instances)"

        if distinct_return:
            cypher = f"""
                UNWIND $names AS name
                CALL (name) {{
                    MATCH (anchor:SynapNode {anchor_match}){anchor_where}
                    WITH anchor LIMIT {anchor_limit}
                    MATCH path = (anchor)-[*1..{max_depth}]-(m)
                    WHERE ALL(node IN nodes(path) WHERE {node_check})
                      AND ALL(rel IN relationships(path) WHERE
                        (rel.customer_id IS NULL OR rel.customer_id = $customer_id)
                        AND {user_filter}{rel_extra}
                      )
                    WITH path
                    LIMIT $per_anchor_limit
                    RETURN path
                }}
                WITH collect(path) AS paths
                CALL (paths) {{
                    UNWIND paths AS p
                    UNWIND nodes(p) AS n
                    RETURN collect(DISTINCT n) AS nodes
                }}
                CALL (paths) {{
                    UNWIND paths AS p
                    UNWIND relationships(p) AS r
                    RETURN collect(DISTINCT r) AS rels
                }}
                RETURN nodes, rels
            """
        else:
            cypher = f"""
                UNWIND $names AS name
                CALL (name) {{
                    MATCH (anchor:SynapNode {anchor_match}){anchor_where}
                    WITH anchor LIMIT {anchor_limit}
                    MATCH path = (anchor)-[*1..{max_depth}]-(m)
                    WHERE ALL(node IN nodes(path) WHERE {node_check})
                      AND ALL(rel IN relationships(path) WHERE
                        (rel.customer_id IS NULL OR rel.customer_id = $customer_id)
                        AND {user_filter}{rel_extra}
                      )
                    RETURN path
                    LIMIT $per_anchor_limit
                }}
                RETURN path
            """

        params = {
            "ns": namespace,
            "names": anchor_names,
            "per_anchor_limit": per_anchor_limit,
            "customer_id": customer_id,
            "user_id": user_id,
        }
        if allowed_instances is not None:
            params["allowed_instances"] = [str(i) for i in allowed_instances]

        def _sync_traverse():
            from neo4j import unit_of_work
            from synap.providers.graph.adapters.neo4j import (
                neo4j_profile_enabled,
                summarize_profile,
            )

            driver = self._provider._get_driver()
            profile = neo4j_profile_enabled()
            run_cypher = ("PROFILE " + cypher) if profile else cypher
            # PR-2: run through execute_read for automatic retry (T1.3) + a tx
            # timeout (T1.4), matching the adapter's _run_query.
            query_timeout = getattr(self._provider, "_query_timeout", 20.0)
            nodes_map = {}
            edges_map = {}
            graph_paths = []

            def _add_node(node):
                node_data = dict(node)
                node_id = node_data.get("id")
                if node_id and node_id not in nodes_map:
                    labels = list(node.labels)
                    node_type = next((l for l in labels if l != "SynapNode"), "Unknown")
                    properties = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}
                    nodes_map[node_id] = Node(id=node_id, node_type=node_type, properties=properties)
                return node_id

            def _add_edge(rel):
                rel_data = dict(rel)
                edge_id = rel_data.get("id")
                if edge_id and edge_id not in edges_map:
                    properties = {k: v for k, v in rel_data.items() if not k.startswith("_") and k != "id"}
                    edges_map[edge_id] = Edge(
                        id=edge_id,
                        from_id=dict(rel.start_node).get("id"),
                        to_id=dict(rel.end_node).get("id"),
                        edge_type=rel.type,
                        properties=properties,
                    )

            @unit_of_work(timeout=query_timeout)
            def _work(tx):
                # Reset accumulators so a transient-error retry starts clean
                # (execute_read re-runs this function; graph_paths would otherwise
                # accumulate duplicates across attempts).
                nodes_map.clear()
                edges_map.clear()
                graph_paths.clear()
                result = tx.run(run_cypher, params)
                if distinct_return:
                    # Single row: the deduped node/rel lists. The zero-match case
                    # still aggregates to one row of empty lists; the guard covers
                    # a defensive None.
                    record = result.single()
                    if record:
                        for node in (record.get("nodes") or []):
                            _add_node(node)
                        for rel in (record.get("rels") or []):
                            _add_edge(rel)
                else:
                    for record in result:
                        path = record.get("path")
                        if not path:
                            continue

                        path_ids = []
                        for node in path.nodes:
                            node_id = _add_node(node)
                            if node_id:
                                path_ids.append(node_id)

                        for rel in path.relationships:
                            _add_edge(rel)

                        if path_ids:
                            graph_paths.append(path_ids)

                return result.consume().profile if profile else None

            try:
                with driver.session(database=self._provider._database) as session:
                    prof = session.execute_read(_work)
                if prof:
                    logger.info(
                        f"neo4j_profile label=traverse ns={namespace} "
                        f"names={len(anchor_names)} {summarize_profile(prof)}"
                    )
            except Exception as e:
                logger.error(f"Neo4j combined_name_traverse failed: {e}", exc_info=True)

            logger.info(
                f"Neo4j combined_name_traverse complete: names={len(anchor_names)}, "
                f"nodes={len(nodes_map)}, edges={len(edges_map)}, paths={len(graph_paths)}"
            )

            return GraphResult(
                nodes=list(nodes_map.values()),
                edges=list(edges_map.values()),
                paths=graph_paths,
            )

        return await asyncio.to_thread(_sync_traverse)

    async def _find_edges_with_memory(
        self,
        memory_id: str,
        scope: Scope,
    ) -> List[Dict[str, Any]]:
        """Find all edges whose memory_contents JSON contains a given memory_id.

        Returns list of dicts with 'edge_id' and 'memory_contents' (parsed list).
        """
        namespace = self._get_namespace(scope)
        # Anchor on the namespace-indexed SynapNode instead
        # of a label-less relationship pattern. The old
        # `MATCH ()-[r {_namespace: $ns}]->()` planned as a DirectedAllRelationshipsScan
        # over EVERY relationship in the DB (cross-tenant + cross-product on shared
        # infra), which costs tens of thousands of dbHits per call in PROFILE.
        # Anchoring on `(:SynapNode {_namespace})` (backed by namespace_idx) makes
        # the planner NodeIndexSeek → Expand, bounding the scan to this tenant and
        # keeping it flat as the overall DB grows. Results are identical: every edge
        # is created between two same-namespace SynapNodes.
        # See Neo4j_Implementation_Improvement_Design_v1.md (T1.2).
        rows = await asyncio.to_thread(
            self._provider._run_query,
            """
            MATCH (a:SynapNode {_namespace: $ns})-[r]->()
            WHERE r._namespace = $ns AND r.memory_contents CONTAINS $memory_id
            RETURN r.id AS edge_id, r.memory_contents AS mc
            """,
            {"ns": namespace, "memory_id": memory_id},
        )

        results = []
        for row in rows:
            edge_id = row.get("edge_id") if isinstance(row, dict) else row[0]
            mc_raw = row.get("mc") if isinstance(row, dict) else row[1]
            try:
                mc_list = json.loads(mc_raw) if isinstance(mc_raw, str) else mc_raw
            except (json.JSONDecodeError, TypeError):
                mc_list = []
            if isinstance(mc_list, list):
                results.append({"edge_id": edge_id, "memory_contents": mc_list})
        return results

    async def update_memory_on_edges(
        self,
        memory_id: str,
        new_content: str,
        new_memory_type: str,
        new_confidence: float,
        scope: Scope,
    ) -> int:
        """The memory's text was edited: the graph records extracted from the
        OLD text no longer have its support.

        A thin wrapper over ``GraphLifecycle.release`` (mode ``retire``, reason
        ``updated``), so it works in references mode (``memory_ids``) and in
        copies mode (``memory_contents``) alike. It used to rewrite only copies,
        found nothing in references mode and changed nothing, so a sentence
        built from the old text kept being served. The new text is searchable
        through the vector store at once; graph lines are not re-extracted from
        it here (owner decision 3). Returns how many records were touched.
        """
        from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle
        result = await GraphLifecycle(self).release(
            scope, [str(memory_id)], mode="retire", reason="updated")
        return result.edges_pruned + result.nodes_pruned + result.legacy_entries_marked

    async def soft_delete_memory_from_edges(
        self,
        memory_id: str,
        scope: Scope,
        redact: bool = False,
    ) -> int:
        """The memory was deleted: erase its graph support.

        A thin wrapper over ``GraphLifecycle.release`` (mode ``erase``, reason
        ``deleted``). The id is removed from every relationship and node; a
        record no live memory supports any more is deleted, with the nodes it
        leaves orphaned (never the writer's or customer's node). In copies mode
        the entry is marked ``is_deleted``; ``redact=True`` (forget) also blanks
        its text and embedding. It used to understand only copies mode, found
        nothing in references mode and reported success. Graph failures raise.
        Returns how many records were touched.
        """
        from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle
        result = await GraphLifecycle(self).release(
            scope, [str(memory_id)], mode="erase", reason="deleted", redact=redact)
        return (result.edges_pruned + result.nodes_pruned + result.legacy_entries_marked)

    async def restore_memory_on_edges(
        self,
        memory_id: str,
        scope: Scope,
        edge_id: str = None,
        memory_contents: list = None,
    ) -> int:
        """Inverse of soft_delete_memory_from_edges (consolidation C3/J7;
        wired 2026-08-12 audit — graph_image called this and a hasattr guard
        silently no-opped its absence).

        Two modes: with ``edge_id`` + ``memory_contents`` (from the S3
        before-image) the edge's entries are restored EXACTLY as captured;
        without them, every edge entry for the memory gets its deleted flags
        flipped back (re-derivation mode).
        """
        namespace = self._get_namespace(scope)

        if edge_id is not None and memory_contents is not None:
            try:
                await asyncio.to_thread(
                    self._provider.update_edge,
                    namespace,
                    edge_id,
                    {"memory_contents": json.dumps(memory_contents)},
                )
                return 1
            except Exception as e:
                logger.warning(
                    f"Exact edge restore failed edge_id={edge_id} "
                    f"memory_id={memory_id} error={e}")
                return 0

        edges = await self._find_edges_with_memory(memory_id, scope)
        restored_count = 0
        for edge_data in edges:
            mc_list = edge_data["memory_contents"]
            modified = False
            for entry in mc_list:
                if entry.get("memory_id") == memory_id and entry.get("is_deleted"):
                    entry["is_deleted"] = False
                    entry["is_current"] = True
                    entry.pop("deleted_at", None)
                    modified = True
            if modified:
                try:
                    await asyncio.to_thread(
                        self._provider.update_edge,
                        namespace,
                        edge_data["edge_id"],
                        {"memory_contents": json.dumps(mc_list)},
                    )
                    restored_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to restore memory on edge "
                        f"edge_id={edge_data['edge_id']} memory_id={memory_id} error={e}")
        logger.info(
            f"Memory restored on relationship edges memory_id={memory_id} "
            f"edges_restored={restored_count}")
        return restored_count
