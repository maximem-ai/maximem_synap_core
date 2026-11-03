"""Entity resolution service wrapping synap-er PipelineResolver."""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from synap_er.memory.index_manager import ScopeKey
from synap.cloud.agents.ingestion.resolution.registry import WriteThroughManager
from synap.cloud.agents.ingestion.resolution.registry import EntityType
from synap_er.pipeline.resolver import PipelineResolver

from synap.cloud.agents.ingestion.models import ExtractedEntity, Scope
from synap.cloud.agents.ingestion.resolution.descriptions import (
    ABOUT_KEY,
    NEW_FACTS_KEY,
    descriptions_enabled,
    render,
    with_new_fact,
    with_phrase,
)
from synap.cloud.agents.ingestion.resolution.global_dictionary import (
    SOURCE_EXTRACTION,
    SOURCE_JUDGE,
)
from synap.cloud.agents.ingestion.resolution.holds import (
    FOUNDING_HOLD_KEY,
    HOLD_ID_PROPERTY,
    JOINED_BY_NAME,
    JOINED_BY_PROPERTY,
    PENDING_ENTRY_PROPERTY,
    PENDING_KEY,
    hold_unsure_enabled,
    is_pending_entry,
    new_hold_id,
    split_held,
)
from synap.cloud.agents.ingestion.resolution.merge_log import (
    DECIDED_BY_SECOND_LOOK,
    MERGE_ID_PROPERTY,
    new_merge_id,
)
from synap.cloud.agents.ingestion.resolution.mode import adjudicated_mode, node_keys_enabled
from synap.cloud.agents.ingestion.resolution.naming import (
    RENAMED_FROM_PROPERTY,
    stamp_vector_name,
    upgrade_name,
)
from synap.cloud.agents.ingestion.resolution.node_key import (
    NODE_KEY_PROPERTY,
    new_provisional_key,
)
from synap.cloud.memory_config.config_schema import GraphConfig

logger = logging.getLogger(__name__)

# Tries at saving a held mention's links before the note's ingestion is failed.
HOLD_LINK_ATTEMPTS = 3

# Maps codebase entity_type strings to synap-er EntityType enum.
ENTITY_TYPE_MAP: Dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "org": EntityType.ORG,
    "organization": EntityType.ORG,
    "company": EntityType.ORG,
    "term": EntityType.TERM,
    "concept": EntityType.TERM,
    "product": EntityType.DOMAIN_ENTITY,
    "technology": EntityType.DOMAIN_ENTITY,
    "location": EntityType.DOMAIN_ENTITY,
    "team": EntityType.ORG,
    "department": EntityType.ORG,
}


class EntityResolutionService:
    """Wraps synap-er PipelineResolver for use in the ingestion pipeline.

    Resolves extracted entity mentions to canonical names via a multi-stage
    pipeline (exact, fuzzy, semantic, contextual, LLM-optional, manual review).
    """

    def __init__(
        self,
        resolver: PipelineResolver,
        write_through: WriteThroughManager,
        enabled: bool = True,
        merge_similar: bool = True,
        similarity_threshold: float = 0.9,
        adjudicator: Optional[Any] = None,
        review_sink: Optional[Any] = None,
        dictionary: Optional[Any] = None,
        merge_log: Optional[Any] = None,
        world_lookup: Optional[Any] = None,
        index_manager: Optional[Any] = None,
        hold_store: Optional[Any] = None,
        conflict_checker: Optional[Any] = None,
        profiler: Optional[Any] = None,
    ):
        self._resolver = resolver
        self._write_through = write_through
        self._enabled = enabled
        self._merge_similar = merge_similar
        self._similarity_threshold = similarity_threshold
        # Adjudicated mode only. Both optional: with no adjudicator synap-er
        # decides by score thresholds, with no sink ambiguous mentions are
        # tagged on the entity but not queued.
        self._adjudicator = adjudicator
        self._review_sink = review_sink
        # The global dictionary (names generally accepted to mean the same)
        # and the log of merges that can be undone. Both optional.
        self._dictionary = dictionary
        self._merge_log = merge_log
        # Asks an LLM what else a new name is widely known by. Optional.
        self._world_lookup = world_lookup
        # Read access to registry entries, to add to their descriptions. Optional.
        self._index_manager = index_manager
        # Where unsure mentions wait, with their links (see holds.py). With
        # none, an unsure mention gets a placeholder node as before.
        self._hold_store = hold_store
        # Checks an exact-name match against the entry's description. Optional.
        self._conflict_checker = conflict_checker
        # Keeps each entry's profile up to date in the background. Optional.
        self._profiler = profiler

    @classmethod
    def from_config(
        cls,
        graph_config: GraphConfig,
        resolver: PipelineResolver,
        write_through: WriteThroughManager,
        adjudicator: Optional[Any] = None,
        review_sink: Optional[Any] = None,
        dictionary: Optional[Any] = None,
        merge_log: Optional[Any] = None,
        world_lookup: Optional[Any] = None,
        index_manager: Optional[Any] = None,
        hold_store: Optional[Any] = None,
        conflict_checker: Optional[Any] = None,
        profiler: Optional[Any] = None,
    ) -> "EntityResolutionService":
        """Create from GraphConfig's entity_canonicalization dict."""
        canon_config = graph_config.entity_canonicalization
        return cls(
            resolver=resolver,
            write_through=write_through,
            adjudicator=adjudicator,
            review_sink=review_sink,
            dictionary=dictionary,
            merge_log=merge_log,
            world_lookup=world_lookup,
            index_manager=index_manager,
            hold_store=hold_store,
            conflict_checker=conflict_checker,
            profiler=profiler,
            enabled=canon_config.get("enabled", True),
            merge_similar=canon_config.get("merge_similar_entities", True),
            similarity_threshold=canon_config.get("similarity_threshold", 0.9),
        )

    async def resolve_entities(
        self,
        entities: List[ExtractedEntity],
        scope: Scope,
        context: Optional[str] = None,
    ) -> List[ExtractedEntity]:
        """Resolve each entity through synap-er, setting canonical_name.

        If disabled or resolution fails, returns entities unchanged
        (canonical_name stays None, downstream falls back to entity.name).
        """
        if not self._enabled:
            return entities

        # The writer of the document is one node per user, keyed on the user
        # id (services/writer_entity). Matching "the user" by name would merge
        # every user of the customer into one entry.
        from synap.cloud.agents.ingestion.services import writer_entity
        writers = [e for e in entities if writer_entity.is_fixed(e)]
        if writers:
            others = [e for e in entities if not writer_entity.is_fixed(e)]
            return (await self.resolve_entities(others, scope, context)) + writers

        scope_chain = _build_scope_chain(scope)

        if adjudicated_mode() and hasattr(self._resolver, "resolve_many"):
            try:
                return await self._resolve_adjudicated(entities, scope, scope_chain, context)
            except Exception:
                # Resolution must never fail ingestion. Entities go through
                # unresolved, exactly as when the library is not installed.
                logger.warning("Adjudicated entity resolution failed, skipping", exc_info=True)
                return entities

        for entity in entities:
            try:
                entity_type = _map_entity_type(entity.entity_type)
                resolved = await self._resolver.resolve(
                    mention=entity.name,
                    entity_type=entity_type,
                    scope_chain=scope_chain,
                    context=context,
                )

                if resolved.resolved and resolved.confidence >= self._similarity_threshold:
                    entity.canonical_name = resolved.canonical_name
                    entity.properties["er_canonical_id"] = str(resolved.canonical_id)
                    entity.properties["er_confidence"] = resolved.confidence
                    entity.properties["er_resolution_path"] = resolved.resolution_path
                else:
                    # Not resolved — auto-register for future lookups
                    await self._auto_register(entity, scope, entity_type)
            except Exception:
                logger.warning(
                    "Entity resolution failed for %r, skipping",
                    entity.name,
                    exc_info=True,
                )

        return entities

    async def _resolve_adjudicated(
        self,
        entities: List[ExtractedEntity],
        scope: Scope,
        scope_chain: List[ScopeKey],
        context: Optional[str],
    ) -> List[ExtractedEntity]:
        """One resolve_many call for the document, then act on each decision.

        MATCH      use the canonical name and record this spelling as an alias
        NEW        register the entity
        AMBIGUOUS  hold it (see holds.py) and queue it for review. Nothing is
                   written to the graph for it. Its name goes in the registry
                   only as a pending entry, which later notes can see and
                   nothing can merge into. A MATCH to a pending entry is held
                   in that entry's group, for the same reason.
        """
        from synap_er.pipeline.adjudication import Decision, MentionInput

        entities, hints = _split_unverified_aliases(entities)
        types = [_map_entity_type(e.entity_type) for e in entities]

        # What the extraction model knows these names by in general goes into
        # the global dictionary first, so this very document can use it.
        if self._dictionary is not None:
            for entity, entity_type in zip(entities, types):
                for other in entity.known_as:
                    await self._dictionary.learn(
                        entity.name, other, entity_type, SOURCE_EXTRACTION, note=context
                    )

        mentions = [
            MentionInput(
                e.name, t, aliases=list(e.aliases), subtype=e.entity_type,
                candidate_indexes=hints.get(i, []),
            )
            for i, (e, t) in enumerate(zip(entities, types))
        ]
        resolutions = await self._resolve_many(mentions, scope_chain, context)

        if self._world_lookup is not None:
            # A name that came out new may be a widely known one that is on
            # file under another name. Ask once; if anything was learned, the
            # dictionary now offers that entry and the document is resolved again.
            fresh = [
                (e.name, t) for e, t, r in zip(entities, types, resolutions)
                if r.decision is Decision.NEW
            ]
            if fresh and await self._world_lookup.lookup(fresh, context):
                resolutions = await self._resolve_many(mentions, scope_chain, context)
        # A name with an identifier ("INV-7802", "bexley-main") is the same
        # thing as a candidate exactly when they share it (resolution/identifiers).
        from synap.cloud.agents.ingestion.resolution import identifiers
        if identifiers.enabled(getattr(scope, "instance_id", None)) and self._index_manager is not None:
            for entity, res in zip(entities, resolutions):
                settled = identifiers.settle(
                    res, [entity.name, *entity.aliases], entity.entity_type, self._index_manager.get_entry)
                if settled:
                    logger.info("Entity resolution: %r settled by identifier: %s", entity.name, settled)
                elif identifiers.keep_apart(res, [entity.name, *entity.aliases], entity.entity_type,
                                            self._index_manager.get_entry):
                    logger.info("Entity resolution: %r kept apart from a related thing of another kind",
                                entity.name)
            # The extraction returned these as two entities of different kinds
            # ("Lumavate", a drug, and "Lumavate refills", a refill): the note
            # itself says they are not one thing.
            from synap_er.pipeline.adjudication import Decision as _D
            for i, (entity, res) in enumerate(zip(entities, resolutions)):
                j = res.same_as_index
                if j is not None and j != i and identifiers._kind(entities[j].entity_type) != identifiers._kind(entity.entity_type):
                    res.decision, res.same_as_index, res.path = _D.NEW, None, "different_kind_in_note"
        # index -> what the second look decided for that mention, if it ran
        second_look = getattr(self._adjudicator, "last", None) or {}

        # index -> (canonical_id, canonical_name) of what was registered for a NEW mention
        registered: Dict[int, tuple] = {}
        counts: Dict[str, int] = {}
        keyed = node_keys_enabled()
        holding = (
            self._hold_store is not None and self._index_manager is not None
            and hold_unsure_enabled()
        )

        conflicts = await self._exact_conflicts(resolutions, context) if holding else {}

        for i, (entity, entity_type, res) in enumerate(zip(entities, types, resolutions)):
            counts[res.decision.value] = counts.get(res.decision.value, 0) + 1
            entity.properties["er_decision"] = res.decision.value.lower()
            entity.properties["er_resolution_path"] = [res.path]

            if res.decision is Decision.NEW:
                entry = await self._auto_register(entity, scope, entity_type, context)
                if entry is not None:
                    self._profile(entry.canonical_id, scope)
                    registered[i] = (entry.canonical_id, entry.canonical_name)
                    entity.properties["er_canonical_id"] = str(entry.canonical_id)
                    if keyed:
                        entity.properties[NODE_KEY_PROPERTY] = str(entry.canonical_id)
                continue

            target = None
            if res.canonical_id is not None:
                target = (res.canonical_id, res.canonical_name)
            elif res.same_as_index is not None:
                # Same entity as an earlier mention of this document.
                target = registered.get(res.same_as_index)

            group_id, conflict = None, None
            if i in conflicts and target is not None:
                # Spelled like an entry on file, but the note cannot be about it.
                conflict, target = conflicts[i], None
            if holding and target is not None and self._is_pending(target[0]):
                # Matched a name that is itself waiting for review.
                group_id, target = str(target[0]), None
                if res.path in ("exact", "normalized"):
                    entity.properties[JOINED_BY_PROPERTY] = JOINED_BY_NAME

            if res.decision is Decision.MATCH and target is not None:
                canonical_id, canonical_name = target
                entity.canonical_name = canonical_name
                entity.properties["er_canonical_id"] = str(canonical_id)
                known_type = getattr(res, "canonical_subtype", None)
                if known_type and known_type != entity.entity_type.lower().strip():
                    # The entry was typed when it was first seen in full
                    # ("Rajesh Kumar Verma", a person); a later "RKV" typed as
                    # an organisation must not relabel it.
                    entity.properties["er_extracted_type"] = entity.entity_type
                    entity.entity_type = known_type
                if keyed:
                    entity.properties[NODE_KEY_PROPERTY] = str(canonical_id)
                self._stamp_vector_name(canonical_id, entity)
                added = []
                for surface in [entity.name] + list(entity.aliases):
                    if await self._learn_alias(canonical_id, canonical_name, surface):
                        added.append(surface)
                looked = second_look.get(str(i)) if res.path == "adjudicated" else None
                if looked and looked.get("outcome") == "merge":
                    # Not a certain merge: log it so it can be taken back.
                    await self._log_merge(
                        entity, canonical_id, canonical_name, added, scope, looked
                    )
                else:
                    # Only a certain match renames the entry to a fuller name,
                    # or adds to what is known about it: a merge that may be
                    # undone must do neither.
                    renamed = await self._upgrade_name(canonical_id, entity, scope)
                    if renamed:
                        entity.properties[RENAMED_FROM_PROPERTY] = canonical_name
                        entity.canonical_name = renamed
                    await self._describe(canonical_id, entity, scope)
                    self._profile(canonical_id, scope)
                    if res.path == "adjudicated" and self._dictionary is not None:
                        # A judge confirmed two forms of one person's name: the
                        # given-name pair is general knowledge ("Bob" / "Robert").
                        await self._dictionary.learn(
                            entity.name, canonical_name, entity_type, SOURCE_JUDGE, note=None
                        )
                continue

            # AMBIGUOUS (or a MATCH whose target could not be registered).
            entity.properties["er_decision"] = Decision.AMBIGUOUS.value.lower()
            earlier = entities[res.same_as_index] if res.same_as_index is not None else None
            entity.properties["er_candidates"] = [c.canonical_name for c in res.candidates[:5]]
            if holding:
                if group_id is None and earlier is not None:
                    group_id = earlier.properties.get(PENDING_ENTRY_PROPERTY)
                if await self._hold(entity, entity_type, res, scope, context, group_id, conflict):
                    await self._queue_review(entity, entity_type, res, scope, context)
                    continue
            if earlier is not None:
                # Shares the unresolved node of the earlier mention.
                entity.canonical_name = earlier.canonical_name or earlier.name
            if keyed:
                # Its own node until someone settles it: an unresolved "R. Smith"
                # must not land on another unresolved "R. Smith".
                entity.properties[NODE_KEY_PROPERTY] = (
                    earlier.properties.get(NODE_KEY_PROPERTY) if earlier is not None else None
                ) or new_provisional_key()
            await self._queue_review(entity, entity_type, res, scope, context)

        logger.info(
            "Entity resolution (adjudicated): entities=%d match=%d new=%d ambiguous=%d adjudicator=%s",
            len(entities),
            counts.get("MATCH", 0), counts.get("NEW", 0), counts.get("AMBIGUOUS", 0),
            type(self._adjudicator).__name__ if self._adjudicator else "none",
        )
        return entities

    async def _exact_conflicts(self, resolutions: List[Any], context: Optional[str]) -> Dict[int, dict]:
        """index -> the entry an exact-name match was refused for.

        Only matches made in code on the spelling alone, to an entry that has
        a description to compare the note with.
        """
        from synap_er.pipeline.adjudication import Decision

        if self._conflict_checker is None or not descriptions_enabled():
            return {}
        asked: Dict[str, dict] = {}
        pairs = []
        for i, res in enumerate(resolutions):
            if res.decision is not Decision.MATCH or res.path not in ("exact", "normalized"):
                continue
            if res.canonical_id is None:
                continue
            try:
                entry = self._index_manager.get_entry(res.canonical_id)
            except Exception:
                entry = None
            description = render(entry.metadata_) if entry is not None else None
            if not description or is_pending_entry(entry):
                continue
            asked[str(i)] = {
                "canonical_id": str(res.canonical_id),
                "canonical_name": res.canonical_name,
                "verdict": "CONFLICTS_WITH_DESCRIPTION",
                "pending": False,
            }
            pairs.append((str(i), res.mention, description))
        if not pairs:
            return {}
        found = await self._conflict_checker.conflicting(pairs, context)
        return {int(key): asked[key] for key in found}

    def _is_pending(self, canonical_id: Any) -> bool:
        try:
            return is_pending_entry(self._index_manager.get_entry(canonical_id))
        except Exception:
            return False

    async def _hold(
        self,
        entity: ExtractedEntity,
        entity_type: EntityType,
        res: Any,
        scope: Scope,
        context: Optional[str],
        group_id: Optional[str],
        conflict: Optional[dict] = None,
    ) -> bool:
        """Hold an unsure mention. False when it could not be recorded, in
        which case the caller falls back to a placeholder node."""
        hold_id = new_hold_id()
        try:
            if group_id is None:
                # Seen for the first time: a pending entry makes the name
                # visible to later notes without making it an entity.
                pending = await self._write_through.create_entry({
                    "canonical_name": entity.name,
                    "entity_type": entity_type.value,
                    **_scope_fields(scope),
                    "source": "RESOLVED",
                    "confidence": 0.0,
                    "aliases": [],
                    "metadata_": {
                        PENDING_KEY: True,
                        FOUNDING_HOLD_KEY: hold_id,
                        "entity_subtype": entity.entity_type.lower().strip(),
                    },
                })
                group_id = str(pending.canonical_id)
            verdicts = {name: getattr(v, "value", str(v)) for name, v in res.verdicts.items()}
            entity.properties[HOLD_ID_PROPERTY] = hold_id
            entity.properties[PENDING_ENTRY_PROPERTY] = group_id
            await self._hold_store.add({
                "hold_id": hold_id,
                **_scope_fields(scope),
                "pending_entry_id": group_id,
                "mention": entity.name,
                "entity_type": entity.entity_type,
                "joined_by": entity.properties.pop(JOINED_BY_PROPERTY, None),
                "snippet": context_snippet(context, entity.name),
                "candidates": [
                    {
                        "canonical_id": str(c.canonical_id),
                        "canonical_name": c.canonical_name,
                        "verdict": verdicts.get(c.canonical_name),
                        "pending": self._is_pending(c.canonical_id),
                    }
                    for c in res.candidates[:5]
                ] + ([conflict] if conflict else []),
                "tenant": scope.model_dump(mode="json"),
                "entity": entity.model_dump(mode="json"),
            })
            return True
        except Exception:
            entity.properties.pop(HOLD_ID_PROPERTY, None)
            entity.properties.pop(PENDING_ENTRY_PROPERTY, None)
            logger.warning("Could not hold %r; it gets a placeholder node", entity.name, exc_info=True)
            return False

    async def hold_unsure(
        self,
        entities: List[ExtractedEntity],
        relationships: List[Any],
        scope: Scope,
        document_id: Optional[str] = None,
        memories: Optional[List[Any]] = None,
    ) -> tuple:
        """Take held mentions out of a note before anything is persisted.

        Returns (entities, relationships) without the held entities and
        without any relationship that touches one. Those relationships, and
        the note's *memories* that mention a held entity, are saved on the
        hold, to be written when the mention is settled. The memories
        themselves are persisted as always.
        """
        if self._hold_store is None:
            return entities, relationships
        settled, kept, links = split_held(entities, relationships)
        held_ids = {
            e.properties[HOLD_ID_PROPERTY]: e.entity_id
            for e in entities if e.properties.get(HOLD_ID_PROPERTY)
        }
        for hold_id, mine in links.items():
            mentioning = [
                m.model_dump(mode="json") for m in memories or []
                if held_ids.get(hold_id) in (m.entities_mentioned or [])
            ]
            await self._save_links(hold_id, mine, document_id, mentioning)
        return settled, kept

    async def _save_links(self, hold_id: str, links: List[dict], document_id: Optional[str],
                          memories: List[dict]) -> None:
        """Keep a held mention's links and memories on its hold.

        The mention and its relationships are already out of the note, so this
        is the only copy. A failure that survives a few tries is raised, to
        fail the stage and let the note be ingested again; swallowing it would
        lose the links for good.
        """
        attempts = HOLD_LINK_ATTEMPTS
        for attempt in range(1, attempts + 1):
            try:
                await self._hold_store.set_links(hold_id, links, document_id, memories)
                return
            except Exception:
                if attempt == attempts:
                    logger.error("Could not save links of hold %s after %d tries", hold_id, attempts,
                                 exc_info=True)
                    raise
                logger.warning("Could not save links of hold %s (try %d of %d)", hold_id, attempt,
                               attempts, exc_info=True)
                await asyncio.sleep(0.2 * attempt)

    async def _resolve_many(self, mentions: List[Any], scope_chain: List[ScopeKey],
                            context: Optional[str]) -> List[Any]:
        if hasattr(self._adjudicator, "last"):
            # resolve_many does not call the adjudicator when every mention is
            # settled in code; a previous document's answers must not be read.
            self._adjudicator.last = {}
        return await self._resolver.resolve_many(
            mentions, scope_chain, context=context, adjudicator=self._adjudicator,
        )

    async def _learn_alias(self, canonical_id: Any, canonical_name: str, surface: str) -> bool:
        """Record a new spelling so its next occurrence is an exact hit."""
        if surface.strip().lower() == canonical_name.strip().lower():
            return False
        add_alias = getattr(self._write_through, "add_alias", None)
        if add_alias is None:  # synap-er older than 0.2.0
            return False
        try:
            await add_alias(canonical_id, surface)
            return True
        except Exception:
            logger.warning("Could not record alias %r for %r", surface, canonical_name, exc_info=True)
            return False

    async def _describe(self, canonical_id: Any, entity: ExtractedEntity,
                        scope: Optional[Scope] = None) -> None:
        """Add what this note says about the entity to its registry entry."""
        if not entity.about or self._index_manager is None or not descriptions_enabled():
            return
        try:
            entry = self._index_manager.get_entry(canonical_id)
            if entry is None:
                return
            if scope is not None and scoped_registry_enabled(scope.instance_id):
                # The entry was found at a level above the note (a user's note
                # naming something the customer level already holds). What a
                # narrower note says must not be written onto the wider entry,
                # where every other reader of that level would be shown it.
                held_at = (getattr(entry.scope, "value", entry.scope), entry.scope_id)
                if held_at != registry_scope(scope):
                    return
            metadata = dict(entry.metadata_ or {})
            changes = {}
            phrases = with_phrase(metadata.get(ABOUT_KEY), entity.about)
            if phrases is not None:
                changes[ABOUT_KEY] = phrases
            if self._profiler is not None:
                waiting = with_new_fact(metadata, entity.about)
                if waiting is not None:
                    changes[NEW_FACTS_KEY] = waiting
            if changes:
                await self._write_through.update_entry(
                    canonical_id, {"metadata_": {**metadata, **changes}}
                )
        except Exception:
            logger.warning("Could not describe %r", entity.name, exc_info=True)

    def _entry(self, canonical_id: Any) -> Optional[Any]:
        if self._index_manager is None:
            return None
        try:
            return self._index_manager.get_entry(canonical_id)
        except Exception:
            return None

    async def _upgrade_name(self, canonical_id: Any, entity: ExtractedEntity,
                            scope: Optional[Scope] = None) -> Optional[str]:
        """Rename the entry when this mention names it more fully. See naming.py.

        Never to the name of a mention of another kind (er_extracted_type is
        set when the match relabelled it): "Chordline refund policy" is not a
        fuller name for the organization "Chordline" (SYNAP_ER_IDENTIFIERS)."""
        from synap.cloud.agents.ingestion.resolution import identifiers
        if entity.properties.get("er_extracted_type") and identifiers.enabled(
                getattr(scope, "instance_id", None)):
            return None
        try:
            renamed = await upgrade_name(
                self._write_through, self._entry(canonical_id), [entity.name] + list(entity.aliases)
            )
        except Exception:
            logger.warning("Could not rename the entry for %r", entity.name, exc_info=True)
            return None
        if renamed:
            self._stamp_vector_name(canonical_id, entity)
        return renamed

    def _stamp_vector_name(self, canonical_id: Any, entity: ExtractedEntity) -> None:
        stamp_vector_name(entity, self._entry(canonical_id))

    def _profile(self, canonical_id: Any, scope: Scope) -> None:
        """Tell the profiler something certain was learned about an entry."""
        if self._profiler is None:
            return
        try:
            self._profiler.note(canonical_id, scope)
        except Exception:
            logger.warning("Could not schedule a profile refresh", exc_info=True)

    async def _log_merge(
        self,
        entity: ExtractedEntity,
        canonical_id: Any,
        canonical_name: str,
        aliases_added: List[str],
        scope: Scope,
        looked: Dict[str, Any],
    ) -> None:
        """Stamp the entity and write the row an undo reads."""
        merge_id = new_merge_id()
        entity.properties[MERGE_ID_PROPERTY] = merge_id
        entity.properties["er_decided_by"] = DECIDED_BY_SECOND_LOOK
        if self._merge_log is None:
            return
        try:
            await self._merge_log.record(
                merge_id=merge_id,
                **_scope_fields(scope),
                into_canonical_id=str(canonical_id),
                into_canonical_name=canonical_name,
                surface=entity.name,
                entity_type=entity.entity_type,
                aliases_added=aliases_added,
                decided_by=DECIDED_BY_SECOND_LOOK,
                evidence={
                    k: looked[k] for k in ("first", "second", "about", "recheck", "by")
                    if looked.get(k) is not None
                },
            )
        except Exception:
            logger.warning("Could not log merge of %r into %r", entity.name, canonical_name, exc_info=True)

    async def _queue_review(
        self,
        entity: ExtractedEntity,
        entity_type: EntityType,
        res: Any,
        scope: Scope,
        context: Optional[str],
    ) -> None:
        if self._review_sink is None:
            return
        verdicts = {name: getattr(v, "value", str(v)) for name, v in res.verdicts.items()}
        candidates = [
            {
                "canonical_id": str(c.canonical_id),
                "canonical_name": c.canonical_name,
                "confidence": round(c.confidence, 3),
                "source_stage": c.source_stage,
                "verdict": verdicts.get(c.canonical_name),
            }
            for c in res.candidates[:5]
        ]
        try:
            await self._review_sink.enqueue(
                mention=entity.name,
                entity_type=entity_type.value,
                candidates=candidates,
                snippet=context_snippet(context, entity.name),
                scope="CUSTOMER",
                scope_id=str(scope.customer_id),
            )
        except Exception:
            logger.warning("Could not queue %r for review", entity.name, exc_info=True)

    async def _auto_register(
        self,
        entity: ExtractedEntity,
        scope: Scope,
        entity_type: EntityType,
        context: Optional[str] = None,
    ) -> Optional[Any]:
        """Register newly-seen entity in synap-er registry for future resolution."""
        about = with_phrase(None, entity.about) if descriptions_enabled() else None
        try:
            return await self._write_through.create_entry({
                "canonical_name": entity.name,
                "entity_type": entity_type.value,
                **_scope_fields(scope),
                "source": "RESOLVED",
                "confidence": 1.0,
                "aliases": list(entity.aliases),
                # entity_type above is one of synap-er's four coarse values;
                # the type Synap extracted ("team", "location") is kept beside it.
                "metadata_": {
                    "source_entity_id": entity.entity_id,
                    "entity_subtype": entity.entity_type.lower().strip(),
                    **({ABOUT_KEY: about} if about else {}),
                },
            })
        except Exception:
            # A failed registration means this name is re-judged as brand new
            # on every later sighting, so it must be visible in the logs.
            logger.warning(
                "Auto-register failed for %r, skipping", entity.name, exc_info=True
            )
            return None


ALIAS_OF_PROPERTY = "er_alias_of"


def _split_unverified_aliases(entities: List[ExtractedEntity]) -> tuple:
    """Give each alias that code cannot verify a mention of its own.

    The extraction model says "Bob" is "Robert Smith" from reading the note.
    That is a claim, not a fact: it also said "Jurgen Muller" was "J. Möller"
    when the note meant two people. An alias code can verify (an acronym, an
    abbreviation, part of the name) stays on its entity. Any other alias, and
    every alias of a person, becomes its own entity with the first one offered
    as a candidate, and the adjudicator decides like for any other pair.

    Returns the new entity list and {index: [index of the entity it was
    extracted as an alias of]}.
    """
    from synap_er.normalize import is_form_of

    out: List[ExtractedEntity] = []
    hints: Dict[int, List[int]] = {}
    for entity in entities:
        entity_type = _map_entity_type(entity.entity_type)
        kept, split = [], []
        for alias in entity.aliases:
            (kept if is_form_of(alias, entity.name, entity_type) else split).append(alias)
        entity.aliases = kept
        main = len(out)
        out.append(entity)
        for alias in split:
            hints[len(out)] = [main]
            out.append(ExtractedEntity(
                entity_id=str(uuid.uuid4()),
                entity_type=entity.entity_type,
                name=alias,
                source_span=entity.source_span,
                properties={ALIAS_OF_PROPERTY: entity.entity_id},
            ))
    return out, hints


# A review row carries a window around the mention, never the whole document.
REVIEW_SNIPPET_CHARS = 300


def context_snippet(context: Optional[str], mention: str, width: int = REVIEW_SNIPPET_CHARS) -> Optional[str]:
    if not context:
        return None
    at = context.lower().find(mention.lower())
    if at < 0:
        return context[:width]
    half = max(0, (width - len(mention)) // 2)
    return context[max(0, at - half): at + len(mention) + half]


def _map_entity_type(entity_type_str: str) -> EntityType:
    """Map ingestion entity_type string to synap-er EntityType enum."""
    return ENTITY_TYPE_MAP.get(entity_type_str.lower().strip(), EntityType.DOMAIN_ENTITY)


def scoped_registry_enabled(instance_id: Any = None) -> bool:
    """SYNAP_ER_SCOPED_REGISTRY: an entity is registered at the scope its
    document was written at, not always at the customer. Off by default."""
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for("SYNAP_ER_SCOPED_REGISTRY", instance_id)


# A ladder level that is none of client, customer or user (a team, a case).
# The registry's scope column holds a short fixed word; the level is told
# apart by its node id.
LADDER_SCOPE = "NODE"


def registry_scope(scope: Scope) -> ScopeKey:
    """Where a document written at *scope* registers what it names.

    Always the customer until SYNAP_ER_SCOPED_REGISTRY: a name from one user's
    private note became an entry every user of the customer was matched
    against. With the flag it is the document's own scope, so the entry is as
    private as the note, and two users naming one entity hold an entry each,
    as they hold a graph node each (shared/scope_key).
    """
    if not scoped_registry_enabled(scope.instance_id):
        return ("CUSTOMER", str(scope.customer_id))
    level = getattr(scope, "scope_level", None) or (
        "user" if scope.user_id else "customer" if scope.customer_id else "client")
    if level == "client" or not scope.customer_id:
        return ("CLIENT", str(scope.client_id))
    if level == "user" and scope.user_id:
        return ("USER", str(scope.user_id))
    if level == "customer":
        return ("CUSTOMER", str(scope.customer_id))
    node_id = _ladder_node(scope, level)
    return (LADDER_SCOPE, node_id) if node_id else ("CUSTOMER", str(scope.customer_id))


def _ladder_node(scope: Scope, level: str) -> Optional[str]:
    for rung in getattr(scope, "chain", None) or []:
        if rung.level_key == level:
            return str(rung.node_id)
    return None


def _scope_fields(scope: Scope) -> Dict[str, str]:
    name, scope_id = registry_scope(scope)
    return {"scope": name, "scope_id": scope_id}


def _build_scope_chain(scope: Scope) -> List[ScopeKey]:
    """Build scope chain narrowest-first: USER -> CUSTOMER -> CLIENT -> WORLD.

    With SYNAP_ER_SCOPED_REGISTRY the chain starts at the scope the document
    is written at and holds only that scope and the levels above it. A
    customer-level document is no longer matched against a user's entries, and
    a document at a ladder level searches that level and each level above it.
    """
    if scoped_registry_enabled(scope.instance_id):
        own = registry_scope(scope)
        chain = [own]
        if own[0] == LADDER_SCOPE:
            rungs = list(getattr(scope, "chain", None) or [])
            ids = [str(r.node_id) for r in rungs]
            known = {str(scope.customer_id), str(scope.client_id), str(scope.user_id)}
            above = ids[: ids.index(own[1])] if own[1] in ids else []
            chain += [(LADDER_SCOPE, i) for i in reversed(above) if i not in known]
        for key in (("CUSTOMER", str(scope.customer_id)) if scope.customer_id else None,
                    ("CLIENT", str(scope.client_id))):
            if key and key not in chain:
                chain.append(key)
        chain.append(("WORLD", None))
        return chain
    chain: List[ScopeKey] = []
    if scope.user_id:
        chain.append(("USER", str(scope.user_id)))
    chain.append(("CUSTOMER", str(scope.customer_id)))
    chain.append(("CLIENT", str(scope.client_id)))
    chain.append(("WORLD", None))
    return chain
