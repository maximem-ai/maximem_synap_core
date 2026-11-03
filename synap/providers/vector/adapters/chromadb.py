"""ChromaDB adapter implementation."""

import concurrent.futures
import math
import logging
import time
from typing import List, Optional, Dict, Any

from ..interface import VectorStoreInterface
from ..models import VectorQuery, VectorDocument, VectorResult, VectorSearchResult
from synap.cloud.shared.logging_utils import log_data_sample, truncate_dict, truncate_string

logger = logging.getLogger(__name__)

# Timeout for chromadb.HttpClient() constructor which may hang if server is unreachable
CONNECT_TIMEOUT_SECONDS = 10


# R-13 / SHIP-09. See create_namespace for the measured curve behind this value.
_HNSW_SEARCH_EF = 200

# How many vectors a collection must accumulate before Chroma writes its HNSW
# index to disk. Chroma's default is 1000, and its index pool holds the only
# copy until then: the pool defaults to 65,536 entries and never evicts at our
# scale, so nothing was ever lost, but the pool is also why memory grows with
# every collection we touch and never comes back.
#
# Measured on the Chroma image in use, dim 1536. A collection is
# readable after eviction only once it holds at least this many vectors:
#
#   1 vector  / threshold 2  -> never on disk -> second read after eviction FAILS
#   2 vectors / threshold 2  -> on disk       -> second read OK
#   3 vectors / threshold 8  -> never on disk -> second read FAILS
#   9 vectors / threshold 8  -> on disk       -> second read OK
#
# Typical collections are small, often only a few vectors.
# So 2, the lowest value Chroma's validator accepts, is the only setting that
# covers them. 8 would leave most small collections fragile.
#
# It is not a cost. Measured on a 2,000-vector collection, dim 1536:
#
#   threshold        batch p50    batch max    disk written
#      2              134.9ms       300.4ms       15.9 MB
#      8              131.6ms       280.7ms       15.9 MB
#   default (1000)    112.2ms       420.2ms       29.9 MB
#
# 20% slower at the median, a lower worst case, and half the disk.
#
# ⚠ This only takes effect at CREATE time and only through collection metadata.
# Passing it in the `configuration` field is accepted and silently discarded,
# and so is updating it on an existing collection. Both measured.
_HNSW_SYNC_THRESHOLD = 2


class ChromaDBAdapter(VectorStoreInterface):
    """
    ChromaDB implementation of vector store interface.

    Features:
    - Namespace = Collection (tenant isolation per instance_id)
    - Cosine similarity by default
    - Metadata filtering
    - Batch upsert/delete
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Config keys:
        - host: str (for HTTP client)
        - port: int (default 8000)
        - persist_directory: str (for persistent local client, optional)
        - auth_token: str (optional)
        """
        self.config = config
        self._client: Optional[Any] = None
        self._chroma_module = None
        self._settings_cls = None
        # Fallback in-memory namespace store for environments where Chroma
        # import/runtime is unavailable (e.g., Python 3.14 + pydantic-v1 paths).
        self._memory_namespaces: Dict[str, Dict[str, VectorDocument]] = {}
        self._use_fallback_memory = False
        # In-process cache of name → Collection handle. ChromaDB's HTTP
        # client resolves a name to a server-side ID via a roundtrip on
        # every ``client.get_collection(name=...)`` call (~500-2600ms
        # observed on a host with 1.5k collections). The mapping is stable
        # for the lifetime of a collection, so caching here turns a 500ms+
        # blocking HTTP call into a dict lookup. Per-call query time
        # (the actual similarity search) is unaffected.
        #
        # Invalidation: ``create_namespace`` and ``delete_namespace`` clear
        # their own entries. External deletion is detected at query time
        # via the ChromaDB error and the cached entry is dropped + retried.
        self._collection_cache: Dict[str, Any] = {}

    def connect(self) -> None:
        """Initialize ChromaDB client."""
        if self._client is not None:
            return

        if self._chroma_module is None:
            try:
                import chromadb as chroma_mod
                from chromadb.config import Settings as chroma_settings
                self._chroma_module = chroma_mod
                self._settings_cls = chroma_settings
            except Exception:
                self._use_fallback_memory = True
                self._client = object()
                return

        host = self.config.get("host")
        port = self.config.get("port", 8000)
        ssl = bool(self.config.get("ssl", False))
        persist_dir = self.config.get("persist_directory")

        # CHROMA_HOST may be a bare hostname (host + CHROMA_PORT) or a full
        # URL such as an https tunnel (https://<sub>.trycloudflare.com). When a
        # scheme is present, derive ssl from it, strip it from the host, and pick
        # the default https port (443) if one wasn't given explicitly.
        if host and "://" in host:
            from urllib.parse import urlparse
            parsed = urlparse(host)
            ssl = parsed.scheme == "https"
            if parsed.port:
                port = parsed.port
            elif ssl:
                port = 443
            host = parsed.hostname or host

        if host:
            # HttpClient() may hang indefinitely if server is unreachable — bound with timeout
            chroma_mod = self._chroma_module
            settings_cls = self._settings_cls
            config = self.config

            def _create_http_client():
                auth_token = config.get("auth_token")
                if auth_token:
                    return chroma_mod.HttpClient(
                        host=host,
                        port=port,
                        ssl=ssl,
                        settings=settings_cls(
                            chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                            chroma_client_auth_credentials=auth_token,
                        ),
                    )
                else:
                    return chroma_mod.HttpClient(host=host, port=port, ssl=ssl)

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(_create_http_client)
            try:
                self._client = future.result(timeout=CONNECT_TIMEOUT_SECONDS)
            finally:
                pool.shutdown(wait=False)

            # chromadb's HttpClient defaults to ``Timeout(timeout=None)`` — i.e.
            # infinite read. In cloud environments TCP connections silently
            # break (NAT idle timeouts, intermediate proxy resets) and a stuck
            # ``recv()`` will wait forever. Synap's outer FAST-mode 8s timeout
            # fires at the asyncio layer but does NOT interrupt the OS thread
            # doing the synchronous chromadb call — that thread stays stuck on
            # the dead socket indefinitely. Over time the chroma worker pool
            # fills with stuck threads, new batch_search calls have to wait
            # for workers to free up, and observed query_time_ms climbs into
            # the 60-95s range. Override with a finite timeout so a broken
            # connection gets closed cleanly and the worker is freed.
            try:
                import httpx as _httpx
                # ── 1. Finite read timeout ─────────────────────────────────
                # chromadb defaults to ``Timeout(timeout=None)`` — infinite
                # read. In cloud deployments TCP connections silently break
                # (NAT idle timeout, intermediate proxy reset), and a stuck
                # ``recv()`` waits forever. 30s is long enough for legitimate
                # slow operations (bulk ``coll.get(include=embeddings)``
                # during clone, ``get_or_create`` under contention) while
                # short enough that a broken connection clears reasonably.
                self._client._server._session.timeout = _httpx.Timeout(
                    connect=10.0, read=30.0, write=30.0, pool=30.0,
                )
                # ── 2. Short keepalive so connections recycle ──────────────
                # chromadb defaults to ``keepalive_expiry=40s``. A connection
                # idle for 40s in cloud (Synap process → Chroma EIP) is very
                # likely to have been silently closed by an intermediate NAT
                # or load balancer. The httpx pool will then HAPPILY reuse
                # that broken connection on the next request, whose ``read``
                # will hang until our 30s timeout fires above — and the
                # next request after that picks up the SAME stale state
                # because more idle-past-NAT connections sit in the pool.
                # Rebuild the session with ``keepalive_expiry=5s`` so most
                # broken-by-NAT connections are gone before they can be
                # reused. The cost is more TCP handshakes; the gain is the
                # whole class of "stuck on stale connection" failures.
                old_session = self._client._server._session
                new_session = _httpx.Client(
                    base_url=old_session.base_url,
                    headers=old_session.headers,
                    timeout=_httpx.Timeout(
                        connect=10.0, read=30.0, write=30.0, pool=30.0,
                    ),
                    limits=_httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                        keepalive_expiry=5.0,
                    ),
                )
                self._client._server._session = new_session
                try:
                    old_session.close()
                except Exception:
                    pass
            except Exception:
                # Best-effort — if chromadb's internal shape changes, the
                # adapter still works (just without our finite timeout).
                logger.warning("Could not set finite httpx timeout on chromadb client", exc_info=True)

            # Verify server is actually responding
            self._client.heartbeat()
        elif persist_dir:
            # Persistent local client
            self._client = self._chroma_module.PersistentClient(path=persist_dir)
        else:
            # Ephemeral in-memory client (for testing)
            self._client = self._chroma_module.Client()

    def disconnect(self) -> None:
        """Disconnect client."""
        self._client = None
        # Cached Collection handles are bound to the client; drop them
        # when the client goes away so a re-connect doesn't serve stale
        # handles from a different client instance.
        self._collection_cache.clear()

    def _get_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._client

    # --- Namespace Management ---

    def create_namespace(self, namespace: str, dimension: int) -> None:
        """
        Create collection for instance.
        ChromaDB infers dimension from first embedding, but we can set metadata.
        """
        if self._use_fallback_memory:
            self._memory_namespaces.setdefault(self._sanitize_name(namespace), {})
            return
        client = self._get_client()
        name = self._sanitize_name(namespace)
        collection = client.get_or_create_collection(
            name=name,
            metadata={
                "dimension": dimension,
                "hnsw:space": "cosine",
                # R-13 / SHIP-09. Measured 2026-08-22 on a throwaway Chroma of
                # this image digest: 6,000 points, a scope filter matching 10%,
                # recall against an exact cosine scan of the readable set.
                #
                #   ef  10 -> recall 0.775, p50 10.80ms
                #   ef  50 -> recall 0.936, p50 10.62ms
                #   ef 100 -> recall 0.993, p50 11.64ms
                #   ef 200 -> recall 1.000, p50 12.50ms
                #   ef 400 -> recall 1.000, p50 13.47ms
                #
                # A collection created without this key measured recall 0.996,
                # so the default already behaves like 100 and there was never a
                # live quality bug here. 200 buys the last 0.4% for about 0.4ms,
                # which is worth one key on collections we are creating anyway.
                #
                # It is NOT worth a migration: a large fleet of collections for
                # 0.4% recall. Existing collections keep whatever they have, so
                # expect a fleet where new ones sit at 200 and old ones at the
                # default, and that is deliberate.
                #
                # ⚠ If you ever change this in place on an existing collection,
                # `modify()` REPLACES the metadata dict rather than merging it,
                # and it refuses any payload containing `hnsw:space` with an
                # error about the distance function. So the only accepted form
                # erases `hnsw:space` and `dimension` from the record. Measured,
                # that erasure is cosmetic (identical ids and distances, the
                # index keeps the distance function it was built with) but know
                # that you are doing it.
                "hnsw:search_ef": _HNSW_SEARCH_EF,
                # See _HNSW_SYNC_THRESHOLD above. Without this the index
                # for a small collection lives only in Chroma's memory pool.
                "hnsw:sync_threshold": _HNSW_SYNC_THRESHOLD,
            },
        )
        # Cache the fresh handle so the first read after create doesn't
        # pay the name→id resolution cost.
        self._collection_cache[name] = collection

    def delete_namespace(self, namespace: str) -> None:
        """Delete collection and all vectors."""
        if self._use_fallback_memory:
            self._memory_namespaces.pop(self._sanitize_name(namespace), None)
            return
        client = self._get_client()
        name = self._sanitize_name(namespace)
        try:
            client.delete_collection(name=name)
        except ValueError:
            pass  # Collection doesn't exist
        # Drop the cached handle so future calls re-resolve.
        self._collection_cache.pop(name, None)

    def _sanitize_name(self, name: str) -> str:
        """ChromaDB collection names must be 3-63 chars, alphanumeric with underscores."""
        sanitized = name.replace("-", "_")
        if len(sanitized) < 3:
            sanitized = f"ns_{sanitized}"
        return sanitized[:63]

    def _get_collection(self, namespace: str):
        """Get collection by namespace, cached.

        ``client.get_collection(name=...)`` does an HTTP roundtrip to
        resolve the name → server-side ID and fetch metadata. With ~1.5k
        collections on the shared instance, the resolution costs
        ~500-2600ms per call (measured). The handle is stable across the
        collection's lifetime, so we cache it forever in-process.

        Cache invalidation:
          - ``create_namespace`` / ``delete_namespace`` patch their own
            entry directly.
          - If a collection is dropped externally (out-of-band cleanup),
            ``collection.query()`` / ``upsert()`` would raise a Chroma
            error. The wrapper code at the call site catches that and
            evicts via ``invalidate_collection`` so the next attempt
            re-resolves.
        """
        name = self._sanitize_name(namespace)
        cached = self._collection_cache.get(name)
        if cached is not None:
            return cached
        client = self._get_client()
        collection = client.get_collection(name=name)
        self._collection_cache[name] = collection
        return collection

    def invalidate_collection(self, namespace: str) -> None:
        """Drop a cached Collection handle.

        Public hook for callers that detect a stale handle (e.g. a
        ``collection.query`` that returned a "collection not found"
        error). After invalidation, the next ``_get_collection`` call
        re-resolves via Chroma.
        """
        name = self._sanitize_name(namespace)
        self._collection_cache.pop(name, None)

    # --- CRUD Operations ---

    def upsert(self, namespace: str, documents: List[VectorDocument]) -> int:
        """Upsert documents into collection."""
        if not documents:
            return 0

        start = time.time()

        logger.info(f"chromadb_upsert_start namespace={self._sanitize_name(namespace)} document_count={len(documents)} documents_preview={log_data_sample([{'id': d.id, 'content': truncate_string(d.content or '', 80)} for d in documents], sample_size=2, label='documents')}")

        if self._use_fallback_memory:
            ns = self._memory_namespaces.setdefault(self._sanitize_name(namespace), {})
            for doc in documents:
                ns[doc.id] = doc
            return len(documents)

        collection = self._get_collection(namespace)

        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        contents = [doc.content for doc in documents if doc.content]

        # Log detailed metadata fields being stored
        logger.info(f"chromadb_upsert_metadata collection={self._sanitize_name(namespace)} total_documents={len(documents)} metadata_sample={log_data_sample([{'id': d.id, 'metadata_fields': list(d.metadata.keys()), 'metadata': truncate_dict(d.metadata, max_depth=3, max_items=10, max_str_len=200)} for d in documents], sample_size=3, label='metadata')}")

        # ChromaDB upsert
        if contents and len(contents) == len(documents):
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents
            )
        else:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

        duration_ms = (time.time() - start) * 1000
        logger.info(f"chromadb_upsert_complete document_count={len(documents)} duration_ms={duration_ms:.2f}")

        return len(documents)

    def delete(self, namespace: str, ids: List[str]) -> int:
        """Delete documents by ID."""
        if not ids:
            return 0

        if self._use_fallback_memory:
            ns = self._memory_namespaces.setdefault(self._sanitize_name(namespace), {})
            deleted = 0
            for doc_id in ids:
                if doc_id in ns:
                    deleted += 1
                    del ns[doc_id]
            return deleted

        collection = self._get_collection(namespace)
        collection.delete(ids=ids)
        return len(ids)

    def get_metadata(self, namespace: str, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Metadata only, by id: {id: metadata}. No embeddings or text are fetched,
        for callers that only need to know what a document currently says about
        itself (a read before a rewrite), which on a busy collection is the
        difference between a small and a large response per batch."""
        if not ids:
            return {}
        if self._use_fallback_memory:
            ns = self._memory_namespaces.setdefault(self._sanitize_name(namespace), {})
            return {i: dict(ns[i].metadata or {}) for i in ids if i in ns}
        collection = self._get_collection(namespace)
        result = collection.get(ids=ids, include=["metadatas"])
        metadatas = result.get("metadatas")
        if metadatas is None:
            return {}
        return {doc_id: dict(metadatas[i] or {}) for i, doc_id in enumerate(result["ids"])}

    def get(self, namespace: str, ids: List[str]) -> List[VectorDocument]:
        """Get documents by ID."""
        if not ids:
            return []

        if self._use_fallback_memory:
            ns = self._memory_namespaces.setdefault(self._sanitize_name(namespace), {})
            return [ns[doc_id] for doc_id in ids if doc_id in ns]

        collection = self._get_collection(namespace)
        result = collection.get(
            ids=ids,
            include=["embeddings", "metadatas", "documents"]
        )

        documents = []
        embeddings = result.get("embeddings")
        metadatas = result.get("metadatas")
        doc_contents = result.get("documents")
        has_embeddings = embeddings is not None and len(embeddings) > 0
        has_metadatas = metadatas is not None and len(metadatas) > 0
        has_documents = doc_contents is not None and len(doc_contents) > 0

        for i, doc_id in enumerate(result["ids"]):
            documents.append(VectorDocument(
                id=doc_id,
                embedding=embeddings[i] if has_embeddings else [],
                metadata=metadatas[i] if has_metadatas else {},
                content=doc_contents[i] if has_documents else None,
            ))

        return documents

    def get_all(
        self,
        namespace: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 500,
    ) -> List[VectorDocument]:
        """
        Get all documents from a collection with optional metadata filters.

        Returns documents WITH embeddings for building adjacency matrices.
        """
        if self._use_fallback_memory:
            ns = self._memory_namespaces.get(self._sanitize_name(namespace), {})
            docs = list(ns.values())
            if filters:
                docs = [d for d in docs if self._matches_filters(d.metadata, filters)]
            return docs[:limit]

        collection = self._get_collection(namespace)

        where = self._build_where_clause(filters) if filters else None

        result = collection.get(
            where=where,
            limit=limit,
            include=["embeddings", "metadatas", "documents"],
        )

        documents = []
        if result["ids"]:
            embeddings = result.get("embeddings")
            metadatas = result.get("metadatas")
            doc_contents = result.get("documents")
            has_embeddings = embeddings is not None and len(embeddings) > 0
            has_metadatas = metadatas is not None and len(metadatas) > 0
            has_documents = doc_contents is not None and len(doc_contents) > 0

            for i, doc_id in enumerate(result["ids"]):
                documents.append(VectorDocument(
                    id=doc_id,
                    embedding=embeddings[i] if has_embeddings else [],
                    metadata=metadatas[i] if has_metadatas else {},
                    content=doc_contents[i] if has_documents else None,
                ))

        return documents

    # Collections are fetched in pages of this size and reduced to names
    # immediately. Chroma returns collection objects, not names, and holding
    # thousands of them at once can exhaust Chroma's memory: thousands of
    # collections at a few MB each against a fixed cap, from a
    # script that was read-only and looked harmless.
    _LIST_PAGE = 200

    def list_namespaces(self) -> List[str]:
        """Collection names, paged, never all the objects at once."""
        if self._use_fallback_memory:
            return list(self._memory_namespaces)

        client = self._get_client()
        names: List[str] = []
        offset = 0
        while True:
            try:
                page = client.list_collections(limit=self._LIST_PAGE, offset=offset)
            except TypeError:
                # A client too old to page. Fall back to one call rather than
                # silently returning a truncated list, and say why in the log,
                # because on a large store this is the expensive path.
                logger.warning(
                    "chroma client does not support paged list_collections; "
                    "fetching every collection object in one call"
                )
                page = client.list_collections()
                names.extend(c.name if hasattr(c, "name") else str(c) for c in page)
                return names
            if not page:
                break
            names.extend(c.name if hasattr(c, "name") else str(c) for c in page)
            if len(page) < self._LIST_PAGE:
                break
            offset += len(page)
        return names

    # --- Metadata maintenance ---

    def scan_metadata(
        self,
        namespace: str,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Ids and metadata only, paged. No embeddings cross the wire.

        Chroma's `get` returns ids in a stable order for a fixed collection, so
        an offset means the same thing between calls. It is not stable across
        concurrent writes to the same collection, which is why the backfill
        re-reads rather than trusting a count: see S-13.
        """
        if self._use_fallback_memory:
            ns = self._memory_namespaces.get(self._sanitize_name(namespace), {})
            docs = list(ns.values())[offset:offset + limit]
            return [{"id": d.id, "metadata": dict(d.metadata or {})} for d in docs]

        collection = self._get_collection(namespace)
        result = collection.get(limit=limit, offset=offset, include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        return [
            {"id": doc_id, "metadata": dict(metadatas[i] or {}) if i < len(metadatas) else {}}
            for i, doc_id in enumerate(ids)
        ]

    def update_metadata(self, namespace: str, updates: Dict[str, Dict[str, Any]]) -> int:
        """Merge keys into existing documents without touching their vectors.

        Chroma's `update` replaces the metadata document rather than merging
        into it, so the caller's keys are merged onto what is already stored and
        the whole thing is written back. Passing only the new keys would drop
        `customer_id`, `memory_type` and everything else the record carries,
        which would be a far larger outage than the one this backfill fixes.
        """
        if not updates:
            return 0

        if self._use_fallback_memory:
            ns = self._memory_namespaces.get(self._sanitize_name(namespace), {})
            changed = 0
            for doc_id, keys in updates.items():
                doc = ns.get(doc_id)
                if doc is not None:
                    doc.metadata = {**(doc.metadata or {}), **keys}
                    changed += 1
            return changed

        collection = self._get_collection(namespace)
        ids = list(updates)
        existing = collection.get(ids=ids, include=["metadatas"])
        current = {
            doc_id: dict((existing.get("metadatas") or [{}])[i] or {})
            for i, doc_id in enumerate(existing.get("ids") or [])
        }

        merged_ids, merged_metadatas = [], []
        for doc_id in ids:
            if doc_id not in current:
                # Deleted between the scan and now. Updating it would recreate a
                # record somebody removed, which for an erasure is the worst
                # possible outcome of a maintenance job.
                continue
            merged_ids.append(doc_id)
            merged_metadatas.append({**current[doc_id], **updates[doc_id]})

        if not merged_ids:
            return 0
        collection.update(ids=merged_ids, metadatas=merged_metadatas)
        return len(merged_ids)

    # --- Search ---

    def search(self, query: VectorQuery) -> VectorSearchResult:
        """Similarity search."""
        start = time.time()

        namespace = query.namespace
        if not namespace:
            raise ValueError("namespace is required for search")

        collection_name = self._sanitize_name(namespace)

        logger.info(f"chromadb_search_start namespace={collection_name} top_k={query.top_k} filters={truncate_dict(query.filters) if query.filters else None} embedding_dim={len(query.embedding) if query.embedding else 0}")

        if self._use_fallback_memory:
            where = query.filters or {}
            ns = self._memory_namespaces.setdefault(self._sanitize_name(namespace), {})
            candidates = []
            for doc in ns.values():
                if self._matches_filters(doc.metadata, where):
                    score = self._cosine_similarity(query.embedding, doc.embedding)
                    candidates.append(
                        VectorResult(
                            id=doc.id,
                            score=score,
                            metadata=doc.metadata,
                            content=doc.content,
                        )
                    )
            candidates.sort(key=lambda r: r.score, reverse=True)
            query_time = (time.time() - start) * 1000
            return VectorSearchResult(results=candidates[: query.top_k], query_time_ms=query_time)

        collection = self._get_collection(namespace)

        # Log collection info
        try:
            collection_count = collection.count()
            logger.info(f"chromadb_collection_info collection={collection_name} document_count={collection_count} collection_metadata={collection.metadata if hasattr(collection, 'metadata') else None}")
        except Exception as e:
            logger.warning(f"Failed to get collection info: {e}")

        # Build where clause from filters
        where = self._build_where_clause(query.filters) if query.filters else None

        logger.info(f"chromadb_query_details collection={collection_name} where_clause={where} n_results={query.top_k} query_embedding_shape={len(query.embedding)} query_embedding_sample={query.embedding[:5] if len(query.embedding) > 5 else query.embedding}")

        results = collection.query(
            query_embeddings=[query.embedding],
            n_results=query.top_k,
            where=where,
            include=["metadatas", "documents", "distances"]
        )

        _ids_count = len(results["ids"][0]) if results.get("ids") and results["ids"] else 0
        _ids_preview = results["ids"][0][:5] if results.get("ids") and results["ids"] and results["ids"][0] else []
        _distances_preview = results["distances"][0][:5] if results.get("distances") and results["distances"] and results["distances"][0] else []
        _metadatas_sample = results["metadatas"][0][:2] if results.get("metadatas") and results["metadatas"] and results["metadatas"][0] else []
        _documents_sample = [truncate_string(d or "", 100) for d in (results["documents"][0][:2] if results.get("documents") and results["documents"] and results["documents"][0] else [])]
        logger.info(f"chromadb_raw_results collection={collection_name} ids_count={_ids_count} ids={_ids_preview} distances={_distances_preview} metadatas_sample={_metadatas_sample} documents_sample={_documents_sample}")

        # Convert to VectorResult list
        vector_results = []
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            documents = results["documents"][0] if results.get("documents") else [None] * len(ids)

            for i, doc_id in enumerate(ids):
                # ChromaDB returns distance, convert to similarity score
                # For cosine, similarity = 1 - distance
                score = 1 - distances[i] if distances[i] else 1.0

                vector_results.append(VectorResult(
                    id=doc_id,
                    score=score,
                    metadata=metadatas[i] or {},
                    content=documents[i]
                ))

        query_time = (time.time() - start) * 1000

        result = VectorSearchResult(
            results=vector_results,
            query_time_ms=query_time
        )

        logger.info(f"chromadb_search_complete results_count={len(vector_results)} query_time_ms={query_time:.2f} results_preview={log_data_sample([{'id': r.id, 'score': f'{r.score:.4f}', 'content': truncate_string(r.content or '', 100)} for r in vector_results], sample_size=3, label='results')}")

        return result

    def batch_search(self, queries: List[VectorQuery]) -> List[VectorSearchResult]:
        """Batch similarity search — single ChromaDB call with multiple embeddings."""
        if not queries:
            return []

        start = time.time()
        namespace = queries[0].namespace
        top_k = queries[0].top_k
        filters = queries[0].filters
        collection_name = self._sanitize_name(namespace)

        logger.info(f"chromadb_batch_search_start namespace={collection_name} query_count={len(queries)} top_k={top_k}")

        collection = self._get_collection(namespace)
        where = self._build_where_clause(filters) if filters else None
        all_embeddings = [q.embedding for q in queries]

        results = collection.query(
            query_embeddings=all_embeddings,
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"]
        )

        batch_results = []
        for q_idx in range(len(queries)):
            vector_results = []
            if results["ids"] and q_idx < len(results["ids"]):
                ids = results["ids"][q_idx]
                distances = results["distances"][q_idx] if results.get("distances") else [0] * len(ids)
                metadatas = results["metadatas"][q_idx] if results.get("metadatas") else [{}] * len(ids)
                documents = results["documents"][q_idx] if results.get("documents") else [None] * len(ids)

                for i, doc_id in enumerate(ids):
                    score = 1 - distances[i] if distances[i] else 1.0
                    vector_results.append(VectorResult(
                        id=doc_id, score=score,
                        metadata=metadatas[i] or {}, content=documents[i],
                    ))

            batch_results.append(VectorSearchResult(results=vector_results, query_time_ms=0))

        query_time = (time.time() - start) * 1000
        total_results = sum(len(r.results) for r in batch_results)
        logger.info(f"chromadb_batch_search_complete query_count={len(queries)} total_results={total_results} query_time_ms={query_time:.2f}")

        return batch_results

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Best-effort metadata filter matcher for in-memory fallback."""
        if not filters:
            return True
        for key, value in filters.items():
            if key.startswith("$"):
                # Keep fallback matcher intentionally simple.
                continue
            if metadata.get(key) != value:
                return False
        return True

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity, handling empty/zero vectors safely."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict:
        """
        Convert filters dict to ChromaDB where clause.

        Supports:
        - Simple equality: {"field": "value"}
        - Operators: {"field": {"$gt": 10}}
        - Logical: {"$and": [...]} or {"$or": [...]}
        """
        if not filters:
            return None

        # If already in ChromaDB format, return as-is
        if any(key.startswith("$") for key in filters.keys()):
            return filters

        # Convert simple key-value to ChromaDB format
        if len(filters) == 1:
            key, value = list(filters.items())[0]
            return {key: value}

        # Multiple filters = $and
        conditions = [{k: v} for k, v in filters.items()]
        return {"$and": conditions}
