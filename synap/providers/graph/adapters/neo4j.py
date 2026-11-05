"""Neo4j adapter implementation."""

from typing import List, Dict, Any, Optional
import logging
import os
import re
import time

from neo4j import GraphDatabase, Driver, unit_of_work
from ..interface import GraphStoreInterface
from ..models import Node, Edge, TraversalPattern, GraphQuery, GraphResult
from synap.cloud.shared.logging_utils import truncate_string, log_data_sample, truncate_dict
from synap.cloud.shared import graph_boundary

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _union_instances(var: str) -> str:
    """Cypher that adds `$iids` to `var.instance_ids`, in the statement that runs it.

    The list is the set of instances that have written a record, and a record is
    shared by every instance of a customer. It must therefore only ever be
    added to, and the addition must be atomic: read-modify-write in Python would
    let two instances writing the same entity at once drop one of them. The
    newest `$inst_cap` are kept, the same bound `append_to_node_list` uses.

    ONLY a record that already has a list is touched. A record written before
    graph records were stamped has none, and its true owners are unknown: giving
    it the list `[writer]` would claim the writer is its ONLY owner, hide it
    from the instance that really wrote it, and let that writer's deletion
    destroy it. It stays unstamped until the backfill, which knows the owners.
    (Assigning null removes the property, which is a no-op when it is absent.)
    """
    return (
        f"SET {var}.{graph_boundary.INSTANCE_IDS} = "
        f"CASE WHEN {var}.{graph_boundary.INSTANCE_IDS} IS NULL THEN null "
        f"WHEN size(merged) > $inst_cap THEN merged[size(merged) - $inst_cap..] "
        f"ELSE merged END"
    )


_MERGE_INSTANCES = (
    "reduce(acc = coalesce({var}.%s, []), x IN $iids | "
    "CASE WHEN x IN acc THEN acc ELSE acc + x END)" % graph_boundary.INSTANCE_IDS
)


# --- Opt-in query profiling (default OFF) --------------------------------
# When SYNAP_NEO4J_PROFILE is truthy, read/data queries run with a PROFILE
# prefix and their plan (operators + db-hits) is logged. PROFILE makes the
# server materialize per-operator counters, so it has real overhead — this is
# strictly a diagnostic switch, never a steady-state default. Behaviour is
# identical to the un-prefixed query (PROFILE returns the same rows).
_NON_PROFILABLE_PREFIXES = (
    "CREATE INDEX", "CREATE CONSTRAINT", "CREATE RANGE", "CREATE TEXT",
    "CREATE FULLTEXT", "CREATE VECTOR", "CREATE POINT", "CREATE LOOKUP",
    "DROP", "SHOW", "CALL DB.", "CALL DBMS.",
)


def neo4j_profile_enabled() -> bool:
    return os.environ.get("SYNAP_NEO4J_PROFILE", "").lower() in ("1", "true", "yes")


def _is_profilable(cypher: str) -> bool:
    """PROFILE is rejected for schema DDL / SHOW; allow everything else."""
    head = cypher.lstrip().upper()
    return not any(head.startswith(p) for p in _NON_PROFILABLE_PREFIXES)


def summarize_profile(profile) -> str:
    """Flatten a driver profile plan into 'total_db_hits=N ops=[Op:hits, ...]'."""
    ops: List[str] = []
    total = 0

    def _walk(p):
        nonlocal total
        if not p:
            return
        db = p.get("dbHits", 0) or 0
        total += db
        ops.append(f"{p.get('operatorType', '?')}:{db}")
        for child in (p.get("children") or []):
            _walk(child)

    _walk(profile)
    return f"total_db_hits={total} ops=[{', '.join(ops[:14])}]"


def _is_constraint_violation(exc: BaseException) -> bool:
    """A uniqueness constraint refused the write (the same test graph_wrapper uses to fall back to an update)."""
    return "ConstraintValidationFailed" in str(exc) or type(exc).__name__ == "ConstraintError"


def _is_schema_already_exists(exc: BaseException) -> bool:
    """Two workers ran the same CREATE INDEX/CONSTRAINT at once and the other one won: the object exists."""
    return "EquivalentSchemaRuleAlreadyExists" in str(exc)



# --- Lifecycle Cypher (release_memory_ids, set_record_states, erase_records) ---

# The writer's and the customer's node: keyed on an id, never retired or deleted
# by the lifecycle (services/writer_entity.is_fixed). Neo4j may hold the flag as
# a boolean or as its string, so both are read.
_FIXED = ("(coalesce(toString({v}.is_writer), '') IN ['true', 'True'] "
          "OR coalesce(toString({v}.is_customer), '') IN ['true', 'True'])")

# What an id release does to one record, shared by edges and nodes: the ids go
# from `memory_ids` to `released_memory_ids` (no duplicates, newest $cap kept),
# and a record left with none is retired. The first retirement wins: a record
# already retired keeps its reason and date.
_RELEASE_EDGES = """
MATCH (a:SynapNode {_namespace: $ns})-[r]->(b:SynapNode)
WHERE r._namespace = $ns AND any(x IN coalesce(r.memory_ids, []) WHERE x IN $ids)
WITH a, b, r, [x IN r.memory_ids WHERE NOT x IN $ids] AS kept,
     [x IN r.memory_ids WHERE x IN $ids] AS gone, coalesce(r.released_memory_ids, []) AS prev
WITH a, b, r, kept, prev + [x IN gone WHERE NOT x IN prev] AS rel
WITH a, b, r, kept, size(kept) = 0 AS unsupported,
     CASE WHEN size(rel) > $cap THEN rel[size(rel) - $cap..] ELSE rel END AS released
WITH a, b, r, kept, unsupported, released, unsupported AND r.retired_at IS NULL AS newly
SET r.memory_ids = kept,
    r.released_memory_ids = released,
    r.retired_at = CASE WHEN newly THEN $now ELSE r.retired_at END,
    r.retired_reason = CASE WHEN newly THEN $reason ELSE r.retired_reason END,
    r.retired_by_memory_id = CASE WHEN newly THEN $by ELSE r.retired_by_memory_id END
WITH a, b, r, kept, unsupported, released, newly, r.id AS rid, type(r) AS rtype,
     $erase AND unsupported AS erased
FOREACH (_ IN CASE WHEN erased THEN [1] ELSE [] END | DELETE r)
RETURN rid AS id, rtype AS type, kept AS memory_ids, released AS released_memory_ids,
       unsupported, newly, erased AS deleted,
       a.id AS from_id, a.canonical_name AS from_name, a.entity_type AS from_type,
       a.er_vector_name AS from_vector_name,
       b.id AS to_id, b.canonical_name AS to_name, b.entity_type AS to_type,
       b.er_vector_name AS to_vector_name
"""

_RELEASE_NODES = """
MATCH (n:SynapNode {_namespace: $ns})
WHERE any(x IN coalesce(n.memory_ids, []) WHERE x IN $ids)
WITH n, [x IN n.memory_ids WHERE NOT x IN $ids] AS kept,
     [x IN n.memory_ids WHERE x IN $ids] AS gone, coalesce(n.released_memory_ids, []) AS prev,
     %s AS fixed
WITH n, kept, fixed, prev + [x IN gone WHERE NOT x IN prev] AS rel
WITH n, kept, fixed, size(kept) = 0 AND NOT fixed AS unsupported,
     CASE WHEN size(rel) > $cap THEN rel[size(rel) - $cap..] ELSE rel END AS released
WITH n, kept, fixed, unsupported, released, unsupported AND n.retired_at IS NULL AS newly
SET n.memory_ids = kept,
    n.released_memory_ids = released,
    n.retired_at = CASE WHEN newly THEN $now ELSE n.retired_at END,
    n.retired_reason = CASE WHEN newly THEN $reason ELSE n.retired_reason END,
    n.retired_by_memory_id = CASE WHEN newly THEN $by ELSE n.retired_by_memory_id END
RETURN n.id AS id, kept AS memory_ids, released AS released_memory_ids,
       unsupported, newly, fixed AS is_fixed_entity
""" % _FIXED.format(v="n")

# A node nothing holds any more: no relationship, no supporting memory, and not
# the writer or the customer. Only nodes the caller names are considered.
_DELETE_ORPHANS = """
MATCH (n:SynapNode {_namespace: $ns})
WHERE n.id IN $candidates AND size(coalesce(n.memory_ids, [])) = 0
  AND NOT %s AND NOT EXISTS { (n)--() }
WITH n, n.id AS id, n.canonical_name AS name, n.entity_type AS entity_type,
     n.er_vector_name AS vector_name, coalesce(n.memory_ids, []) AS memory_ids
DETACH DELETE n
RETURN id, name, entity_type, vector_name, memory_ids
""" % _FIXED.format(v="n")

_SET_EDGE_STATES = """
UNWIND $rows AS row
MATCH (:SynapNode {_namespace: $ns})-[x {_namespace: $ns, id: row.id}]->(:SynapNode)
SET x.retired_at = CASE WHEN row.open THEN null ELSE coalesce(x.retired_at, row.now) END,
    x.retired_reason = CASE WHEN row.open THEN null ELSE row.reason END,
    x.retired_by_memory_id = CASE WHEN row.open THEN null ELSE row.by END,
    x.valid_to = CASE WHEN row.open THEN null ELSE row.valid_to END
RETURN count(x) AS n
"""

_SET_NODE_STATES = """
UNWIND $rows AS row
MATCH (x:SynapNode {_namespace: $ns, id: row.id})
WHERE row.open OR NOT %s
SET x.retired_at = CASE WHEN row.open THEN null ELSE coalesce(x.retired_at, row.now) END,
    x.retired_reason = CASE WHEN row.open THEN null ELSE row.reason END,
    x.retired_by_memory_id = CASE WHEN row.open THEN null ELSE row.by END,
    x.valid_to = CASE WHEN row.open THEN null ELSE row.valid_to END
RETURN count(x) AS n
""" % _FIXED.format(v="x")

# erase_records: the named edges and every edge of a named node, then the named
# nodes (never the writer's or the customer's).
_ERASE_EDGES = """
MATCH (a:SynapNode {_namespace: $ns})-[r]->(b:SynapNode)
WHERE r._namespace = $ns
  AND (r.id IN $edge_ids OR (a.id IN $node_ids AND NOT %s) OR (b.id IN $node_ids AND NOT %s))
WITH a, b, r, r.id AS rid, type(r) AS rtype, coalesce(r.memory_ids, []) AS memory_ids
DELETE r
RETURN rid AS id, rtype AS type, memory_ids,
       a.id AS from_id, a.canonical_name AS from_name, a.entity_type AS from_type,
       a.er_vector_name AS from_vector_name,
       b.id AS to_id, b.canonical_name AS to_name, b.entity_type AS to_type,
       b.er_vector_name AS to_vector_name
""" % (_FIXED.format(v="a"), _FIXED.format(v="b"))

_ERASE_NODES = """
MATCH (n:SynapNode {_namespace: $ns})
WHERE n.id IN $node_ids AND NOT %s
WITH n, n.id AS id, n.canonical_name AS name, n.entity_type AS entity_type,
     n.er_vector_name AS vector_name, coalesce(n.memory_ids, []) AS memory_ids
DETACH DELETE n
RETURN id, name, entity_type, vector_name, memory_ids
""" % _FIXED.format(v="n")

# Undo of a supersession: the ids go back from `released_memory_ids` to
# `memory_ids` (no duplicates, newest $cap kept). The state is NOT changed
# here: the caller re-applies the rule, which reopens the record only when an
# alive memory supports it.
_RESTORE = """
MATCH (x:SynapNode {{_namespace: $ns}}){pattern}
WHERE {where} any(i IN coalesce(x.released_memory_ids, []) WHERE i IN $ids)
WITH x, coalesce(x.memory_ids, []) AS current,
     [i IN x.released_memory_ids WHERE i IN $ids] AS back
WITH x, current + [i IN back WHERE NOT i IN current] AS merged
SET x.memory_ids = CASE WHEN size(merged) > $cap THEN merged[size(merged) - $cap..] ELSE merged END,
    x.released_memory_ids = [i IN x.released_memory_ids WHERE NOT i IN $ids]
RETURN x.id AS id, x.memory_ids AS memory_ids, x.released_memory_ids AS released_memory_ids,
       {fixed} AS is_fixed_entity, x.retired_at AS retired_at
"""
_RESTORE_EDGES = _RESTORE.replace("MATCH (x:SynapNode {{_namespace: $ns}}){pattern}",
                                  "MATCH (:SynapNode {{_namespace: $ns}})-[x]->(:SynapNode)").format(
    where="x._namespace = $ns AND", fixed="false")
_RESTORE_NODES = _RESTORE.format(pattern="", where="", fixed=_FIXED.format(v="x"))

_LIST_NAMESPACES = """
MATCH (n:SynapNode) WHERE n._namespace IS NOT NULL
RETURN DISTINCT n._namespace AS ns
"""

_LIFECYCLE_EDGES = """
MATCH (a:SynapNode {_namespace: $ns})-[r]->(:SynapNode)
WHERE r._namespace = $ns AND ($ids IS NULL OR r.id IN $ids)
RETURN r.id AS id, coalesce(r.memory_ids, []) AS memory_ids,
       coalesce(r.released_memory_ids, []) AS released_memory_ids,
       r.memory_contents AS memory_contents, r.retired_at AS retired_at,
       r.retired_reason AS retired_reason, r.retired_by_memory_id AS retired_by_memory_id
"""

_LIFECYCLE_NODES = """
MATCH (n:SynapNode {_namespace: $ns})
WHERE $ids IS NULL OR n.id IN $ids
RETURN n.id AS id, coalesce(n.memory_ids, []) AS memory_ids,
       coalesce(n.released_memory_ids, []) AS released_memory_ids,
       %s AS is_fixed_entity, n.retired_at AS retired_at,
       n.retired_reason AS retired_reason, n.retired_by_memory_id AS retired_by_memory_id
""" % _FIXED.format(v="n")

# Copies mode: the relationships whose memory_contents JSON names one of $ids.
_EDGES_WITH_CONTENTS = """
MATCH (a:SynapNode {_namespace: $ns})-[r]->()
WHERE r._namespace = $ns AND r.memory_contents IS NOT NULL
  AND any(x IN $ids WHERE r.memory_contents CONTAINS x)
RETURN r.id AS id, r.memory_contents AS memory_contents
"""

RELEASE_KEYS = ("edges_pruned", "edges_retired", "edges_deleted",
                "nodes_pruned", "nodes_retired", "nodes_deleted")


def _empty_release() -> Dict[str, Any]:
    out: Dict[str, Any] = {k: 0 for k in RELEASE_KEYS}
    out.update(records=[], deleted_edges=[], deleted_nodes=[])
    return out


def _edge_names(e: Dict) -> Dict[str, Any]:
    return {k: e.get(k) for k in ("id", "type", "from_id", "from_name", "from_type", "from_vector_name",
                                  "to_id", "to_name", "to_type", "to_vector_name")}


def _node_names(n: Dict) -> Dict[str, Any]:
    return {k: n.get(k) for k in ("id", "name", "entity_type", "vector_name")}


def _release_summary(edges: List[Dict], nodes: List[Dict], orphans: List[Dict]) -> Dict[str, Any]:
    """Counts and touched records from the rows of the three release statements."""
    out = _empty_release()
    out["edges_pruned"] = len(edges)
    out["edges_retired"] = sum(1 for e in edges if e["newly"] and not e["deleted"])
    out["edges_deleted"] = sum(1 for e in edges if e["deleted"])
    out["nodes_pruned"] = len(nodes)
    deleted_ids = {o["id"] for o in orphans}
    out["nodes_retired"] = sum(1 for n in nodes if n["newly"] and n["id"] not in deleted_ids)
    out["nodes_deleted"] = len(orphans)
    out["records"] = (
        [{"kind": "edge", "id": e["id"], "memory_ids": e["memory_ids"] or [],
          "released_memory_ids": e["released_memory_ids"] or [], "deleted": bool(e["deleted"]),
          "is_fixed_entity": False} for e in edges]
        + [{"kind": "node", "id": n["id"], "memory_ids": n["memory_ids"] or [],
            "released_memory_ids": n["released_memory_ids"] or [], "deleted": n["id"] in deleted_ids,
            "is_fixed_entity": bool(n["is_fixed_entity"])} for n in nodes])
    out["deleted_edges"] = [_edge_names(e) for e in edges if e["deleted"]]
    out["deleted_nodes"] = [_node_names(o) for o in orphans]
    return out


class Neo4jAdapter(GraphStoreInterface):
    """
    Neo4j implementation of graph store interface.

    Features:
    - Traversal-based API → internally builds optimized Cypher
    - Namespace isolation via _namespace property on all nodes
    - Connection pooling via Neo4j driver
    - Index creation for performance
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Config keys:
        - uri: str (e.g., "bolt://localhost:7687")
        - user: str
        - password: str
        - database: str (default "neo4j")
        - pool_size: int (default 50)
        """
        self.config = config
        self._driver: Optional[Driver] = None
        self._database = config.get("database", "neo4j")
        # Per-query server-side transaction timeout (PR-2 / T1.4). Set for real in
        # connect(); default here guards any pre-connect call path.
        self._query_timeout: float = 20.0
        # Schema (indexes + constraint) is global to the :SynapNode label, not
        # per-namespace, so it only needs to be ensured once per process (PR-3).
        self._schema_ensured: bool = False

    def _cfg_float(self, key: str, env: str, default: float) -> float:
        if key in self.config:
            return float(self.config[key])
        val = os.environ.get(env)
        return float(val) if val is not None else default

    def connect(self) -> None:
        """Initialize Neo4j driver."""
        if self._driver is not None:
            return

        # PR-2 (T1.4): per-query transaction timeout so a runaway query is
        # cancelled server-side instead of pinning a pooled connection.
        self._query_timeout = self._cfg_float("query_timeout", "NEO4J_QUERY_TIMEOUT_S", 20.0)

        self._driver = GraphDatabase.driver(
            self.config["uri"],
            auth=(self.config["user"], self.config["password"]),
            max_connection_pool_size=self.config.get("pool_size", 50),
            # PR-2 (T3.2): explicit pool/connection tuning. Defaults left the pool
            # able to hang on acquisition and hold connections indefinitely.
            connection_acquisition_timeout=self._cfg_float(
                "connection_acquisition_timeout", "NEO4J_CONN_ACQUISITION_TIMEOUT_S", 30.0),
            max_connection_lifetime=self._cfg_float(
                "max_connection_lifetime", "NEO4J_MAX_CONN_LIFETIME_S", 3600.0),
            connection_timeout=self._cfg_float(
                "connection_timeout", "NEO4J_CONN_TIMEOUT_S", 15.0),
            # PR-2 (T1.3): bound total time spent retrying transient failures.
            max_transaction_retry_time=self._cfg_float(
                "max_transaction_retry_time", "NEO4J_MAX_TX_RETRY_TIME_S", 30.0),
        )
        # Verify connectivity
        self._driver.verify_connectivity()

    def disconnect(self) -> None:
        """Close driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def _get_driver(self) -> Driver:
        if self._driver is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._driver

    def _run_query(self, cypher: str, params: dict = None) -> List[Dict]:
        """Execute Cypher query and return results."""
        start = time.time()
        query_preview = truncate_string(cypher, 300)

        logger.info(f"neo4j_query_start query={query_preview} param_keys={sorted(params) if params else None}")

        driver = self._get_driver()
        profile = neo4j_profile_enabled() and _is_profilable(cypher)
        run_cypher = ("PROFILE " + cypher) if profile else cypher
        # Schema DDL (CREATE INDEX/CONSTRAINT, DROP, SHOW, CALL db.*) cannot run in
        # a managed transaction — keep those as auto-commit. Everything else (reads)
        # runs via execute_read for automatic retry (T1.3) + a tx timeout (T1.4).
        is_schema = not _is_profilable(cypher)
        try:
            with driver.session(database=self._database) as session:
                if is_schema:
                    result = session.run(run_cypher, params or {})
                    records = [record.data() for record in result]
                else:
                    @unit_of_work(timeout=self._query_timeout)
                    def _work(tx):
                        res = tx.run(run_cypher, params or {})
                        recs = [record.data() for record in res]
                        prof = res.consume().profile if profile else None
                        return recs, prof

                    records, prof = session.execute_read(_work)
                    if prof:
                        logger.info(f"neo4j_profile label=query {summarize_profile(prof)}")

                duration_ms = (time.time() - start) * 1000
                logger.info(f"neo4j_query_complete result_count={len(records)} result_preview={log_data_sample(records, sample_size=2, label='records')} duration_ms={duration_ms:.2f}")

                return records
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            if _is_schema_already_exists(e):
                logger.info(f"neo4j_schema_already_exists error={e} duration_ms={duration_ms:.2f}")
            else:
                logger.error(f"neo4j_query_failed error={e} duration_ms={duration_ms:.2f}", exc_info=True)
            raise

    def _run_write(self, cypher: str, params: dict = None) -> Any:
        """Execute write query."""
        start = time.time()
        query_preview = truncate_string(cypher, 300)

        logger.info(f"neo4j_write_start query={query_preview} param_keys={sorted(params) if params else None}")

        driver = self._get_driver()
        profile = neo4j_profile_enabled() and _is_profilable(cypher)
        run_cypher = ("PROFILE " + cypher) if profile else cypher
        # Data writes (CREATE/MERGE/SET/DELETE) run via execute_write for automatic
        # retry (T1.3) + a tx timeout (T1.4). Managed retries only re-run when the tx
        # failed to commit, so re-running is safe (no double-writes). Schema DDL, if
        # ever routed here, stays auto-commit (can't run in a managed transaction).
        is_schema = not _is_profilable(cypher)
        try:
            with driver.session(database=self._database) as session:
                if is_schema:
                    result = session.run(run_cypher, params or {})
                    single_result = result.single()
                else:
                    @unit_of_work(timeout=self._query_timeout)
                    def _work(tx):
                        res = tx.run(run_cypher, params or {})
                        single = res.single()
                        prof = res.consume().profile if profile else None
                        return single, prof

                    single_result, prof = session.execute_write(_work)
                    if prof:
                        logger.info(f"neo4j_profile label=write {summarize_profile(prof)}")

                duration_ms = (time.time() - start) * 1000
                logger.info(f"neo4j_write_complete result={str(single_result)[:200] if single_result else None} duration_ms={duration_ms:.2f}")

                return single_result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            if _is_constraint_violation(e):
                # Two writers creating the same node at once: the loser gets this, and the caller
                # (graph_wrapper) reads the winner's node and updates it. It is expected, not an error.
                logger.info(f"neo4j_write_constraint_violation error={e} duration_ms={duration_ms:.2f}")
            else:
                logger.error(f"neo4j_write_failed error={e} duration_ms={duration_ms:.2f}", exc_info=True)
            raise

    def _run_write_rows(self, cypher: str, params: dict = None) -> List[Dict]:
        """Execute a data-changing statement in a WRITE transaction and return every row.

        `_run_query` always runs in a read transaction, which Neo4j answers
        with "Writing in read access mode not allowed" for DETACH DELETE, SET,
        MERGE and CREATE. `_run_write` is a write transaction but returns only
        the first row. A caller that needs the rows a deleting statement
        returns (a count, for example) uses this one. Failures raise; they are
        never turned into an empty list.
        """
        start = time.time()
        logger.info(f"neo4j_write_start query={truncate_string(cypher, 300)} param_keys={sorted(params) if params else None}")
        driver = self._get_driver()

        @unit_of_work(timeout=self._query_timeout)
        def _work(tx):
            return [record.data() for record in tx.run(cypher, params or {})]

        try:
            with driver.session(database=self._database) as session:
                records = session.execute_write(_work)
            logger.info(f"neo4j_write_complete rows={len(records)} duration_ms={(time.time() - start) * 1000:.2f}")
            return records
        except Exception as e:
            logger.error(f"neo4j_write_failed error={e} duration_ms={(time.time() - start) * 1000:.2f}", exc_info=True)
            raise

    # --- Namespace Management ---

    # Global :SynapNode schema (PR-3). Fixed names, not per-namespace: every
    # node carries :SynapNode, so one physical index/constraint serves all
    # namespaces (the old per-namespace CREATE … IF NOT EXISTS collapsed to one
    # object by equivalent schema anyway). `id` index is new (T1.5): id is a
    # unique uuid, so a lookup by id seeks straight to one node.
    _SCHEMA_INDEXES = (
        ("synap_synapnode_namespace",
         "CREATE INDEX synap_synapnode_namespace IF NOT EXISTS "
         "FOR (n:SynapNode) ON (n._namespace)"),
        ("synap_synapnode_id",
         "CREATE INDEX synap_synapnode_id IF NOT EXISTS "
         "FOR (n:SynapNode) ON (n.id)"),
        # Entity resolution finds a renamed entity's node by its key. Only
        # nodes written in adjudicated mode carry one, so the index is small.
        ("synap_synapnode_er_node_key",
         "CREATE INDEX synap_synapnode_er_node_key IF NOT EXISTS "
         "FOR (n:SynapNode) ON (n._namespace, n.er_node_key)"),
    )
    _CANONICAL_CONSTRAINT = (
        "synap_synapnode_canonical",
        "CREATE CONSTRAINT synap_synapnode_canonical IF NOT EXISTS "
        "FOR (n:SynapNode) REQUIRE (n.canonical_name, n._namespace) IS UNIQUE",
    )

    def create_namespace(self, namespace: str) -> None:
        """Ensure the :SynapNode schema exists so this namespace is usable.

        Schema is global (not per-namespace); ``namespace`` is accepted for API
        compatibility. Idempotent and cheap after the first call per process.
        """
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the global :SynapNode indexes + uniqueness constraint (once).

        Unlike the old per-namespace best-effort creation, a constraint failure is
        NOT silently swallowed: it means duplicate (canonical_name, _namespace)
        rows exist, which both blocks the constraint (so the anchor lookup loses
        its backing index) and leaves the graph split. We log an actionable ERROR
        pointing at the dedup remediation rather than hiding it in a warning.
        """
        if self._schema_ensured:
            return

        for _name, cypher in self._SCHEMA_INDEXES:
            try:
                self._run_query(cypher)
            except Exception as e:
                if not _is_schema_already_exists(e):
                    raise

        name, cypher = self._CANONICAL_CONSTRAINT
        try:
            self._run_query(cypher)
        except Exception as e:
            # A concurrent worker created it first: the constraint exists, which is the point.
            if not _is_schema_already_exists(e):
                logger.error(
                    "neo4j_constraint_creation_failed constraint=%s error=%s — likely "
                    "duplicate (canonical_name, _namespace) rows: the anchor lookup "
                    "loses its unique index and duplicates persist. Remediate with "
                    "synap/scripts/cleanup_neo4j_duplicates.py, then reconnect.",
                    name, e,
                )

        self._schema_ensured = True
        # Best-effort verification (log-only); never blocks setup.
        try:
            self.verify_schema()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"neo4j_verify_schema_error error={e}")

    def verify_schema(self) -> Dict[str, bool]:
        """Check (via SHOW INDEXES/CONSTRAINTS) that the expected :SynapNode
        schema is present, matched by *schema* (label+properties), not name — so
        it detects the objects regardless of their historical per-namespace names.
        Logs a WARNING for anything missing. Returns a presence map.
        """
        idx_rows = self._run_query(
            "SHOW INDEXES YIELD labelsOrTypes, properties, entityType "
            "RETURN labelsOrTypes, properties, entityType")
        con_rows = self._run_query(
            "SHOW CONSTRAINTS YIELD labelsOrTypes, properties "
            "RETURN labelsOrTypes, properties")

        def _has(rows, props):
            want = set(props)
            return any(
                (r.get("labelsOrTypes") == ["SynapNode"]
                 and set(r.get("properties") or []) == want)
                for r in rows
            )

        present = {
            "namespace_index": _has(idx_rows, ["_namespace"]),
            "id_index": _has(idx_rows, ["id"]),
            "canonical_constraint": _has(con_rows, ["canonical_name", "_namespace"]),
        }
        for key, ok in present.items():
            if not ok:
                logger.warning(
                    "neo4j_schema_missing object=%s — SynapNode lookups may degrade; "
                    "run schema bootstrap (create_namespace) / dedup script", key)
        return present

    def delete_namespace(self, namespace: str) -> None:
        """Delete all nodes and edges in namespace."""
        # DETACH DELETE is a write — must go through _run_write (execute_write).
        # Previously used _run_query, which now routes non-schema statements to a
        # read transaction (a write there is rejected under access-mode READ).
        self._run_write("""
            MATCH (n:SynapNode {_namespace: $ns})
            DETACH DELETE n
        """, {"ns": namespace})

    # --- Node Operations ---

    def add_node(self, namespace: str, node_type: str, properties: Dict[str, Any]) -> str:
        """Add node. Returns node ID."""
        import uuid
        node_id = str(uuid.uuid4())

        props = {**properties, "_namespace": namespace, "id": node_id}
        if graph_boundary.INSTANCE_IDS in props:
            # A new node starts with exactly the instances that wrote it.
            _, _ids = graph_boundary.split_instance_ids(props)
            if _ids:
                props[graph_boundary.INSTANCE_IDS] = _ids
            else:
                props.pop(graph_boundary.INSTANCE_IDS)

        # Log node creation details
        logger.info(f"neo4j_add_node namespace={namespace} node_type={node_type} node_id={node_id} properties_keys={list(properties.keys())} properties={truncate_dict(properties, max_depth=3, max_items=10, max_str_len=200)}")

        # Build property string for Cypher
        self._run_write(f"""
            CREATE (n:SynapNode:{node_type} $props)
            RETURN n.id as id
        """, {"props": props})

        return node_id

    def get_node(self, namespace: str, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        result = self._run_query("""
            MATCH (n:SynapNode {_namespace: $ns, id: $id})
            RETURN n, labels(n) as labels
        """, {"ns": namespace, "id": node_id})

        if not result:
            return None

        record = result[0]
        node_data = dict(record["n"])
        labels = record["labels"]

        # Extract node_type (exclude SynapNode base label)
        node_type = next((l for l in labels if l != "SynapNode"), "Unknown")

        # Remove internal properties
        properties = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}

        return Node(
            id=node_data["id"],
            node_type=node_type,
            properties=properties
        )

    def update_node(self, namespace: str, node_id: str, properties: Dict[str, Any]) -> None:
        """Update node properties.

        `instance_ids` is reserved: it is added to what the node already holds,
        never written over it (see `_union_instances`).
        """
        props, iids = graph_boundary.split_instance_ids(properties)
        if not iids:
            self._run_write("""
                MATCH (n:SynapNode {_namespace: $ns, id: $id})
                SET n += $props
            """, {"ns": namespace, "id": node_id, "props": props})
            return
        self._run_write(f"""
            MATCH (n:SynapNode {{_namespace: $ns, id: $id}})
            SET n += $props
            WITH n, {_MERGE_INSTANCES.format(var="n")} AS merged
            {_union_instances("n")}
        """, {"ns": namespace, "id": node_id, "props": props, "iids": iids,
              "inst_cap": graph_boundary.MAX_INSTANCES_PER_RECORD})

    def append_to_node_list(self, namespace: str, node_id: str, key: str,
                            values: List[str], cap: int) -> None:
        """Add *values* to a list property of one node, in one statement:
        no duplicates, and only the newest *cap* kept. A retired record is NOT
        reopened here: the caller re-evaluates it with the shared rule
        (GraphLifecycle.reevaluate), which reopens it only when an alive
        memory supports it."""
        if not _SAFE_IDENTIFIER.match(key):
            raise ValueError(f"Invalid property key: {key!r}")
        self._run_write(f"""
            MATCH (n:SynapNode {{_namespace: $ns, id: $id}})
            WITH n, coalesce(n.{key}, []) AS current
            WITH n, current, current + [v IN $values WHERE NOT v IN current] AS merged
            SET n.{key} = CASE WHEN size(merged) > $cap
                               THEN merged[size(merged) - $cap..] ELSE merged END
        """, {"ns": namespace, "id": node_id, "values": list(values), "cap": cap})

    def delete_node(self, namespace: str, node_id: str) -> None:
        """Delete node and connected edges."""
        self._run_write("""
            MATCH (n:SynapNode {_namespace: $ns, id: $id})
            DETACH DELETE n
        """, {"ns": namespace, "id": node_id})

    def find_nodes_by_properties(
        self, namespace: str, node_type: str, properties: Dict[str, Any], limit: int = 10
    ) -> List[Node]:
        """Find nodes matching property values within namespace."""
        if not _SAFE_IDENTIFIER.match(node_type):
            raise ValueError(f"Invalid node_type: {node_type!r}")
        for k in properties:
            if not _SAFE_IDENTIFIER.match(k):
                raise ValueError(f"Invalid property key: {k!r}")
        conditions = " AND ".join([f"n.{k} = $prop_{k}" for k in properties.keys()])
        cypher = f"""
            MATCH (n:SynapNode:{node_type} {{_namespace: $ns}})
            WHERE {conditions}
            RETURN n, labels(n) as labels
            LIMIT $limit
        """
        params = {"ns": namespace, "limit": limit, **{f"prop_{k}": v for k, v in properties.items()}}
        result = self._run_query(cypher, params)

        nodes = []
        for record in result:
            node_data = dict(record["n"])
            labels = record["labels"]
            ntype = next((l for l in labels if l != "SynapNode"), "Unknown")
            props = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}
            nodes.append(Node(id=node_data["id"], node_type=ntype, properties=props))

        return nodes

    def find_nodes_by_properties_any_type(
        self, namespace: str, properties: Dict[str, Any], limit: int = 10
    ) -> List[Node]:
        """Find nodes matching properties within namespace, ignoring node_type label."""
        for k in properties:
            if not _SAFE_IDENTIFIER.match(k):
                raise ValueError(f"Invalid property key: {k!r}")
        conditions = " AND ".join([f"n.{k} = $prop_{k}" for k in properties.keys()])
        cypher = f"""
            MATCH (n:SynapNode {{_namespace: $ns}})
            WHERE {conditions}
            RETURN n, labels(n) as labels
            LIMIT $limit
        """
        params = {"ns": namespace, "limit": limit, **{f"prop_{k}": v for k, v in properties.items()}}
        result = self._run_query(cypher, params)

        nodes = []
        for record in result:
            node_data = dict(record["n"])
            labels = record["labels"]
            ntype = next((l for l in labels if l != "SynapNode"), "Unknown")
            props = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}
            nodes.append(Node(id=node_data["id"], node_type=ntype, properties=props))

        return nodes

    # --- Edge Operations ---

    def add_edge(self, namespace: str, from_id: str, to_id: str, edge_type: str, properties: Dict = None) -> Optional[str]:
        """Add or merge edge. Returns edge ID, or None if either endpoint not found.

        Uses MERGE to prevent duplicate edges between the same two nodes with
        the same type and namespace. ON CREATE sets full props; ON MATCH updates
        props but preserves the original edge ID.
        """
        import uuid
        edge_id = str(uuid.uuid4())

        base_props, iids = graph_boundary.split_instance_ids(properties)
        create_props = {**base_props, "_namespace": namespace, "id": edge_id}
        update_props = {k: v for k, v in create_props.items() if k != "id"}
        if iids:
            # A NEW edge starts with exactly the instances that wrote it. A
            # matched edge is only ever added to, and only if it was stamped.
            create_props[graph_boundary.INSTANCE_IDS] = list(iids)

        # Log relationship creation details
        props_dict = properties or {}
        logger.info(f"neo4j_add_edge namespace={namespace} edge_type={edge_type} edge_id={edge_id} from_node_id={from_id} to_node_id={to_id} properties_keys={list(props_dict.keys())}")

        if not iids:
            result = self._run_write(f"""
                MATCH (a:SynapNode {{_namespace: $ns, id: $from_id}})
                MATCH (b:SynapNode {{_namespace: $ns, id: $to_id}})
                MERGE (a)-[r:{edge_type} {{_namespace: $ns}}]->(b)
                ON CREATE SET r = $create_props
                ON MATCH SET r += $update_props
                RETURN r.id as id
            """, {"ns": namespace, "from_id": from_id, "to_id": to_id, "create_props": create_props, "update_props": update_props})
        else:
            # The same MERGE, then the instance list is unioned into whatever the
            # edge already held (only if it held one: see `_union_instances`).
            result = self._run_write(f"""
                MATCH (a:SynapNode {{_namespace: $ns, id: $from_id}})
                MATCH (b:SynapNode {{_namespace: $ns, id: $to_id}})
                MERGE (a)-[r:{edge_type} {{_namespace: $ns}}]->(b)
                ON CREATE SET r = $create_props
                ON MATCH SET r += $update_props
                WITH r, {_MERGE_INSTANCES.format(var="r")} AS merged
                {_union_instances("r")}
                RETURN r.id as id
            """, {"ns": namespace, "from_id": from_id, "to_id": to_id, "create_props": create_props,
                  "update_props": update_props, "iids": iids,
                  "inst_cap": graph_boundary.MAX_INSTANCES_PER_RECORD})

        if result is None:
            logger.warning(f"neo4j_add_edge_no_match edge_id={edge_id} from_id={from_id} to_id={to_id} - one or both nodes not found")
            return None

        # On MERGE-MATCH, return the existing edge's ID
        return result["id"] if hasattr(result, "__getitem__") else edge_id

    def get_edge(self, namespace: str, edge_id: str) -> Optional[Edge]:
        """Get edge by ID."""
        result = self._run_query("""
            MATCH (a:SynapNode {_namespace: $ns})-[r {_namespace: $ns, id: $id}]->(b)
            RETURN properties(r) as props, type(r) as edge_type, a.id as from_id, b.id as to_id
        """, {"ns": namespace, "id": edge_id})

        if not result:
            return None

        record = result[0]
        edge_data = record["props"]
        properties = {k: v for k, v in edge_data.items() if not k.startswith("_") and k != "id"}

        return Edge(
            id=edge_data["id"],
            from_id=record["from_id"],
            to_id=record["to_id"],
            edge_type=record["edge_type"],
            properties=properties
        )

    def update_edge(self, namespace: str, edge_id: str, properties: Dict[str, Any]) -> None:
        """Update edge properties (merge into existing).

        `instance_ids` is reserved: added to, never written over.
        """
        props, iids = graph_boundary.split_instance_ids(properties)
        if not iids:
            self._run_write("""
                MATCH (a:SynapNode {_namespace: $ns})-[r {_namespace: $ns, id: $id}]->()
                SET r += $props
            """, {"ns": namespace, "id": edge_id, "props": props})
            return
        self._run_write(f"""
            MATCH (a:SynapNode {{_namespace: $ns}})-[r {{_namespace: $ns, id: $id}}]->()
            SET r += $props
            WITH r, {_MERGE_INSTANCES.format(var="r")} AS merged
            {_union_instances("r")}
        """, {"ns": namespace, "id": edge_id, "props": props, "iids": iids,
              "inst_cap": graph_boundary.MAX_INSTANCES_PER_RECORD})

    def append_to_edge_list(self, namespace: str, edge_id: str, key: str,
                            values: List[str], cap: int) -> None:
        """Add *values* to a list property of one edge, in one statement:
        no duplicates, and only the newest *cap* kept. A retired record is NOT
        reopened here: the caller re-evaluates it with the shared rule
        (GraphLifecycle.reevaluate), which reopens it only when an alive
        memory supports it."""
        if not _SAFE_IDENTIFIER.match(key):
            raise ValueError(f"Invalid property key: {key!r}")
        self._run_write(f"""
            MATCH (:SynapNode {{_namespace: $ns}})-[r {{_namespace: $ns, id: $id}}]->()
            WITH r, coalesce(r.{key}, []) AS current
            WITH r, current, current + [v IN $values WHERE NOT v IN current] AS merged
            SET r.{key} = CASE WHEN size(merged) > $cap
                               THEN merged[size(merged) - $cap..] ELSE merged END
        """, {"ns": namespace, "id": edge_id, "values": list(values), "cap": cap})

    def stamp_unstamped(self, namespace: str, kind: str, rows: List[Dict[str, Any]]) -> int:
        """Give records that name no instance the instances that own them (backfill).

        `kind` is "node" or "edge"; `rows` are `{"id": ..., "ids": [instance, ...]}`.
        Ordinary writes deliberately never stamp a record that was written before
        stamping existed (they cannot know its other owners); this is the one
        operation that does, because the caller has worked the owners out. It acts
        only on a record that is STILL unstamped, so a stamp a live writer added in
        the meantime is never overwritten. Returns how many records it stamped.
        """
        iid = graph_boundary.INSTANCE_IDS
        rows = [{"id": r["id"], "ids": [str(i) for i in dict.fromkeys(r["ids"]) if i]}
                for r in rows if r.get("ids")]
        if not rows:
            return 0
        if kind == "node":
            match = "MATCH (x:SynapNode {_namespace: $ns, id: row.id})"
        elif kind == "edge":
            match = "MATCH (:SynapNode {_namespace: $ns})-[x {_namespace: $ns, id: row.id}]->(:SynapNode)"
        else:
            raise ValueError(f"kind must be node or edge, not {kind!r}")
        out = self._run_write_rows(f"""
            UNWIND $rows AS row
            {match}
            WHERE x.{iid} IS NULL OR size(x.{iid}) = 0
            SET x.{iid} = row.ids
            RETURN count(x) AS n
        """, {"ns": namespace, "rows": rows})
        return int(out[0]["n"]) if out else 0

    def rewrite_memory_contents(self, namespace: str, edge_id: str, transform,
                                attempts: int = 5) -> int:
        """Rewrite one relationship's memory list, without losing a concurrent append.

        `transform(memories)` returns the new list, or None for "no change". The
        text is read, transformed, and written back ONLY IF it is still what was
        read; otherwise it is read again. Ingestion appends memories to the same
        relationship, and a plain read-modify-write here would silently throw
        those away. Returns how many memories the transform removed (the length
        difference), or 0 when nothing changed. Raises if it cannot win after
        `attempts` tries: that is a failure to report, not to hide.
        """
        import json as _json
        for _ in range(max(1, attempts)):
            rows = self._run_query(
                "MATCH (:SynapNode {_namespace: $ns})-[r {_namespace: $ns, id: $id}]->() RETURN r.memory_contents AS mc",
                {"ns": namespace, "id": edge_id})
            if not rows or rows[0]["mc"] is None:
                return 0
            current = rows[0]["mc"]
            try:
                memories = _json.loads(current)
            except (TypeError, ValueError):
                return 0
            if not isinstance(memories, list):
                return 0
            updated = transform(memories)
            if updated is None:
                return 0
            written = self._run_write_rows(
                "MATCH (:SynapNode {_namespace: $ns})-[r {_namespace: $ns, id: $id}]->() "
                "WHERE r.memory_contents = $expected "
                "SET r.memory_contents = $new RETURN count(r) AS n",
                {"ns": namespace, "id": edge_id, "expected": current,
                 "new": _json.dumps(updated)})
            if written and int(written[0]["n"]) > 0:
                return max(0, len(memories) - len(updated))
        raise RuntimeError(
            f"memory_contents of relationship {edge_id} kept changing; gave up after {attempts} attempts")

    # --- Lifecycle: graph records follow the memories that support them ---
    #
    # A record (edge or node) is supported by the memories in `memory_ids`. When
    # a memory is retired or deleted its id is released from every record in the
    # namespace in ONE write transaction, so an `append_to_edge_list` running at
    # the same time cannot interleave (Neo4j holds the record's write lock until
    # commit). See issue12_graph.md section 5.

    def _run_in_write_tx(self, work) -> Any:
        """Run `work(tx)` in one managed write transaction and return its result.

        For a change made of several statements that must commit together. A
        managed transaction re-runs `work` only when it failed to commit, so the
        statements must be idempotent (every lifecycle statement is)."""
        start = time.time()
        driver = self._get_driver()

        @unit_of_work(timeout=self._query_timeout)
        def _work(tx):
            return work(tx)

        try:
            with driver.session(database=self._database) as session:
                out = session.execute_write(_work)
            logger.info(f"neo4j_write_tx_complete duration_ms={(time.time() - start) * 1000:.2f}")
            return out
        except Exception as e:
            logger.error(f"neo4j_write_tx_failed error={e} duration_ms={(time.time() - start) * 1000:.2f}", exc_info=True)
            raise

    def release_memory_ids(self, namespace: str, ids: List[str], mode: str = "retire",
                           now: Optional[str] = None, reason: str = "superseded",
                           superseded_by: Optional[str] = None, cap: int = 500) -> Dict[str, Any]:
        """See GraphStoreInterface.release_memory_ids."""
        if mode not in ("retire", "erase"):
            raise ValueError(f"mode must be retire or erase, not {mode!r}")
        ids = [str(i) for i in dict.fromkeys(ids or []) if i]
        out = _empty_release()
        if not ids:
            return out
        from datetime import datetime, timezone
        params = {"ns": namespace, "ids": ids, "cap": int(cap),
                  "now": now or datetime.now(timezone.utc).isoformat(),
                  "reason": reason, "by": str(superseded_by) if superseded_by else None,
                  "erase": mode == "erase"}

        def work(tx):
            edges = [r.data() for r in tx.run(_RELEASE_EDGES, params)]
            nodes = [r.data() for r in tx.run(_RELEASE_NODES, params)]
            orphans: List[Dict] = []
            if mode == "erase":
                candidates = {n["id"] for n in nodes if not n["memory_ids"]}
                for e in edges:
                    if e["deleted"]:
                        candidates.update((e["from_id"], e["to_id"]))
                # An endpoint another transaction deleted between our MATCH and
                # our write lock reads back as null: nothing to clean up there.
                candidates.discard(None)
                if candidates:
                    orphans = [r.data() for r in tx.run(
                        _DELETE_ORPHANS, {"ns": namespace, "candidates": sorted(candidates)})]
            return edges, nodes, orphans

        edges, nodes, orphans = self._run_in_write_tx(work)
        return _release_summary(edges, nodes, orphans)

    def set_record_states(self, namespace: str, states: List[Dict[str, Any]]) -> int:
        """See GraphStoreInterface.set_record_states."""
        rows = {"edge": [], "node": []}
        for s in states or []:
            kind = s.get("kind")
            if kind not in rows:
                raise ValueError(f"kind must be node or edge, not {kind!r}")
            valid_to = s.get("valid_to")
            rows[kind].append({
                "id": str(s["id"]), "open": bool(s.get("open")),
                "now": s.get("now"), "reason": s.get("reason"),
                "by": str(s["retired_by_memory_id"]) if s.get("retired_by_memory_id") else None,
                "valid_to": valid_to.isoformat() if hasattr(valid_to, "isoformat") else valid_to,
            })
        if not rows["edge"] and not rows["node"]:
            return 0
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for kind in rows:
            for r in rows[kind]:
                r["now"] = r["now"] or now

        def work(tx):
            n = 0
            if rows["edge"]:
                rec = tx.run(_SET_EDGE_STATES, {"ns": namespace, "rows": rows["edge"]}).single()
                n += int(rec["n"]) if rec else 0
            if rows["node"]:
                rec = tx.run(_SET_NODE_STATES, {"ns": namespace, "rows": rows["node"]}).single()
                n += int(rec["n"]) if rec else 0
            return n

        return int(self._run_in_write_tx(work))

    def erase_records(self, namespace: str, edge_ids: List[str], node_ids: List[str]) -> Dict[str, Any]:
        """See GraphStoreInterface.erase_records."""
        params = {"ns": namespace, "edge_ids": [str(i) for i in edge_ids or []],
                  "node_ids": [str(i) for i in node_ids or []]}
        if not params["edge_ids"] and not params["node_ids"]:
            return {"deleted_edges": [], "deleted_nodes": [], "records": []}

        def work(tx):
            edges = [r.data() for r in tx.run(_ERASE_EDGES, params)]
            nodes = [r.data() for r in tx.run(_ERASE_NODES, params)]
            candidates = set()
            for e in edges:
                candidates.update((e["from_id"], e["to_id"]))
            candidates -= set(params["node_ids"])
            candidates.discard(None)
            orphans = []
            if candidates:
                orphans = [r.data() for r in tx.run(
                    _DELETE_ORPHANS, {"ns": namespace, "candidates": sorted(candidates)})]
            return edges, nodes, orphans

        edges, nodes, orphans = self._run_in_write_tx(work)
        return {
            "deleted_edges": [_edge_names(e) for e in edges],
            "deleted_nodes": [_node_names(n) for n in nodes + orphans],
            "records": ([{"kind": "edge", "id": e["id"], "memory_ids": e.get("memory_ids") or []} for e in edges]
                        + [{"kind": "node", "id": n["id"], "memory_ids": n.get("memory_ids") or []} for n in nodes]),
        }

    def restore_memory_ids(self, namespace: str, ids: List[str], cap: int = 500) -> List[Dict[str, Any]]:
        """See GraphStoreInterface.restore_memory_ids."""
        ids = [str(i) for i in dict.fromkeys(ids or []) if i]
        if not ids:
            return []
        params = {"ns": namespace, "ids": ids, "cap": int(cap)}

        def work(tx):
            edges = [dict(r.data(), kind="edge") for r in tx.run(_RESTORE_EDGES, params)]
            nodes = [dict(r.data(), kind="node") for r in tx.run(_RESTORE_NODES, params)]
            return edges + nodes

        return self._run_in_write_tx(work)

    def list_namespaces(self) -> List[str]:
        """Every graph namespace that holds a node (for the reconciler)."""
        return [r["ns"] for r in self._run_query(_LIST_NAMESPACES, {})]

    def lifecycle_records(self, namespace: str, ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Every edge and node of *namespace* (only those with an id in *ids*
        when given) with what the lifecycle reads:
        kind, id, memory_ids, released_memory_ids, legacy_ids (the memory ids
        in a copies-mode ``memory_contents``), is_fixed_entity, retired_at."""
        import json as _json
        out: List[Dict[str, Any]] = []
        for e in self._run_query(_LIFECYCLE_EDGES, {"ns": namespace, "ids": None if ids is None else [str(i) for i in ids]}):
            legacy: List[str] = []
            if e.get("memory_contents"):
                try:
                    legacy = [str(m.get("memory_id")) for m in _json.loads(e["memory_contents"])
                              if isinstance(m, dict) and m.get("memory_id")]
                except (TypeError, ValueError):
                    legacy = []
            out.append({"kind": "edge", "id": e["id"], "memory_ids": list(e["memory_ids"]),
                        "released_memory_ids": list(e["released_memory_ids"]), "legacy_ids": legacy,
                        "is_fixed_entity": False, "retired_at": e.get("retired_at"),
                        "retired_reason": e.get("retired_reason"),
                        "retired_by_memory_id": e.get("retired_by_memory_id")})
        for n in self._run_query(_LIFECYCLE_NODES, {"ns": namespace, "ids": None if ids is None else [str(i) for i in ids]}):
            out.append({"kind": "node", "id": n["id"], "memory_ids": list(n["memory_ids"]),
                        "released_memory_ids": list(n["released_memory_ids"]), "legacy_ids": [],
                        "is_fixed_entity": bool(n["is_fixed_entity"]), "retired_at": n.get("retired_at"),
                        "retired_reason": n.get("retired_reason"),
                        "retired_by_memory_id": n.get("retired_by_memory_id")})
        return out

    def edges_with_memory_contents(self, namespace: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Copies mode: ``[{id, memory_contents}]`` of the relationships whose
        ``memory_contents`` names one of *ids*."""
        ids = [str(i) for i in ids or [] if i]
        if not ids:
            return []
        return self._run_query(_EDGES_WITH_CONTENTS, {"ns": namespace, "ids": ids})

    def strip_person_memories(self, client_id: str, field: str, value: str,
                              instance_ids: List[str], namespaces: Optional[List[str]] = None,
                              include_unstamped: bool = False, batch: int = 200):
        """Remove one person's memories, written by these instances, from every
        relationship that survives an erase.

        An erase that keeps a node because another instance still uses it must not
        leave the erased person's memory TEXT behind on that node's relationships.
        `field` is `user_id` or `customer_id`. `instance_ids=None` removes the
        person's memories whatever instance wrote them (a whole-person erase). A
        memory that names no instance (written before stamping) is removed only when
        `include_unstamped` (nobody else holds this person's data); otherwise it is
        left and counted.
        With `namespaces` the search is confined to them, through the namespace
        index; without, it scans the client's relationships.

        Returns (memories removed, this person's unstamped memories left).
        """
        if field not in ("user_id", "customer_id"):
            raise ValueError(f"cannot strip memories by {field!r}")
        # None means "any instance": a whole-person erase.
        iids = None if instance_ids is None else [str(i) for i in dict.fromkeys(instance_ids) if i]
        if iids is not None and not iids:
            return 0, 0
        left = [0]       # total over relationships
        this_pass = [0]  # the pass in progress: a compare-and-set retry runs it again

        def _strip(memories):
            this_pass[0] = 0
            kept = []
            for m in memories:
                theirs = isinstance(m, dict) and str(m.get(field)) == str(value)
                inst = m.get(graph_boundary.INSTANCE_ID) if isinstance(m, dict) else None
                if theirs and (iids is None or (str(inst) in iids if inst else include_unstamped)):
                    continue
                if theirs and not inst:
                    this_pass[0] += 1
                kept.append(m)
            return kept if len(kept) != len(memories) else None

        removed = 0
        scopes = [str(n) for n in namespaces] if namespaces else [None]
        for ns in scopes:
            last = ""
            while True:
                if ns is None:
                    match = "MATCH (:SynapNode)-[r]->(:SynapNode) WHERE r.client_id = $client_id AND"
                    qparams = {"client_id": str(client_id)}
                else:
                    match = "MATCH (:SynapNode {_namespace: $ns})-[r {_namespace: $ns}]->(:SynapNode) WHERE"
                    qparams = {"ns": ns}
                rows = self._run_query(f"""
                    {match} r.memory_contents IS NOT NULL AND r.id > $last
                      AND r.memory_contents CONTAINS $value
                    RETURN r.id AS id, r._namespace AS ns
                    ORDER BY r.id LIMIT $batch
                """, {**qparams, "value": str(value), "last": last, "batch": int(batch)})
                if not rows:
                    break
                for r in rows:
                    this_pass[0] = 0
                    removed += self.rewrite_memory_contents(r["ns"], r["id"], _strip)
                    left[0] += this_pass[0]
                last = rows[-1]["id"]
        return removed, left[0]

    def remove_instances(self, client_id: str, instance_ids: List[str],
                         batch: int = 500) -> Dict[str, int]:
        """Take instances out of every graph record of one client.

        A node or relationship is shared by every instance of a customer and
        carries the LIST of instances that wrote it. Deleting an instance must
        therefore take that instance out of each list, delete only the records
        left owned by no one, and take that instance's memories off the
        relationships that survive. Deleting by "node.instance_id = X" (what the
        instance-deletion worker did) matched nothing, because nothing ever wrote
        a scalar, and would have destroyed other instances' data had it matched.

        Records that name no instance at all (written before they were stamped)
        are NOT touched: their owner cannot be known. They are counted so the
        caller can say so. Every statement is a WRITE transaction and failures
        raise; this is the function whose predecessor ran DETACH DELETE in a read
        transaction, failed, and was logged as success.

        Returns counts: edges_updated, edges_deleted, memories_removed,
        nodes_updated, nodes_deleted, unstamped_nodes, unstamped_edges.
        """
        iids = [str(i) for i in dict.fromkeys(instance_ids) if i]
        counts = {"edges_updated": 0, "edges_deleted": 0, "memories_removed": 0,
                  "nodes_updated": 0, "nodes_deleted": 0,
                  "unstamped_nodes": 0, "unstamped_edges": 0}
        if not iids:
            return counts
        params = {"client_id": str(client_id), "iids": iids, "batch": int(batch)}
        iid = graph_boundary.INSTANCE_IDS

        def drain(cypher: str) -> int:
            total = 0
            while True:
                rows = self._run_write_rows(cypher, params)
                n = int(rows[0]["n"]) if rows else 0
                total += n
                if n < batch:
                    return total

        # 1. Relationships: take the instances off the list, then delete the
        #    ones nobody owns.
        counts["edges_updated"] = drain(f"""
            MATCH (:SynapNode)-[r]->(:SynapNode)
            WHERE r.client_id = $client_id
              AND ANY(i IN coalesce(r.{iid}, []) WHERE i IN $iids)
            WITH r LIMIT $batch
            SET r.{iid} = [i IN r.{iid} WHERE NOT i IN $iids]
            RETURN count(r) AS n
        """)
        counts["edges_deleted"] = drain(f"""
            MATCH (:SynapNode)-[r]->(:SynapNode)
            WHERE r.client_id = $client_id AND r.{iid} IS NOT NULL AND size(r.{iid}) = 0
            WITH r LIMIT $batch
            DELETE r
            RETURN count(r) AS n
        """)

        # 2. Memories on the relationships that survive. They are a JSON text, so
        #    this is done here, and only for relationships whose text names the
        #    instance at all. Paged by edge id (not by "the first N that still
        #    match", which stops early when a page matches only by coincidence of
        #    text) and written back only if nobody changed the text meanwhile.
        def _strip(memories):
            kept = [m for m in memories
                    if not (isinstance(m, dict) and str(m.get(graph_boundary.INSTANCE_ID)) in iids)]
            return kept if len(kept) != len(memories) else None

        last = ""
        while True:
            rows = self._run_query("""
                MATCH (:SynapNode)-[r]->(:SynapNode)
                WHERE r.client_id = $client_id AND r.memory_contents IS NOT NULL AND r.id > $last
                  AND ANY(x IN $iids WHERE r.memory_contents CONTAINS x)
                RETURN r.id AS id, r._namespace AS ns
                ORDER BY r.id LIMIT $batch
            """, {**params, "last": last})
            if not rows:
                break
            for r in rows:
                counts["memories_removed"] += self.rewrite_memory_contents(r["ns"], r["id"], _strip)
            last = rows[-1]["id"]

        # 3. Nodes, the same two steps. A node's edges go with it.
        counts["nodes_updated"] = drain(f"""
            MATCH (n:SynapNode)
            WHERE n.client_id = $client_id
              AND ANY(i IN coalesce(n.{iid}, []) WHERE i IN $iids)
            WITH n LIMIT $batch
            SET n.{iid} = [i IN n.{iid} WHERE NOT i IN $iids]
            RETURN count(n) AS n
        """)
        counts["nodes_deleted"] = drain(f"""
            MATCH (n:SynapNode)
            WHERE n.client_id = $client_id AND n.{iid} IS NOT NULL AND size(n.{iid}) = 0
            WITH n LIMIT $batch
            DETACH DELETE n
            RETURN count(n) AS n
        """)

        # 4. What could not be attributed, so the caller can say so.
        for key, label in (("unstamped_nodes", "(n:SynapNode)"), ("unstamped_edges", "(:SynapNode)-[n]->(:SynapNode)")):
            rows = self._run_query(
                f"MATCH {label} WHERE n.client_id = $client_id AND n.{iid} IS NULL RETURN count(n) AS c",
                params)
            counts[key] = int(rows[0]["c"]) if rows else 0
        return counts

    def edges_between(self, namespace: str, a_id: str, b_id: str) -> List[Edge]:
        """Every edge between two nodes, in either direction."""
        rows = self._run_query("""
            MATCH (a:SynapNode {_namespace: $ns})-[r {_namespace: $ns}]->(b:SynapNode {_namespace: $ns})
            WHERE (a.id = $a AND b.id = $b) OR (a.id = $b AND b.id = $a)
            RETURN r.id AS id, type(r) AS edge_type, a.id AS from_id, b.id AS to_id,
                   r.inverse_type AS inverse_type
        """, {"ns": namespace, "a": a_id, "b": b_id})
        return [Edge(id=row["id"], from_id=row["from_id"], to_id=row["to_id"], edge_type=row["edge_type"],
                     properties={"inverse_type": row["inverse_type"]}) for row in rows or []]

    def delete_edge(self, namespace: str, edge_id: str) -> None:
        """Delete edge."""
        self._run_write("""
            MATCH (a:SynapNode {_namespace: $ns})-[r {_namespace: $ns, id: $id}]->()
            DELETE r
        """, {"ns": namespace, "id": edge_id})

    # --- Traversal Operations ---

    def get_neighbors(self, namespace: str, node_id: str, edge_type: str = None, direction: str = "out") -> List[Node]:
        """Get neighboring nodes."""
        # Build direction pattern
        if direction == "out":
            pattern = "(n)-[r]->(m)"
        elif direction == "in":
            pattern = "(n)<-[r]-(m)"
        else:  # both
            pattern = "(n)-[r]-(m)"

        # Build edge type filter
        edge_filter = f":{edge_type}" if edge_type else ""

        cypher = f"""
            MATCH {pattern.replace('[r]', f'[r{edge_filter}]')}
            WHERE n._namespace = $ns AND n.id = $id AND m._namespace = $ns
            RETURN m, labels(m) as labels
        """

        result = self._run_query(cypher, {"ns": namespace, "id": node_id})

        nodes = []
        for record in result:
            node_data = dict(record["m"])
            labels = record["labels"]
            node_type = next((l for l in labels if l != "SynapNode"), "Unknown")
            properties = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}

            nodes.append(Node(
                id=node_data["id"],
                node_type=node_type,
                properties=properties
            ))

        return nodes

    def traverse(self, namespace: str, start_id: str, pattern: TraversalPattern) -> GraphResult:
        """
        Traverse graph from start node.
        Builds optimized Cypher from TraversalPattern.
        """
        # Build edge type pattern
        if pattern.edge_types:
            edge_pattern = ":" + "|".join(pattern.edge_types)
        else:
            edge_pattern = ""

        # Build direction
        if pattern.direction == "out":
            rel_pattern = f"-[r{edge_pattern}*1..{pattern.max_depth}]->"
        elif pattern.direction == "in":
            rel_pattern = f"<-[r{edge_pattern}*1..{pattern.max_depth}]-"
        else:
            rel_pattern = f"-[r{edge_pattern}*1..{pattern.max_depth}]-"

        # Build node filter
        node_filter = ""
        if pattern.node_filters:
            conditions = " AND ".join([f"m.{k} = ${k}" for k in pattern.node_filters.keys()])
            node_filter = f"AND ({conditions})"

        cypher = f"""
            MATCH path = (n {{_namespace: $ns, id: $start_id}}){rel_pattern}(m)
            WHERE ALL(node IN nodes(path) WHERE node._namespace = $ns)
              {node_filter}
            RETURN path
            LIMIT $limit
        """

        params = {
            "ns": namespace,
            "start_id": start_id,
            "limit": pattern.limit,
            **(pattern.node_filters or {})
        }

        # Process paths inside the session to preserve Neo4j Path objects
        # (_run_query calls .data() which serializes paths into plain dicts)
        from synap.providers.graph.models import Node, Edge, GraphResult
        driver = self._get_driver()
        nodes_map = {}
        edges_map = {}
        graph_paths = []

        try:
            with driver.session(database=self._database) as session:
                result = session.run(cypher, params)
                for record in result:
                    path = record.get("path")
                    if not path:
                        continue

                    path_ids = []
                    for node in path.nodes:
                        node_data = dict(node)
                        node_id = node_data.get("id")
                        if node_id and node_id not in nodes_map:
                            labels = list(node.labels)
                            node_type = next((l for l in labels if l != "SynapNode"), "Unknown")
                            properties = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}
                            nodes_map[node_id] = Node(id=node_id, node_type=node_type, properties=properties)
                        if node_id:
                            path_ids.append(node_id)

                    for rel in path.relationships:
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

                    if path_ids:
                        graph_paths.append(path_ids)
        except Exception as e:
            logger.error(f"Neo4j traverse failed: {e}", exc_info=True)

        logger.info(f"Neo4j traverse complete nodes={len(nodes_map)} edges={len(edges_map)} paths={len(graph_paths)}")

        return GraphResult(
            nodes=list(nodes_map.values()),
            edges=list(edges_map.values()),
            paths=graph_paths,
        )

    def batch_traverse(self, namespace: str, start_ids: List[str], pattern: TraversalPattern) -> GraphResult:
        """
        Traverse graph from multiple start nodes in a single Cypher query.

        Uses UNWIND + CALL {} subquery to give each anchor its own LIMIT,
        reducing N round-trips to 1 while maintaining per-anchor fairness.
        """
        if not start_ids:
            return GraphResult(nodes=[], edges=[], paths=[])

        # Build direction pattern (same logic as traverse)
        if pattern.edge_types:
            edge_pattern = ":" + "|".join(pattern.edge_types)
        else:
            edge_pattern = ""

        if pattern.direction == "out":
            rel_pattern = f"-[r{edge_pattern}*1..{pattern.max_depth}]->"
        elif pattern.direction == "in":
            rel_pattern = f"<-[r{edge_pattern}*1..{pattern.max_depth}]-"
        else:
            rel_pattern = f"-[r{edge_pattern}*1..{pattern.max_depth}]-"

        cypher = f"""
            UNWIND $start_ids AS sid
            CALL (sid) {{
                MATCH path = (n {{_namespace: $ns, id: sid}}){rel_pattern}(m)
                WHERE ALL(node IN nodes(path) WHERE node._namespace = $ns)
                RETURN path
                LIMIT $per_anchor_limit
            }}
            RETURN path
        """

        params = {
            "ns": namespace,
            "start_ids": start_ids,
            "per_anchor_limit": pattern.limit,
        }

        nodes_map = {}
        edges_map = {}
        graph_paths = []

        try:
            driver = self._get_driver()
            with driver.session(database=self._database) as session:
                result = session.run(cypher, params)
                for record in result:
                    path = record.get("path")
                    if not path:
                        continue

                    path_ids = []
                    for node in path.nodes:
                        node_data = dict(node)
                        node_id = node_data.get("id")
                        if node_id and node_id not in nodes_map:
                            labels = list(node.labels)
                            node_type = next((l for l in labels if l != "SynapNode"), "Unknown")
                            properties = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}
                            nodes_map[node_id] = Node(id=node_id, node_type=node_type, properties=properties)
                        if node_id:
                            path_ids.append(node_id)

                    for rel in path.relationships:
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

                    if path_ids:
                        graph_paths.append(path_ids)
        except Exception as e:
            logger.error(f"Neo4j batch_traverse failed: {e}", exc_info=True)

        logger.info(f"Neo4j batch_traverse complete: start_ids={len(start_ids)}, nodes={len(nodes_map)}, edges={len(edges_map)}, paths={len(graph_paths)}")

        return GraphResult(
            nodes=list(nodes_map.values()),
            edges=list(edges_map.values()),
            paths=graph_paths,
        )

    def query(self, query: GraphQuery) -> GraphResult:
        """Execute graph query."""
        if query.pattern:
            return self.traverse(query.namespace, query.start_node_id, query.pattern)

        # Simple node lookup by type
        if query.start_node_type:
            cypher = """
                MATCH (n:SynapNode {_namespace: $ns})
                WHERE $type IN labels(n)
                RETURN n, labels(n) as labels
                LIMIT 100
            """
            result = self._run_query(cypher, {"ns": query.namespace, "type": query.start_node_type})

            nodes = []
            for record in result:
                node_data = dict(record["n"])
                labels = record["labels"]
                node_type = next((l for l in labels if l != "SynapNode"), "Unknown")
                properties = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "id"}
                nodes.append(Node(id=node_data["id"], node_type=node_type, properties=properties))

            return GraphResult(nodes=nodes, edges=[])

        return GraphResult(nodes=[], edges=[])

    def get_namespace_subgraph(self, namespace: str, limit: int = 5000):
        """Return (nodes, edges) for a namespace as lightweight dicts.

        nodes: [{"id", "name"}] (name = canonical_name|name)
        edges: [{"edge_id", "from_id", "to_id"}]
        Used by cross-document memory->edge relink. Bounded by ``limit``.
        """
        node_rows = self._run_query(
            "MATCH (n:SynapNode {_namespace: $ns}) "
            "RETURN n.id AS id, coalesce(n.canonical_name, n.name) AS name LIMIT $limit",
            {"ns": namespace, "limit": limit},
        )
        nodes = [{"id": r["id"], "name": r["name"]} for r in node_rows if r.get("id")]

        edge_rows = self._run_query(
            "MATCH (a:SynapNode {_namespace: $ns})-[r]->(b:SynapNode {_namespace: $ns}) "
            "RETURN r.id AS edge_id, a.id AS from_id, b.id AS to_id LIMIT $limit",
            {"ns": namespace, "limit": limit},
        )
        edges = [
            {"edge_id": r["edge_id"], "from_id": r["from_id"], "to_id": r["to_id"]}
            for r in edge_rows if r.get("edge_id")
        ]
        return nodes, edges

