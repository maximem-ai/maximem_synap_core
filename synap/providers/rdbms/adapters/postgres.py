"""PostgreSQL adapter implementation."""

from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool, extras
import logging
import time

from ..interface import RDBMSInterface
from ..models import SQLQuery, SQLResult
from synap.cloud.shared.logging_utils import format_sql_query, log_data_sample, truncate_dict

logger = logging.getLogger(__name__)


class PostgresAdapter(RDBMSInterface):
    """
    PostgreSQL implementation of RDBMS interface.

    Features:
    - Connection pooling (configurable size)
    - Transaction support via context manager
    - Dict cursor for easy row access
    - Multi-tenancy via tenant_id column (handled by caller)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Config keys:
        - host: str
        - port: int (default 5432)
        - database: str
        - user: str
        - password: str
        - pool_size: int (default 10)
        - pool_max_overflow: int (default 5)
        """
        self.config = config
        self._pool: Optional[pool.ThreadedConnectionPool] = None

    def connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return

        self._pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=self.config.get("pool_size", 10),
            host=self.config["host"],
            port=self.config.get("port", 5432),
            database=self.config["database"],
            user=self.config["user"],
            password=self.config["password"]
        )

    def disconnect(self) -> None:
        """Close all connections in pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None

    def _get_connection(self):
        """Get connection from pool."""
        if self._pool is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._pool.getconn()

    def _put_connection(self, conn):
        """Return connection to pool."""
        if self._pool:
            self._pool.putconn(conn)

    def execute(self, query: SQLQuery) -> SQLResult:
        """Execute single query (INSERT, UPDATE, DELETE)."""
        start_time = time.time()
        formatted_query = format_sql_query(query.sql, query.params)

        logger.info(f"postgres_execute_start query={formatted_query}")

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(query.sql, query.params)
                conn.commit()

                # Try to get last insert id if available
                last_id = None
                if cur.description is None:  # Non-SELECT
                    try:
                        cur.execute("SELECT lastval()")
                        last_id = cur.fetchone()["lastval"]
                    except:
                        pass

                result = SQLResult(
                    rows=[],
                    row_count=cur.rowcount,
                    last_insert_id=last_id
                )

                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"postgres_execute_complete row_count={result.row_count} last_insert_id={last_id} duration_ms={duration_ms:.2f}")

                return result
        except Exception as e:
            conn.rollback()
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"postgres_execute_failed error={e} duration_ms={duration_ms:.2f}", exc_info=True)
            raise
        finally:
            self._put_connection(conn)

    def execute_many(self, query: SQLQuery, params_list: List[tuple]) -> SQLResult:
        """Execute query with multiple param sets (batch insert/update)."""
        start_time = time.time()
        formatted_query = format_sql_query(query.sql, None)

        logger.info(f"postgres_execute_many_start query={formatted_query} batch_size={len(params_list)} params_preview={str(params_list[:2]) if len(params_list) > 0 else '[]'}")

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.executemany(query.sql, params_list)
                conn.commit()
                result = SQLResult(
                    rows=[],
                    row_count=cur.rowcount
                )

                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"postgres_execute_many_complete row_count={result.row_count} batch_size={len(params_list)} duration_ms={duration_ms:.2f}")

                return result
        except Exception as e:
            conn.rollback()
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"postgres_execute_many_failed error={e} batch_size={len(params_list)} duration_ms={duration_ms:.2f}", exc_info=True)
            raise
        finally:
            self._put_connection(conn)

    def fetch_one(self, query: SQLQuery) -> Optional[Dict[str, Any]]:
        """Fetch single row."""
        start_time = time.time()
        formatted_query = format_sql_query(query.sql, query.params)

        logger.info(f"postgres_fetch_one_start query={formatted_query}")

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(query.sql, query.params)
                row = cur.fetchone()
                result = dict(row) if row else None

                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"postgres_fetch_one_complete found={result is not None} result_preview={truncate_dict(result, max_depth=2, max_items=10) if result else None} duration_ms={duration_ms:.2f}")

                return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"postgres_fetch_one_failed error={e} duration_ms={duration_ms:.2f}", exc_info=True)
            raise
        finally:
            self._put_connection(conn)

    def fetch_all(self, query: SQLQuery) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        start_time = time.time()
        formatted_query = format_sql_query(query.sql, query.params)

        logger.info(f"postgres_fetch_all_start query={formatted_query}")

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(query.sql, query.params)
                rows = cur.fetchall()
                result = [dict(row) for row in rows]

                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"postgres_fetch_all_complete row_count={len(result)} result_preview={log_data_sample(result, sample_size=2, label='rows')} duration_ms={duration_ms:.2f}")

                return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"postgres_fetch_all_failed error={e} duration_ms={duration_ms:.2f}", exc_info=True)
            raise
        finally:
            self._put_connection(conn)

    @contextmanager
    def transaction(self):
        """
        Context manager for transactions.

        Usage:
            with adapter.transaction() as tx:
                tx.execute(SQLQuery("INSERT INTO ..."))
                tx.execute(SQLQuery("UPDATE ..."))
            # Auto-commits on success, rolls back on exception
        """
        conn = self._get_connection()
        try:
            yield _TransactionContext(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_connection(conn)


class _TransactionContext:
    """Helper class for transaction context."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, query: SQLQuery) -> SQLResult:
        with self._conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(query.sql, query.params)
            return SQLResult(rows=[], row_count=cur.rowcount)

    def fetch_one(self, query: SQLQuery) -> Optional[Dict[str, Any]]:
        with self._conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(query.sql, query.params)
            row = cur.fetchone()
            return dict(row) if row else None

    def fetch_all(self, query: SQLQuery) -> List[Dict[str, Any]]:
        with self._conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(query.sql, query.params)
            return [dict(row) for row in cur.fetchall()]
