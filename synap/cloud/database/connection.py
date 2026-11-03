"""Database connection management for Synap Cloud.

Provides async SQLAlchemy session management with connection pooling.
"""

import os
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.exc import TimeoutError as SQLTimeoutError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# Global engine and session factory
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    """Get database URL from environment variables.
    
    Returns:
        PostgreSQL connection URL for asyncpg driver.
    """
    # Support both DATABASE_URL and individual components
    database_url = os.environ.get("DATABASE_URL")
    
    if database_url:
        # Convert postgres:// to postgresql+asyncpg://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return database_url
    
    # Build from components
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    user = os.environ.get("DB_USER", "synap")
    password = os.environ.get("DB_PASSWORD", "")
    database = os.environ.get("DB_NAME", "synap")
    
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"


def get_engine() -> AsyncEngine:
    """Get or create the async database engine.
    
    Returns:
        AsyncEngine instance with connection pooling.
    """
    global _engine
    
    if _engine is None:
        _engine = create_async_engine(
            get_database_url(),
            echo=os.environ.get("DB_ECHO", "false").lower() == "true",
            pool_size=int(os.environ.get("DB_POOL_SIZE", "5")),
            max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "10")),
            # Stated rather than inherited. 30 is SQLAlchemy's own default, so this
            # changes nothing, but the pool-timeout signal reports how long a caller
            # waited and that number should come from a value someone can see and set,
            # not from a library default nobody had written down.
            pool_timeout=float(os.environ.get("DB_POOL_TIMEOUT", "30")),
            pool_pre_ping=True,  # Verify connections before use
        )
    
    return _engine


_autocommit_engine: AsyncEngine | None = None


def get_autocommit_engine() -> AsyncEngine:
    """Engine view with AUTOCOMMIT isolation, sharing get_engine()'s pool.

    For single-statement hot-path reads: skips the implicit BEGIN/COMMIT
    pair SQLAlchemy sessions wrap around every statement — two wire round
    trips per query, which is material when the database is cross-cloud
    (tens of ms each). Never use this for multi-statement transactions
    or anything needing row locks.
    """
    global _autocommit_engine

    if _autocommit_engine is None:
        _autocommit_engine = get_engine().execution_options(
            isolation_level="AUTOCOMMIT"
        )

    return _autocommit_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.
    
    Returns:
        Session factory for creating new database sessions.
    """
    global _async_session_factory
    
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    
    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions.
    
    Yields:
        AsyncSession that auto-commits on success and rolls back on error.
    
    Usage:
        @router.get("/users")
        async def get_users(session: AsyncSession = Depends(get_async_session)):
            ...
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except SQLTimeoutError as exc:
            # Pool exhaustion is invisible to the database: the caller never got a
            # connection, so nothing appears in pg_stat_activity and the connection-count
            # half of the Postgres signal stays calm while every request queues behind a
            # 30 second wait. It surfaces everywhere as latency and nowhere as an error,
            # which is exactly the shape this signal was written for. SQLAlchemy puts the
            # pool geometry in the message and nothing machine-readable anywhere, so the
            # event is emitted as fields here.
            await session.rollback()
            # Imported here, not at module level (a cycle through the shared
            # package otherwise). Cloud's hook calls
            # observability.request_fields.log_pool_timeout; Core logs nothing.
            from synap.cloud.shared.hook_signals import log_pool_timeout

            log_pool_timeout(
                "sqlalchemy_async",
                waited_seconds=float(os.environ.get("DB_POOL_TIMEOUT", "30")),
                size=int(os.environ.get("DB_POOL_SIZE", "5"))
                + int(os.environ.get("DB_MAX_OVERFLOW", "10")),
            )
            raise
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions.
    
    Yields:
        AsyncSession that auto-commits on success and rolls back on error.
    
    Usage:
        async with get_db_session() as session:
            result = await session.execute(...)
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize the database connection.
    
    Call this on application startup to verify connectivity.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        # Just verify we can connect
        await conn.run_sync(lambda _: None)


async def close_db() -> None:
    """Close the database connection pool.
    
    Call this on application shutdown.
    """
    global _engine, _async_session_factory
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
