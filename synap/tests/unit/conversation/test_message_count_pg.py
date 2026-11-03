"""``ConversationTracker.count_messages`` against a real Postgres.

The SDK decides whether its local short-term cache is complete by comparing
what it holds with this number. So the number has to be the count of exactly
the rows ``/context-for-prompt`` assembles from, which is what
``CompactionStorage.load_conversation_messages`` reads. Both are checked here
against the real ``message`` table, built from the real migrations.

Needs ``SYNAP_TEST_DATABASE_URL``. Everything is created in a schema of its
own and dropped afterwards.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

TEST_DSN = os.environ.get("SYNAP_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(not TEST_DSN, reason="SYNAP_TEST_DATABASE_URL is not set")

_MIGRATIONS = Path(__file__).resolve().parents[3] / "db" / "migrations"
# The enums, the tables `message` has foreign keys to, and `message` itself.
_FILES = (
    "001_enums.sql",
    "002_setup_entities.sql",
    "003_core_entities.sql",
)


def _raw(dsn: str) -> str:
    return dsn.replace("postgresql+asyncpg://", "postgresql://").replace("+asyncpg", "")


def _async(dsn: str) -> str:
    return _raw(dsn).replace("postgresql://", "postgresql+asyncpg://", 1)


@pytest_asyncio.fixture
async def sessions(monkeypatch):
    """A session factory on a throwaway schema holding the real tables."""
    import asyncpg

    schema = f"msgcount_{uuid.uuid4().hex[:12]}"
    settings = {"search_path": schema}

    # Raw asyncpg for the migrations: a file is many statements, and the
    # SQLAlchemy asyncpg dialect sends everything as one prepared statement.
    conn = await asyncpg.connect(_raw(TEST_DSN), server_settings=settings)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        for name in _FILES:
            await conn.execute((_MIGRATIONS / name).read_text())
    finally:
        await conn.close()

    engine = create_async_engine(
        _async(TEST_DSN), connect_args={"server_settings": settings},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    from synap.cloud.application.compaction import storage as storage_module
    from synap.cloud.application.conversation import tracker as tracker_module

    monkeypatch.setattr(tracker_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(storage_module, "get_session_factory", lambda: factory)
    try:
        yield factory
    finally:
        await engine.dispose()
        conn = await asyncpg.connect(_raw(TEST_DSN))
        try:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        finally:
            await conn.close()


async def _conversation(factory, message_types) -> str:
    """One conversation holding a message of each given type, in order."""
    conversation_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    async with factory() as session:
        # The rows above `message` (client, instance, session) are not what is
        # under test, so their foreign keys are not enforced for this insert.
        await session.execute(text("SET LOCAL session_replication_role = replica"))
        await session.execute(
            text(
                "INSERT INTO conversation (conversation_id, instance_id, user_id) "
                "VALUES (CAST(:c AS uuid), 'inst_test', 'u1')"
            ),
            {"c": conversation_id},
        )
        for i, message_type in enumerate(message_types):
            await session.execute(
                text(
                    "INSERT INTO message "
                    "(conversation_id, session_id, message_type, content, timestamp) "
                    "VALUES (CAST(:c AS uuid), CAST(:s AS uuid), "
                    "        CAST(:t AS message_type), CAST(:body AS jsonb), "
                    "        NOW() + make_interval(secs => :i))"
                ),
                {"c": conversation_id, "s": session_id, "t": message_type,
                 "body": f'"turn {i}"', "i": i},
            )
        await session.commit()
    return conversation_id


@pytest.mark.asyncio
async def test_it_counts_the_user_and_assistant_messages(sessions):
    from synap.cloud.application.conversation.tracker import ConversationTracker

    conversation_id = await _conversation(sessions, ["user", "assistant", "user"])
    assert await ConversationTracker().count_messages(conversation_id) == 3


@pytest.mark.asyncio
async def test_a_conversation_with_no_messages_counts_zero(sessions):
    from synap.cloud.application.conversation.tracker import ConversationTracker

    assert await ConversationTracker().count_messages(str(uuid.uuid4())) == 0


@pytest.mark.asyncio
async def test_it_counts_one_conversation_and_not_its_neighbour(sessions):
    from synap.cloud.application.conversation.tracker import ConversationTracker

    first = await _conversation(sessions, ["user", "assistant"])
    await _conversation(sessions, ["user", "assistant", "user", "assistant"])
    assert await ConversationTracker().count_messages(first) == 2


@pytest.mark.asyncio
async def test_it_is_the_number_of_rows_the_prompt_route_reads(sessions):
    """Reasoning and plan rows are stored and are not part of the prompt context.

    That is why the count is a query over ``message`` and not a read of
    ``conversation.message_count``: that column's trigger adds one for every
    insert, whatever the message type.
    """
    from synap.cloud.application.compaction.storage import CompactionStorage
    from synap.cloud.application.conversation.tracker import ConversationTracker

    conversation_id = await _conversation(
        sessions,
        ["user", "assistant_reasoning", "assistant", "assistant_plan", "user"],
    )
    loaded = await CompactionStorage().load_conversation_messages(conversation_id)
    assert await ConversationTracker().count_messages(conversation_id) == len(loaded) == 3
