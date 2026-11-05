-- Graph lifecycle outbox (issue12_graph.md section 5.2 and 6).
--
-- A memory retired or deleted in Postgres must also be released from the graph
-- (Neo4j), and the two stores cannot share a transaction. The memory change
-- writes one row here IN THE SAME TRANSACTION; the graph step runs after commit
-- and marks the row done. A row that is still pending (graph down, process
-- killed) is retried by the drain worker with backoff. Releasing an id that is
-- no longer in the graph changes nothing, so a retry is always safe.
create table if not exists graph_sync_pending (
    id              uuid primary key,
    memory_id       uuid not null,
    client_id       text not null,
    instance_id     text,
    customer_id     text,
    user_id         text,
    scope_level     text,
    -- Graph namespaces the memory can be on, computed when the row is written
    -- (the scope chain is not stored anywhere else).
    namespaces      text[] not null default '{}',
    op              text not null check (op in ('retire', 'erase')),
    reason          text not null,
    superseded_by   uuid,
    redact          boolean not null default false,
    ingestion_id    uuid,
    created_at      timestamptz not null default now(),
    attempts        integer not null default 0,
    last_error      text,
    next_attempt_at timestamptz not null default now(),
    leased_until    timestamptz,
    done_at         timestamptz
);

create index if not exists graph_sync_pending_due
    on graph_sync_pending (next_attempt_at) where done_at is null;
create index if not exists graph_sync_pending_ingestion
    on graph_sync_pending (ingestion_id) where done_at is null;
