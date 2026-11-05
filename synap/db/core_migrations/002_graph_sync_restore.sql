-- Undo of a supersession on the graph side (owner decision: deleting a newer
-- fact restores the older fact it replaced).
--
-- When memory N is deleted, the foreign key on memories.superseded_by clears
-- it on every memory S that N had superseded, so S is alive again. The graph
-- must put S back on the records that released it. Once N's delete commits
-- nobody can tell which memories N had superseded, so the erase row carries
-- them, read in the same transaction before the delete.
alter table graph_sync_pending
    add column if not exists restore_memory_ids uuid[] not null default '{}';
