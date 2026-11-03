"""Let existing KU tests keep asserting on the calls they were written for.

The coordinator used to retire a memory with `soft_delete_memory` or
`set_deprecated_at`, then write the trail row with `insert_memory_action` in a
second transaction. It now does all of it in one call,
`retire_memory_with_audit`, so that a memory cannot be retired without the
before-state a reversal needs.

The tests around it assert on the old calls, and those assertions are still
the right ones: the same rows must still be written, with the same arguments,
in the same cases. So rather than rewriting them to a new shape, this shim
gives a double the new method and implements it by calling the old ones. What
each test checks is unchanged; only the route to it is.
"""
from __future__ import annotations


def attach_retire_shim(rdbms):
    """Give a test double `retire_memory_with_audit`, in terms of its own
    existing methods. Returns the double for chaining."""

    async def retire_memory_with_audit(
        *, memory_id, action, scope, previous_content, content=None,
        memory_type=None, superseded_by=None, replaced_by_memory_id=None,
        correlation_id=None, ingestion_id=None, reason=None,
        deprecate_with_link=False, link_kind="DEPRECATES",
        link_to_memory_id=None, require_retire=True,
    ):
        if superseded_by is not None:
            retired = await rdbms.soft_delete_memory(
                memory_id=memory_id, superseded_by=superseded_by)
        else:
            retired = await rdbms.set_deprecated_at(memory_id=memory_id)

        if require_retire and not retired:
            return False

        if deprecate_with_link and superseded_by is not None:
            await rdbms.set_deprecated_at(memory_id=memory_id)
            await rdbms.insert_memory_link(
                from_memory_id=memory_id,
                to_memory_id=link_to_memory_id or superseded_by,
                kind=link_kind, reason=reason)

        # Unconditional now, which is the whole point of the change: the audit
        # is no longer skipped when there is no correlation id.
        await rdbms.insert_memory_action(
            correlation_id=correlation_id, ingestion_id=ingestion_id,
            scope=scope, action=action, memory_id=memory_id,
            replaced_by_memory_id=replaced_by_memory_id or superseded_by,
            previous_content=previous_content, content=content,
            memory_type=memory_type, reason=reason)
        return True

    rdbms.retire_memory_with_audit = retire_memory_with_audit
    return rdbms
