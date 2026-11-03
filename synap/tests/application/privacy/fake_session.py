"""The savepoint half of the session protocol, for the in-memory doubles.

Every statement in the privacy package that degrades on an unapplied migration
runs inside `reporting.savepoint`, which opens `session.begin_nested()`. So a
double that only has `execute` and `commit` can no longer stand in for a
session, and every double in this directory that reaches one of those functions
inherits from here.

**What this models honestly.** A statement that fails inside a savepoint leaves
the session able to run the next one. That is the property the whole fix rests
on, and a double can represent it exactly: `aborted` goes back to False on the
way out, the same way `ROLLBACK TO SAVEPOINT` makes Postgres willing to accept
the next statement.

**What it deliberately does not model.** Undoing rows the failed block already
wrote. A list of dictionaries has no undo, and writing one here would be a
double agreeing with an implementation nobody had checked. So the property that
matters most, that a write made on the same session before the swallow is still
there afterwards, is not asserted against any double in this directory. It is
asserted against a real database in `test_missing_table_savepoints.py`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager


class SavepointSupport:
    """`begin_nested` for a session double.

    The counters are here for tests that want to assert a savepoint was taken
    at all, rather than inferring it from behaviour a stub would also produce.
    """

    # Set by a double whose `execute` models Postgres refusing every statement
    # after a failure, which is the state a savepoint has to get out of.
    aborted = False

    savepoints_opened = 0
    savepoints_rolled_back = 0

    def begin_nested(self):
        return self._savepoint()

    @asynccontextmanager
    async def _savepoint(self):
        self.savepoints_opened += 1
        try:
            yield self
        except BaseException:
            self.savepoints_rolled_back += 1
            self.aborted = False
            raise
