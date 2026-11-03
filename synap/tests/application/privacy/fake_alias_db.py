"""An in-memory stand-in for the two alias tables.

Worth writing rather than mocking, because the things being tested are things a
mock would simply agree with: that a unique index actually stops a second row,
that a collision on one index is told apart from a collision on the other, and
that a savepoint leaves the transaction usable afterwards. A mock that returns
whatever the test wants proves none of that.

It enforces both unique indexes from migration 088 and raises the same
`IntegrityError` SQLAlchemy would, so the retry path in `alias_store` runs for
real.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import IntegrityError


class _Result:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self.rowcount = len(rows)

    def mappings(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows

    def scalar(self):
        if not self._rows:
            return None
        return list(self._rows[0].values())[0]


class FakeAliasDB:
    """A session-shaped object over two dictionaries."""

    def __init__(self):
        self.client_keys: Dict[tuple, Dict[str, Any]] = {}
        self.aliases: List[Dict[str, Any]] = []
        self.commits = 0
        self.statements: List[str] = []
        # Set to a label to make the next insert collide on the label index, so
        # the retry path can be exercised without waiting for 10^17 rows.
        self.force_label_collision: Optional[str] = None
        self._in_savepoint = False

    # ---- session protocol -------------------------------------------------

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None

    @asynccontextmanager
    async def _savepoint(self):
        self._in_savepoint = True
        try:
            yield self
        finally:
            self._in_savepoint = False

    def begin_nested(self):
        return self._savepoint()

    async def execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.statements.append(sql)
        params = params or {}

        if "INSERT INTO pii_client_key" in sql:
            return self._insert_client_key(params)
        if "FROM pii_client_key" in sql and "SELECT wrapped_key" in sql:
            return self._select_client_key(params)
        if "DELETE FROM pii_client_key" in sql:
            return self._delete_client_keys(params)
        if "UPDATE pii_client_key" in sql:
            return self._update_client_key(params)
        if "SELECT client_id, key_version, wrapped_key" in sql:
            return self._select_keys_for_rotation(params)
        if "SELECT master_version, key_version" in sql:
            return self._key_operations()

        if "DELETE FROM pii_alias" in sql:
            return self._delete_aliases(sql, params)
        if "INSERT INTO pii_alias" in sql:
            return self._insert_alias(params)
        if "UPDATE pii_alias" in sql:
            return self._bump(params)
        if "SELECT alias_key, label" in sql:
            return self._select_by_alias_key(params)
        if "SELECT label, alias_key, ciphertext" in sql:
            return self._select_by_label(params)
        if "SELECT field_type, category, sharing_scope" in sql:
            return self._alias_summary(params)

        return _Result([])

    # ---- pii_client_key ---------------------------------------------------

    def _insert_client_key(self, p):
        key = (str(p["c"]), int(p["v"]))
        if key in self.client_keys:
            return _Result([])  # ON CONFLICT DO NOTHING
        self.client_keys[key] = {
            "client_id": str(p["c"]),
            "key_version": int(p["v"]),
            "wrapped_key": p["w"],
            "nonce": p["n"],
            "master_version": int(p["m"]),
            "created_at": datetime.now(timezone.utc),
            "retired_at": None,
        }
        return _Result([])

    def _select_client_key(self, p):
        row = self.client_keys.get((str(p["c"]), int(p["v"])))
        if not row or row["retired_at"] is not None:
            return _Result([])
        return _Result([dict(row)])

    def _delete_client_keys(self, p):
        gone = [k for k in self.client_keys if k[0] == str(p["c"])]
        for k in gone:
            del self.client_keys[k]
        return _Result([{"n": i} for i in range(len(gone))])

    def _update_client_key(self, p):
        row = self.client_keys.get((str(p["c"]), int(p["v"])))
        if row:
            row["wrapped_key"] = p["w"]
            row["nonce"] = p["n"]
            row["master_version"] = int(p["m"])
        return _Result([])

    def _select_keys_for_rotation(self, p):
        target = int(p["target"])
        return _Result(
            [
                dict(r)
                for r in self.client_keys.values()
                if r["master_version"] != target and r["retired_at"] is None
            ]
        )

    def _key_operations(self):
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for r in self.client_keys.values():
            if r["retired_at"] is not None:
                continue
            k = (r["master_version"], r["key_version"])
            g = grouped.setdefault(
                k,
                {
                    "master_version": r["master_version"],
                    "key_version": r["key_version"],
                    "clients": 0,
                    "oldest": r["created_at"],
                },
            )
            g["clients"] += 1
        return _Result(sorted(grouped.values(), key=lambda g: (g["master_version"], g["key_version"])))

    # ---- pii_alias --------------------------------------------------------

    def _insert_alias(self, p):
        client_id = str(p["c"])
        label = self.force_label_collision or p["label"]
        if self.force_label_collision:
            self.force_label_collision = None

        for row in self.aliases:
            if row["client_id"] == client_id and row["alias_key"] == p["ak"]:
                raise IntegrityError(
                    "idx_pii_alias_key", {}, Exception("duplicate key alias_key")
                )
            if (
                row["client_id"] == client_id
                and row["label"] == label
                and row["sharing_scope"] == p["scope"]
            ):
                raise IntegrityError(
                    "idx_pii_alias_label", {}, Exception("duplicate key label")
                )

        self.aliases.append(
            {
                "client_id": client_id,
                "alias_key": p["ak"],
                "label": label,
                "sharing_scope": p["scope"],
                "instance_id": p["instance_id"],
                "customer_id": p["customer_id"],
                "user_id": p["user_id"],
                "field_type": p["field_type"],
                "category": p["category"],
                "ciphertext": p["ct"],
                "nonce": p["nonce"],
                "key_version": int(p["kv"]),
                "occurrences": 1,
                "first_seen_at": datetime.now(timezone.utc),
                "last_seen_at": datetime.now(timezone.utc),
            }
        )
        return _Result([])

    def _delete_aliases(self, sql, p):
        """Applies the clauses that are actually in the SQL, and only those.

        Reads the statement rather than reproducing what it is expected to say.
        The first version of this hard-coded the two shapes `forget_person`
        builds, and that quietly defeated the point: widening the real WHERE to
        drop the scope check left the tests green, because the fake kept
        applying a filter the code no longer had. A test double that disagrees
        with the code in the safe direction is worse than no double at all.
        """
        conditions = []

        # ⚠️ The client clause is read out of the statement like every other
        # clause, and it was not.
        #
        # The docstring above was written after this double was caught
        # hardcoding the scope check, and the fix stopped one clause short: the
        # client filter kept being applied on the double's own authority. So
        # `forget_person` could drop `client_id = CAST(:c AS UUID)` entirely and
        # the suite stayed green, while forgetting one person would have deleted
        # the aliases of every person on the platform sharing their user id.
        #
        # Third time in this class. The rule, stated rather than implied: a
        # double reads the statement, all of it. A clause it applies by itself
        # is a clause nobody is testing.
        if "client_id = CAST(:c AS UUID)" in sql:
            conditions.append(lambda r: r["client_id"] == str(p["c"]))
        if "user_id = :u" in sql:
            conditions.append(lambda r: r["user_id"] == str(p["u"]))
        if "customer_id = :cu" in sql:
            conditions.append(lambda r: r["customer_id"] == str(p["cu"]))
        if "sharing_scope = 'user'" in sql:
            conditions.append(lambda r: r["sharing_scope"] == "user")
        if "sharing_scope IN ('user', 'customer')" in sql:
            conditions.append(lambda r: r["sharing_scope"] in ("user", "customer"))

        def matches(row):
            return all(condition(row) for condition in conditions)

        before = len(self.aliases)
        self.aliases = [r for r in self.aliases if not matches(r)]
        return _Result([{"n": i} for i in range(before - len(self.aliases))])

    def _bump(self, p):
        keys = set(p["keys"])
        for row in self.aliases:
            if row["client_id"] == str(p["c"]) and row["alias_key"] in keys:
                row["occurrences"] += int(p["n"])
                row["last_seen_at"] = datetime.now(timezone.utc)
        return _Result([])

    def _select_by_alias_key(self, p):
        keys = set(p["keys"])
        return _Result(
            [
                {
                    "alias_key": r["alias_key"],
                    "label": r["label"],
                    "field_type": r["field_type"],
                    "sharing_scope": r["sharing_scope"],
                    "key_version": r["key_version"],
                }
                for r in self.aliases
                if r["client_id"] == str(p["c"]) and r["alias_key"] in keys
            ]
        )

    def _select_by_label(self, p):
        labels = set(p["labels"])
        return _Result(
            [
                {
                    "label": r["label"],
                    "alias_key": r["alias_key"],
                    "ciphertext": r["ciphertext"],
                    "nonce": r["nonce"],
                    "key_version": r["key_version"],
                    "field_type": r["field_type"],
                }
                for r in self.aliases
                if r["client_id"] == str(p["c"]) and r["label"] in labels
            ]
        )

    def _alias_summary(self, p):
        grouped: Dict[tuple, Dict[str, Any]] = {}
        for r in self.aliases:
            if r["client_id"] != str(p["c"]):
                continue
            k = (r["field_type"], r["category"], r["sharing_scope"])
            g = grouped.setdefault(
                k,
                {
                    "field_type": r["field_type"],
                    "category": r["category"],
                    "sharing_scope": r["sharing_scope"],
                    "distinct_values": 0,
                    "total_occurrences": 0,
                    "last_seen": r["last_seen_at"],
                    "not_stored": 0,
                },
            )
            g["distinct_values"] += 1
            g["total_occurrences"] += r["occurrences"]
            if r["ciphertext"] is None:
                g["not_stored"] += 1
        return _Result(list(grouped.values()))

    # ---- helpers for assertions ------------------------------------------

    def raw_bytes(self) -> bytes:
        """Every byte the alias table holds, concatenated.

        Used by the canary sweeps: a value present anywhere in here, in any
        column, is a value that would be in the database.
        """
        blob = b""
        for r in self.aliases:
            for column in ("alias_key", "ciphertext", "nonce"):
                if r[column]:
                    blob += bytes(r[column])
            blob += r["label"].encode("utf-8")
            blob += r["field_type"].encode("utf-8")
        for r in self.client_keys.values():
            blob += bytes(r["wrapped_key"]) + bytes(r["nonce"])
        return blob
