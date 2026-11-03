"""No id read out of the database goes back through an external-id mapper.

WHY THIS IS A REPO-WIDE GUARD AND NOT TWO ASSERTIONS

`test_delete_targets_the_right_collection.py` pins the two routes where this
was found. This file is the class of defect, because the class is what cost
four months: the same one-line mistake produced a delete that deleted nothing,
an update that updated nothing readable, and 561 phantom customer rows, and
none of the three raised.

The shape is simple. Some functions map an EXTERNAL id, what a client sends, to
an INTERNAL uuid, usually `uuid5(client_uuid, external_id)`. Feed one an id
that already came out of a table and it hashes a uuid into a new uuid that
belongs to nobody, and, because the mappers auto-create on first sight, writes
a row for it.

HOW THE ALLOWLIST WORKS, AND WHY THERE IS ONE

Some tables genuinely store external ids. `playground_session_v2.user_id` is
`Text` and holds `user_pgv2_...`, so mapping it is correct. Verified against
stored rows rather than by reading the comment above it, because the
comment would say the same thing either way. Each entry names its reason, and
adding one is a deliberate act that shows up in review.

THIS CHECKER WAS WRONG ON ITS FIRST RUN

It walked only the outermost expression, so it missed both of the sites it was
written for: they are written `str(existing_row.user_id) if ... else None` and
the walk stopped at the conditional. It found 2 of 6. Now it searches the whole
argument expression. A checker that misses the case it exists for is worse than
none, because it reports zero and is believed.
"""
from __future__ import annotations

import ast
import os

# Functions whose job is to turn an external id into an internal uuid.
MAPPERS = {
    "resolve_customer", "resolve_user", "resolve_identity",
    "customer_deterministic_uuid", "user_deterministic_uuid", "uuid5_hash",
    "_resolve_scope", "_resolve_scope_from_conversation",
    "resolve_chain_rungs", "resolve_chain_rungs_for_path",
}
# Names that mean "this came out of a table" rather than "a client sent this".
ROW_HINTS = ("row", "record", "rec", "existing", "conv", "conversation",
             "memory", "msg", "message", "job", "entry", "item", "doc", "mem")
ID_SLOTS = ("customer_id", "user_id", "external_id", "external_customer_id",
            "external_user_id", "customer_uuid", "user_uuid")

# path:function -> why this one is genuinely an external id.
ALLOWED = {
    "synap/cloud/application/admin_api/routes/playground_v2.py:resolve_user":
        "playground_session_v2.user_id and .customer_id are Text columns "
        "holding the visitor's EXTERNAL ids (user_pgv2_..., cust_pgv2_...), "
        "confirmed against production rows. Mapping them is correct.",
    "synap/cloud/application/admin_api/routes/playground_v2.py:resolve_customer":
        "Same two Text columns, same reason, reached from `switch-rung`, which "
        "moves a session to a sibling rung and needs the tenant's data-plane "
        "uuid for the N-14 node hint. Added deliberately rather than dodged by "
        "renaming the local: `customer_deterministic_uuid(client_uuid, "
        "row.customer_id)` would compute the same value without a round trip "
        "and would trip this checker identically, and it would be WORSE, "
        "because it derives the hint instead of reading the value the customer "
        "row actually carries. A derived hint that disagrees with the stored "
        "one is exactly B-15: both read arms miss and the visitor gets HTTP "
        "200 with nothing in it.",
}

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "cloud")


def _row_names(expr):
    out = set()
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and any(h in n.id.lower() for h in ROW_HINTS):
            out.add(n.id)
        if isinstance(n, ast.Attribute) and any(h in n.attr.lower() for h in ROW_HINTS):
            out.add(n.attr)
    return out


def _scan():
    hits = []
    for base, _, files in os.walk(ROOT):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(base, f)
            try:
                tree = ast.parse(open(p).read())
            except SyntaxError:
                continue
            rel = os.path.relpath(p, os.path.join(ROOT, "..", "..")).replace(os.sep, "/")
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                fn = (n.func.attr if isinstance(n.func, ast.Attribute)
                      else getattr(n.func, "id", None))
                if fn not in MAPPERS:
                    continue
                for kw in n.keywords:
                    if kw.arg in ID_SLOTS and _row_names(kw.value):
                        hits.append((rel, n.lineno, fn, kw.arg,
                                     ",".join(sorted(_row_names(kw.value)))))
    return sorted(set(hits))


def test_no_stored_id_is_fed_back_into_a_mapper():
    offenders = [h for h in _scan() if f"{h[0]}:{h[2]}" not in ALLOWED]
    assert not offenders, (
        "an id read out of the database is being mapped as if a client had "
        "sent it, which hashes a uuid into a new one and creates a phantom "
        "row:\n" + "\n".join(f"  {p}:{ln}  {fn}({slot}= ... from `{src}`)"
                             for p, ln, fn, slot, src in offenders)
    )


def test_the_checker_can_actually_see_the_shape_it_looks_for():
    """The positive control, in the same run.

    A guard that finds nothing and a guard that looks for nothing are the same
    output. This one asserts the scanner still detects the original defect when
    it is written out, so a green result above means something.
    """
    sample = ast.parse(
        "await _resolve_scope(auth, "
        "user_id=str(existing_row.user_id) if existing_row.user_id else None, "
        "customer_id=str(existing_row.customer_id) if existing_row.customer_id else None)"
    )
    found = []
    for n in ast.walk(sample):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) in MAPPERS:
            for kw in n.keywords:
                if kw.arg in ID_SLOTS and _row_names(kw.value):
                    found.append(kw.arg)
    assert sorted(found) == ["customer_id", "user_id"], (
        "the scanner no longer recognises the defect it was written for, so a "
        "clean run above proves nothing"
    )
