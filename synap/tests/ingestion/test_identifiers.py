"""Names with an identifier, settled by it (resolution/identifiers)."""

import uuid
from types import SimpleNamespace

from synap_er.pipeline.adjudication import Decision

from synap.cloud.agents.ingestion.resolution import identifiers


def candidate(name, kind, aliases=()):
    cid = uuid.uuid4()
    entry = SimpleNamespace(aliases=list(aliases), metadata_={"entity_subtype": kind})
    return SimpleNamespace(canonical_id=cid, canonical_name=name), (cid, entry)


def resolve(names, kind, *candidates, decision=Decision.AMBIGUOUS, path="adjudicated"):
    entries = dict(e for _, e in candidates)
    res = SimpleNamespace(decision=decision, path=path, candidates=[c for c, _ in candidates],
                          canonical_id=None, canonical_name=None, same_as_index=None)
    return identifiers.settle(res, names, kind, entries.get), res


def test_what_counts_as_an_identifier():
    assert identifiers.identifiers(["Invoice INV-7802", "Visa ending 4421", "workspace bexley-main"]) == {
        "inv-7802", "4421", "bexley-main"}
    assert identifiers.identifiers(["Bexley Studios", "the workspace", "Dr. Imogen Hartley"]) == set()


def test_the_same_identifier_is_the_same_thing():
    how, res = resolve(["INV-7802"], "invoice", candidate("Invoice INV-7802", "invoice"))
    assert how == "match" and res.decision is Decision.MATCH and res.canonical_name == "Invoice INV-7802"


def test_an_identifier_against_a_different_kind_of_thing_is_new():
    how, res = resolve(["bexley-main"], "workspace", candidate("Bexley Studios", "organization"),
                       candidate("Starter plan", "plan"))
    assert how == "new" and res.decision is Decision.NEW


def test_different_identifiers_are_different_things():
    how, _ = resolve(["INV-7802"], "invoice", candidate("INV-7710", "invoice"))
    assert how == "new"


def test_what_identifiers_cannot_decide_is_left_alone():
    # a generic word against an identifier of the same kind
    assert resolve(["workspace"], "workspace", candidate("workspace corliss-ops", "workspace"))[0] is None
    # no identifier on either side
    assert resolve(["annual plans"], "plan", candidate("Business plan", "plan"))[0] is None
    # one candidate decided, one not
    assert resolve(["bexley-main"], "workspace", candidate("Bexley Studios", "organization"),
                   candidate("main workspace", "workspace"))[0] is None
    # a match the spelling already made
    assert resolve(["INV-7802"], "invoice", candidate("INV-7710", "invoice"),
                   decision=Decision.MATCH, path="exact")[0] is None


def test_a_longer_name_of_another_kind_is_a_different_thing():
    c, (cid, entry) = candidate("Chordline", "organization")
    res = SimpleNamespace(decision=Decision.MATCH, path="adjudicated", candidates=[c],
                          canonical_id=cid, canonical_name="Chordline", same_as_index=None)
    assert identifiers.keep_apart(res, ["Chordline refund policy"], "policy", {cid: entry}.get) == "new"
    assert res.decision is Decision.NEW and res.canonical_id is None


def test_a_longer_name_of_the_same_or_an_unknown_kind_is_left_alone():
    for kind in ("organization", ""):
        c, (cid, entry) = candidate("Chordline", "organization")
        res = SimpleNamespace(decision=Decision.MATCH, path="adjudicated", candidates=[c],
                              canonical_id=cid, canonical_name="Chordline", same_as_index=None)
        assert identifiers.keep_apart(res, ["Chordline Inc"], kind, {cid: entry}.get) is None
    # another kind, but not one name extending the other: "company" and "organization" may be one thing
    c, (cid, entry) = candidate("Chordline", "organization")
    res = SimpleNamespace(decision=Decision.MATCH, path="adjudicated", candidates=[c],
                          canonical_id=cid, canonical_name="Chordline", same_as_index=None)
    assert identifiers.keep_apart(res, ["Chordline"], "company", {cid: entry}.get) is None


def test_unsure_between_related_things_of_another_kind_is_new():
    a, (ida, ea) = candidate("Lumavate", "drug")
    res = SimpleNamespace(decision=Decision.AMBIGUOUS, path="adjudicated", candidates=[a],
                          canonical_id=None, canonical_name=None, same_as_index=None)
    assert identifiers.keep_apart(res, ["Lumavate refills"], "refill", {ida: ea}.get) == "new"
