"""The copies of a forgotten memory's text that carry no memory id."""

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.memory_forget import erase_copies as ec
from synap.cloud.application.memory_forget.models import Candidate

SCOPE = SimpleNamespace(client_id=uuid.uuid4(), customer_id=uuid.uuid4(),
                        user_id=uuid.uuid4(), instance_id=uuid.uuid4())
MID = "11111111-1111-1111-1111-111111111111"
CONTENT = {MID: 'The user\'s number is "512-555-0142"'}


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeSession:
    def __init__(self, candidate_rows=(), fail_on=()):
        self.candidate_rows, self.fail_on = list(candidate_rows), set(fail_on)
        self.statements, self.commits = [], 0

    async def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        for needle in self.fail_on:
            if needle in sql:
                raise RuntimeError(f"{needle} unavailable")
        self.statements.append((sql, params))
        if sql.startswith("SELECT id, content, memory_type FROM candidates"):
            return _Rows(self.candidate_rows)
        return _Rows([])

    async def commit(self):
        self.commits += 1

    def ran(self, prefix):
        return [(s, p) for s, p in self.statements if s.startswith(prefix)]


def pick_containing(word):
    async def judge(description, cands):
        return [c for c in cands if word in c.content]
    return judge


# ------------------------------------------------------------------ payload patterns

def test_payload_patterns_cover_the_id_and_both_spellings_of_the_text():
    pats = ec.payload_patterns([MID], CONTENT)
    assert f"%{MID}%" in pats
    assert '%The user\'s number is "512-555-0142"%' in pats
    assert '%The user\'s number is \\\\"512-555-0142\\\\"%' in pats     # JSON-escaped quotes, LIKE-escaped


def test_a_very_short_memory_text_is_matched_by_id_only():
    pats = ec.payload_patterns([MID], {MID: "ok"})
    assert pats == [f"%{MID}%"]


def test_like_metacharacters_in_a_memory_text_are_escaped():
    pats = ec.payload_patterns([], {MID: "100% sure_thing"})
    assert any("100\\% sure\\_thing" in p for p in pats)


# ------------------------------------------------------------------ pure profile logic

PROFILE = {
    "attributes": {"phone": {"value": "512-555-0142"}, "language": {"value": "en"}},
    "extras": {"contact_note": "call after 5", "pet": "dog"},
    "overview": "A user in Austin, reachable on 512-555-0142.",
}


def test_every_part_of_a_profile_is_a_candidate():
    ids = [c.memory_id for c in ec.profile_candidates(PROFILE)]
    assert ids == ["attributes:phone", "attributes:language", "extras:contact_note",
                   "extras:pet", "overview"]


def test_strip_profile_removes_only_what_was_picked_and_blanks_a_picked_overview():
    out = ec.strip_profile(PROFILE, ["attributes:phone", "overview"])
    assert "phone" not in out["attributes"] and "language" in out["attributes"]
    assert out["overview"] == "" and out["extras"] == PROFILE["extras"]
    assert PROFILE["attributes"]["phone"]                       # the input is not mutated


def test_strip_profile_with_nothing_picked_changes_nothing():
    assert ec.strip_profile(PROFILE, []) == PROFILE


# ------------------------------------------------------------------ the copy steps

class FakeVector:
    def __init__(self, docs=None, fail_search_on=(), fail_delete_on=()):
        self.docs = docs or {}
        self.fail_search_on, self.fail_delete_on = set(fail_search_on), set(fail_delete_on)
        self.searches, self.deleted = [], []

    async def search_in(self, qvec, scope, collection, top_k, filters):
        if collection in self.fail_search_on:
            raise RuntimeError("chroma down")
        self.searches.append((collection, filters, top_k))
        return [{"id": i, "content": c, "score": 0.9, "metadata": {}}
                for i, c in self.docs.get(collection, {}).items()][:top_k]

    async def delete_in(self, scope, collection, ids):
        if collection in self.fail_delete_on:
            raise RuntimeError("chroma down")
        self.deleted.append((collection, list(ids)))
        return len(ids)


class FakeProfileStore:
    def __init__(self, profile=PROFILE):
        self.profile, self.written = profile, None

    async def resolve_owner_by_scope_uuids(self, client, user, customer):
        return ("user", "u1")

    async def read(self, owner):
        return self.profile

    async def write(self, owner, merge):
        self.written = merge(self.profile)
        return self.written


QV = [0.1, 0.2]


def cand(store, id_, content=""):
    return Candidate(id_, content, store=store)


# ------------------------------------------------------------------ gathering

async def test_every_store_contributes_candidates_tagged_with_where_they_live():
    vec = FakeVector({
        "chunks": {"c1": "my number is 512-555-0142"},
        "sentences": {"s1": "My number is 512-555-0142."},
        "entities": {"e1": "512-555-0142"},
        "triples": {"t1": "user HAS_PHONE 512-555-0142"},
    })
    rows = [(uuid.uuid4(), "number 512-555-0142", "fact")]
    cands, unreadable = await ec.gather_copies(
        FakeSession(candidate_rows=rows), SCOPE, QV, vector=vec, profile_store=FakeProfileStore())
    stores = {c.store for c in cands}
    assert stores == {"chunk", "sentence", "entity", "triple", "profile", "candidate"}
    assert unreadable == []


async def test_each_collection_is_searched_once_for_this_user_with_a_small_page():
    vec = FakeVector({"chunks": {"c1": "x y z"}})
    await ec.gather_copies(FakeSession(), SCOPE, QV, vector=vec)
    assert [c for c, _, _ in vec.searches] == list(ec.VECTOR_COLLECTIONS)
    assert all(f == {"user_id": str(SCOPE.user_id)} and k == ec.PER_STORE_HITS
               for _, f, k in vec.searches)


async def test_empty_text_is_not_a_candidate():
    vec = FakeVector({"chunks": {"c1": "   ", "c2": "real text"}})
    cands, _ = await ec.gather_copies(FakeSession(), SCOPE, QV, vector=vec)
    assert [c.memory_id for c in cands] == ["c2"]


async def test_a_store_that_cannot_be_searched_is_named_and_the_others_still_run():
    vec = FakeVector({"sentences": {"s1": "My number is 512-555-0142."}},
                     fail_search_on=("chunks",))
    cands, unreadable = await ec.gather_copies(FakeSession(), SCOPE, QV, vector=vec)
    assert unreadable == ["chunks"]
    assert [c.memory_id for c in cands] == ["s1"]


async def test_without_a_vector_store_the_collections_are_named_not_skipped():
    _, unreadable = await ec.gather_copies(FakeSession(), SCOPE, QV, vector=None)
    assert unreadable == list(ec.VECTOR_COLLECTIONS)


async def test_the_candidates_table_failing_is_named():
    _, unreadable = await ec.gather_copies(
        FakeSession(fail_on=["FROM candidates"]), SCOPE, QV, vector=FakeVector())
    assert unreadable == ["candidates"]


async def test_nothing_is_judged_while_gathering():
    # gather_copies has no judge parameter at all: pooling and the single call
    # happen in the service.
    import inspect
    assert "judge" not in inspect.signature(ec.gather_copies).parameters


# ------------------------------------------------------------------ erasing the picked

async def erase(session=None, **kw):
    session = session or FakeSession()
    args = dict(scope=SCOPE, memory_ids=[MID], contents=CONTENT, copies=[],
                vector=FakeVector(), profile_store=FakeProfileStore())
    args.update(kw)
    failed = await ec.erase_other_copies(session, **args)
    return session, failed, args


async def test_the_audit_log_is_blanked_for_the_memory_and_for_rows_it_replaced():
    session, failed, _ = await erase()
    sql, params = session.ran("UPDATE memory_actions")[0]
    assert "content = NULL, previous_content = NULL" in sql
    assert "memory_id = ANY" in sql and "replaced_by_memory_id = ANY" in sql
    assert params["ids"] == [MID] and params["client_id"] == str(SCOPE.client_id)
    assert failed == []


async def test_loss_manifests_are_deactivated_and_emptied_for_this_user_only():
    session, _, _ = await erase()
    sql, params = session.ran("UPDATE loss_manifests")[0]
    assert "active = FALSE" in sql and "dropped_details = '[]'" in sql
    assert "summary_memory_id = ANY" in sql and "source_memory_ids &&" in sql
    assert params["customer_id"] == str(SCOPE.customer_id)


async def test_saved_payloads_are_scoped_to_client_and_instance_and_matched_on_id_and_text():
    session, _, _ = await erase()
    sql, params = session.ran("UPDATE request_payloads")[0]
    # Real rows have NULL customer_id and user_id, so neither can be a predicate.
    assert "customer_id" not in sql and "user_id" not in sql
    assert "client_id = CAST(:client_id" in sql and "instance_id" in sql
    assert params["instance_id"] == str(SCOPE.instance_id)
    assert f"%{MID}%" in params["pats"]


async def test_the_logged_copy_of_an_ingest_request_is_left_alone():
    session, _, _ = await erase()
    sql, _ = session.ran("UPDATE request_payloads")[0]
    for path in ("/memories/create", "/memories/batch", "/memories/upload", "ingest"):
        assert f"path NOT LIKE '%{path}%'" in sql


async def test_payloads_that_quote_a_removed_chunk_are_cleared_in_the_same_statement():
    # A fetch served for a SURVIVING memory carries the chunk it came from as
    # evidence; the chunk text is not the forgotten memory's own text.
    chunk = "I said my number is 512-555-0142 to the agent. I also like jazz."
    session, failed, _ = await erase(copies=[cand("chunk", "c1", chunk)])
    updates = session.ran("UPDATE request_payloads")
    assert len(updates) == 1 and failed == []
    assert any(chunk in p for p in updates[0][1]["pats"])
    assert f"%{MID}%" in updates[0][1]["pats"]


async def test_the_database_copies_are_committed_once():
    session, _, _ = await erase(copies=[cand("chunk", "c1", "I said my number is 512-555-0142")])
    assert session.commits == 1


async def test_only_the_picked_candidates_are_deleted():
    picked = uuid.uuid4()
    session, _, _ = await erase(copies=[cand("candidate", str(picked), "number 512-555-0142")])
    sql, params = session.ran("DELETE FROM candidates")[0]
    assert params["ids"] == [str(picked)] and params["user_id"] == str(SCOPE.user_id)


async def test_no_picked_candidate_means_no_delete():
    session, _, _ = await erase(copies=[])
    assert session.ran("DELETE FROM candidates") == []


async def test_the_profile_loses_only_the_picked_parts():
    store = FakeProfileStore()
    _, failed, _ = await erase(
        profile_store=store,
        copies=[cand("profile", "attributes:phone"), cand("profile", "overview")])
    assert failed == []
    assert "phone" not in store.written["attributes"]
    assert "language" in store.written["attributes"] and store.written["extras"]["pet"] == "dog"
    assert store.written["overview"] == ""


async def test_a_profile_nothing_was_picked_from_is_left_alone():
    store = FakeProfileStore()
    await erase(profile_store=store, copies=[cand("chunk", "c1", "x y z w")])
    assert store.written is None


async def test_the_vector_copies_are_deleted_by_exact_id_per_collection():
    vec = FakeVector()
    _, failed, _ = await erase(vector=vec, copies=[
        cand("chunk", "c1", "a b c d e f"), cand("sentence", "s1", "a b c d e f"),
        cand("entity", "e1", "a b c d e f"), cand("triple", "t1", "a b c d e f"),
        cand("triple", "t2", "a b c d e f")])
    assert failed == []
    assert dict(vec.deleted) == {"chunks": ["c1"], "sentences": ["s1"],
                                 "entities": ["e1"], "triples": ["t1", "t2"]}


async def test_a_collection_nothing_was_picked_from_is_not_touched():
    vec = FakeVector()
    await erase(vector=vec, copies=[cand("chunk", "c1", "a b c d e f")])
    assert [c for c, _ in vec.deleted] == ["chunks"]


async def test_picked_vectors_without_a_vector_store_are_named():
    _, failed, _ = await erase(vector=None, copies=[cand("chunk", "c1", "a b c d e f")])
    assert failed == ["chunks"]


async def test_a_collection_that_fails_to_delete_is_named_and_the_others_still_run():
    vec = FakeVector(fail_delete_on=("chunks",))
    _, failed, _ = await erase(vector=vec, copies=[
        cand("chunk", "c1", "a b c d e f"), cand("sentence", "s1", "a b c d e f")])
    assert failed == ["chunks"]
    assert ("sentences", ["s1"]) in vec.deleted


async def test_each_database_step_can_fail_alone():
    picked = [cand("candidate", str(uuid.uuid4()), "number 512-555-0142")]
    for needle, label in (("UPDATE memory_actions", "audit log"),
                          ("UPDATE loss_manifests", "loss manifests"),
                          ("UPDATE request_payloads", "saved payloads"),
                          ("DELETE FROM candidates", "candidates")):
        session = FakeSession(fail_on=[needle])
        done, failed, _ = await erase(session, copies=picked)
        assert failed == [label], needle
        assert len(done.statements) >= 3                         # the rest still ran


async def test_a_failed_commit_is_reported():
    session = FakeSession()

    async def bad_commit():
        raise RuntimeError("commit failed")
    session.commit = bad_commit
    _, failed, _ = await erase(session)
    assert "database copies" in failed


async def test_nothing_in_erase_judges_anything():
    import inspect
    assert "judge" not in inspect.signature(ec.erase_other_copies).parameters
