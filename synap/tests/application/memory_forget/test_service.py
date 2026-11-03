"""Preview, approve, reject and direct delete, with the stores faked."""

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.memory_forget import requests_store as store
from synap.cloud.application.memory_forget import service
from synap.cloud.application.memory_forget.judge import JudgeUnavailable
from synap.cloud.application.memory_forget.models import Candidate
from synap.cloud.application.memory_forget.soft_delete import SoftDeleteResult


def _scope():
    return SimpleNamespace(client_id=uuid.uuid4(), instance_id=uuid.uuid4(),
                           customer_id=uuid.uuid4(), user_id=uuid.uuid4(), scope_level="user")


class FakeStore:
    """In-memory stand-in for requests_store, patched in over the module functions."""

    def __init__(self):
        self.rows = {}
        self.created = []

    async def create_pending(self, session, *, scope, query, candidates, requested_by):
        rid = str(uuid.uuid4())
        self.rows[rid] = {"id": rid, "client_id": str(scope.client_id),
                          "instance_id": str(scope.instance_id),
                          "customer_id": str(scope.customer_id), "user_id": str(scope.user_id),
                          "scope_level": "user", "query": query, "status": "pending",
                          "candidates": [c.as_dict() for c in candidates], "result": None}
        self.created.append(rid)
        return rid

    async def get_request(self, session, *, client_id, request_id):
        r = self.rows.get(request_id)
        return dict(r) if r and r["client_id"] == client_id else None

    async def claim_pending(self, session, *, client_id, request_id, decided_by, new_status):
        r = self.rows.get(request_id)
        if not r or r["client_id"] != client_id or r["status"] != "pending":
            return False
        r["status"] = new_status
        return True

    async def finish(self, session, *, request_id, status, result):
        r = self.rows[request_id]
        r["status"], r["result"] = status, result
        r["candidates"] = [{"memory_id": c["memory_id"], "store": c.get("store", "memory")}
                           for c in r["candidates"]]


    async def record_direct(self, session, *, scope, query, candidates, requested_by,
                            status, result):
        if getattr(self, "record_raises", False):
            raise RuntimeError("db down")
        rid = str(uuid.uuid4())
        self.rows[rid] = {"id": rid, "client_id": str(scope.client_id), "query": query,
                          "status": status, "requested_by": requested_by, "result": result,
                          "candidates": [{"memory_id": c.memory_id, "store": c.store}
                                         for c in candidates]}
        return rid


class FakeSession:
    """Enough of AsyncSession for the direct path's savepoint."""

    def begin_nested(self):
        class _Savepoint:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False
        return _Savepoint()


@pytest.fixture
def fake_store(monkeypatch):
    fs = FakeStore()
    for name in ("create_pending", "get_request", "claim_pending", "finish", "record_direct"):
        monkeypatch.setattr(store, name, getattr(fs, name))
    return fs


class Deps:
    """The pool is whatever was gathered; the judge flags every candidate whose id is in
    ``flag`` (default: all of them) and counts how many times it was asked."""

    def __init__(self, pool, flag=None, judge_raises=False, delete_result=None,
                 unreadable=(), page_full=False):
        self.pool, self.flag = list(pool), flag
        self.judge_raises, self.delete_result = judge_raises, delete_result
        self.unreadable, self.page_full = list(unreadable), page_full
        self.deleted, self.gathers, self.judge_calls = [], 0, []

    def build(self):
        async def gather(description):
            self.gathers += 1
            mems = [c.memory_id for c in self.pool if c.store == "memory"]
            return service.Gathered(candidates=list(self.pool), unreadable=self.unreadable,
                                    memory_ids_ranked=mems, page_full=self.page_full)

        async def judge(desc, cands):
            self.judge_calls.append(len(cands))
            if self.judge_raises:
                raise JudgeUnavailable("down")
            return [c for c in cands if self.flag is None or c.memory_id in self.flag]

        async def delete(scope, picked, reason):
            self.deleted.append((list(picked), reason))
            if self.delete_result is not None:
                return self.delete_result
            return SoftDeleteResult(deleted_ids=[c.memory_id for c in picked
                                                 if c.store == "memory"])

        return service.ForgetDeps(gather=gather, judge=judge, delete=delete)


PHONE = Candidate("11111111-1111-1111-1111-111111111111", "number is 512-555-0142", "fact")
ADDR = Candidate("22222222-2222-2222-2222-222222222222", "lives at 4 Elm St", "fact")
CHUNK = Candidate("c1", "I said my number is 512-555-0142", "source text", "chunk")
TRIPLE = Candidate("t1", "user HAS_PHONE 512-555-0142", "relationship", "triple")
PROFILE = Candidate("attributes:phone", "phone: 512-555-0142", "profile", "profile")


# ------------------------------------------------------------------ one call per forget

async def test_a_forget_asks_the_judge_once_over_memories_and_copies_together(fake_store):
    d = Deps([PHONE, ADDR, CHUNK, TRIPLE, PROFILE], flag={PHONE.memory_id, "c1", "t1"})
    out = await service.direct(None, scope=_scope(), description="my phone", deps=d.build())
    assert d.judge_calls == [5]                   # ONE call, all five candidates
    assert d.gathers == 1
    assert [c.memory_id for c in d.deleted[0][0]] == [PHONE.memory_id, "c1", "t1"]
    assert out.status == "deleted"


async def test_preview_is_also_one_call(fake_store):
    d = Deps([PHONE, CHUNK])
    await service.preview(None, scope=_scope(), description="x", deps=d.build(),
                          requested_by=None)
    assert d.judge_calls == [1 + 1]


async def test_approving_costs_no_judge_call_and_no_search(fake_store):
    scope = _scope()
    first = Deps([PHONE, ADDR, CHUNK], flag={PHONE.memory_id, "c1"})
    prev = await service.preview(None, scope=scope, description="x", deps=first.build(),
                                 requested_by=None)
    later = Deps([PHONE, ADDR, CHUNK])
    await _approve(fake_store, prev.request_id, later, str(scope.client_id))
    assert later.judge_calls == [] and later.gathers == 0


# ------------------------------------------------------------------ preview

async def test_preview_files_a_request_holding_the_memories_and_the_copies(fake_store):
    d = Deps([PHONE, ADDR, CHUNK, TRIPLE], flag={PHONE.memory_id, "c1", "t1"})
    out = await service.preview(None, scope=_scope(), description="my details",
                                deps=d.build(), requested_by="sdk:k1")
    assert out.status == "preview" and out.request_id == fake_store.created[0]
    assert [(c.memory_id, c.store) for c in out.candidates] == [
        (PHONE.memory_id, "memory"), ("c1", "chunk"), ("t1", "triple")]
    assert d.deleted == []                                  # nothing deleted by a preview
    stored = fake_store.rows[out.request_id]
    assert stored["status"] == "pending"
    assert [c["store"] for c in stored["candidates"]] == ["memory", "chunk", "triple"]


async def test_nothing_matching_files_nothing_and_deletes_nothing(fake_store):
    out = await service.preview(None, scope=_scope(), description="x",
                                deps=Deps([PHONE], flag=set()).build(), requested_by=None)
    assert out.status == "nothing_matches" and fake_store.created == []
    d = Deps([PHONE], flag=set())
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.status == "nothing_matches" and d.deleted == []


async def test_an_empty_pool_is_nothing_matches_without_asking_the_judge(fake_store):
    d = Deps([])
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.status == "nothing_matches" and d.judge_calls == []


async def test_an_unavailable_judge_files_nothing_and_deletes_nothing(fake_store):
    d = Deps([PHONE], judge_raises=True)
    with pytest.raises(JudgeUnavailable):
        await service.preview(None, scope=_scope(), description="x", deps=d.build(),
                              requested_by=None)
    with pytest.raises(JudgeUnavailable):
        await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert fake_store.created == [] and d.deleted == []


# ------------------------------------------------------------------ direct

async def test_direct_deletes_in_one_call_with_no_request(fake_store):
    d = Deps([PHONE, ADDR])
    out = await service.direct(None, scope=_scope(), description="my details", deps=d.build())
    assert out.status == "deleted" and out.deleted_ids == [PHONE.memory_id, ADDR.memory_id]
    assert [c.memory_id for c in d.deleted[0][0]] == [PHONE.memory_id, ADDR.memory_id]
    assert d.deleted[0][1] == service.REASON_DIRECT
    assert fake_store.created == []                        # direct skips the review queue


async def test_direct_reports_a_partial_when_a_store_still_holds_the_text(fake_store):
    d = Deps([PHONE], delete_result=SoftDeleteResult(deleted_ids=[PHONE.memory_id],
                                                     still_present_in=["cache"]))
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.status == "partial" and out.still_present_in == ["cache"]


async def test_a_store_that_could_not_be_searched_makes_a_direct_delete_partial(fake_store):
    d = Deps([PHONE], unreadable=["chunks"])
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.status == "partial" and out.still_present_in == ["chunks"]
    assert out.may_be_incomplete is True


async def test_a_copy_that_is_the_only_match_is_still_removed(fake_store):
    d = Deps([ADDR, PROFILE], flag={"attributes:phone"})
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert [c.store for c in d.deleted[0][0]] == ["profile"]
    assert out.status == "deleted" and out.deleted_ids == []


# ------------------------------------------------------------------ may_be_incomplete

async def test_a_full_page_with_a_match_near_the_bottom_may_be_incomplete(fake_store):
    pool = [Candidate(f"{i:08d}-0000-0000-0000-000000000000", f"m{i}", "fact") for i in range(50)]
    d = Deps(pool, flag={pool[-1].memory_id}, page_full=True)
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.may_be_incomplete is True


async def test_a_full_page_whose_matches_are_all_near_the_top_is_complete(fake_store):
    pool = [Candidate(f"{i:08d}-0000-0000-0000-000000000000", f"m{i}", "fact") for i in range(50)]
    d = Deps(pool, flag={pool[0].memory_id}, page_full=True)
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.may_be_incomplete is False


async def test_a_page_that_was_not_full_is_complete_whatever_matched(fake_store):
    d = Deps([PHONE, ADDR], page_full=False)
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.may_be_incomplete is False


async def test_a_preview_with_an_unreadable_store_says_it_may_be_incomplete(fake_store):
    d = Deps([PHONE], unreadable=["triples"])
    out = await service.preview(None, scope=_scope(), description="x", deps=d.build(),
                                requested_by=None)
    assert out.status == "preview" and out.may_be_incomplete is True


@pytest.mark.parametrize("mode", ["preview", "direct"])
async def test_nothing_matched_but_a_store_was_unreadable_says_it_may_be_incomplete(
        fake_store, mode):
    d = Deps([PHONE], flag=set(), unreadable=["chunks"])
    if mode == "preview":
        out = await service.preview(None, scope=_scope(), description="x", deps=d.build(),
                                    requested_by=None)
    else:
        out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.status == "nothing_matches" and out.may_be_incomplete is True


async def test_nothing_matched_with_every_store_read_is_complete(fake_store):
    d = Deps([PHONE], flag=set())
    out = await service.direct(None, scope=_scope(), description="x", deps=d.build())
    assert out.status == "nothing_matches" and out.may_be_incomplete is False


async def test_a_direct_forget_files_a_decided_request_and_returns_its_id(fake_store):
    d = Deps([PHONE, CHUNK])
    out = await service.direct(FakeSession(), scope=_scope(), description="my phone",
                               deps=d.build(), requested_by="sdk:cred-1")
    assert out.status == "deleted" and out.request_id in fake_store.rows
    row = fake_store.rows[out.request_id]
    assert row["status"] == "applied" and row["requested_by"] == "sdk:cred-1"
    assert row["result"]["deleted"] == [PHONE.memory_id]
    assert all("content" not in c for c in row["candidates"]), "the text is not kept"


async def test_a_direct_forget_that_deleted_nothing_files_a_failed_request(fake_store):
    d = Deps([PHONE], delete_result=SoftDeleteResult(skipped_ids=[PHONE.memory_id]))
    out = await service.direct(FakeSession(), scope=_scope(), description="x", deps=d.build())
    assert fake_store.rows[out.request_id]["status"] == "failed"


async def test_failing_to_file_the_record_does_not_fail_a_done_deletion(fake_store):
    fake_store.record_raises = True
    d = Deps([PHONE])
    out = await service.direct(FakeSession(), scope=_scope(), description="x", deps=d.build())
    assert out.status == "deleted" and out.deleted_ids == [PHONE.memory_id]
    assert out.request_id is None


# ------------------------------------------------------------------ approve / reject

def _scope_of(req):
    return SimpleNamespace(client_id=req["client_id"], customer_id=req["customer_id"],
                           user_id=req["user_id"], instance_id=req["instance_id"],
                           scope_level="user")


async def _approve(fake_store, rid, deps, client_id):
    return await service.approve(
        None, client_id=client_id, request_id=rid, decided_by="dana@acme.test",
        scope_of=_scope_of, deps_for=lambda scope: deps.build())


async def test_approve_deletes_exactly_the_reviewed_items_memories_and_copies(fake_store):
    scope = _scope()
    first = Deps([PHONE, ADDR, CHUNK], flag={PHONE.memory_id, "c1"})
    prev = await service.preview(None, scope=scope, description="x", deps=first.build(),
                                 requested_by=None)
    # By approval time the index would return a different set. It must not matter.
    later = Deps([PHONE, ADDR, CHUNK, Candidate("33333333-3333-3333-3333-333333333333", "new")])
    out = await _approve(fake_store, prev.request_id, later, str(scope.client_id))
    assert out.status == "deleted"
    assert [(c.memory_id, c.store) for c in later.deleted[0][0]] == [
        (PHONE.memory_id, "memory"), ("c1", "chunk")]
    assert later.deleted[0][1] == service.REASON_APPROVED
    row = fake_store.rows[prev.request_id]
    assert row["status"] == "applied"
    assert all(set(c) == {"memory_id", "store"} for c in row["candidates"])   # text blanked


async def test_a_second_approval_does_nothing(fake_store):
    scope = _scope()
    prev = await service.preview(None, scope=scope, description="x",
                                 deps=Deps([PHONE]).build(), requested_by=None)
    d1, d2 = Deps([PHONE]), Deps([PHONE])
    assert await _approve(fake_store, prev.request_id, d1, str(scope.client_id)) is not None
    assert await _approve(fake_store, prev.request_id, d2, str(scope.client_id)) is None
    assert d2.deleted == []


async def test_another_clients_request_cannot_be_approved(fake_store):
    scope = _scope()
    prev = await service.preview(None, scope=scope, description="x",
                                 deps=Deps([PHONE]).build(), requested_by=None)
    d = Deps([PHONE])
    assert await _approve(fake_store, prev.request_id, d, str(uuid.uuid4())) is None
    assert d.deleted == []


async def test_reject_deletes_nothing_and_blocks_a_later_approval(fake_store):
    scope = _scope()
    prev = await service.preview(None, scope=scope, description="x",
                                 deps=Deps([PHONE]).build(), requested_by=None)
    assert await service.reject(None, client_id=str(scope.client_id),
                                request_id=prev.request_id, decided_by="dana") is True
    d = Deps([PHONE])
    assert await _approve(fake_store, prev.request_id, d, str(scope.client_id)) is None
    assert d.deleted == [] and fake_store.rows[prev.request_id]["status"] == "rejected"


async def test_an_approval_whose_memories_were_all_refused_is_marked_failed(fake_store):
    scope = _scope()
    prev = await service.preview(None, scope=scope, description="x",
                                 deps=Deps([PHONE]).build(), requested_by=None)
    d = Deps([PHONE], delete_result=SoftDeleteResult(skipped_ids=[PHONE.memory_id]))
    await _approve(fake_store, prev.request_id, d, str(scope.client_id))
    assert fake_store.rows[prev.request_id]["status"] == "failed"


async def test_an_approval_of_copies_only_is_applied_not_failed(fake_store):
    scope = _scope()
    prev = await service.preview(None, scope=scope, description="x",
                                 deps=Deps([ADDR, CHUNK], flag={"c1"}).build(),
                                 requested_by=None)
    d = Deps([], delete_result=SoftDeleteResult())
    await _approve(fake_store, prev.request_id, d, str(scope.client_id))
    assert fake_store.rows[prev.request_id]["status"] == "applied"


async def test_a_crash_during_approval_marks_the_request_failed_and_reraises(fake_store):
    scope = _scope()
    prev = await service.preview(None, scope=scope, description="x",
                                 deps=Deps([PHONE]).build(), requested_by=None)

    class Boom(Deps):
        def build(self):
            deps = super().build()

            async def delete(scope, picked, reason):
                raise RuntimeError("db down")
            deps.delete = delete
            return deps

    with pytest.raises(RuntimeError):
        await _approve(fake_store, prev.request_id, Boom([PHONE]), str(scope.client_id))
    assert fake_store.rows[prev.request_id]["status"] == "failed"


def test_scope_from_request_uses_the_stored_internal_ids_unchanged():
    pytest.importorskip("langchain_anthropic")   # the real Scope lives behind the MACA import
    ids = {k: str(uuid.uuid4()) for k in ("client_id", "instance_id", "customer_id", "user_id")}
    s = service.scope_from_request({**ids, "scope_level": "user"})
    assert str(s.client_id) == ids["client_id"] and str(s.user_id) == ids["user_id"]
    assert s.scope_level == "user"
