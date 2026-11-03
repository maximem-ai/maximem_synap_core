"""Forget by description: gather what may hold the text, ask JEV once, soft-delete.

    erase_copies.py   gathers the copies kept outside the memories table, and removes
                      the picked ones by id
    judge.py          JEV decides, in ONE call over the pooled candidates, which of
                      them really are the thing to forget
    soft_delete.py    mark deleted, then remove every copy that could serve the text
    requests_store.py the dashboard's "Deletion requests" queue
    service.py        preview, approve, reject, and direct delete

Delete by ID (DELETE /api/v1/memories/{id}) is a different, hard delete and is
deliberately left alone.
"""

from synap.cloud.application.memory_forget.models import Candidate, ForgetOutcome

__all__ = ["Candidate", "ForgetOutcome"]
