"""Plain data carried between the forget steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Candidate:
    """One live memory of the user, read from Postgres (never from the vector hit)."""

    memory_id: str          # the id inside its own store (a chunk id for a chunk)
    content: str
    memory_type: Optional[str] = None
    # Where it lives. "memory" is the user's stored memory; anything else is a copy of
    # that text kept elsewhere ("chunk", "sentence", "entity", "triple", "profile",
    # "candidate"). One judge call sees all of them together.
    store: str = "memory"

    def as_dict(self) -> Dict[str, Any]:
        return {"memory_id": self.memory_id, "content": self.content,
                "memory_type": self.memory_type, "store": self.store}


@dataclass
class ForgetOutcome:
    # "nothing_matches" | "preview" | "deleted" | "partial"
    status: str
    candidates: List[Candidate] = field(default_factory=list)
    request_id: Optional[str] = None
    deleted_ids: List[str] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)
    # Stores a delete did not reach, named for a person. Empty when all worked.
    still_present_in: List[str] = field(default_factory=list)
    may_be_incomplete: bool = False
    # A direct forget that matched nothing and was queued to be searched again later.
    retry_scheduled: bool = False
