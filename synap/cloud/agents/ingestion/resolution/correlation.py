"""The request an entity-resolution LLM call is being made for.

The ingestion charge sums usage rows keyed on the request id, so a call that
does not carry it writes tokens nothing can attribute. The three LLM calls in
this package (second look, world lookup, profiles) sit several layers below
Stage 4, behind wrappers whose signatures are shared with synap-er, so the id
travels in a ContextVar instead of through every one of them. A ContextVar
rather than a module global so concurrent ingestions cannot see each other's.
Profile refreshes run as tasks, and a task copies the context it is created in.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

_request_id: ContextVar[Optional[str]] = ContextVar("entity_resolution_request_id", default=None)


def bind(request_id: Optional[str]) -> Token:
    return _request_id.set(request_id)


def unbind(token: Token) -> None:
    _request_id.reset(token)


def current() -> Optional[str]:
    return _request_id.get()
