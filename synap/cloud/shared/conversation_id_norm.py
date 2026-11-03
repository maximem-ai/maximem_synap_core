"""Conversation-ID normalization helper.

`ConversationTracker.record_message` (and friends) coerce non-UUID
``conversation_id`` strings into a deterministic ``uuid5(NAMESPACE_URL, value)``
before any write to Postgres / Redis. Every reader of those rows must apply
the SAME coercion or it will look up the raw key and miss.

Public callers (playground v2, SDK conversation routes, benchmarks, etc.)
hand us arbitrary strings — most commonly ``pgv2_<hex32>``. ``_coerce_uuid``
was historically a private helper inside ``tracker.py``; this module hoists
it to a shared location so reader paths can apply the same coercion without
importing the writer module (avoids cycles).

The helper is idempotent on already-valid UUIDs, so it is safe to call
defensively anywhere a ``conversation_id`` string flows.
"""
import uuid as _uuid


def coerce_conversation_id(value: str, namespace: _uuid.UUID = _uuid.NAMESPACE_URL) -> str:
    """Return ``value`` if it parses as a UUID, else a UUID5 derived from it.

    The helper is idempotent: ``coerce(coerce(x)) == coerce(x)``.

    The same default namespace (``NAMESPACE_URL``) is used as in
    ``ConversationTracker._coerce_uuid`` so the two halves of the
    writer/reader pipeline collide on the same key.
    """
    try:
        _uuid.UUID(value)
        return value
    except (ValueError, AttributeError):
        return str(_uuid.uuid5(namespace, value))
