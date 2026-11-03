"""
Scope resolution utilities for multi-tenant context retrieval.

Synap hierarchy: client → instance → customer → user
Context can be retrieved at any level and rolled up/down.
"""

import logging
import os
import uuid as _uuid
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Mapping, Tuple, TYPE_CHECKING

from synap.cloud.shared import edition

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from synap.cloud.shared.id_mapper import IdMapper

NAMESPACE_URL = _uuid.NAMESPACE_URL

# Sentinel stored in vector metadata's `user_id` field for customer-scoped
# shared memories. ChromaDB drops None metadata values silently, so a distinct
# string is the only way to express "this doc is shared across all users of a
# customer" in a way that equality filters (and $in) can match.
CUSTOMER_SCOPE_SENTINEL = "__CUSTOMER_SCOPE__"


# Nested scoping, off until a deployment turns it on. When off, not one line of
# the code below runs and every scope is built exactly as it is today.
async def _ladder_is_ratified(client_uuid) -> bool:
    """A-02. Has a person accepted this client's ladder?

    Never raises. A ladder that cannot prove it was accepted is a draft, which
    is the safe reading: treating an unreadable record as consent would let a
    shape nobody agreed to start deciding who can read whose memories.
    """
    try:
        from synap.cloud.shared.scope_ladder import get_scope_ladder

        return await get_scope_ladder().is_ratified(client_uuid)
    except Exception:  # noqa: BLE001
        return False


async def _nested_scoping_enabled(instance_id: Optional[str] = None) -> bool:
    """Deployment-wide flag, overridable per instance (Z-03).

    Without an instance the answer is the deployment-wide flag alone, which is
    what a caller that has not resolved an instance yet should get: the
    conservative default rather than an assumption about which instance it
    might turn out to be.
    """
    if not edition.nested_scoping:
        # Core has no scope ladder and no rollout gate: the answer is no.
        return False
    if instance_id:
        try:
            from synap.cloud.shared.scope_rollout import get_rollout_gate

            return await get_rollout_gate().enabled_for(instance_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("rollout gate unavailable, falling back to the env flag: %s", exc)
    from synap.cloud.shared.scope_rollout import deployment_enabled

    return deployment_enabled()


async def resolve_chain_rungs(
    client_uuid: _uuid.UUID,
    *,
    external_customer_id: Optional[str] = None,
    external_user_id: Optional[str] = None,
    customer_uuid: Optional[_uuid.UUID] = None,
    user_uuid: Optional[_uuid.UUID] = None,
    instance_id: Optional[str] = None,
    create: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """The chain for a request, as plain rungs, or None.

    The one place that owns the feature flag and the promise never to raise, so
    that neither has to be repeated at each call site. Two models carry a chain,
    `ScopeHierarchy` on the retrieval side and the ingestion `Scope`, and both
    get it from here rather than each resolving separately, because two
    resolutions are two chances to resolve differently.

    Returns None when the feature is off, when the client has no ladder, when
    there is nothing to resolve, or when resolution failed. Every one of those
    is an ordinary request that proceeds exactly as it does today.
    """
    if not await _nested_scoping_enabled(instance_id):
        return None
    if not external_customer_id and not external_user_id:
        return None
    # A-02. An unratified ladder is never used at runtime.
    #
    # Every client is seeded a ladder at signup, so a ladder EXISTING says
    # nothing about whether the client wanted that shape: it is true of every
    # client. Until a person accepts it, this request behaves exactly as it
    # did before the feature existed. No chain, no node_id, the legacy arm
    # scopes it the old way. That is a known-good state, not a degraded one.
    #
    # After the flag check on purpose, not before: an off deployment must not
    # touch the ladder at all, and checking consent for a feature nobody has
    # switched on is a query with no question behind it.
    #
    # Deliberately NOT inside `resolve_legacy_chain`, because the explain panel
    # and the memory detail page call that directly to PREVIEW a chain. A client
    # reviewing a draft has to be able to see what they are being asked to
    # accept.
    if not await _ladder_is_ratified(client_uuid):
        return None
    try:
        from synap.cloud.shared.scope_ladder import (
            chain_to_rungs,
            get_scope_ladder,
            resolve_legacy_chain,
        )

        chain = await resolve_legacy_chain(
            get_scope_ladder(),
            client_uuid,
            external_customer_id=external_customer_id,
            external_user_id=external_user_id,
            customer_uuid=customer_uuid,
            user_uuid=user_uuid,
            create=create,
        )
        return chain_to_rungs(chain) if chain is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("scope chain not resolved for client %s: %s", client_uuid, exc)
        return None


class ScopePathRejected(ValueError):
    """A caller supplied a scope path that cannot be honoured.

    ⚠️ **Subclasses `ValueError` on purpose, and that is load-bearing.** Three
    docstrings promised this becomes a 400: the `scope` field, this class, and
    `_build_scope`. None of them made it happen. Every fetch route catches
    `ValueError` and converts it, and nothing anywhere catches this type, so
    when it inherited from `Exception` a rejected scope path reached FastAPI
    unhandled and became a **500**. An audit caught it; no test asserted a
    status code.

    A bad scope path is a bad value, so `ValueError` is also the honest parent
    rather than a trick to reach a handler.

    Deliberately NOT handled the way `resolve_chain_rungs` handles its failures.
    That function returns None and the request proceeds on today's behaviour,
    which is right when nobody asked for anything in particular.

    An explicit path is different. If a caller says "this conversation is
    happening at customer=acme, team=payments" and we cannot honour it, the only
    safe answers are to refuse or to serve exactly that. Falling back to the
    legacy chain would serve a **wider** scope than was asked for, silently, and
    P-01 exists precisely because a wrong inference at or above the tenant of
    record is a cross-tenant leak. So this raises and the route returns 4xx.
    """


async def resolve_chain_rungs_for_path(
    client_uuid: _uuid.UUID,
    scope_path: Mapping[str, str],
    *,
    instance_id: Optional[str] = None,
    create: bool = True,
    tenant_node_id: Optional[_uuid.UUID] = None,
    identity_node_id: Optional[_uuid.UUID] = None,
) -> List[Dict[str, Any]]:
    """G-03. Resolve an explicit caller-supplied scope path to chain rungs.

    The strict counterpart to `resolve_chain_rungs`. Every failure raises
    `ScopePathRejected`; nothing here ever falls back or infers.

    Including the case where nested scoping is switched OFF for the instance.
    That deserves saying out loud, because "the flag is off so ignore what they
    sent" looks like the tolerant choice and is the dangerous one: the caller
    asked to be narrowed to a rung, and ignoring them serves the whole tenant
    instead. Off means a path cannot be honoured, so a path is an error.
    """
    if not scope_path:
        raise ScopePathRejected("scope path is empty")
    if not await _nested_scoping_enabled(instance_id):
        raise ScopePathRejected(
            "nested scoping is not enabled for this instance, so a scope path "
            "cannot be honoured. Sending one and having it ignored would widen "
            "the scope rather than narrow it, so it is refused instead."
        )
    if not await _ladder_is_ratified(client_uuid):
        # Same reasoning as the line above, one step further in: a path names
        # rungs of a ladder nobody has agreed to yet. Honouring it would let an
        # unratified shape decide what a caller reads.
        raise ScopePathRejected(
            "this client's scope ladder has not been accepted yet, so a scope "
            "path cannot be honoured. Accept the ladder in the dashboard first."
        )
    from synap.cloud.shared.scope_ladder import (
        ScopeLadderError,
        chain_to_rungs,
        get_scope_ladder,
    )

    # N-14 for the path transport. Without these the tenant and identity nodes
    # were minted with `gen_random_uuid()`, and a random id at a rung that has a
    # data-plane uuid is not a cosmetic difference: it makes BOTH read arms
    # miss. See `ScopeLadder._normalise` for the full account and for the
    # measurement that found it.
    #
    # The caller is the one place that holds both, and it holds them for the
    # customer and the person NAMED IN THE PATH: `resolve_identity` reads
    # `scope_path["customer"]` and `scope_path["user"]` over the flat ids before
    # resolving either uuid, so these cannot be the uuids of somebody else.
    role_hints = {
        k: v for k, v in (
            ("tenant_of_record", tenant_node_id),
            ("identity", identity_node_id),
        ) if v is not None
    }

    try:
        chain = await get_scope_ladder().resolve(
            client_uuid, scope_path, create=create,
            role_node_ids=role_hints or None,
        )
    except ScopeLadderError as exc:
        # Typed ladder errors already say which level was unknown or which rung
        # the path skipped (P-02), so pass the message straight through rather
        # than replacing it with something vaguer.
        raise ScopePathRejected(str(exc)) from exc
    rungs = chain_to_rungs(chain)
    if not rungs:
        raise ScopePathRejected("scope path resolved to an empty chain")
    return rungs


def _as_uuid(value: Any) -> Optional[_uuid.UUID]:
    """A uuid or None. Never raises, because a malformed id here must cost the
    hint and not the request."""
    if not value:
        return None
    try:
        return _uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


async def attach_scope_chain(scope: "ScopeHierarchy", *, create: bool = True) -> None:
    """N-12. Make sure every node on this request's chain exists, and record it.

    Called from the mapped scope builders, which is the one place that holds
    both the external ids the client sent and the uuids we derived from them.
    Deeper in the pipeline only the uuids survive, and a node cannot be created
    from a uuid alone because the registry stores what the client calls it.

    Why on the write path at all: many distinct scope uuids own vector data
    while having no row in the control-plane `customer` table. A registry seeded
    only by backfill therefore cannot be complete, and under `node_id IN chain`
    an unresolvable chain returns nothing rather than something slightly worse.

    `create` defaults to True even though this runs on reads as well as writes,
    which deserves saying out loud because a read that inserts is normally a
    smell. Two reasons it is right here. IdMapper already does exactly this from
    the same call: `resolve_customer` and `resolve_user` insert a control-plane
    row on first sight, on any path. And the node this would create carries the
    uuid IdMapper just wrote, so declining to create it would leave the registry
    permanently one step behind the table it mirrors, which is the state that
    produced the 321 orphans in the first place.

    Never raises. A client with no ladder yet, a database blip, or a conflict
    must not fail an ingestion that would otherwise have succeeded: the chain is
    additive metadata and the request is correct without it.
    """
    try:
        client_uuid = _uuid.UUID(str(scope.client_id))
    except (TypeError, ValueError):
        return
    explicit = getattr(scope, "scope_path", None)
    if explicit:
        # Strict on purpose, and NOT inside the try below. The whole point of an
        # explicit path is that it is honoured or refused; letting it fall into
        # the tolerant handler would serve a wider scope than was asked for,
        # which is the failure this contract exists to prevent.
        scope.chain_rungs = await resolve_chain_rungs_for_path(
            client_uuid, explicit, instance_id=scope.instance_id, create=create,
            # The uuids this record is about to be stamped with. Handing them
            # over is what makes `node_id IN chain` match the record it just
            # wrote, instead of a surrogate that matches nothing.
            tenant_node_id=_as_uuid(scope.customer_id),
            identity_node_id=_as_uuid(scope.user_id),
        )
        return

    try:
        scope.chain_rungs = await resolve_chain_rungs(
            client_uuid,
            external_customer_id=scope.external_customer_id,
            external_user_id=scope.external_user_id,
            customer_uuid=_uuid.UUID(str(scope.customer_id)) if scope.customer_id else None,
            user_uuid=_uuid.UUID(str(scope.user_id)) if scope.user_id else None,
            instance_id=scope.instance_id,
            create=create,
        )
    except Exception as exc:  # noqa: BLE001
        # "Never raises" was a claim, not a guarantee: only the uuid parse above
        # was guarded, while `resolve_chain_rungs` and the two inline uuid
        # conversions were not. The consequence was not a failed chain, which is
        # survivable and the point of this function. It was that the caller,
        # `build_scope_with_maca_mapped`, sits inside a bare `except Exception`
        # in `_build_scope` that falls back to `build_scope_with_maca`. That
        # fallback derives the SAME uuids and calls IdMapper for none of them,
        # so a single error in here silently stopped `customer` and `user` rows
        # being created for that request, while everything looked fine.
        #
        # That is the SHIP-22 symptom exactly, and `CannotPlaceWrite` is one of
        # the exceptions that could reach here. The chain is additive metadata;
        # losing it must never cost a control-plane row.
        logger.warning(
            "scope chain not attached (%s: %s); request continues without it",
            type(exc).__name__, exc,
        )


def derive_item_scope(
    user_id: Optional[Any] = None,
    customer_id: Optional[Any] = None,
) -> str:
    """Tag a retrieved item with its narrowest non-empty owning scope.

    A row/doc/node carries the IDs it was ingested under. The narrowest non-empty
    one is its scope: user → customer → client. The customer-scope sentinel
    (CUSTOMER_SCOPE_SENTINEL) counts as no user_id.
    """
    has_user = bool(user_id) and str(user_id) != CUSTOMER_SCOPE_SENTINEL
    if has_user:
        return "user"
    if customer_id:
        return "customer"
    return "client"


def derive_scope_level(
    user_id: Optional[str],
    customer_id: Optional[str],
    user_context_isolation: str,
) -> str:
    """Compute the narrowest scope level for a request from the IDs passed.

    B2C (`user_context_isolation == "equals_customer"`): customer == user. If
    either ID is passed the request is user-scoped; if neither, client-scoped.

    B2B (strict): user+customer → user; customer only → customer; neither →
    client; user only (no customer_id) is invalid and raises ValueError.

    Ingestion uses the returned level to pick a destination (MEMORIES /
    CUSTOMER_KNOWLEDGE / CLIENT_KNOWLEDGE). Retrieval always rolls
    narrow-to-broad and uses the level only to derive filter shapes.
    """
    is_b2c = user_context_isolation == "equals_customer"
    if is_b2c:
        return "user" if (user_id or customer_id) else "client"
    # B2B
    if user_id and customer_id:
        return "user"
    if customer_id and not user_id:
        return "customer"
    if not customer_id and not user_id:
        return "client"
    # user_id without customer_id in B2B — cannot locate the customer namespace
    raise ValueError(
        "B2B instance: user_id provided without customer_id. "
        "customer_id is required to resolve the customer namespace."
    )


B2C_ISOLATION = "equals_customer"


def collapse_b2c_identity(
    user_id: Optional[str],
    customer_id: Optional[str],
    *,
    is_b2c: bool,
    where: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """Collapse the two ids onto the ONE identity a B2C request actually has.

    On a B2C instance (`user_context_isolation == "equals_customer"`) the
    customer and the user are the same node, so a request carries one identity
    under two names. Every id that reaches the store has to be derived from the
    same string, or the write and the read address different scopes.

    The identity is `user_id` when the caller sent one, else `customer_id`.
    Returns the pair `(user_id, customer_id)` with both set to it. B2B is
    returned untouched.

    ⚠ Why the user wins, and not the customer. This used to be decided
    separately on each side and the two sides disagreed: the read builders took
    `user_id or customer_id` while every write path only filled in whichever id
    was MISSING, so a caller sending both — say `user_id="<person>"` and a fixed
    `customer_id="<brand>"` — wrote under the brand and read under the person.
    Nothing errored; the write returned an id, the read returned an empty
    bundle, and a client reported it as "you can store but you cannot recall".
    This was seen across several B2C instances.

    The user is the right anchor for three reasons. The read side is the one
    that cannot be changed without stranding data, because every context route
    and the retrieval manager already resolve this way. The data already on disk
    is keyed that way: for the reporting client, nearly all B2C scopes
    are derived from a person id and almost none from the brand. And keying
    on the customer would file every end user of a client into a single pool,
    which is a data-isolation regression, not a fix.

    A caller sending two DIFFERENT strings on a B2C instance is telling us
    something is off — usually that the instance is really B2B — so it is
    logged. It is not refused: refusing would break live traffic, and the
    collapse is well-defined.
    """
    if not is_b2c:
        return user_id, customer_id
    identity = user_id or customer_id
    if user_id and customer_id and user_id != customer_id:
        logger.warning(
            "b2c_identity_collapsed%s: user_id and customer_id differ on a B2C "
            "instance; both resolve to the user_id. If this instance really has "
            "customers with several users under them it should not be "
            "user_context_isolation=equals_customer.",
            f" ({where})" if where else "",
        )
    return identity, identity


def customer_deterministic_uuid(
    client_uuid: _uuid.UUID, external_customer_id: str
) -> _uuid.UUID:
    """Deterministic customer UUID scoped under client.

    Ensures two clients with the same external customer_id get different UUIDs.
    """
    return _uuid.uuid5(client_uuid, str(external_customer_id))


def user_deterministic_uuid(
    customer_uuid: _uuid.UUID, external_user_id: str, *, is_b2c: bool = False
) -> _uuid.UUID:
    """Deterministic user UUID scoped under customer.

    For B2C (user_context_isolation == "equals_customer"), returns customer_uuid
    directly so user == customer identity is enforced at the UUID level.
    """
    if is_b2c:
        return customer_uuid
    return _uuid.uuid5(customer_uuid, str(external_user_id))


def parse_or_deterministic_uuid(value: str) -> _uuid.UUID:
    """Parse a UUID string, falling back to deterministic UUIDv5.

    This is the canonical conversion for Synap identifiers (instance_id,
    client_id, etc.) which may be raw UUIDs or prefixed strings like
    ``inst_<hex>`` or ``cli_<hex>``.  The fallback uses UUID5 with
    NAMESPACE_URL so the same input always produces the same UUID —
    matching ingestion behaviour.
    """
    try:
        return _uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return _uuid.uuid5(NAMESPACE_URL, str(value))


def build_scope_from_ids(
    client_id: str,
    instance_id: str,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> "ScopeHierarchy":
    """Build a ScopeHierarchy with proper UUID conversion on every field.

    This mirrors the ``_build_scope`` helper in ``sdk_context.py`` and
    ensures the gRPC path produces the same scope values as the HTTP path.

    customer_id is required. If not provided, raises ValueError.
    """
    if not customer_id:
        raise ValueError(
            "customer_id is required on a B2B instance. On a B2C instance "
            "(user_context_isolation=equals_customer) do NOT send customer_id "
            "at all: the user_id is the whole identity."
        )
    effective_user_id = user_id if user_id and user_id != "anonymous" else None
    client_uuid = parse_or_deterministic_uuid(client_id)
    cust_uuid = customer_deterministic_uuid(client_uuid, customer_id)
    return ScopeHierarchy(
        instance_id=str(parse_or_deterministic_uuid(instance_id)),
        client_id=str(client_uuid),
        customer_id=str(cust_uuid),
        user_id=str(user_deterministic_uuid(cust_uuid, effective_user_id)) if effective_user_id else None,
    )


def build_scope_with_maca(
    client_id: str,
    instance_id: str,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    maca_scoping: Optional[Dict[str, Any]] = None,
) -> "ScopeHierarchy":
    """MACA-aware scope builder.

    Resolves customer_id based on MACA config:
    - B2C (user_context_isolation == "equals_customer"): customer_id = user_id
    - B2B: customer_id must be explicitly provided
    - Client-scope: customer_id = None is valid (queries client-level collections)

    Args:
        client_id: Client identifier
        instance_id: Instance identifier
        user_id: User identifier (may be None for customer/client-level ops)
        customer_id: Customer identifier (may be None for B2C auto-resolution or client-level ops)
        maca_scoping: Raw scoping dict from MACA config (optional)
    """
    scoping = maca_scoping or {}
    user_context_isolation = scoping.get("user_context_isolation", "strict")
    is_b2c = user_context_isolation == B2C_ISOLATION

    # B2C: one identity under two names. `collapse_b2c_identity` is the single
    # place that decides which string it is derived from, and every write path
    # calls the same function, so a scope written and a scope read cannot
    # disagree.
    if is_b2c:
        user_id, customer_id = collapse_b2c_identity(
            user_id, customer_id, is_b2c=True, where="build_scope_with_maca",
        )
        resolved_customer = customer_id
    elif customer_id:
        resolved_customer = customer_id
    elif user_id:
        # B2B with user_id but no customer_id — cannot resolve customer scope
        raise ValueError(
            "customer_id is required for B2B instances. "
            "Please specify customer_id in the request."
        )
    else:
        # Neither provided — only valid for client-scope operations
        resolved_customer = None

    effective_user_id = user_id if user_id and user_id != "anonymous" else None
    client_uuid = parse_or_deterministic_uuid(client_id)
    cust_uuid = customer_deterministic_uuid(client_uuid, resolved_customer) if resolved_customer else None
    return ScopeHierarchy(
        instance_id=str(parse_or_deterministic_uuid(instance_id)),
        client_id=str(client_uuid),
        customer_id=str(cust_uuid) if cust_uuid else None,
        user_id=str(user_deterministic_uuid(cust_uuid, effective_user_id, is_b2c=is_b2c))
        if effective_user_id and cust_uuid
        else None,
    )


async def build_scope_from_ids_mapped(
    mapper: "IdMapper",
    client_id: str,
    instance_id: str,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> "ScopeHierarchy":
    """Build ScopeHierarchy using IdMapper for persistent ID mapping.

    Same semantics as build_scope_from_ids but uses IdMapper to
    resolve and persist mappings. customer_id is required.
    """
    if not customer_id:
        raise ValueError(
            "customer_id is required on a B2B instance. On a B2C instance "
            "(user_context_isolation=equals_customer) do NOT send customer_id "
            "at all: the user_id is the whole identity."
        )
    effective_user_id = user_id if user_id and user_id != "anonymous" else None

    client_uuid = await mapper.resolve_client(client_id)
    customer_uuid = await mapper.resolve_customer(client_id, customer_id, client_uuid=client_uuid)

    scope = ScopeHierarchy(
        instance_id=str(await mapper.resolve_instance(instance_id)),
        client_id=str(client_uuid),
        customer_id=str(customer_uuid),
        user_id=str(
            await mapper.resolve_user(client_id, effective_user_id, customer_id, customer_uuid=customer_uuid)
        )
        if effective_user_id
        else None,
        external_user_id=effective_user_id,
        external_customer_id=customer_id,
    )
    await attach_scope_chain(scope)
    return scope


@dataclass
class ResolvedIdentity:
    """What a request's ids resolve to, before anyone decides what shape to
    carry it in.

    This exists because a scope used to be resolved by two different functions
    with two different return types, `sdk_memories._resolve_scope` for writes
    and `build_scope_with_maca_mapped` for reads. They agreed by convention,
    not by construction, and when they stopped agreeing about B2C the write and
    the read addressed different buckets while both returned 200. One client saw a long
    run of consecutive empty reads.

    So resolution happens once, here, and `Scope` and `ScopeHierarchy` become
    adapters over the result. The two pipelines can still want different
    shapes; they can no longer want different answers.
    """
    client_uuid: _uuid.UUID
    instance_uuid: _uuid.UUID
    customer_uuid: Optional[_uuid.UUID]
    user_uuid: Optional[_uuid.UUID]
    external_user_id: Optional[str]
    external_customer_id: Optional[str]
    scope_level: str
    is_b2c: bool
    mapped: bool           # False when IdMapper was unavailable and ids were derived
    chain_rungs: Optional[List[Dict[str, Any]]] = None


async def effective_user_context_isolation(
    instance_id: str,
    maca_scoping: Optional[Dict[str, Any]] = None,
) -> str:
    """The instance's isolation mode, asked once, in one place.

    A caller that already knows wins: if `maca_scoping` carries the key, that is
    the answer. Only silence reaches the authority.

    Silence used to mean `"strict"`, hardcoded here and again in
    `sdk_memories`. That is a default wearing the clothes of a fact, and it is
    wrong for exactly the instances that cannot correct it: an instance has no
    MACA config for the whole time MACA takes to run, and permanently if that
    run is lost.

    Example: a B2C instance whose config had not landed yet. For just under
    seven minutes `whoami` reported `user_context_isolation: null` and this
    function's callers read `"strict"`:

        1st  create  user_id only              -> 400  "B2B instance:
                                                       customer_id is required"
        2nd  create  user_id + customer_id     -> 400  customer_id_not_accepted_on_b2c
        3rd  create  user_id only              -> 400  "B2B instance: ..."

    No request shape worked. The second error is the ingress guard, which
    `b2c_contract.load_isolation` had already been taught to answer correctly
    from the instance itself (e88cbe91). This path never learned, so the two
    sides told one caller opposite things within about a minute.

    So both sides now end at the same authority. `load_isolation` reads the
    config first and falls back to the instance's own `user_relationship`, the
    field the client chose at creation and the same one MACA reads to decide
    isolation. Still `"strict"` when even the instance does not say, because
    that is the historical behaviour and an instance that has never declared a
    relationship has given us nothing better to go on.
    """
    scoping = maca_scoping or {}
    declared = scoping.get("user_context_isolation")
    if declared is not None:
        return declared

    try:
        from synap.cloud.shared.b2c_contract import load_isolation
        resolved = await load_isolation(instance_id)
    except Exception as exc:  # noqa: BLE001 — never fail a request to answer this
        logger.debug("isolation lookup failed for %s: %s", instance_id, exc)
        resolved = None

    if resolved is None:
        return "strict"

    logger.info(
        "scope_isolation_resolved instance_id=%s isolation=%s reason=caller_did_not_say",
        instance_id, resolved,
    )
    return resolved


async def resolve_identity(
    client_id: str,
    instance_id: str,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    maca_scoping: Optional[Dict[str, Any]] = None,
    *,
    mapper: Optional["IdMapper"] = None,
    resolve_chain: bool = True,
    derive_on_failure: bool = False,
    scope_path: Optional[Mapping[str, str]] = None,
) -> ResolvedIdentity:
    """The one place external ids become scope uuids.

    Raises ValueError for a B2B request naming a user with no customer, which
    is the one shape that cannot be resolved at all.

    `mapper=None` resolves through the process-wide IdMapper. If that is
    unavailable the ids are DERIVED instead, and `mapped=False` records it: a
    derived id matches the mapped one, but no control-plane row is created, and
    a node registered under a derived id would match no vector record. Both
    sides degrade identically now; previously each had its own fallback.
    """
    user_context_isolation = await effective_user_context_isolation(
        instance_id, maca_scoping,
    )
    is_b2c = user_context_isolation == B2C_ISOLATION

    # A path names the rung this request is happening at, so it decides the
    # level. `derive_scope_level` cannot: it reads which of two ids were sent,
    # and the three levels it knows are the three reserved roles. That is why a
    # client could define a `team` rung and never file anything on it. The
    # deepest rung the caller named is the one they meant, and it is the LAST
    # key in a path that has already been checked for gaps by the resolver.
    if scope_path:
        user_id = scope_path.get("user", user_id)
        customer_id = scope_path.get("customer", customer_id)
    scope_level = derive_scope_level(user_id, customer_id, user_context_isolation)

    if is_b2c:
        user_id, customer_id = collapse_b2c_identity(
            user_id, customer_id, is_b2c=True, where="resolve_identity",
        )

    effective_user_id = user_id if user_id and user_id != "anonymous" else None
    resolved_customer = customer_id

    # `derive_on_failure` decides what an IdMapper failure MEANS, and the two
    # sides genuinely differ here, so it is a parameter rather than a silent
    # default. The write path has always fallen back to derived ids inline. The
    # read path has always let the failure propagate, because its callers have
    # their own fallbacks, and swallowing it here would silently disable them.
    #
    # What both sides now share is the resolution itself. Only the failure
    # policy is chosen by the caller, and it is chosen out loud.
    if mapper is None:
        try:
            from synap.cloud.shared.id_mapper import get_id_mapper
            mapper = get_id_mapper()
        except Exception as exc:  # noqa: BLE001
            if not derive_on_failure:
                raise
            logger.warning(
                "scope resolved WITHOUT id mapping (%s: %s); customer/user "
                "control-plane rows will not be created for this request",
                type(exc).__name__, exc,
            )
            mapper = None

    if mapper is not None:
        try:
            client_uuid = await mapper.resolve_client(client_id)
            instance_uuid = await mapper.resolve_instance(instance_id)
            customer_uuid = (
                await mapper.resolve_customer(client_id, resolved_customer, client_uuid=client_uuid)
                if resolved_customer else None
            )
            user_uuid = (
                await mapper.resolve_user(
                    client_id, effective_user_id, resolved_customer,
                    customer_uuid=customer_uuid, is_b2c=is_b2c,
                )
                if effective_user_id else None
            )
            mapped = True
        except Exception as exc:  # noqa: BLE001
            if not derive_on_failure:
                raise
            logger.warning(
                "id mapping failed mid-resolution (%s: %s); falling back to derived ids",
                type(exc).__name__, exc,
            )
            mapper = None

    if mapper is None:
        client_uuid = parse_or_deterministic_uuid(client_id)
        instance_uuid = parse_or_deterministic_uuid(instance_id)
        customer_uuid = (
            customer_deterministic_uuid(client_uuid, resolved_customer)
            if resolved_customer else None
        )
        user_uuid = (
            user_deterministic_uuid(customer_uuid, effective_user_id, is_b2c=is_b2c)
            if effective_user_id and customer_uuid else None
        )
        mapped = False

    rungs = None
    # No chain on the derived path, deliberately. N-14 measured that derivation
    # disagrees with the stored value for some clients, and a node registered
    # under a wrong id returns nothing while looking correct, which is worse
    # than having no node at all.
    if resolve_chain and mapped and scope_path:
        # The strict resolver, the same one the read path uses for a path. It
        # never infers and never falls back: a gap is refused by name rather
        # than guessed, because a wrong guess at or above the tenant crosses a
        # customer boundary. Deliberately NOT wrapped in the tolerant helper
        # below, whose promise never to raise is right for three ids and wrong
        # for an explicit instruction.
        rungs = await resolve_chain_rungs_for_path(
            client_uuid, scope_path, instance_id=str(instance_uuid),
            # N-14. `customer_uuid` and `user_uuid` above were resolved from the
            # PATH's own customer and user values, not from the flat ids: lines
            # 622-624 let the path override them before any mapping happens. So
            # these are the authority uuids for exactly the customer and person
            # this path names, and they are what the record is stamped with.
            tenant_node_id=customer_uuid,
            identity_node_id=user_uuid,
        )
        # The rung the caller actually named, which is the deepest one the
        # resolved chain reaches. Read off the chain rather than off the path,
        # so it cannot disagree with where the record is about to be filed.
        if rungs:
            scope_level = rungs[-1]["level_key"]
    elif resolve_chain and mapped:
        rungs = await resolve_chain_rungs(
            client_uuid,
            external_customer_id=resolved_customer,
            external_user_id=effective_user_id,
            customer_uuid=customer_uuid,
            user_uuid=user_uuid,
            instance_id=str(instance_uuid),
        )

    return ResolvedIdentity(
        client_uuid=client_uuid,
        instance_uuid=instance_uuid,
        customer_uuid=customer_uuid,
        user_uuid=user_uuid,
        external_user_id=effective_user_id,
        external_customer_id=resolved_customer,
        scope_level=scope_level,
        is_b2c=is_b2c,
        mapped=mapped,
        chain_rungs=rungs,
    )


async def build_scope_with_maca_mapped(
    mapper: "IdMapper",
    client_id: str,
    instance_id: str,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    maca_scoping: Optional[Dict[str, Any]] = None,
    scope_path: Optional[Dict[str, str]] = None,
) -> "ScopeHierarchy":
    """MACA-aware scope builder using IdMapper for persistent ID mapping.

    Same B2C auto-resolution logic as build_scope_with_maca.

    `scope_path` (G-03) is the caller's explicit rung path. It has to be set on
    the scope BEFORE `attach_scope_chain` runs at the end of this function,
    because that is where the chain is resolved; setting it afterwards would be
    too late and would silently give the caller the legacy chain instead.
    """
    # A thin adapter over `resolve_identity`. Everything that decides WHICH
    # uuids this request means lives there and is shared with the write path,
    # so the two can no longer drift apart. That drift is the entire bug this
    # change exists to make impossible.
    #
    # `resolve_chain=False` because `attach_scope_chain` below resolves the
    # chain itself, and it must run AFTER `scope_path` is set: that is where
    # the caller's explicit rung path is honoured. Resolving in both places
    # would be two chances to resolve differently.
    ident = await resolve_identity(
        client_id, instance_id, user_id, customer_id, maca_scoping,
        mapper=mapper, resolve_chain=False,
    )
    scope = ScopeHierarchy(
        instance_id=str(ident.instance_uuid),
        client_id=str(ident.client_uuid),
        customer_id=str(ident.customer_uuid) if ident.customer_uuid else None,
        user_id=str(ident.user_uuid) if ident.user_uuid else None,
        external_user_id=ident.external_user_id,
        external_customer_id=ident.external_customer_id,
        scope_path=scope_path,
    )
    await attach_scope_chain(scope)
    return scope


@dataclass
class ScopeHierarchy:
    """Resolved scope hierarchy for a request."""
    instance_id: str
    client_id: str
    customer_id: Optional[str] = None
    user_id: Optional[str] = None

    # Internal IDs (Synap's IDs, not client's)
    internal_user_id: Optional[str] = None
    internal_customer_id: Optional[str] = None

    # Original external IDs (for logging/debugging)
    external_user_id: Optional[str] = None
    external_customer_id: Optional[str] = None

    # Nested scoping: the resolved chain, root to leaf, as plain dicts with
    # node_id / level_key / role. None when the feature is off, when the client
    # has no ladder, or when resolution failed, and every one of those is an
    # ordinary request that proceeds unchanged.
    chain_rungs: Optional[List[Dict[str, Any]]] = None

    # G-03. The path the CALLER explicitly asked for, e.g.
    # {"customer": "acme", "team": "payments", "user": "d-42"}. None means they
    # sent ids only and the legacy adapter builds the chain, which is every
    # request today. When it is set, resolution is strict: it is honoured
    # exactly or the request is refused, never widened. See
    # `resolve_chain_rungs_for_path`.
    scope_path: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        # chain_rungs itself is deliberately absent. This dict feeds logs, and
        # a depth is what a log line can usefully carry.
        #
        # ⚠ It is NOT enough for a cache key, and this comment used to say it
        # was: "the rungs are derived from ids already in the key". That held
        # while the legacy adapter was the only source of a chain, because it
        # derives one from `customer_id` and `user_id` alone. G-03 ended it. An
        # explicit `scope_path` resolves a chain those ids do not determine, so
        # two requests with identical ids and different paths are two different
        # people with one depth. `RetrievalManager._build_cache_key` therefore
        # puts the chain node ids in the key itself and does not use this dict.
        #
        # ⚠ But a marker IS carried, and only when the feature is on. Without
        # it the cache key was byte-identical with nested scoping on and off,
        # so toggling it OFF left the fleet serving bundles that had been ranked
        # with it ON for the whole cache TTL. Z-03's verify line is "toggle off
        # mid-flight and assert three-level behaviour", and that could not hold.
        # Found by audit 2026-08-23.
        #
        # Conditional on purpose: adding a key unconditionally would cold-start
        # every cached bundle in the fleet, which is the cost the original
        # comment was right to avoid. This way the cold start lands only on
        # instances that actually turn the feature on, which is the staged
        # rollout anyway, and the key reverts the moment they turn it off.
        #
        # Depth rather than a boolean, so inserting or removing a rung also
        # flips it: the ranking changes and the cached bundles are stale. A
        # re-parent at the SAME depth does not flip it; that is what
        # `ScopeLadder.invalidate` and the TTL are for.
        out = {
            "instance_id": self.instance_id,
            "client_id": self.client_id,
            "customer_id": self.customer_id,
            "user_id": self.user_id,
            "internal_user_id": self.internal_user_id,
            "internal_customer_id": self.internal_customer_id,
            "external_user_id": self.external_user_id,
            "external_customer_id": self.external_customer_id,
        }
        if self.chain_rungs:
            out["nested_depth"] = len(self.chain_rungs)
        return out

    def get_scopes_to_query(self) -> List[str]:
        """
        Get list of scopes to query in order of specificity.
        More specific scopes first (user → customer → client).
        """
        scopes = []
        if self.user_id:
            scopes.append("user")
        if self.customer_id:
            scopes.append("customer")
        scopes.append("client")  # Always include client-level context
        return scopes


class ScopeResolver:
    """
    Resolves external IDs to internal IDs and builds scope hierarchy.

    Uses database lookups (via providers) to:
    - Map external_user_id → internal_user_id
    - Find customer_id for a user
    - Find client_id for an instance
    """

    def __init__(self, db_provider=None):
        """
        Args:
            db_provider: Database provider for ID lookups (injected)
        """
        self.db = db_provider

    async def resolve(
        self,
        instance_id: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> ScopeHierarchy:
        """
        Resolve full scope hierarchy from instance and user.

        Args:
            instance_id: The Synap instance ID
            user_id: External user ID from client's system
            conversation_id: Optional conversation ID for session context

        Returns:
            ScopeHierarchy with all resolved IDs
        """
        # Get instance → client mapping
        client_id = await self._get_client_id(instance_id)

        # Resolve user if provided
        internal_user_id = None
        internal_customer_id = None
        customer_id = None

        if user_id:
            # Map external → internal user ID
            internal_user_id = await self._resolve_user_id(client_id, user_id)

            # Get customer for this user
            customer_id, internal_customer_id = await self._get_customer_for_user(
                client_id, internal_user_id
            )

        client_uuid = parse_or_deterministic_uuid(client_id)
        cust_uuid = customer_deterministic_uuid(client_uuid, customer_id) if customer_id else None
        return ScopeHierarchy(
            instance_id=str(parse_or_deterministic_uuid(instance_id)),
            client_id=str(client_uuid),
            customer_id=str(cust_uuid) if cust_uuid else None,
            user_id=str(user_deterministic_uuid(cust_uuid, user_id)) if user_id and cust_uuid else (str(parse_or_deterministic_uuid(user_id)) if user_id else None),
            internal_user_id=internal_user_id,
            internal_customer_id=internal_customer_id,
        )

    async def _get_client_id(self, instance_id: str) -> str:
        """Get client_id for an instance."""
        if self.db is None:
            # No DB — derive a deterministic UUID from instance_id so
            # downstream IngestionScope(client_id=UUID(...)) doesn't crash.
            return str(parse_or_deterministic_uuid(instance_id))

        # Real implementation: query database
        result = await self.db.fetch_one(
            "SELECT client_id FROM instances WHERE instance_id = %s",
            (instance_id,)
        )
        if not result:
            raise ValueError(f"Instance not found: {instance_id}")
        return result["client_id"]

    async def _resolve_user_id(self, client_id: str, external_user_id: str) -> str:
        """Map external user ID to internal Synap user ID."""
        if self.db is None:
            # Stub: return hash of external ID
            import hashlib
            return hashlib.md5(f"{client_id}:{external_user_id}".encode()).hexdigest()[:16]

        # Real implementation: query or create user mapping
        result = await self.db.fetch_one(
            """
            SELECT internal_user_id FROM user_mappings
            WHERE client_id = %s AND external_user_id = %s
            """,
            (client_id, external_user_id)
        )
        if result:
            return result["internal_user_id"]

        # Create new mapping (in real impl, this might be a separate flow)
        import uuid
        internal_id = str(uuid.uuid4())
        await self.db.execute(
            """
            INSERT INTO user_mappings (client_id, external_user_id, internal_user_id)
            VALUES (%s, %s, %s)
            """,
            (client_id, external_user_id, internal_id)
        )
        return internal_id

    async def _get_customer_for_user(
        self, client_id: str, internal_user_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Get customer ID for a user."""
        if self.db is None:
            # Stub: no customer association
            return None, None

        result = await self.db.fetch_one(
            """
            SELECT c.external_customer_id, c.internal_customer_id
            FROM users u
            JOIN customers c ON u.customer_id = c.internal_customer_id
            WHERE u.internal_user_id = %s AND u.client_id = %s
            """,
            (internal_user_id, client_id)
        )
        if result:
            return result["external_customer_id"], result["internal_customer_id"]
        return None, None
