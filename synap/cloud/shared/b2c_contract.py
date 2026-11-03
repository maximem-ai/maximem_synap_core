"""The B2C identifier contract, in one place.

On an instance with ``user_context_isolation = "equals_customer"`` the customer
and the user are the same node. The contract is therefore:

    B2C   send user_id, and nothing else. customer_id is not accepted.
    B2B   customer_id is required. user_id without customer_id is an error.

Those are mirror images on purpose. Before this module the two shapes were
silently accepted everywhere, which is what made the modes indistinguishable to
clients and to coding agents writing integrations.

Why a 400 rather than a silent collapse. On a B2C instance a request carrying
both ids wrote under the customer and read under the user, and **neither the
server nor the SDK raised**. A client had a long run of consecutive fetches
return an empty body with HTTP 200, while the dashboard showed a near-zero
error rate, because an empty result scores as success. A client cannot debug what never complains.

Rollout is a single global switch, not a per-instance flag. A flag would mean
two instances with identical config behaving differently, which is the class of
bug this exists to remove.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

B2C_ISOLATION = "equals_customer"

# Rollout states.
#   off      today's behaviour, collapse silently. Kept only as a kill switch.
#   warn     observe and log, change nothing. Safe to deploy at any time.
#   enforce  reject with a 400.
POLICY_OFF = "off"
POLICY_WARN = "warn"
POLICY_ENFORCE = "enforce"
_VALID_POLICIES = (POLICY_OFF, POLICY_WARN, POLICY_ENFORCE)

_ENV_VAR = "SYNAP_B2C_ID_POLICY"


class B2CCustomerIdRejected(ValueError):
    """A B2C instance was sent a customer_id, or a customer-scoped route.

    A ValueError subclass because several routes already convert ValueError to
    a 400; the dedicated type lets the app-level handler produce a message that
    says what to do instead of a bare validation error.

    `customer_id=None` means the whole ROUTE was the problem rather than one
    field, and the message says so. It used to interpolate a placeholder string
    into "Received customer_id=..." , which put "<customer-scoped route>" in
    front of a client as if it were their value.
    """

    def __init__(self, where: str, customer_id: Optional[str] = None):
        self.where = where
        self.customer_id = customer_id
        if customer_id is None:
            msg = (
                f"{where} is a customer-scoped endpoint and is not available on a "
                "B2C instance (user_context_isolation=equals_customer), which has "
                "no customer scope. Fetch at user scope instead: "
                "POST /v1/context/user/fetch with user_id."
            )
        else:
            msg = (
                "customer_id is not accepted on a B2C instance "
                "(user_context_isolation=equals_customer). Send user_id only; it "
                f"is the whole identity. Received customer_id={customer_id!r} at "
                f"{where}."
            )
        super().__init__(msg)


def policy() -> str:
    """Current rollout state, read per call so it can be changed without a code
    deploy. An unrecognised value falls back to `warn` rather than to `enforce`,
    because a typo must never start rejecting live traffic."""
    raw = (os.environ.get(_ENV_VAR) or "").strip().lower()
    if raw in _VALID_POLICIES:
        return raw
    if raw:
        logger.warning(
            "%s=%r is not one of %s; falling back to %r",
            _ENV_VAR, raw, _VALID_POLICIES, POLICY_WARN,
        )
    return POLICY_WARN


async def load_isolation(instance_id: str) -> Optional[str]:
    """The instance's scoping mode, or None when it cannot be read.

    None on failure, never a default. Defaulting to "strict" would tell a B2C
    caller their customer_id is required, and defaulting to B2C would start
    rejecting a B2B client's mandatory field. Not knowing is its own answer, and
    every caller below treats it as "do nothing".
    """
    try:
        from synap.cloud.application.client_api.retrieval_manager import _load_maca_config
        maca = await _load_maca_config(instance_id)
        if maca:
            return maca.user_context_isolation
    except Exception as exc:  # noqa: BLE001
        logger.debug("MACA read failed for %s: %s", instance_id, exc)

    # No config. Ask the instance itself before giving up.
    #
    # The docstring above is right that a DEFAULT would be wrong in both
    # directions. This is not a default: `user_relationship` is chosen by the
    # client when they create the instance, and it is the same field MACA reads
    # to decide `user_context_isolation` in the first place. Reading it here is
    # going one layer up to the original answer, not guessing.
    #
    # Why it matters: an instance has no config for the whole time MACA takes to
    # run, and permanently if that run is lost. During that window this returned
    # None, every caller treated None as "do nothing", and a B2C instance with
    # SYNAP_B2C_ID_POLICY=enforce accepted the very field the contract exists to
    # refuse. A B2C instance can take over a minute to get its config, and
    # customer_id requests were accepted in that window. The first one after
    # the config landed was correctly refused.
    derived = await _isolation_from_relationship(instance_id)
    if derived is not None:
        logger.info(
            "b2c_isolation_from_instance instance_id=%s isolation=%s reason=no_maca_config",
            instance_id, derived,
        )
    return derived


async def _isolation_from_relationship(instance_id: str) -> Optional[str]:
    """The instance's declared relationship, mapped to an isolation mode.

    Still None when the instance itself does not say, which is the honest
    answer for a row that predates the field. Never raises: this is a fallback,
    and a fallback that can fail the request is not one.
    """
    try:
        from sqlalchemy import text

        from synap.cloud.database.connection import get_db_session

        async with get_db_session() as session:
            row = (await session.execute(
                text("SELECT instance_metadata FROM instance WHERE instance_id = :iid"),
                {"iid": instance_id},
            )).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001
        logger.debug("instance relationship read failed for %s: %s", instance_id, exc)
        return None

    if not row:
        return None
    meta = row
    if isinstance(meta, str):
        import json as _json

        try:
            meta = _json.loads(meta)
        except ValueError:
            return None
    if not isinstance(meta, dict):
        return None

    rel = meta.get("user_relationship")
    if not rel:
        return None

    # The same mapping `resolve_instance_type_config` uses, so the two cannot
    # disagree about what a relationship means. Imported rather than restated:
    # a second copy of this table is how the write path and the read path
    # stopped agreeing about B2C the last time.
    try:
        from synap.cloud.memory_config.enums import (
            resolve_instance_type_config,
        )

        return resolve_instance_type_config(str(rel))["scope_defaults"][
            "user_context_isolation"
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("unknown user_relationship %r for %s: %s", rel, instance_id, exc)
        return None


async def is_b2c(instance_id: str) -> bool:
    return (await load_isolation(instance_id)) == B2C_ISOLATION


async def check_customer_id(
    instance_id: str,
    customer_id: Optional[str],
    *,
    where: str,
    user_id: Optional[str] = None,
) -> None:
    """Apply the contract to one inbound request.

    Does nothing unless the instance is B2C **and** a customer_id was actually
    sent. Under `warn` it records the violation and returns; under `enforce` it
    raises. Callers do not branch on the policy themselves, so the rollout state
    lives in exactly one place.
    """
    if not customer_id:
        return
    mode = policy()
    if mode == POLICY_OFF:
        return
    if not await is_b2c(instance_id):
        return

    # One structured line, because the point of `warn` is to be able to answer
    # "who is still sending this, and from where" without reading bodies.
    logger.warning(
        "b2c_customer_id_sent policy=%s instance_id=%s where=%s "
        "customer_id=%s user_id=%s",
        mode, instance_id, where, customer_id, user_id,
    )
    if mode == POLICY_ENFORCE:
        raise B2CCustomerIdRejected(where, customer_id)


async def reject_customer_route(instance_id: str, *, where: str) -> None:
    """A customer-scoped route was called on a B2C instance.

    Separate from `check_customer_id` because the whole ROUTE is meaningless
    here, not just one field: a B2C instance has no customer scope to fetch.

    ⚠ This is NOT free, and an earlier version of this docstring claimed it was.
    The claim was that the route "already returns nothing on every B2C instance",
    so refusing it "costs no working behaviour". That was reasoned from the code
    and never measured, and it is false. A client called
    it repeatedly and got populated facts back every time, while
    their `/v1/context/user/fetch` returned empty. For that client this route was
    the ONLY read that worked, and refusing it takes their working read away.

    It is refused anyway, deliberately. On a B2C instance the customer scope is
    an artefact of the client sending an identifier we tell them not to send, so
    anything it returns is filed under a key their own user reads can never
    reach. Serving it would keep that split alive and hide it behind a result
    that looks fine. The cost is real and is paid once, at migration, by clients
    who must move to `user_id`; leaving it open costs them silently forever.

    Anyone changing this: measure the route against live traffic before you
    reason about it. That is the specific mistake this paragraph exists to stop
    from happening twice.
    """
    mode = policy()
    if mode == POLICY_OFF:
        return
    if not await is_b2c(instance_id):
        return
    logger.warning(
        "b2c_customer_route_called policy=%s instance_id=%s where=%s", mode, instance_id, where,
    )
    if mode == POLICY_ENFORCE:
        raise B2CCustomerIdRejected(where)
