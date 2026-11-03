"""An instance with no MACA config is not recognised as B2C.

`load_isolation` returns None when it cannot read a config, and every caller
treats None as "do nothing". That is a deliberate and defensible choice — the
docstring argues that defaulting either way breaks the other shape — but it has
a consequence nobody wrote down: an instance whose config never got written is a
B2C instance with `SYNAP_B2C_ID_POLICY=enforce` and no enforcement.

That is reachable in a running deployment. MACA generation is a fire-and-forget
`asyncio.create_task`, so a deploy during the ~5 minutes it takes leaves the
instance with no config, no error and no retry. An instance
(inst_00000000000000a1) created seconds before a restart can still have zero
config rows much later. Some instances end up with none.

This pins the combination rather than either half, because each half on its own
reads as reasonable.
"""
from __future__ import annotations

import pytest

from synap.cloud.shared import b2c_contract


async def test_no_config_reads_as_unknown_not_as_b2b(monkeypatch):
    async def _none(_iid):
        return None
    monkeypatch.setattr(b2c_contract, "load_isolation", _none)
    assert await b2c_contract.is_b2c("inst_orphaned") is False, (
        "unknown is not B2C, which is the documented choice"
    )


async def test_the_customer_id_contract_does_not_fire_without_a_config(monkeypatch):
    """The consequence, stated out loud. A B2C instance that lost its config
    accepts the very field the contract exists to refuse."""
    monkeypatch.setattr(b2c_contract, "policy", lambda: b2c_contract.POLICY_ENFORCE)

    async def _none(_iid):
        return None
    monkeypatch.setattr(b2c_contract, "load_isolation", _none)

    # No raise. This is the finding, not a passing behaviour to celebrate.
    await b2c_contract.check_customer_id(
        "inst_orphaned", "some-org", where="test", user_id="person-7")


async def test_the_same_call_is_refused_once_the_config_exists(monkeypatch):
    """The control. With a config saying B2C, the identical call raises, so the
    test above is measuring the missing config and not a broken contract."""
    monkeypatch.setattr(b2c_contract, "policy", lambda: b2c_contract.POLICY_ENFORCE)

    async def _b2c(_iid):
        return b2c_contract.B2C_ISOLATION
    monkeypatch.setattr(b2c_contract, "load_isolation", _b2c)

    with pytest.raises(Exception) as exc:
        await b2c_contract.check_customer_id(
            "inst_ok", "some-org", where="test", user_id="person-7")
    assert "customer" in str(exc.value).lower()
