#!/usr/bin/env python3
"""Synap Core verification: a real write, then a real recall, through the SDK.

Run this as the LAST step of a Synap Core setup, against the running server. It proves, in order:

  1. /health answers and the server says it is Synap Core,
  2. which scoping mode the install uses (equals_customer or strict),
  3. which memory config is active (the starter sample, or the developer's own),
  4. the SDK connects with no API key,
  5. a memory can be WRITTEN (a throwaway fact with a unique marker) and is stored,
  6. the same fact can be RECALLED (polled, because extraction is eventual),
  7. a second user cannot see the first user's memory (isolation),
  8. forgetting for a user who stored nothing answers "nothing matches", not an error,
  9. the memory browser is read-only (it refuses writes),
 10. the test memories are deleted again (unless --keep).

Every line starts with [OK], [WARN] or [ERROR]; the recalled text is printed verbatim. Quote the
output to the human as it is. The output ends with exactly two lines: `RESULT: PASS` or
`RESULT: FAIL <reason>`, then `EXIT CODE: <n>`. A [WARN] is explained, never dropped.

    python3 scripts/verify_core.py
    python3 scripts/verify_core.py --keep          # leave the test memories in place
    python3 scripts/verify_core.py --url http://127.0.0.1:8000

The base URL is --url, else SYNAP_BASE_URL, else http://127.0.0.1:8000. If the server has a token
(SYNAP_API_TOKEN), the script reads it from the environment or from ./.env (that one line only) and
never prints it. Needs `pip install maximem-synap` (0.6.13 or newer) in the Python that runs this.

Optional environment:
  SYNAP_VERIFY_TIMEOUT=240        seconds to keep polling for the write and for the recall
  SYNAP_VERIFY_CUSTOMER_ID=acme   strict (business) installs only; default "synap-verify". Ignored on
                                  equals_customer installs, where a customer id is refused.

Test users are named test_user_verify_<random>. Deleting a memory removes it and its search entries;
the user ids themselves leave no profile behind but a conversation or graph entity written from the
test text can remain; they carry the test_user_ prefix so you can recognise them in the browser.

Exit codes: 0 pass. 1 something failed. 2 the server could not be reached.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

lines: list[str] = []


def say(level: str, text: str) -> None:
    line = f"[{level}] {text}"
    lines.append(line)
    print(line, flush=True)


def load_token() -> str:
    token = os.environ.get("SYNAP_API_TOKEN", "")
    if not token and os.path.isfile(".env"):
        for raw in open(".env").read().splitlines():
            if raw.startswith("SYNAP_API_TOKEN="):
                token = raw.partition("=")[2].strip()
    return token


class Http:
    def __init__(self, base: str, token: str):
        self.base, self.token = base.rstrip("/"), token

    def call(self, method: str, path: str) -> tuple[int, str]:
        req = urllib.request.Request(self.base + path, method=method, data=b"{}" if method not in ("GET", "DELETE") else None)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
        except Exception as e:  # noqa: BLE001
            return 0, f"{type(e).__name__}: {e}"


def texts(ctx) -> list[str]:
    out = []
    for name in ("facts", "preferences", "episodes", "emotions", "temporal_events"):
        for item in getattr(ctx, name, None) or []:
            out.append(getattr(item, "content", str(item)))
    return out


async def verify(args) -> None:
    base = args.url or os.environ.get("SYNAP_BASE_URL") or "http://127.0.0.1:8000"
    token = load_token()
    http = Http(base, token)
    timeout = float(os.environ.get("SYNAP_VERIFY_TIMEOUT", "240"))

    status, body = http.call("GET", "/health")
    if status == 0:
        say("ERROR", f"cannot reach {base}: {body}. Is the stack up? Run scripts/wait_healthy.py.")
        finish(2)
    if status != 200:
        say("ERROR", f"/health answered {status}: {body[:160]}. The server is not ready; run scripts/wait_healthy.py and read its output.")
        finish(1)
    try:
        edition = json.loads(body).get("edition")
    except ValueError:
        edition = None
    if edition != "core":
        say("ERROR", f"/health says edition={edition!r}, not 'core'. This is not a Synap Core server.")
        finish(1)
    say("OK", f"{base} is healthy and is Synap Core")

    status, body = http.call("GET", "/api/v1/auth/whoami")
    if status in (401, 403):
        say("ERROR", "the server requires a token and none was accepted. Set SYNAP_API_TOKEN in this shell (or in ./.env) to the value in the server's .env.")
        finish(1)
    mode = None
    if status == 200:
        mode = json.loads(body).get("user_context_isolation")
    if mode == "strict":
        say("OK", "scoping mode: strict (business app): every call carries a customer id and a user id")
    elif mode == "equals_customer":
        say("OK", "scoping mode: equals_customer (consumer app): every call carries a user id only")
    else:
        say("WARN", f"could not read the scoping mode from whoami (status {status}); sending a user id only")
    customer = os.environ.get("SYNAP_VERIFY_CUSTOMER_ID", "synap-verify") if mode == "strict" else None

    status, body = http.call("GET", "/browser/api/config")
    if status == 200:
        cfg = json.loads(body)
        if cfg.get("valid") is False:
            say("ERROR", "the memory config has problems: " + "; ".join(cfg.get("problems", [])[:2]))
        elif cfg.get("starter"):
            say("WARN", "the active memory config is still the starter sample (it only sets the scoping mode; everything else is default). That is fine to start; see reference/config-and-maca.md to tune it.")
        else:
            say("OK", "the active memory config is the developer's own and is accepted: " + ", ".join(s["name"] for s in cfg.get("sections", [])))
    else:
        say("WARN", f"could not read /browser/api/config (status {status})")

    for name in [n for n in os.environ if n.startswith("SYNAP_") and n not in ("SYNAP_API_TOKEN",)]:
        del os.environ[name]
    os.environ["SYNAP_BASE_URL"] = base
    if token:
        os.environ["SYNAP_API_TOKEN"] = token
    try:
        from maximem_synap import MaximemSynapSDK
    except ImportError:
        v = sys.version_info
        say("ERROR", f"the maximem-synap package is not installed in this Python ({v.major}.{v.minor}). The SDK needs Python 3.11 or newer: make a virtualenv with it, run `pip install maximem-synap` (0.6.13 or newer) there, and run this script with that Python.")
        finish(1)

    sdk = MaximemSynapSDK()
    await sdk.initialize()
    say("OK", "the SDK connected with no API key")
    ids: list[str] = []
    suffix = uuid.uuid4().hex[:8]
    user_a, user_b = f"test_user_verify_{suffix}_a", f"test_user_verify_{suffix}_b"
    extra = {"customer_id": customer} if customer else {}
    try:
        async def store(user: str, document: str):
            created = await sdk.memories.create(document=document, document_type="ai-chat-conversation", user_id=user, **extra)
            return await sdk.memories.wait_for_completion(created.ingestion_id, timeout_seconds=int(timeout))

        done = await store(user_a, "User: My favourite tea is Lapsang Souchong and I drink it at five every afternoon.\nAssistant: Noted, Lapsang Souchong at five.")
        ids += list(done.memory_ids)
        if done.status.value == "completed" and done.memories_created >= 1:
            say("OK", f"write: stored {done.memories_created} memories for {user_a}")
        else:
            say("ERROR", f"write: ingestion ended as {done.status.value} with {done.memories_created} memories. {done.error_message or ''} Look at `docker compose logs synap`.")
            return

        recalled, end = [], time.time() + timeout
        while time.time() < end:
            ctx = await sdk.user.context.fetch(user_id=user_a, search_query=["favourite tea", "afternoon drink"], **extra)
            recalled = texts(ctx)
            if any("lapsang" in t.lower() for t in recalled):
                break
            await asyncio.sleep(5)
        if any("lapsang" in t.lower() for t in recalled):
            say("OK", "recall: the fact came back, verbatim below")
            for t in recalled[:4]:
                print(f"        \"{t}\"")
        else:
            say("ERROR", "recall: the stored fact did not come back within the time limit. Recalled: " + (" | ".join(recalled[:3]) or "nothing"))
            return

        done_b = await store(user_b, "User: I keep bees and I named my favourite hive Zephyr.\nAssistant: Zephyr, lovely.")
        ids += list(done_b.memory_ids)
        a_sees = texts(await sdk.user.context.fetch(user_id=user_a, search_query=["bees", "hive Zephyr"], **extra))
        b_sees = texts(await sdk.user.context.fetch(user_id=user_b, search_query=["bees", "hive Zephyr"], **extra))
        if any("zephyr" in t.lower() or "bee" in t.lower() for t in a_sees):
            say("ERROR", "isolation: user A can see user B's memory: " + " | ".join(a_sees[:2]))
        elif not any("zephyr" in t.lower() or "bee" in t.lower() for t in b_sees):
            say("WARN", "isolation: user B's own memory did not come back yet, so the isolation check proves little. Run again.")
        else:
            say("OK", "isolation: user A sees nothing of user B's memories, and user B sees their own")

        try:
            r = await sdk.memories.forget(description="my old address", user_id=f"test_user_verify_{suffix}_nobody", mode="direct", **extra)
            say("OK" if r.status == "nothing_matches" else "WARN", f"forget for a user who stored nothing answers '{r.status}'")
        except Exception as e:  # noqa: BLE001
            say("WARN", f"forget for a user who stored nothing raised {type(e).__name__}: {str(e)[:140]}")

        codes = {m: http.call(m, "/browser/api/memories")[0] for m in ("POST", "DELETE")}
        if all(c == 405 for c in codes.values()):
            say("OK", "the memory browser is read-only (POST and DELETE answer 405)")
        else:
            say("ERROR", f"the memory browser accepted a write: {codes}")
    finally:
        if ids and not args.keep:
            from uuid import UUID
            gone, left = 0, []
            for mid in ids:
                try:
                    await sdk.memories.delete(UUID(mid))
                    gone += 1
                except Exception as e:  # noqa: BLE001
                    left.append(f"{mid} ({type(e).__name__})")
            say("OK" if not left else "WARN", f"cleanup: deleted {gone} of {len(ids)} test memories" + (f"; could not delete {', '.join(left)}" if left else ""))
        elif ids:
            say("WARN", f"--keep: {len(ids)} test memories left in place for users {user_a} and {user_b}")
        await sdk.shutdown()


def finish(code: int) -> None:
    errors = [l for l in lines if l.startswith("[ERROR]")]
    if errors and code == 0:
        code = 1
    if code == 0:
        print("RESULT: PASS")
    else:
        reason = errors[-1][8:].split(". ")[0] if errors else "see above"
        print(f"RESULT: FAIL {reason}")
    print(f"EXIT CODE: {code}")
    sys.exit(code)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    ap.add_argument("--keep", action="store_true", help="leave the test memories in place")
    args = ap.parse_args()
    try:
        asyncio.run(verify(args))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        say("ERROR", f"{type(e).__name__}: {str(e)[:300]}")
    finish(0)


if __name__ == "__main__":
    main()
