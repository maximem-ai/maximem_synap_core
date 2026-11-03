#!/usr/bin/env python3
"""Check that the memory config the server is using is accepted and really in force.

Run it after switching to a new config/memory.yml (your own, or one the MACA plugin wrote).

    python3 scripts/check_config.py
    python3 scripts/check_config.py --rate-limit     # also test the retrieval limit, if the config sets one (run last)

It reads the active config from the memory browser's API, then stores and recalls a throwaway fact
through the SDK for two test users and checks they stay apart. Output: [OK]/[WARN]/[ERROR] lines,
then `RESULT:` and `EXIT CODE:`. Needs `pip install maximem-synap` and, for the rate-limit test,
`pip install pyyaml`. Reads SYNAP_BASE_URL and SYNAP_API_TOKEN from the environment.
"""
import argparse
import asyncio
import json
import os
import sys
import urllib.request
import uuid

lines = []


def say(level, text):
    lines.append(f"[{level}] {text}")
    print(lines[-1], flush=True)


def load_token() -> str:
    """SYNAP_API_TOKEN from the environment, else from ./.env (that one line only; never printed)."""
    token = os.environ.get("SYNAP_API_TOKEN", "")
    if not token and os.path.isfile(".env"):
        for raw in open(".env").read().splitlines():
            if raw.startswith("SYNAP_API_TOKEN="):
                token = raw.partition("=")[2].strip()
    return token


def texts(ctx):
    return [getattr(i, "content", str(i)) for n in ("facts", "preferences", "episodes", "emotions", "temporal_events") for i in (getattr(ctx, n, None) or [])]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate-limit", action="store_true")
    ap.add_argument("--url", default=os.environ.get("SYNAP_BASE_URL", "http://127.0.0.1:8000"))
    args = ap.parse_args()
    token = load_token()

    def get(path):
        req = urllib.request.Request(args.url + path, headers={"Authorization": f"Bearer {token}"} if token else {})
        return json.load(urllib.request.urlopen(req, timeout=20))

    cfg = get("/browser/api/config")
    if cfg.get("valid") is not True:
        say("ERROR", "the config is not accepted: " + "; ".join(cfg.get("problems", [])[:2]))
    else:
        say("OK", "the config is accepted: " + ", ".join(f"{s['name']} ({s['settings']})" for s in cfg["sections"]))
    say("OK" if not cfg.get("starter") else "WARN", "this is the developer's own config" if not cfg.get("starter") else "this is still the starter sample, not a config of your own")
    mode = None
    try:
        mode = get("/api/v1/auth/whoami").get("user_context_isolation")
    except Exception:
        pass
    say("OK", f"scoping mode: {mode}")
    customer = {"customer_id": "synap-verify"} if mode == "strict" else {}

    for name in [n for n in os.environ if n.startswith("SYNAP_") and n != "SYNAP_API_TOKEN"]:
        del os.environ[name]
    os.environ["SYNAP_BASE_URL"] = args.url
    if token:
        os.environ["SYNAP_API_TOKEN"] = token
    try:
        from maximem_synap import MaximemSynapSDK
    except ImportError:
        say("ERROR", "the maximem-synap package is not installed in this Python. It needs Python 3.11 or newer: use a virtualenv with it and `pip install maximem-synap`.")
        print("RESULT: FAIL maximem-synap not installed")
        print("EXIT CODE: 1")
        sys.exit(1)

    sdk = MaximemSynapSDK()
    await sdk.initialize()
    run = uuid.uuid4().hex[:6]
    ana, ben = f"test_user_config_{run}_a", f"test_user_config_{run}_b"
    ids = []
    try:
        for user, doc in ((ana, "User: I am vegetarian, no fish either. I run on Tuesday and Saturday mornings.\nAssistant: Understood."),
                          (ben, "User: My sister Priya's birthday is on 14 March and she loves orchids.\nAssistant: Noted.")):
            created = await sdk.memories.create(document=doc, document_type="ai-chat-conversation", user_id=user, **customer)
            done = await sdk.memories.wait_for_completion(created.ingestion_id, timeout_seconds=400)
            ids += list(done.memory_ids)
            say("OK" if done.memories_created >= 1 else "ERROR", f"{user}: stored {done.memories_created} memories")
        a = texts(await sdk.user.context.fetch(user_id=ana, search_query=["diet", "running", "Priya"], **customer))
        b = texts(await sdk.user.context.fetch(user_id=ben, search_query=["diet", "running", "Priya"], **customer))
        say("OK" if any("vegetarian" in t.lower() for t in a) else "ERROR", "user A's diet is recalled")
        say("OK" if any("priya" in t.lower() for t in b) else "ERROR", "user B's sister is recalled")
        say("OK" if not any("priya" in t.lower() for t in a) and not any("vegetarian" in t.lower() for t in b) else "ERROR", "the two users stay apart")
        if args.rate_limit:
            try:
                import yaml
                limit = ((yaml.safe_load(cfg.get("text") or "") or {}).get("guardrails") or {}).get("rate_limits", {}).get("max_retrieval_calls_per_minute")
            except ImportError:
                limit = None
                say("WARN", "pyyaml is not installed, so the retrieval limit cannot be read; `pip install pyyaml`")
            if not limit:
                say("OK", "the config sets no retrieval limit, nothing to test")
            else:
                refused = 0
                for i in range(int(limit) + 15):
                    try:
                        await sdk.user.context.fetch(user_id=f"test_user_probe_{run}_{i}", search_query=[f"probe {i}"], **customer)
                    except Exception as exc:  # noqa: BLE001
                        refused += any(w in str(exc).lower() for w in ("rate", "limit"))
                say("OK" if refused else "ERROR", f"more than {limit} fetches in a minute: {refused} were refused (fetches stay blocked for about a minute)")
    finally:
        from uuid import UUID
        left = []
        for mid in ids:
            try:
                await sdk.memories.delete(UUID(mid))
            except Exception as exc:  # noqa: BLE001
                left.append(f"{mid} ({type(exc).__name__})")
        say("OK" if not left else "WARN", f"cleanup: removed {len(ids) - len(left)} of {len(ids)} test memories" + (f"; could not remove {', '.join(left)}" if left else ""))
        await sdk.shutdown()
    errors = [l for l in lines if l.startswith("[ERROR]")]
    print("RESULT: PASS" if not errors else f"RESULT: FAIL {errors[0][8:]}")
    print(f"EXIT CODE: {1 if errors else 0}")
    sys.exit(1 if errors else 0)


asyncio.run(main())
