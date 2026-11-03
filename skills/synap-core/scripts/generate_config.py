#!/usr/bin/env python3
"""Ask Synap Cloud (through your Synap Core server) to write a memory config from a use-case file.

This is the MACA plugin. It needs, BEFORE you run it:
  * SYNAP_CLOUD_API_KEY in Core's .env: a Synap Cloud MANAGEMENT key (see reference/config-and-maca.md),
    and the stack restarted after you set it (`docker compose -f deploy/docker-compose.yml up -d`);
  * a use-case markdown file you wrote;
  * maximem-synap 0.6.13 or newer (`sdk.maca`).

    python3 scripts/generate_config.py --use-case use-case.md --relationship b2c

--relationship must match the install: b2c for equals_customer (consumer), b2b for strict (business).
Once memories exist the server refuses the other one.

A run takes 15 to 20 minutes. Core saves the result as config/memory.generated.yml and NEVER touches
config/memory.yml; reviewing and switching is a separate step. Exit 0 = succeeded, 1 = the job ended
invalid or failed, 2 = it could not be started.
"""
import argparse
import asyncio
import os
import sys
import time


def load_token() -> str:
    """SYNAP_API_TOKEN from the environment, else from ./.env (that one line only; never printed)."""
    token = os.environ.get("SYNAP_API_TOKEN", "")
    if not token and os.path.isfile(".env"):
        for raw in open(".env").read().splitlines():
            if raw.startswith("SYNAP_API_TOKEN="):
                token = raw.partition("=")[2].strip()
    return token


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use-case", required=True)
    ap.add_argument("--relationship", required=True, choices=["b2c", "b2b"])
    ap.add_argument("--url", default=os.environ.get("SYNAP_BASE_URL", "http://127.0.0.1:8000"))
    args = ap.parse_args()
    token = load_token()
    for name in [n for n in os.environ if n.startswith("SYNAP_")]:
        del os.environ[name]
    os.environ["SYNAP_BASE_URL"] = args.url
    if token:
        os.environ["SYNAP_API_TOKEN"] = token
    from maximem_synap import MaximemSynapSDK

    sdk = MaximemSynapSDK()
    await sdk.initialize()
    try:
        try:
            job = await sdk.maca.generate(file_path=args.use_case, user_relationship=args.relationship)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] could not start the job: {str(exc)[:400]}")
            print("RESULT: FAIL could not start")
            print("EXIT CODE: 2")
            sys.exit(2)
        print(f"[OK] {time.strftime('%H:%M:%S')} started {job.job_id}", flush=True)
        last = None
        while True:
            job = await sdk.maca.get_job(job.job_id)
            if job.status != last:
                print(f"[OK] {time.strftime('%H:%M:%S')} {job.status}: {job.message}", flush=True)
                last = job.status
            if job.status in ("succeeded", "invalid", "failed"):
                break
            await asyncio.sleep(15)
        print(f"     config file : {job.config_file}")
        print(f"     left out    : {len(job.removed_fields)} fields Core does not read")
        for problem in job.problems:
            print(f"[ERROR] {problem}")
        ok = job.status == "succeeded"
        print("RESULT: PASS" if ok else f"RESULT: FAIL job {job.status}")
        print(f"EXIT CODE: {0 if ok else 1}")
        sys.exit(0 if ok else 1)
    finally:
        await sdk.shutdown()


asyncio.run(main())
