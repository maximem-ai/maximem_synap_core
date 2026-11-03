#!/usr/bin/env python3
"""Wait for Synap Core to become healthy after `docker compose up -d`, and show what it is doing.

The first start takes a while: it builds (or pulls) the image, generates the database passwords,
downloads the embedding model (about 2.3 GiB, once) and creates the schema. This script polls
quietly, prints one progress line when something changes, and stops when /health answers ok.

    python3 scripts/wait_healthy.py                              # from the folder that holds deploy/
    python3 scripts/wait_healthy.py --project synap-core --timeout 3600

If you ran Compose with SYNAP_IMAGE / SYNAP_PULL_POLICY (the published-image path), run this with
the same two variables in the environment so the Compose calls it makes agree with yours.

Exit 0 = healthy. 1 = a service failed or keeps restarting (the log tail of that service is printed). 2 = timed out.

A tool call that blocks for more than about ten minutes is cut off, and the first start can take longer.
Run it in the background with its output in a file and read the file until it ends with `RESULT:`:
    python3 scripts/wait_healthy.py > wait.out 2>&1 &
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def parse_ps(text: str) -> list:
    """`compose ps --format json` is one JSON array on Compose before 2.21 and one object per line after."""
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            return json.loads(text)
        except ValueError:
            return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def health(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception:
        return 0, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="deploy/docker-compose.yml")
    ap.add_argument("--project", default=None, help="the Compose project name, if you used -p")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    base = ["docker", "compose"] + (["-p", args.project] if args.project else []) + ["-f", args.file]

    start, last, last_progress, warned_empty = time.time(), "", 0.0, False
    while time.time() - start < args.timeout:
        status, body = health(args.url)
        if status == 200:
            print(f"[OK] healthy after {int(time.time() - start)}s: {body.strip()[:80]}")
            print("RESULT: PASS")
            print("EXIT CODE: 0")
            sys.exit(0)

        rows = parse_ps(run(base + ["ps", "-a", "--format", "json"]))
        state = {r.get("Service"): (r.get("State"), r.get("ExitCode"), r.get("Health")) for r in rows}
        failed = [s for s, (st, code, _) in state.items() if (st == "exited" and code not in (0, None)) or st in ("restarting", "dead")]
        if not state and time.time() - start > 90 and not warned_empty:
            print("[WARN] no containers after 90s. Did `docker compose up -d` start, and did it fail? Read its output (the log file you redirected it to).", flush=True)
            warned_empty = True
        if failed:
            svc = failed[0]
            tail = ANSI.sub("", run(base + ["logs", "--no-color", "--tail", "40", svc]))
            print(f"[ERROR] service '{svc}' {'is restarting in a loop' if state[svc][0] == 'restarting' else 'exited with a failure'}:")
            print(tail[-3000:])
            if svc == "first-start":
                print("[WARN] first-start stops when a model line in .env is still a placeholder, or when the download fails (network drop, Mac asleep). Fix the cause and run `docker compose up -d` again: the download continues where it stopped.")
            print(f"RESULT: FAIL service {svc} failed")
            print("EXIT CODE: 1")
            sys.exit(1)

        summary = ", ".join(f"{s}={st}{'/' + h if h else ''}" for s, (st, _, h) in sorted(state.items())) or "no containers yet"
        if summary != last:
            print(f"[{int(time.time() - start):>4}s] {summary}", flush=True)
            last = summary
        if state.get("first-start", ("",))[0] == "running" and time.time() - last_progress > 30:
            line = [l for l in ANSI.sub("", run(base + ["logs", "--no-color", "--tail", "3", "first-start"])).splitlines() if "first-start" in l]
            if line:
                print("       " + line[-1].split("|", 1)[-1].strip(), flush=True)
            last_progress = time.time()
        time.sleep(5)

    print(f"[ERROR] not healthy after {args.timeout}s. Last state: {last}")
    print("RESULT: FAIL timed out waiting for the server")
    print("EXIT CODE: 2")
    sys.exit(2)


if __name__ == "__main__":
    main()
