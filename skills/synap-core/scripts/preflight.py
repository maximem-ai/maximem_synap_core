#!/usr/bin/env python3
"""Synap Core preflight: can this machine run Synap Core, and is anything in the way?

Run it BEFORE installing anything. It changes nothing. It checks, in order:

  1. the operating system and CPU,
  2. Docker and Docker Compose (installed, daemon reachable, version),
  3. the memory Docker may use and the free disk where Synap Core will live,
  4. port 8000 (Synap Core publishes the API on 127.0.0.1:8000 only),
  5. an existing Synap Core stack: containers and Docker volumes named synap-core_*,
     which a new start would REUSE (old memories, old passwords),
  6. git and Python (for the SDK), and on Linux whether SYNAP_USER is set,
  7. whether the registries the first start needs can be reached,
  8. a .env in the target folder, by NAME only (values are never read out).

Every line starts with [OK], [WARN] or [ERROR]. The output ends with exactly two lines:
`RESULT: PASS` or `RESULT: FAIL <reason>`, then `EXIT CODE: <n>`. Quote the whole output to
the human as it is; a [WARN] is explained, never dropped.

    python3 scripts/preflight.py                 # checks the current folder
    python3 scripts/preflight.py --dir ~/synap   # the folder you plan to use
    python3 scripts/preflight.py --image         # you plan the published-image path (no git, no build)

Exit code 0 = no [ERROR]. 1 = at least one [ERROR]. Standard library only.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request

PORT = 8000
results: list[str] = []


def say(level: str, text: str) -> None:
    line = f"[{level}] {text}"
    results.append(line)
    print(line, flush=True)


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr).strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def check_os() -> None:
    system, arch = platform.system(), platform.machine()
    say("OK", f"{system} {platform.release()} on {arch}")
    if system == "Windows":
        say("WARN", "Windows: run everything inside WSL2 (Ubuntu) with Docker Desktop's WSL integration on. Synap Core has not been tested natively on Windows.")
    if system == "Linux":
        if not os.environ.get("SYNAP_USER"):
            say("WARN", "Linux: SYNAP_USER is not set in this shell. Run `export SYNAP_USER=$(id -u):$(id -g)` before `docker compose up`, in the same terminal, so the server can write the originals/ folder.")
        else:
            say("OK", "SYNAP_USER is set for this shell")
    if arch in ("x86_64", "AMD64") and system == "Darwin":
        say("OK", "Intel Mac: the amd64 image is used")


def check_docker() -> bool:
    code, out = run(["docker", "--version"])
    if code != 0:
        say("ERROR", "Docker is not installed or not on PATH. Install Docker Desktop (Mac, Windows) or Docker Engine with the Compose plugin (Linux), then run this again.")
        return False
    say("OK", out.splitlines()[0])
    code, out = run(["docker", "compose", "version"])
    if code != 0:
        say("ERROR", "Docker Compose v2 is missing (`docker compose version` failed). Install the Compose plugin; the old `docker-compose` v1 is not supported.")
        return False
    say("OK", out.splitlines()[0])
    code, out = run(["docker", "info", "--format", "{{.ServerVersion}}|{{.MemTotal}}|{{.OperatingSystem}}"], timeout=30)
    if code != 0:
        say("ERROR", "The Docker daemon is not reachable. Start Docker Desktop (or `sudo systemctl start docker`) and wait until it says it is running.")
        return False
    version, mem, osname = (out.split("|") + ["", "", ""])[:3]
    gib = int(mem) / 2**30 if mem.isdigit() else 0
    if gib and gib < 6:
        say("WARN", f"Docker has {gib:.1f} GiB of memory. The stack uses about 4 GiB when idle and more while it loads models; give Docker at least 6 GiB (Docker Desktop: Settings, Resources).")
    elif gib:
        say("OK", f"Docker daemon {version} reachable, {gib:.1f} GiB of memory available to it")
    else:
        say("OK", f"Docker daemon {version} reachable")
    return True


def check_disk(path: str, image_path: bool) -> None:
    probe = path
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe or ".").free / 2**30
    need = "about 8 GiB (the server image is about 3 GiB, the models about 2.5 GiB, the stores grow with use)"
    if free < 8:
        say("ERROR", f"Only {free:.1f} GiB free where Synap Core will live ({probe}); it needs {need}.")
    elif free < 12:
        say("WARN", f"{free:.1f} GiB free at {probe}. Enough to start; it needs {need} and your memories add to that.")
    else:
        say("OK", f"{free:.0f} GiB free at {probe} (needs {need})")


def check_port() -> None:
    with socket.socket() as s:
        s.settimeout(1)
        busy = s.connect_ex(("127.0.0.1", PORT)) == 0
    if not busy:
        say("OK", f"port {PORT} is free")
        return
    code, out = run(["docker", "ps", "--filter", f"publish={PORT}", "--format", "{{.Names}}"])
    owner = out.strip() if code == 0 and out.strip() else "something that is not a Docker container"
    say("WARN", f"port {PORT} is already in use by {owner}. If that is an earlier Synap Core, stop it (`docker compose stop`) before starting another; two stacks cannot share the port.")


def check_existing() -> None:
    code, out = run(["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}"])
    names = [l for l in out.splitlines() if l.startswith("synap-core-")] if code == 0 else []
    if names:
        say("WARN", f"an earlier Synap Core stack exists: {len(names)} containers named synap-core-* (" + "; ".join(n.split('|')[0] + ' ' + n.split('|')[1] for n in names[:3]) + (" ..." if len(names) > 3 else "") + "). Starting again from the same Compose project reuses them.")
    code, out = run(["docker", "volume", "ls", "--format", "{{.Name}}"])
    vols = [v for v in out.splitlines() if v.startswith("synap-core_")] if code == 0 else []
    if vols:
        say("WARN", f"Docker volumes from an earlier Synap Core exist ({', '.join(sorted(vols))}). A new start under the same project name REUSES them: old memories, old passwords, and a scoping mode that can no longer be changed. Ask the human: keep them, or start fresh. Never delete them without an explicit yes.")
    if not names and not vols:
        say("OK", "no earlier Synap Core containers or volumes: this will be a fresh install")


def check_tools(image_path: bool) -> None:
    if not image_path:
        code, out = run(["git", "--version"])
        say("OK" if code == 0 else "ERROR", out if code == 0 else "git is not installed (needed to clone the repository). The published-image path does not need it; or install git.")
    py = sys.version_info
    if py >= (3, 11):
        say("OK", f"Python {py.major}.{py.minor}.{py.micro} (the SDK needs 3.11 or newer)")
    else:
        say("WARN", f"This Python is {py.major}.{py.minor}; the Python SDK needs 3.11 or newer. Use a newer Python for the SDK, or use the JavaScript SDK (Node 20+).")
    code, out = run(["node", "--version"])
    say("OK", f"Node {out} (JavaScript SDK, needs 20 or newer)" if code == 0 else "Node is not installed (only needed for the JavaScript SDK)")


def reachable(url: str) -> bool:
    """True when the host answers at all. Uses curl when there is one (it has its own certificates);
    otherwise urllib, and a certificate problem on this Python is not counted as 'unreachable'."""
    if shutil.which("curl"):
        code, out = run(["curl", "-s", "-o", "/dev/null", "-I", "--max-time", "10", "-w", "%{http_code}", url], timeout=20)
        return out.isdigit() and out != "000"
    import ssl
    for context in (None, ssl._create_unverified_context()):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "synap-core-preflight"})
            urllib.request.urlopen(req, timeout=10, context=context)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            continue
    return False


def check_network() -> None:
    for name, url, why in (
        ("Hugging Face", "https://huggingface.co", "the first start downloads the embedding model (about 2.3 GiB) once"),
        ("GitHub container registry", "https://ghcr.io", "the Chroma store image, and the Synap Core image on the published-image path"),
        ("Docker Hub", "https://registry-1.docker.io/v2/", "the Postgres, Neo4j and Redis images"),
    ):
        if reachable(url):
            say("OK", f"{name} is reachable ({why})")
        else:
            say("WARN", f"{name} could not be reached from here ({why}). Check the network or a proxy before the first start.")


def check_env(folder: str) -> None:
    path = os.path.join(folder, ".env")
    if not os.path.isfile(path):
        say("OK", f"no .env in {folder} yet (it is created from .env.example)")
        return
    names = []
    for line in open(path):
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            names.append((k.strip(), bool(v.strip())))
    filled = [k for k, has in names if has]
    say("OK", f".env exists with these lines filled in (names only): {', '.join(filled) or 'none'}")
    mode = oct(os.stat(path).st_mode & 0o777)
    if os.stat(path).st_mode & 0o077:
        say("WARN", f".env has mode {mode}; run `chmod 600 .env` so other users of this machine cannot read your keys.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.getcwd(), help="the folder Synap Core will live in")
    ap.add_argument("--image", action="store_true", help="the published-image path (no git, no build)")
    args = ap.parse_args()
    folder = os.path.abspath(os.path.expanduser(args.dir))

    check_os()
    if check_docker():
        check_disk(folder, args.image)
        check_port()
        check_existing()
    check_tools(args.image)
    check_network()
    check_env(folder)

    errors = [r for r in results if r.startswith("[ERROR]")]
    if errors:
        print(f"RESULT: FAIL {errors[0][8:].split('.')[0]}")
        print("EXIT CODE: 1")
        sys.exit(1)
    print("RESULT: PASS")
    print("EXIT CODE: 0")


if __name__ == "__main__":
    main()
