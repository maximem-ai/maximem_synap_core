"""Brings up a complete Synap Core stack from this repository, for the end-to-end tests.

What it does, in order:
  1. Copies the tracked files of this repository into a temporary folder, so a run behaves like a
     fresh clone and never touches your own .env, config/ or originals/.
  2. Writes a .env there from E2E_ENV_FILE (default: ./.env). Your model settings are used; any Jev,
     token and Synap Cloud key lines are dropped, so every run starts without them.
  3. Runs `docker compose up -d` under its own project name (core-e2e-<random>), so it cannot reuse
     the volumes or containers of another Core stack on the machine.
  4. At the end removes that project and its volumes (`down -v`), and the temporary folder.

The API publishes port 8000 (or SYNAP_PORT, if you set it in your shell), so that port must be free.

Environment:
  E2E_ENV_FILE             a filled-in .env (model provider, model, key). Default: ./.env
  E2E_SEED_MODELS_VOLUME   name of a Docker volume that already holds the downloaded models, to skip
                           the 2.3 GB download (copied into the test project's models volume)
  E2E_KEEP=1               leave the stack and the temporary folder in place afterwards
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("SYNAP_PORT", "8000"))
DROPPED_ENV_KEYS = ("SYNAP_API_TOKEN", "JEV_API_KEY", "SYNAP_CLOUD_API_KEY")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Stack:
    def __init__(self, project: str, root: Path):
        self.project = project
        self.root = root
        self.env_path = root / ".env"
        self.compose_file = root / "deploy" / "docker-compose.yml"
        self.url = f"http://127.0.0.1:{PORT}"

    # -- docker -----------------------------------------------------------------------------------
    def compose(self, *args: str, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["docker", "compose", "-p", self.project, "-f", str(self.compose_file), *args]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and done.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{done.stdout[-1500:]}\n{done.stderr[-2500:]}")
        return done

    def docker(self, *args: str, timeout: int = 120) -> str:
        done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
        return done.stdout

    def logs(self, service: str) -> str:
        return ANSI.sub("", self.compose("logs", service, "--no-color", check=False).stdout)

    def exit_code(self, service: str) -> int:
        return int(self.docker("inspect", f"{self.project}-{service}-1", "--format", "{{.State.ExitCode}}").strip())

    # -- the .env ---------------------------------------------------------------------------------
    def set_env(self, key: str, value: str | None) -> None:
        lines = [l for l in self.env_path.read_text().splitlines() if not l.startswith(f"{key}=")]
        if value is not None:
            lines.append(f"{key}={value}")
        self.env_path.write_text("\n".join(lines) + "\n")

    # -- http -------------------------------------------------------------------------------------
    def get(self, path: str, headers: dict | None = None) -> tuple[int, str]:
        req = urllib.request.Request(self.url + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def send(self, method: str, path: str) -> int:
        req = urllib.request.Request(self.url + path, method=method, data=b"{}" if method != "DELETE" else None)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def wait_healthy(self, timeout: int = 300) -> None:
        end = time.time() + timeout
        while time.time() < end:
            try:
                status, body = self.get("/health")
                if status == 200:
                    return
            except Exception:
                pass
            time.sleep(3)
        raise RuntimeError("the server did not become healthy:\n" + self.logs("synap")[-3000:])


def _port_is_free() -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", PORT)) != 0


@pytest.fixture(scope="session")
def stack():
    env_file = Path(os.environ.get("E2E_ENV_FILE", REPO / ".env"))
    if not env_file.is_file():
        pytest.exit(
            f"No env file at {env_file}. Copy .env.example to .env and fill in the model lines, "
            "or point E2E_ENV_FILE at a filled-in file.", returncode=2)
    if not _port_is_free():
        pytest.exit(f"Port {PORT} is in use. Stop the Core stack that is running (docker compose stop), "
                    "then run again.", returncode=2)

    root = Path(tempfile.mkdtemp(prefix="core-e2e-"))
    st = None
    try:
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
        for rel in filter(None, tracked.decode().split("\0")):
            src = REPO / rel
            if src.is_file():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, root / rel)
        kept = [l for l in env_file.read_text().splitlines() if not l.startswith(tuple(f"{k}=" for k in DROPPED_ENV_KEYS))]
        (root / ".env").write_text("\n".join(kept) + "\n")
        (root / ".env").chmod(0o600)

        st = Stack("core-e2e-" + secrets.token_hex(4), root)
        seed = os.environ.get("E2E_SEED_MODELS_VOLUME")
        if seed:
            st.docker("volume", "create", f"{st.project}_models")
            st.docker("run", "--rm", "-v", f"{seed}:/from:ro", "-v", f"{st.project}_models:/to",
                      "alpine:3.20", "sh", "-c", "cp -a /from/. /to/", timeout=900)
        st.compose("up", "-d", timeout=3600)
        st.wait_healthy()
        yield st
    finally:
        if os.environ.get("E2E_KEEP") == "1" and st is not None:
            print(f"\nE2E_KEEP=1: stack '{st.project}' and {root} left in place")
        else:
            if st is not None:
                try:
                    st.compose("down", "-v", check=False, timeout=300)
                except Exception:
                    pass
            shutil.rmtree(root, ignore_errors=True)
