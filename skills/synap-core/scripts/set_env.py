#!/usr/bin/env python3
"""Write settings into Synap Core's .env without ever printing a secret.

The coding agent must never see or repeat an API key. So:

  * non-secret settings (provider, model, base URL ...) are given as KEY=VALUE;
  * secret settings (anything whose name ends in KEY, TOKEN, SECRET or PASSWORD) are NEVER
    taken from the command line. They come from the human's own shell with --from-env NAME
    (the human ran `export OPENAI_API_KEY=...` in their terminal), or they are generated
    here (--generate-token), or the human types them into the file themselves.

Output shows names and lengths only. The file is created from .env.example when missing and is
kept at mode 600.

    python3 scripts/set_env.py SYNAP_LLM_PROVIDER=openai SYNAP_LLM_MODEL=gpt-4o-mini
    python3 scripts/set_env.py --from-env OPENAI_API_KEY
    python3 scripts/set_env.py --from-env SYNAP_LLM_API_KEY=MY_GEMINI_KEY      # copy MY_GEMINI_KEY into SYNAP_LLM_API_KEY
    python3 scripts/set_env.py --generate-token                                # SYNAP_API_TOKEN, random, not printed
    python3 scripts/set_env.py --unset SYNAP_CLOUD_API_KEY                     # empty a setting (the line stays, with no value)
    python3 scripts/set_env.py --file /path/to/.env ...

Exit 0 on success, 2 on a refused request. Standard library only.
"""
from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import sys

ALLOWED = {
    "SYNAP_LLM_PROVIDER", "SYNAP_LLM_MODEL", "SYNAP_LLM_BASE_URL", "SYNAP_LLM_API_KEY", "SYNAP_LLM_CONTEXT_WINDOW",
    "SYNAP_LLM_MODEL_INGESTION", "SYNAP_LLM_MODEL_GRAPH", "SYNAP_LLM_MODEL_KNOWLEDGE_UPDATE",
    "SYNAP_LLM_MODEL_COMPACTION", "SYNAP_LLM_MODEL_VISION",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_NUM_CTX", "OLLAMA_KEEP_ALIVE",
    "EMBEDDING_PROVIDER", "SYNAP_API_TOKEN", "JEV_API_KEY", "SYNAP_CLOUD_API_KEY",
}
PROVIDERS = {"openai", "anthropic", "ollama", "openai-compatible"}
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$")


def mask_url(url: str) -> str:
    """scheme://host:port/path only. A gateway URL can carry a token in its user info or query string."""
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        return "<not a URL>"
    shown = f"{parts.scheme}://{parts.hostname}" + (f":{parts.port}" if parts.port else "") + parts.path
    hidden = bool(parts.username or parts.password or parts.query or parts.fragment)
    return shown + (" (credentials or query hidden)" if hidden else "")


def refuse(msg: str) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(2)


def set_line(lines: list[str], key: str, value: str) -> list[str]:
    """Replace `KEY=...`, else un-comment `# KEY=...`, else append."""
    active = re.compile(rf"^\s*{re.escape(key)}=")
    commented = re.compile(rf"^\s*#\s*{re.escape(key)}=")
    for i, line in enumerate(lines):
        if active.match(line):
            lines[i] = f"{key}={value}"
            return lines
    for i, line in enumerate(lines):
        if commented.match(line):
            lines[i] = f"{key}={value}"
            return lines
    lines.append(f"{key}={value}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", help="KEY=VALUE for non-secret settings")
    ap.add_argument("--file", default=".env")
    ap.add_argument("--from-env", action="append", default=[], metavar="NAME[=SOURCE]",
                    help="copy a secret from an environment variable of this shell (NAME is the .env key; SOURCE the variable to read, default NAME)")
    ap.add_argument("--generate-token", action="store_true", help="write a random SYNAP_API_TOKEN")
    ap.add_argument("--unset", action="append", default=[], metavar="NAME", help="empty a setting, for example SYNAP_CLOUD_API_KEY after the MACA plugin has run")
    args = ap.parse_args()

    path = os.path.abspath(args.file)
    if not os.path.exists(path):
        example = os.path.join(os.path.dirname(path), ".env.example")
        if not os.path.exists(example):
            refuse(f"{path} does not exist and there is no .env.example next to it to copy.")
        shutil.copyfile(example, path)
        print(f"[OK] created {path} from .env.example")
    lines = open(path).read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    done: list[str] = []
    for pair in args.pairs:
        if "=" not in pair:
            refuse(f"'{pair}' is not KEY=VALUE.")
        key, _, value = pair.partition("=")
        if key not in ALLOWED:
            refuse(f"{key} is not a setting Synap Core reads. Allowed: {', '.join(sorted(ALLOWED))}")
        if SECRET_NAME.search(key):
            refuse(f"{key} is a secret. Do not pass it on the command line: use --from-env {key} (the human exports it in their own shell), or let the human type it into {path}.")
        if value != value.strip() or value[:1] in "\"'" or not value:
            refuse(f"The value for {key} is empty or has quotes or spaces around it; write it bare.")
        if key == "SYNAP_LLM_PROVIDER" and value not in PROVIDERS:
            refuse(f"SYNAP_LLM_PROVIDER must be one of {', '.join(sorted(PROVIDERS))}, not '{value}'.")
        lines = set_line(lines, key, value)
        done.append(f"{key} = {mask_url(value) if key.endswith('_URL') else value}")

    for spec in args.from_env:
        key, _, source = spec.partition("=")
        source = source or key
        if key not in ALLOWED or not SECRET_NAME.search(key):
            refuse(f"--from-env is for secrets only (one of: {', '.join(sorted(k for k in ALLOWED if SECRET_NAME.search(k)))}).")
        value = os.environ.get(source, "")
        if not value:
            refuse(f"The environment variable {source} is not set in this shell, so there is nothing to copy. Ask the human to run `export {source}=...` in their own terminal and start this command from that same terminal; or to type the value into {path} themselves. Do not ask them to paste it here.")
        lines = set_line(lines, key, value)
        done.append(f"{key} = <copied from ${source}, {len(value)} characters, not shown>")

    if args.generate_token:
        token = secrets.token_urlsafe(32)
        lines = set_line(lines, "SYNAP_API_TOKEN", token)
        done.append(f"SYNAP_API_TOKEN = <generated, {len(token)} characters, not shown; give it to your backend as SYNAP_API_TOKEN>")

    for key in args.unset:
        if key not in ALLOWED:
            refuse(f"{key} is not a setting Synap Core reads.")
        lines = set_line(lines, key, "")
        done.append(f"{key} = <emptied>")

    if not done:
        refuse("Nothing to do. Give KEY=VALUE pairs, --from-env, --generate-token or --unset.")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)
    for item in done:
        print(f"[OK] {item}")
    print(f"[OK] {path} saved (mode 600). After changing .env run `docker compose -f deploy/docker-compose.yml up -d` so the server picks it up.")


if __name__ == "__main__":
    main()
