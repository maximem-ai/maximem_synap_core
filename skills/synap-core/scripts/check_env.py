#!/usr/bin/env python3
"""Check Synap Core's .env before the first start. Reads values, prints names and lengths only.

It answers: will the server start with this file, and does the combination make sense?

    python3 scripts/check_env.py                  # ./.env
    python3 scripts/check_env.py --file /path/.env

Every line starts with [OK], [WARN] or [ERROR]; the output ends with `RESULT:` and `EXIT CODE:`.
Exit 0 = the server can start with this file. 1 = it cannot, the [ERROR] lines say why.
Standard library only. A secret is never printed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

PROVIDERS = ("openai", "anthropic", "ollama", "openai-compatible")
lines: list[str] = []


def say(level: str, text: str) -> None:
    line = f"[{level}] {text}"
    lines.append(line)
    print(line)


def mask_url(url: str) -> str:
    """scheme://host:port/path only. A gateway URL can carry a token in its user info or query string."""
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        return "<not a URL>"
    shown = f"{parts.scheme}://{parts.hostname}" + (f":{parts.port}" if parts.port else "") + parts.path
    hidden = bool(parts.username or parts.password or parts.query or parts.fragment)
    return shown + (" (credentials or query hidden)" if hidden else "")


def read(path: str) -> dict[str, str]:
    vals: dict[str, str] = {}
    for raw in open(path).read().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        vals[key.strip()] = value.split("  #")[0].rstrip()
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=".env")
    path = os.path.abspath(ap.parse_args().file)
    if not os.path.isfile(path):
        say("ERROR", f"{path} does not exist. Copy .env.example to .env first.")
        finish()
    v = read(path)

    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        say("WARN", f".env has mode {oct(mode)}; run `chmod 600 .env` so other users cannot read your keys.")
    else:
        say("OK", ".env is readable by you only (mode 600)")

    provider, model = v.get("SYNAP_LLM_PROVIDER", ""), v.get("SYNAP_LLM_MODEL", "")
    if not provider or provider.startswith("add-"):
        say("ERROR", "SYNAP_LLM_PROVIDER is still the placeholder. Set it to one of: " + ", ".join(PROVIDERS) + ". The first start stops until you do.")
    elif provider not in PROVIDERS:
        say("ERROR", f"SYNAP_LLM_PROVIDER is '{provider}'. It must be one of: {', '.join(PROVIDERS)}.")
    else:
        say("OK", f"SYNAP_LLM_PROVIDER = {provider}")
    if not model or model.startswith("add-"):
        say("ERROR", "SYNAP_LLM_MODEL is still the placeholder. Write the model name exactly as your provider names it. The first start stops until you do.")
    else:
        say("OK", f"SYNAP_LLM_MODEL = {model}")

    def has(key: str) -> bool:
        return bool(v.get(key))

    if provider == "openai":
        say("OK" if has("OPENAI_API_KEY") else "ERROR", f"OPENAI_API_KEY is {'set (%d characters)' % len(v['OPENAI_API_KEY']) if has('OPENAI_API_KEY') else 'empty: the provider is openai, so the key is required'}")
    elif provider == "anthropic":
        say("OK" if has("ANTHROPIC_API_KEY") else "ERROR", f"ANTHROPIC_API_KEY is {'set (%d characters)' % len(v['ANTHROPIC_API_KEY']) if has('ANTHROPIC_API_KEY') else 'empty: the provider is anthropic, so the key is required'}")
    elif provider == "openai-compatible":
        url = v.get("SYNAP_LLM_BASE_URL", "")
        if url and "@" in url.split("//", 1)[-1].split("/", 1)[0]:
            say("WARN", "SYNAP_LLM_BASE_URL has credentials in its address. They are not shown here; make sure that is intended.")
        if not url:
            say("ERROR", "SYNAP_LLM_BASE_URL is empty. The provider is openai-compatible, so the server needs the address of the endpoint (for Gemini: https://generativelanguage.googleapis.com/v1beta/openai/).")
        elif not re.match(r"^https?://", url):
            say("ERROR", "SYNAP_LLM_BASE_URL must start with http:// or https://.")
        else:
            say("OK", f"SYNAP_LLM_BASE_URL = {mask_url(url)}")
            if "localhost" in url or "127.0.0.1" in url:
                say("WARN", "SYNAP_LLM_BASE_URL points at localhost, which inside the server's container is the container itself. Use http://host.docker.internal:<port>/v1 to reach a server on this machine.")
        if has("SYNAP_LLM_API_KEY"):
            say("OK", f"SYNAP_LLM_API_KEY is set ({len(v['SYNAP_LLM_API_KEY'])} characters)")
        else:
            say("WARN", "SYNAP_LLM_API_KEY is empty. Most hosted endpoints need one; a local server may not.")
    elif provider == "ollama":
        say("OK", "ollama needs no key. Make sure `ollama pull <model>` was run for the model named above, and see reference/model-setup.md for the Linux network step.")

    unused = []
    if provider != "openai" and has("OPENAI_API_KEY"):
        unused.append("OPENAI_API_KEY")
    if provider != "anthropic" and has("ANTHROPIC_API_KEY"):
        unused.append("ANTHROPIC_API_KEY")
    if provider not in ("openai-compatible",) and has("SYNAP_LLM_API_KEY"):
        unused.append("SYNAP_LLM_API_KEY")
    if unused:
        say("WARN", f"{', '.join(unused)} is set but the provider is {provider or 'not chosen'}, so it is not used. Harmless; remove it if you do not want the key on disk.")

    embed = v.get("EMBEDDING_PROVIDER", "")
    say("OK" if embed in ("sentence-transformers", "") else "WARN",
        f"EMBEDDING_PROVIDER = {embed or 'default (sentence-transformers, runs inside the server)'}")

    for key in v:
        value = v[key]
        if value and (value != value.strip() or value[0] in "\"'"):
            say("ERROR", f"{key} has quotes or spaces around its value. Write the value bare: {key}=value")
        odd = [i for i, c in enumerate(value) if ord(c) > 126 or ord(c) < 32]
        if odd:
            say("ERROR", f"{key} has a character that is not plain ASCII (code {hex(ord(value[odd[0]]))} at position {odd[0] + 1} of {len(value)}). A stray keystroke while pasting is the usual cause; delete it and save. The server cannot send it in a request header.")
        elif key.endswith(("_KEY", "_TOKEN", "_PASSWORD")) and any(c.isspace() for c in value):
            say("ERROR", f"{key} has a space inside it. Keys and tokens have none; check the paste.")

    if has("SYNAP_API_TOKEN"):
        say("OK", f"SYNAP_API_TOKEN is set ({len(v['SYNAP_API_TOKEN'])} characters): every request must carry it")
    else:
        say("WARN", "SYNAP_API_TOKEN is not set: anyone who can reach the port can read and write every memory. The port is bound to 127.0.0.1, so on one machine that is you; set a token before exposing it anywhere (reference/production.md).")
    if has("SYNAP_CLOUD_API_KEY"):
        say("WARN", "SYNAP_CLOUD_API_KEY is set. It is only needed while the MACA plugin writes a config; remove it afterwards (reference/config-and-maca.md).")
    if has("JEV_API_KEY"):
        say("OK", "JEV_API_KEY is set: forget by description is available")
    if not has("SYNAP_LLM_MODEL_VISION"):
        say("WARN", "SYNAP_LLM_MODEL_VISION is not set. Images are read by the main model if it has vision; if you will send images and the model is not one Synap knows, set a vision model here. Ignore if you will not send images.")
    finish()


def finish() -> None:
    errors = [l for l in lines if l.startswith("[ERROR]")]
    if errors:
        print(f"RESULT: FAIL {errors[0][8:].split('. ')[0]}")
        print("EXIT CODE: 1")
        sys.exit(1)
    print("RESULT: PASS")
    print("EXIT CODE: 0")
    sys.exit(0)


if __name__ == "__main__":
    main()
