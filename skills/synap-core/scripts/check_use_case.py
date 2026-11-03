#!/usr/bin/env python3
"""Check use-case.md before it is shown to the human or sent to Synap Cloud. Reads one file, changes nothing.

It answers: does the file have what MACA needs (the three required sections, real examples), and is there
anything in it that must not leave the machine (a key, a token, a card number)?

    python3 scripts/check_use_case.py                    # ./use-case.md
    python3 scripts/check_use_case.py --file path/to/use-case.md

Every line starts with [OK], [WARN] or [ERROR]; the output ends with `RESULT:` and `EXIT CODE:`.
Exit 0 = nothing blocks sending it (WARN lines are explained, not dropped). 1 = fix the [ERROR] lines first.
Standard library only. A secret is never printed: only the line number and what kind it looks like.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

MAX_BYTES = 256 * 1024
SECTIONS = [
    "Agent Objective", "Target Users", "Interaction Pattern", "Task Examples", "Behavioral Guidelines",
    "Role Descriptions", "Compliance and Data Sensitivity", "Memory Priorities", "Additional Context",
]
REQUIRED = ("Agent Objective", "Target Users", "Task Examples")
SECRETS = [
    ("an OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("an Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}")),
    ("an AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("a Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("a bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("a private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a key, token or password assignment", re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)[A-Za-z_]*\s*[:=]\s*[\"']?[^\s\"']{12,}")),
]
PERSONAL = [
    ("an email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("a long digit run (a card or phone number?)", re.compile(r"(?<![\w.])\d(?:[ -]?\d){11,18}(?![\w.])")),
]
PLACEHOLDER = re.compile(r"(?i)(\bTODO\b|\bTBD\b|\blorem ipsum\b|<[^>\n]{2,40}>|\[your [^\]]*\]|\{\{[^}]*\}\})")
lines_out: list[str] = []


def say(level: str, text: str) -> None:
    line = f"[{level}] {text}"
    lines_out.append(line)
    print(line)


def finish() -> None:
    errors = [l for l in lines_out if l.startswith("[ERROR]")]
    if errors:
        print(f"RESULT: FAIL {errors[0][8:].split('. ')[0]}")
        print("EXIT CODE: 1")
        sys.exit(1)
    print("RESULT: PASS")
    print("EXIT CODE: 0")
    sys.exit(0)


def split_sections(text: str) -> dict[str, str]:
    """Section name -> body, matching headings loosely (case, '&' for 'and', trailing words)."""
    found: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.+?)\s*$", raw)
        if m:
            title = re.sub(r"\s+", " ", m.group(1).replace("&", "and")).strip().lower()
            current = next((s for s in SECTIONS if title == s.lower() or title.startswith(s.lower())), None)
            if current:
                found.setdefault(current, [])
            continue
        if current:
            found[current].append(raw)
    return {k: "\n".join(v).strip() for k, v in found.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="use-case.md")
    path = os.path.abspath(ap.parse_args().file)
    if not os.path.isfile(path):
        say("ERROR", f"{path} does not exist. Write it from the human's answers first (reference/use-case-questions.md).")
        finish()
    data = open(path, "rb").read()
    if not data.strip():
        say("ERROR", "the file is empty.")
        finish()
    if len(data) > MAX_BYTES:
        say("ERROR", f"the file is {len(data) // 1024} KB; the limit is {MAX_BYTES // 1024} KB. Shorten it.")
    else:
        say("OK", f"{len(data)} bytes, under the {MAX_BYTES // 1024} KB limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        say("ERROR", "the file is not valid UTF-8. Save it as UTF-8.")
        finish()
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        say("ERROR", "the file has control characters in it. Retype the affected lines.")

    secret_lines = []
    for n, line in enumerate(text.splitlines(), 1):
        for kind, pat in SECRETS:
            if pat.search(line):
                secret_lines.append((n, kind))
                break
    for n, kind in secret_lines:
        say("ERROR", f"line {n} looks like {kind}. Remove it: this file is sent to Synap Cloud and a key must never be in it.")
    if not secret_lines:
        say("OK", "no key, token or password found")

    for n, line in enumerate(text.splitlines(), 1):
        for kind, pat in PERSONAL:
            if pat.search(line):
                say("WARN", f"line {n} has {kind}. Use made-up people in the examples; real customer data must not be in this file.")
                break

    sections = split_sections(text)
    for name in SECTIONS:
        body = sections.get(name)
        required = name in REQUIRED
        if body is None:
            say("ERROR" if required else "WARN", f"the section '{name}' is missing." + (" MACA needs it." if required else " Add it, or write `not stated` under it."))
            continue
        bare = re.sub(r"[\s`*_.\-]+", " ", body).strip().lower()
        if not bare:
            say("ERROR" if required else "WARN", f"the section '{name}' is empty." + (" MACA needs it." if required else ""))
        elif bare == "not stated":
            say("ERROR" if required else "WARN", f"'{name}' says only `not stated`." + (" MACA needs real content here: ask the human." if required else " That is fine if they had nothing to add."))
        elif required:
            say("OK", f"required section '{name}' is there ({len(body.split())} words)")
    examples = sections.get("Task Examples", "")
    n_examples = len(re.findall(r"(?mi)^\s*[-*]\s+\*\*user\*\*", examples))
    if examples and "not stated" not in examples.lower()[:30]:
        if n_examples == 0:
            say("ERROR", "Task Examples has no examples in the form `- **User**: \"...\"` then `**Agent**: ...`.")
        elif n_examples < 3:
            say("WARN", f"Task Examples has {n_examples} example(s); three to five give MACA much more to work with.")
        else:
            say("OK", f"{n_examples} task examples")

    ph = [n for n, line in enumerate(text.splitlines(), 1) if PLACEHOLDER.search(line)]
    if ph:
        say("WARN", "placeholder text on line(s) " + ", ".join(map(str, ph[:8])) + ". Replace it with the human's words or `not stated`.")
    finish()


if __name__ == "__main__":
    main()
