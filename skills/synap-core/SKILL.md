---
name: synap-core
description: >-
  Set up Synap Core, the open-source, self-hosted edition of Maximem Synap's memory engine, end to end
  on the user's own machine with Docker: check the machine, pick the install path (clone or published
  image), choose the model, choose consumer or business scoping, write .env without ever exposing a
  key, start the stack, learn about the human's agent and write its use-case file, optionally design a memory
  config for it with the MACA plugin, and verify a real write and recall. Use whenever the user says "set up Synap Core", "self-hosted Synap", "open-source Synap",
  "run Synap locally", "Synap in Docker", "install maximem_synap_core", or asks for self-hosted, local or
  open-source long-term memory for an AI agent with docker compose, or when a prompt points at
  github.com/maximem-ai/maximem_synap_core or ghcr.io/maximem-ai/maximem_synap_core. Not for the hosted
  Synap Cloud service (use the `synap` skill for that), and not for wiring memory into an agent framework
  (that skill does it, with the differences in reference/agent-framework-handoff.md).
allowed-tools: Read, Write, Edit, Bash
---

# Synap Core: end-to-end setup

Synap Core is Synap's memory engine as a self-hosted Docker stack: eight services, one Compose file, the API on `127.0.0.1:8000`, your data in Docker volumes on your disk. It uses the same SDKs as Synap Cloud with no API key. This skill takes a human from nothing to a **verified, running** server. It does not need a Synap account (only the optional MACA step does).

**Orientation first:** `reference/mental-model.md` (what runs, where data lives, what Core does not have). Scripts are in `scripts/` and use only the Python standard library, except the three that call the SDK (`verify_core.py`, `generate_config.py`, `check_config.py`).

## Conventions (read once; every command in this skill uses them)

- `<skill>` is the folder that contains this `SKILL.md`. `<folder>` is the Synap Core folder (the one that contains `deploy/`). In every reference file, `scripts/x.py` means `<skill>/scripts/x.py`.
- **Each shell call starts fresh**, so a `cd` or an `export` does not carry over. Run scripts as one command: `cd <folder> && python3 <skill>/scripts/x.py`.
- `<compose>` means `docker compose -f deploy/docker-compose.yml`, run from `<folder>`, with these prefixes **in the same command** whenever they apply: on **Linux** `SYNAP_USER=$(id -u):$(id -g)`; on the **published-image path** `SYNAP_IMAGE=ghcr.io/maximem-ai/maximem_synap_core:<tag> SYNAP_PULL_POLICY=always`. Give the human the same prefixes in the commands you leave with them.
- **Never open, read, grep or print `.env`, or the output of `<compose> config`, with any tool** (an `Edit` needs a `Read` first, which would put the key in your context). Write only through `scripts/set_env.py`; check only through `scripts/check_env.py`.
- Long commands (`<compose> up -d`, `wait_healthy.py`, `generate_config.py`) run in the BACKGROUND with output redirected to a file (`> up.log 2>&1 &`); read the file until it ends (`RESULT:` for the scripts). A foreground call that blocks for more than about ten minutes is cut off.

## Procedure

**Never guess. If anything is unclear, stop and ask the human. Every question carries your recommendation and a tip: `reference/ask-rules.md`.** Quote script output to the human unedited.

0. **Orient.** Read `reference/mental-model.md` and `reference/ask-rules.md`. Tell the human in four lines what you are about to do and roughly how long it takes: 10 to 25 minutes on the clone path when the models must be downloaded (the build is 5 to 10 and the 2.3 GiB download up to 15), a few minutes if they are already on the machine. Say that while it starts you will ask about their agent, and that an optional memory config designed for it takes about 20 minutes more.
1. **Ask where it lives and which install path** (`reference/choose-install-path.md`): clone the repository (recommended) or use the published image. Recommend a folder of its own outside any project.
2. **Preflight.** `cd <folder-or-its-parent> && python3 <skill>/scripts/preflight.py --dir <folder>` (add `--image` for the image path; the folder need not exist yet). It checks the OS, Docker and Compose, Docker memory, free disk, port 8000, an earlier Synap Core (containers and `synap-core_*` volumes that a new start would REUSE), git, Python, Node and the network. Fix every `[ERROR]`; explain every `[WARN]`. `reference/prerequisites.md`. If an earlier install exists, ask: keep or start fresh. Deleting needs an explicit yes (`reference/uninstall-and-reset.md`); a kept install keeps its scoping mode and its `.env`.
3. **Get the files, never overwriting anything that exists.** Clone: `git clone https://github.com/maximem-ai/maximem_synap_core <folder>`. Image: download the files in `choose-install-path.md` only where the file is missing. Then `cp -n .env.example .env && chmod 600 .env` (`-n`: an existing `.env` is never replaced).
4. **Choose the scoping mode** (`reference/scoping-mode.md`) unless a kept install already fixed it: consumer (`equals_customer`, the default in `config/memory.yml`) or business (`strict`: copy `config/samples/business-app.yml` over `config/memory.yml`). **Final once the first memory exists, and the verify step in 9 stores one**: decide before you verify.
5. **Choose the model and write `.env`** (`reference/model-setup.md`). Non-secret lines with `python3 scripts/set_env.py KEY=VALUE`. **The key: never ask for it in chat, never print it.** First ask whether this agent was started from the terminal where they exported the key. If yes, run `python3 scripts/set_env.py --from-env OPENAI_API_KEY` (or `--from-env SYNAP_LLM_API_KEY=GEMINI_API_KEY`). If not, they type it into `<folder>/.env` themselves, save, and tell you. Then `python3 scripts/check_env.py` must end `RESULT: PASS`. Optional now: a token (`set_env.py --generate-token`, `reference/production.md`).
6. **Say what the first start does, get a yes, start it.** It builds or pulls the image, generates the database passwords, downloads about 2.3 GiB of models once (do not let the machine sleep) and creates the schema. `<compose> up -d > up.log 2>&1 &` (with the prefixes from Conventions), then `python3 <skill>/scripts/wait_healthy.py > wait.out 2>&1 &` and read `wait.out` and `up.log` every minute or so until `wait.out` ends with `RESULT:`; a failure in `up.log` (a build error, a port in use) is yours to report. Details and what "ready" means: `reference/start-and-verify.md`.
7. **While it starts, learn about their agent and write `use-case.md`.** The first start takes ten minutes or more; use that time. Ask in three short rounds, each question with your recommendation and a tip (`reference/use-case-questions.md`): what the agent does and who uses it; what a user says and what it must remember or never store; anything else. If there is a project in sight (a README, a system prompt, a tool list), read it first and propose answers for them to correct. Write `<folder>/use-case.md` from what they said, with `not stated` for gaps and never an invented fact, never a key or real user data. Run `python3 <skill>/scripts/check_use_case.py`, show them the whole file, and get a yes before it goes anywhere.
8. **Ask: design the memory config now, or stay on the starter?** Once `wait.out` says healthy. Recommend yes if they have a Synap Cloud account with credits (create one or add credits at https://synap.maximem.ai); otherwise say the starter config works and how to do it later. The MACA plugin sends `use-case.md` to Synap Cloud and the finished config comes straight back into `config/memory.generated.yml`: no dashboard step. It takes about 20 minutes; `reference/config-and-maca.md` has every command (management key through the CLI login the human approves in a browser, `scripts/generate_config.py --use-case use-case.md --relationship b2c|b2b`, review the diff, switch, `scripts/check_config.py`, then remove and revoke the Cloud key). **Finish this before step 9**: the verify step stores a memory, and the scoping mode is final after that. Never start it without a yes.
9. **Verify.** In a virtualenv: `pip install maximem-synap`, then `python scripts/verify_core.py > verify.out 2>&1; echo "exit=$?"`. Paste the WHOLE of `verify.out` unedited, `[WARN]` lines included, ending with `RESULT:` and `EXIT CODE:`. It writes and recalls a real memory, checks that two users stay apart, and deletes its own test memories. Do not report success it did not print. On failure: `reference/troubleshooting.md`, never a destructive fix.
10. **Show the human around.** The memory browser at `http://127.0.0.1:8000/browser/` (read-only; its Config page says whether the config is the starter sample or their own), the commands to keep (`start-and-verify.md`), and `reference/sdk-connect.md` for connecting their code (`SYNAP_BASE_URL`, no key).
11. **Report.** Fill `reference/final-report-template.md`, with the verify output after it. Say what was NOT done and what waits on the human. If they want the memory wired into an agent framework, that is the `synap` skill, with the differences in `reference/agent-framework-handoff.md`.

## Hard rules

The human never has to paste a secret into chat; never guess; ask, with a recommendation and a tip; **never print, echo, open or commit a key, token or password** (use `set_env.py` and `check_env.py`; never read `.env` or `<compose> config` with any tool); **never run `git commit`, `git add -A`, `git push`, `git stash` or an amend**, leave changes uncommitted and new files untracked; **never delete a volume, run `down -v`, delete `originals/`, the Synap Core folder, `.env`, `config/memory*.yml` or `~/.synap`, or change the scoping mode after memories exist, without an explicit yes after listing what is lost**; never overwrite an existing `.env`, `config/memory.yml` or Compose file; if `git pull` conflicts, stop and ask, never stash or reset; never delete the `state` volume alone; touch only Synap Core's own Compose project and never `docker system prune` or `docker volume prune`; do not edit `synap/` (the engine) or the Compose file's ports; do not offer Cloud-only features (credits, delete by scope, the live stream, a dashboard) beyond the MACA config of step 8; never claim a result a script did not print; quote every `[WARN]` and explain it; stop and ask when a step fails in a way `troubleshooting.md` does not cover.

## Reference map

| For | File |
|---|---|
| What Core is, what it lacks | `reference/mental-model.md` |
| Machine requirements, platforms | `reference/prerequisites.md` |
| Clone vs image | `reference/choose-install-path.md` |
| Model and `.env` | `reference/model-setup.md` |
| Consumer vs business | `reference/scoping-mode.md` |
| Start, wait, verify | `reference/start-and-verify.md` |
| Connect code, SDK surprises | `reference/sdk-connect.md` |
| Memory config and MACA | `reference/config-and-maca.md` |
| Token, TLS, backups, updates | `reference/production.md` |
| Errors | `reference/troubleshooting.md` |
| What to ask | `reference/ask-rules.md` |
| Stop, reset, remove | `reference/uninstall-and-reset.md` |
| Agent frameworks | `reference/agent-framework-handoff.md` |
| The final report | `reference/final-report-template.md` |
| Other sources | `reference/docs-map.md` |
