# Synap Core setup skill

This is the skill that teaches a coding agent to set up **Synap Core** end to end: check the machine, pick the install path, choose the model and the scoping mode, write `.env` without ever exposing a key, start the Docker stack, learn about your agent and write its `use-case.md`, optionally have Synap design a memory config for it with the MACA plugin, and verify a real write and recall. It is the Synap Core counterpart of the [`synap` skill](https://github.com/maximem-ai/maximem_synap_sdk/tree/main/skills/synap), which does the same for the hosted Synap Cloud.

## Copy this into your coding agent

Claude Code, Codex, Cursor, Windsurf, Cline, Aider: paste it into an empty folder's session.

> Set up Synap Core on this machine. Install the Synap Core skill (`npx skills add maximem-ai/maximem_synap_core --skill synap-core -y`, or read https://github.com/maximem-ai/maximem_synap_core/tree/main/skills/synap-core) and follow it end to end. Ask me before anything that takes a long time or deletes data, and never ask me to paste an API key into this chat: I will put it in `.env` myself.

The agent then runs the checks, asks you a handful of questions (each with a recommendation), waits while you add your model key, starts the stack, and shows you the verification output.

## Install the skill yourself

| Agent | How |
|---|---|
| Claude Code | `git clone --depth 1 https://github.com/maximem-ai/maximem_synap_core /tmp/synap-core && cp -r /tmp/synap-core/skills/synap-core ~/.claude/skills/synap-core` |
| Codex, Cursor, Aider, Cline, Continue, Windsurf, Zed | `npx skills add maximem-ai/maximem_synap_core --skill synap-core -y` (the `skills` tool chooses the folder, usually `.agents/skills/`), or copy the single file [`AGENTS.md`](AGENTS.md) alone into your project as `AGENTS.md`, `.cursor/rules/synap-core.mdc` or `.clinerules` |
| Any agent, no install | tell it to read `SKILL.md` in this folder |

## What is in here

```
skills/synap-core/
├── SKILL.md                     # entry point: triggers, the 11-step procedure, hard rules
├── AGENTS.md                    # self-sufficient single-file edition
├── README.md                    # this file
├── reference/                   # the detail, read as needed
│   ├── mental-model.md          # what Core is, what runs, what it does not have
│   ├── prerequisites.md         # Docker, memory, disk, platforms (Mac, Linux, WSL)
│   ├── choose-install-path.md   # clone vs published image
│   ├── model-setup.md           # providers, the .env lines, Ollama, cost
│   ├── scoping-mode.md          # consumer vs business, final after the first memory
│   ├── start-and-verify.md      # start, wait, verify, commands to keep
│   ├── sdk-connect.md           # SYNAP_BASE_URL, no key, SDK surprises
│   ├── use-case-questions.md    # what to ask about the agent, and how the answers become use-case.md
│   ├── config-and-maca.md       # the starter config; generating one with MACA
│   ├── production.md            # token, TLS, backups, updates, forget
│   ├── troubleshooting.md       # symptom, cause, fix
│   ├── ask-rules.md             # what to ask, with a recommendation and a tip
│   ├── uninstall-and-reset.md   # stop, down, down -v, what each deletes
│   ├── agent-framework-handoff.md # wiring an agent: the synap skill, and what differs on Core
│   ├── final-report-template.md
│   └── docs-map.md
├── scripts/                     # run from the Synap Core folder
│   ├── preflight.py             # machine, Docker, disk, port 8000, earlier installs, network
│   ├── set_env.py               # write .env lines; secrets only from the shell, never printed
│   ├── check_env.py             # validate .env by name and length
│   ├── wait_healthy.py          # watch the first start until healthy
│   ├── verify_core.py           # real write, recall, isolation, forget, browser read-only, cleanup
│   ├── check_use_case.py        # is use-case.md complete, and free of keys and personal data
│   ├── generate_config.py       # the MACA plugin: use-case file in, memory config out
│   └── check_config.py          # is the active config accepted and really in force
└── examples/
    ├── python-minimal.py        # store and recall against a local server
    ├── typescript-minimal.ts    # the same in TypeScript
    ├── multi-user-scoping.md    # scoping in practice
    ├── use-case.md              # a sample use-case file for the MACA plugin
    └── use-case-personal-chatbot.md # a sample use-case file for a personal chatbot
```

## What the agent does to your machine

- **It never commits.** No `git commit`, `git add -A`, `git push`, `git stash` or amend. New files stay untracked for you to review.
- **It never sees your key.** Secrets go into `.env` only through `set_env.py --from-env` (copied from your own shell) or by you typing them in; every script prints names and lengths, never values, and the agent is told never to open `.env`.
- **It does not delete your data.** `down -v`, removing volumes, deleting `originals/` and changing the scoping mode after memories exist all need your explicit yes after it lists what you would lose.
- **It touches only Synap Core's own Docker project.** No `system prune`, no other containers.
- **Every question comes with a recommendation and a tip.**

## Keeping it true

The reference files describe Core as of its first release. When the Compose file, the settings in `.env.example`, the scripts or the SDK change, change the matching file here in the same pull request. `scripts/` use only the Python standard library, except `verify_core.py`, `generate_config.py` and `check_config.py`, which need `pip install maximem-synap`.
