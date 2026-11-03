# The memory config, and generating one with the MACA plugin

> **Commands here are written for the clone path on a Mac.** Put the prefixes from `SKILL.md` Conventions in front of every `docker compose` command: `SYNAP_USER=$(id -u):$(id -g)` on Linux, and `SYNAP_IMAGE=... SYNAP_PULL_POLICY=always` on the published-image path. In this file `scripts/x.py` means `<skill>/scripts/x.py`, run from the Synap Core folder.


Synap Core is configured in two places:

| Place | What it holds | Who writes it |
|---|---|---|
| `.env` | how the server runs: the model and key, access, optional keys | the human, with `set_env.py` |
| `config/memory.yml` | how memory behaves: memory types, extraction guidance, graph types, chunking, ranking weights, compaction thresholds, scoping, user profile | the human by hand, or the MACA plugin |

## The starter config

The repository ships a small `config/memory.yml`: it only sets the scoping mode to `equals_customer`, everything else is a built-in default. It is enough to run and test. It is NOT tuned to the human's app. The memory browser's Config page says "You are on the starter config" until the file is changed, and `verify_core.py` reports it as a WARN. Mention it, offer the next step, and do not insist.

- No file: the server starts on built-in defaults whose scoping mode is `strict`. Do not delete the starter.
- A file with a mistake: the server refuses to start and names the line. A change while it runs is reloaded; a bad change keeps the last good config and logs the error. Settings that only Synap Cloud reads are rejected with a message.
- Ranking weights default to relevance 0.70, importance 0.20, recency 0.10.

## Option A: write it by hand

Edit `config/memory.yml` (reload is automatic; `docker compose -f deploy/docker-compose.yml restart synap` is the safe way). Document types and short text: `ai-chat-conversation` is read from 10 characters; `document` from 100 and `email` from 50; shorter text is skipped and the ingestion status says why.

## Option B: have MACA write it (the MACA plugin)

The human describes the use case in a short markdown file. The Core server sends it to Synap Cloud, which designs a config built for it and returns it. Core saves it next to the human's own and never overwrites `memory.yml`. It needs a **Synap Cloud account with credits** and takes **15 to 20 minutes**. A run creates an instance named `maca-plugin-...` in their Cloud account, tagged *Synap Core*. It counts toward the instance limit and stays after the config is saved: they can download the Core config again from its page in the dashboard, move it to Synap Cloud, or delete it (deleting needs their explicit yes). Only the description file leaves the machine: no memories, no conversations, no user ids.

In the setup flow this is step 8. Step 7 already wrote `use-case.md` from the human's answers (`use-case-questions.md`), so the file exists before you start here.

Ask first (`ask-rules.md`): does the human have a Synap Cloud account with credits, and do they want this now? They can create an account or add credits at https://synap.maximem.ai (the login code in step 1 below is approved on that same site). The finished config comes straight back to the server through the plugin: there is no dashboard step. If they do not want it now, stay on the starter config and say how to come back to it.

### 1. Mint a management key (the human approves in a browser)

The plugin does not use an instance API key from the dashboard; it needs a **management key**.

```bash
npx @maximem/synap-cli@latest auth login --profile core-config     # needs Node 16+ and Python 3.10+
```

The first run installs the `maximem-synap` command (under `~/.local/bin`, which may not be on PATH); if it is not found afterwards, use `npx @maximem/synap-cli@latest` in its place everywhere below.

It prints a short code and a link. **The human opens the link, checks the code matches, picks their client and clicks Approve.** The code expires in a few minutes: tell them before you run it, and rerun if it expires. Then `maximem-synap whoami --profile core-config` should list the scope `management:*` (or `*`).

This key can manage the whole Cloud account, not only the plugin. Treat it as the most sensitive thing in the session: it must never be printed, pasted into chat, or committed.

### 2. Give the server the key without showing it

```bash
SYNAP_CLOUD_KEY=$(env -u SYNAP_API_KEY maximem-synap auth print-access-token --profile core-config) \
  python3 scripts/set_env.py --from-env SYNAP_CLOUD_API_KEY=SYNAP_CLOUD_KEY
docker compose -f deploy/docker-compose.yml up -d
```

`env -u SYNAP_API_KEY` matters: if that variable is set in the shell (an instance key from another project), the CLI prints IT instead of the management key, and the wrong key would be written. Check first that `maximem-synap whoami --profile core-config` shows the scope `management:*` or `*`.

The key stays inside that one command. The server sends it to Synap Cloud and nowhere else, and never returns it or logs it.

### 3. Write the use-case file

This is step 7 of the setup flow. If `use-case.md` is not in `<folder>` yet, ask the questions in `use-case-questions.md` first, then run `python3 scripts/check_use_case.py`. The Synap use-case template is at https://docs.maximem.ai/getting-started/use-case, with finished examples in `examples/use-case.md` and `examples/use-case-personal-chatbot.md` here. Sections: Agent Objective, Target Users, Interaction Pattern, Task Examples, Behavioral Guidelines, Role Descriptions, Compliance and Data Sensitivity, Memory Priorities, Additional Context. **Describe only what the human told you; write `not stated` for gaps, never invent.** Show the exact file in full and get approval before sending it. Maximum 256 KB, UTF-8.

### 4. Run it

```bash
pip install maximem-synap            # 0.6.13 or newer has sdk.maca
python3 scripts/generate_config.py --use-case use-case.md --relationship b2c
```

`--relationship` must match the install: `b2c` for `equals_customer`, `b2b` for `strict`. Once memories exist the server refuses the other one ("This install already holds memories stored for ..."). Run it in the BACKGROUND with its output in a file, `python3 <skill>/scripts/generate_config.py --use-case use-case.md --relationship b2c > maca.out 2>&1 &`, and read `maca.out` every minute or two until it ends with `RESULT:`. It prints a line each time the job's state changes. A job survives a server restart.

The job ends as `succeeded` (saved as `config/memory.generated.yml`), `invalid` (saved as `config/memory.generated.invalid.yml` with the problems listed; fix by hand or run again) or `failed` (the message says why).

### 5. Review and switch

```bash
head -5 config/memory.generated.yml                          # the header lists the fields Core left out
grep -n user_context_isolation config/memory.generated.yml   # must match the install's mode
cp config/memory.yml config/memory.yml.bak
mv config/memory.generated.yml config/memory.yml
docker compose -f deploy/docker-compose.yml restart synap
python3 scripts/check_config.py
```

Show the human the diff before you switch. Then **clean up the key**: `python3 scripts/set_env.py --unset SYNAP_CLOUD_API_KEY`, `docker compose -f deploy/docker-compose.yml up -d`, and `maximem-synap auth logout --profile core-config` (revokes the key). Say the key is revoked only if `auth logout` exited 0 and reported it.

### Failures

| What you see | Meaning |
|---|---|
| "The MACA plugin is not configured" | `SYNAP_CLOUD_API_KEY` is empty, or the server was not restarted after setting it |
| "Synap Cloud did not accept the key" | expired or revoked key, or an instance API key instead of a management key |
| "could not start the run, usually because the account has too few credits" | add credits, then try again |
| "is not accepting a run for this key right now" | the daily limit, or a free trial that used its credits |
| "Synap Cloud answered 409" | another run is in progress on the account |
| "Could not reach Synap Cloud" | a network error; Core keeps working |
| "This install already holds memories stored for ..." | `--relationship` disagrees with the install's scoping mode |
| `invalid` | read `job.problems` / the `.invalid.yml` file |
