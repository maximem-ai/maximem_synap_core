# Troubleshooting

> Commands are for the clone path. Add `SYNAP_USER=...` (Linux) and the two `SYNAP_IMAGE` variables (image path) in front of each `docker compose` command (`SKILL.md` Conventions).

First, always: `docker compose -f deploy/docker-compose.yml ps -a` and `... logs <service>` for the service that is not healthy (`first-start`, `schema`, `synap`). Strip colour with `sed -E 's/\x1b\[[0-9;]*m//g'`. Never paste a log that contains a key; the server does not log keys, but check before you quote.

| Symptom | Cause | Fix |
|---|---|---|
| `first-start` exits 1 and the log names `SYNAP_LLM_PROVIDER` or `SYNAP_LLM_MODEL` | a placeholder (`add-your-...`) is still in `.env` | `python3 scripts/check_env.py`, fix the lines, `docker compose ... up -d` |
| `first-start` exits 1 at some percent of `model.safetensors`; "Name or service not known", connection reset | the download dropped: network, VPN, or the Mac went to sleep | run `docker compose ... up -d` again; it continues from where it stopped. Keep the machine awake |
| `up -d` prints almost nothing for many minutes | normal on the first start: it waits for `first-start` to finish the 2.3 GiB download | `python3 scripts/wait_healthy.py` or `logs -f first-start` |
| `/health` answers 503 `not_ready` | models not downloaded yet | wait for `first-start` to exit 0 |
| `bind: address already in use` on port 8000 | another stack (often an earlier Synap Core) or another program | `docker ps --filter publish=8000`; `docker compose stop` that stack. To keep both, set another port in your shell before `up -d` (`export SYNAP_PORT=8010`) and use it in `SYNAP_BASE_URL`. Do not change the published address in the Compose file |
| `synap` restarts; the log says `PermissionError: the folder /app/originals is not writable` | `originals/` is owned by root (Docker created it) and the server runs as another user. Linux, or an old checkout | `git pull` (the `secrets` step now fixes ownership); on Linux `export SYNAP_USER=$(id -u):$(id -g)` and `up -d`. Never `chmod 777` |
| `synap` or `schema` fails: `password authentication failed` | the `state` volume (passwords) was deleted while the data volumes were kept, or the reverse | the volumes must go together. Ask the human; if the data is expendable, remove all five data volumes (`uninstall-and-reset.md`) and start again |
| `pull access denied for synap-core` | the build did not run: Compose was run from the wrong folder, or without `-f deploy/docker-compose.yml`, or on the image path without `SYNAP_PULL_POLICY=always` | run from the repository root with the exact command in `start-and-verify.md` |
| `unauthorized` pulling `ghcr.io/maximem-ai/maximem_synap_core` | the package is still private | use the clone path, or `docker login ghcr.io` with a token that has `read:packages` |
| container killed, exit code 137 | Docker ran out of memory | give Docker more (Settings, Resources), at least 6 GiB; close other heavy containers |
| the server refuses to start after a config edit: "already holds memories stored under ..." | the scoping mode line was changed after memories exist | put `user_context_isolation` back to the original value (`scoping-mode.md`) |
| the server refuses to start after a config edit, naming a line | a mistake or a Cloud-only setting in `config/memory.yml` | fix the named line; `config/memory.yml.bak` or the starter is the way back |
| `customer_id not accepted on b2c` / "customer" in an error | a customer id on an `equals_customer` install | send a user id only, or choose the business mode before storing anything |
| a `strict` install errors on a call with only a user id | a customer id is required | add `customer_id` to every call |
| ingestion `completed` with `memories_created` 0 and "the text is shorter than 100 characters" | `document_type="document"` is read from 100 characters (`email` 50) | send short text as `ai-chat-conversation`, or send more text |
| ingestion stays queued/in progress for minutes | the model is slow, or the queue is behind; first memory after a start can take a minute | `logs -f synap`; check the provider key and quota; an error in the log from the provider (401, 404 model not found, 429) names the cause |
| provider errors: 401, 403 | wrong or expired key, or the key is in the wrong line for the provider | `python3 scripts/check_env.py`; fix with `set_env.py` |
| provider error: 404 / "model not found" | the model name is not exactly as the provider names it, or the base URL is wrong | fix `SYNAP_LLM_MODEL` / `SYNAP_LLM_BASE_URL` |
| provider error: 429 | rate limit or quota | wait, raise the quota, or choose another model |
| the log warns "Compaction model ... is not in the model window table" | the model is not a well-known hosted one | set `SYNAP_LLM_CONTEXT_WINDOW` to the model's real window |
| the log warns "No vision model is configured" | images cannot be read by the main model, as far as Synap can tell | only if images matter: set `SYNAP_LLM_MODEL_VISION` |
| SDK: "Connection closed by peer" right after a restart | the server is still loading | wait for healthy, retry |
| SDK: `ServiceUnavailableError ... needs a Jev API key` (after three retries) | forget by description without `JEV_API_KEY` | set the key (`production.md`), or delete the memory by id |
| SDK returns an old answer after a delete or update | the SDK's five-minute cache | `sdk.cache.clear()` |
| SDK: "not available in the open-source server" | a Cloud-only call (credits, delete by scope, live stream) | that is by design: `mental-model.md` |
| `401` from every call, and `/browser` asks for a token | `SYNAP_API_TOKEN` is set on the server | export the same value in the SDK's environment |
| Ollama: connection refused (Linux) | Ollama listens on 127.0.0.1 only | `model-setup.md`, the Linux step |
| MACA plugin errors | see the table in `config-and-maca.md` | |
| nothing at all works on Windows | Docker is running in a Windows-path project | use WSL2 and keep the project in the WSL filesystem |

## When it is none of these

Collect, in this order: the output of `python3 scripts/preflight.py`, `docker compose ... ps -a`, the last 60 log lines of the failing service (check for secrets), the commit of the repository (`git rev-parse --short HEAD`) or the image tag, and the OS and Docker version. Say exactly what you ran and what happened, then stop and tell the human; the issue form at https://github.com/maximem-ai/maximem_synap_core/issues asks for the same things. Do not try destructive fixes (`down -v`, deleting volumes, `chmod 777`) to make an error go away.
