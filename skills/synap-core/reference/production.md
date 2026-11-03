# Running it for real

> **Commands here are written for the clone path on a Mac.** Put the prefixes from `SKILL.md` Conventions in front of every `docker compose` command: `SYNAP_USER=$(id -u):$(id -g)` on Linux, and `SYNAP_IMAGE=... SYNAP_PULL_POLICY=always` on the published-image path. In this file `scripts/x.py` means `<skill>/scripts/x.py`, run from the Synap Core folder.


Synap Core is built to sit **behind the human's own backend**, never to face the internet or be called from a browser or a mobile app.

## What it assumes

- Their backend calls the server; their users' apps do not.
- **Anyone who can reach the port can read, write and delete every memory,** and ask for any user's memories. There is no login by default.
- A memory someone plants can carry instructions into the agent's prompt.
- Open ingestion runs up the model bill, because every request uses their model.

## What it already does

The API listens on `127.0.0.1:8000` only. No database port is published. The three database passwords are generated on first start, one per store, in a Docker volume. Downloaded files are checked against fixed hashes.

## What to do

1. **Keep it off the public network.** A private network or a firewall.
2. **Set a token.** `python3 scripts/set_env.py --generate-token`, then `docker compose -f deploy/docker-compose.yml up -d`. The token is written to `.env` and never printed, so you do not know it and must not read it out of the file: tell the human that it is on the line `SYNAP_API_TOKEN=` of `<folder>/.env`, and that THEY copy it into the secret manager or environment of the backend that uses the SDK (the SDK reads it from `SYNAP_API_TOKEN` in ITS environment; never the repository). The skill's scripts pick it up from `./.env` on their own when run from the Synap Core folder. Every request must carry it as a bearer token. It is one token for the whole install, not a user identity. The memory browser asks for it. Without it the server logs a loud warning at start.
3. **TLS through a reverse proxy.** The server does not terminate TLS. Put Caddy, nginx or a cloud load balancer in front, and let only the proxy reach the server. To reach it from another machine, put the proxy in front rather than changing the published address in the Compose file.
4. **Back up the data.** The volumes `synap-core_postgres-data`, `_neo4j-data`, `_chroma-data` and `_redis-data`, the `state` volume (the passwords: losing it with the data kept makes the databases unreadable) and the `originals/` folder. `docker compose down -v` deletes the volumes. A simple approach: stop the stack, archive each volume with a throwaway container, start it again. This has not been tested by the maintainers; take a copy before relying on it and test a restore.
5. **Do not call it from a browser or mobile app.** Their backend only.
6. **Keep `.env` private** (`chmod 600`, not in version control). It holds the model key.

## Updating

- Clone path: `git pull`, then `docker compose -f deploy/docker-compose.yml up -d --build`.
- Image path: `docker compose -f deploy/docker-compose.yml pull`, then `up -d`, with the two `SYNAP_IMAGE` variables.

Upgrading never touches stored memories. New releases may add later database changes, which the `schema` step applies at start. If `git pull` reports a conflict (for example in `config/memory.yml`, which the human may have changed), stop and ask: never stash, reset or overwrite their file.

## Deleting is not erasing

`memories.delete` and forget remove a memory and its search entries. They do **not** remove the copy of the original text kept in the `originals/` folder, and a conversation or graph entity written from that text can remain. Never tell a human that a person's data is erased because a memory was deleted; say what was and was not removed. Erasing a person completely also means removing their text from `originals/` and deciding what to do with the conversations and graph.


## Forget by description needs the human's own Jev key

`memories.forget(description=...)` uses Jev to decide which memories match; the text of the candidate memories is sent to Jev. Set `JEV_API_KEY` (via `set_env.py --from-env`) and restart. Without it the call answers "Forget by description needs a Jev API key ... Nothing was deleted." Deleting one memory by id needs no key. Do not set this up unless the human wants it.

## Limits and expiry (YAML, off unless written)

`guardrails.rate_limits.max_memories_per_user_per_day`, `max_retrieval_calls_per_minute`, `guardrails.retention.max_memory_age_days`, `guardrails.compliance.prohibited_memory_content`, `memory_extraction.filters.require_entity_grounding`. Each acts only when the human writes it. Configs from the MACA plugin set the retrieval limit to 60 a minute.
