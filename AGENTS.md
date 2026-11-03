# AGENTS.md

Instructions for coding agents working in this repository. Humans: see [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## What this is

Synap Core is the open-source, self-hosted edition of Maximem Synap's memory engine: a FastAPI server plus Postgres, Redis, ChromaDB and Neo4j, started by one Compose file. It exposes the same API as Synap Cloud, so the published SDKs work against it with no API key. The server runs as the Core edition (`SYNAP_EDITION=core`, set in the `Dockerfile` and `deploy/docker-compose.yml`). It has no login and is single tenant.

## Layout

- `synap/`: the engine (`cloud/` API, ingestion, retrieval, compaction and the memory browser; `providers/`; `contracts/`; `db/`; `boot/`; `tests/`). Mirrored from a private repository.
- `deploy/docker-compose.yml`: the server and its four stores. The API binds to `127.0.0.1:8000`.
- `config/`: `memory.yml` (the starter memory config) and `samples/business-app.yml`.
- `skills/synap-core/`: the setup skill for coding agents, with `reference/` and `scripts/`.
- `e2e/`: the end-to-end suite, driven through the real SDK and Docker.
- `benchmarks/`: LongMemEval results and how to re-run them.
- Root: `Dockerfile`, `requirements.txt`, `.env.example`, `originals/` (runtime copy of stored text, not source).

## Run it

```bash
cp .env.example .env && chmod 600 .env   # then set SYNAP_LLM_PROVIDER, SYNAP_LLM_MODEL and the key line
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps         # ready when `synap` is healthy
docker compose -f deploy/docker-compose.yml logs -f synap
```

The first start downloads about 2.3 GiB of models. On Linux, prefix `SYNAP_USER=$(id -u):$(id -g)`. For the published image path, see the README (option 3). The read-only memory browser is at `http://127.0.0.1:8000/browser/`.

## Tests

- End-to-end (the suite that ships and is meant to be run here):
  `python -m venv .venv && . .venv/bin/activate && pip install -r e2e/requirements.txt && cd e2e && pytest`.
  It needs Docker, port 8000 free and a model in `.env`. It starts its own Compose project and never touches your `.env`, `config/` or `originals/`. Details: [e2e/README.md](e2e/README.md).
- Run it before proposing a change to `deploy/`, `config/`, the `Dockerfile` or `.env.example`.
- The unit tests in `synap/tests/` are run in the source repository (CONTRIBUTING.md). No command to run them here is documented; do not guess one, ask the human or read `synap/tests/conftest.py`.
- There is no CI on pushes or pull requests in this repository.

## Rules

- Never commit, print, echo or open `.env` or any key, token or password. Check settings by name only. Do not paste keys into issues or pull requests.
- Do not edit `synap/`. It is overwritten on every sync from a private repository, so pull requests that change it are closed. For an engine bug or gap, open an issue instead. Pull requests are welcome for `deploy/`, `config/`, `e2e/`, `skills/`, the `Dockerfile` and the docs files.
- Sign off every commit (`git commit -s`, DCO). Keep changes small and do one thing per pull request.
- Do not run `docker compose down -v`, delete volumes, or delete `originals/` without an explicit yes from the human: that deletes stored memories.
- The scoping mode (`equals_customer` or `strict` in `config/memory.yml`) is final once the first memory exists.
- Cloud-only features (credits, delete by scope, the live stream, the dashboard) answer "not available in the open-source server" on purpose. Do not add them.

## Pointers

- Asked to set up Synap Core for a user: follow [skills/synap-core/SKILL.md](skills/synap-core/SKILL.md). The single-file rules for tools that take one file are in [skills/synap-core/AGENTS.md](skills/synap-core/AGENTS.md); do not duplicate them here.
- Asked how to use the SDK against Core: [skills/synap-core/reference/sdk-connect.md](skills/synap-core/reference/sdk-connect.md). Use only the public SDK (`maximem-synap`, `@maximem/synap-js-sdk`) with `SYNAP_BASE_URL=http://127.0.0.1:8000`.
- A leftover `SYNAP_API_KEY` in the shell makes the SDK act as a Cloud client even with `SYNAP_BASE_URL` set: run `unset SYNAP_API_KEY`.
- Framework integrations against Core need `SYNAP_AUTO_LISTEN=false`, because Core has no live stream.
- Security reports go through [SECURITY.md](SECURITY.md), never a public issue.
