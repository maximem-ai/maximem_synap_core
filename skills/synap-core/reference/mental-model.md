# What Synap Core is

Synap Core is the open-source, self-hosted edition of Synap's memory engine. It extracts memories from what users say, resolves who and what they are talking about, understands time, and finds the right context when an agent asks. It speaks the same API as Synap Cloud, so the same SDKs and framework integrations work with both. Moving to Cloud later is a new base URL and an API key.

## What runs (one Compose file, eight services)

| Service | What it is | After start |
|---|---|---|
| `synap` | the server, API on `127.0.0.1:8000` | stays up |
| `postgres` | memories and metadata | stays up |
| `chroma` | vectors (search) | stays up |
| `neo4j` | the knowledge graph | stays up |
| `redis` | the ingestion queue | stays up |
| `secrets` | generates the three database passwords once, into a Docker volume | exits 0 |
| `schema` | creates the database tables | exits 0 |
| `first-start` | downloads the embedding model (about 2.3 GiB) and a small spaCy model once, checks their hashes | exits 0 |

`secrets`, `schema` and `first-start` finish and exit. Compose shows them as exited. That is correct.

No store publishes a port. Only the API does, on loopback. The passwords are never in a file you edit.

## Where the data lives

Docker volumes named `synap-core_*` (Compose project `synap-core`): `state` (the passwords), `postgres-data`, `chroma-data`, `neo4j-data`, `redis-data`, `models` (the downloaded models). A copy of the text you send is kept in the `originals/` folder next to `deploy/`; deleting a memory does not remove that copy. `docker compose down` keeps all of it. `down -v` deletes the volumes (not `originals/`).

## What leaves the machine

1. Your model provider (or your own Ollama): every memory step uses your model.
2. Hugging Face and GitHub, once, on first start: the embedding models come from Hugging Face and the small English text model (spaCy) from GitHub. After that nothing is downloaded.
3. Synap Cloud, only if you set `SYNAP_CLOUD_API_KEY` for the MACA plugin.
4. Jev, only if you set `JEV_API_KEY`, for forget by description.

Synap Core sends no usage statistics, and the SDK sends none when pointed at a local server.

## What Core has, and what it does not

Has: memory extraction, entity resolution, temporal reasoning, vector and graph retrieval, compaction, user and customer scoping, delete by id, forget by description (needs your own Jev key), image reading (when the model has vision), a read-only memory browser at `/browser`, the MACA plugin.

Does not have, and answers "not available in the open-source server": credits and billing, delete by scope (everything for a user), the live stream (`instance.listen`) and predictive prefetch, PII management, a dashboard, user logins and multi-tenancy (it is single tenant: one fixed client `cli_core_default` and one instance `inst_core_default`; the only access control is an optional shared token, `SYNAP_API_TOKEN`).

When the human asks for a Cloud-only thing, say so plainly and point at Synap Cloud (https://synap.maximem.ai). Do not work around it.

## Words

- **Scoping mode**: how users relate to customers. `equals_customer` for consumer apps (send a user id only), `strict` for business apps (send a customer id and a user id). Chosen once; see `scoping-mode.md`.
- **Memory config**: one YAML file, `config/memory.yml`, that decides what is remembered and how. The repository ships a small starter; see `config-and-maca.md`.
- **MACA**: the Synap Cloud service that designs a memory config from a description of the use case. Core reaches it through the MACA plugin.
