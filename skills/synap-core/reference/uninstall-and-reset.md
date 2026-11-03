# Stop, uninstall and reset

All commands from the repository root. Add the two `SYNAP_IMAGE` variables on the published-image path.

| Goal | Command | Data |
|---|---|---|
| Pause | `docker compose -f deploy/docker-compose.yml stop` | kept |
| Start again, or pick up a `.env` change | `docker compose -f deploy/docker-compose.yml up -d` | kept |
| Remove the containers | `docker compose -f deploy/docker-compose.yml down` | **kept** (volumes stay) |
| Remove containers AND all data | `docker compose -f deploy/docker-compose.yml down -v` | **deleted**, including the passwords and the downloaded models (a full 2.3 GiB download follows the next start) |
| Start over but keep the models | `down`, then `docker volume rm synap-core_state synap-core_postgres-data synap-core_redis-data synap-core_chroma-data synap-core_neo4j-data` | stored memories deleted; models kept, so no re-download |
| Also remove the stored text | delete the `originals/` folder next to `deploy/` | the copy of the text that was sent |

> Commands are for the clone path. Add `SYNAP_USER=...` (Linux) and the two `SYNAP_IMAGE` variables (image path) in front of each `docker compose` command, as in `SKILL.md` Conventions.

## Rules

- **Never run `down -v`, `docker volume rm` or delete `originals/` without the human's explicit yes**, and say what will be lost first: the stored memories, the generated passwords, the scoping choice, possibly the model download.
- **Never delete the `state` volume alone.** It holds the database passwords; the databases were created with them. Without it, new passwords no longer match the old data and Postgres refuses the connection. Remove `state` together with the data volumes, or none of them.
- Only touch volumes named `synap-core_*` (or the project the human chose). Never `docker system prune`, never `docker volume prune`: they remove other projects' data.
- To run a second, throwaway stack next to the first, use a different Compose project name (`docker compose -p other -f deploy/docker-compose.yml ...`) and stop the first one: both publish port 8000.

## Fully remove Synap Core from a machine

1. `docker compose -f deploy/docker-compose.yml down -v` (after the yes).
2. `docker rmi ghcr.io/maximem-ai/maximem_synap_core:<tag>` or `docker rmi synap-core:local`.
3. Delete the folder. **This also deletes** the `originals/` folder (the copy of every text that was sent), `.env` (with the model key) and any `config/memory.yml` the human tuned or generated. Name all of that and get an explicit yes first.
4. The SDK's cache is in `~/.synap`; delete it only if the human wants it gone (another install's SDK may use it).
