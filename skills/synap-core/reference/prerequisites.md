# Prerequisites

Check them with `python3 scripts/preflight.py`; this file explains the numbers.

| Need | Detail |
|---|---|
| Docker with Compose v2 | Docker Desktop (Mac, Windows) or Docker Engine plus the Compose plugin (Linux). Tested with Docker 29 and Compose 5. The old `docker-compose` v1 does not work. |
| Memory for Docker | The stack idles at about 4 GiB (server 1.3, Chroma 1.3, Neo4j 1.2, Postgres 0.2) and uses more while it loads models and ingests. Give Docker at least 6 GiB; 8 is comfortable. Docker Desktop: Settings, Resources. |
| Disk | About 8 GiB free: the server image about 3.2 GiB, the models about 2.5 GiB, and the stores grow with use. |
| Port | `127.0.0.1:8000` must be free. |
| A model | An API key for OpenAI, Anthropic or any OpenAI-compatible endpoint, or a model you run with Ollama. Synap Core ships no default model. |
| Network, first start only | Hugging Face for the models; Docker Hub for Postgres, Neo4j and Redis; the GitHub container registry for Chroma (and for the Synap Core image on the published-image path). |
| For the SDK | Python 3.11 or newer, or Node 20 or newer. |
| For the clone path | `git`. |

## Platforms

- **Mac (Apple Silicon and Intel):** tested. Docker Desktop maps file ownership itself, so nothing extra is needed. The first start downloads 2.3 GiB: **do not let the Mac sleep** during it, a sleep drops the download (the fix is to run `docker compose up -d` again; it continues where it stopped).
- **Linux:** put `SYNAP_USER=$(id -u):$(id -g)` in front of every `docker compose` command (an `export` does not survive between an agent's shell calls, and a later `up -d` without it recreates the server as another user), so the server runs as you and can write the `originals/` folder. Not yet tested by the maintainers: if something fails, say so and read `troubleshooting.md`.
- **Windows:** use WSL2 (Ubuntu) with Docker Desktop's WSL integration, and keep the project inside the WSL filesystem (not `/mnt/c`). Not tested natively.
- **Both CPU types:** the published image is built for `linux/amd64` and `linux/arm64`.

## Before you start: is something already there?

`preflight.py` warns if port 8000 is busy, or if containers or volumes from an earlier Synap Core exist. A new start under the same Compose project name (`synap-core`) **reuses those volumes**: old memories, old passwords, and a scoping mode that can no longer be changed. Ask the human whether to keep them or start fresh; never delete volumes without an explicit yes (`uninstall-and-reset.md`).
