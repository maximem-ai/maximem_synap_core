# Choose the install path

Ask the human which one, with your recommendation (see `ask-rules.md`). Both end in the same running stack.

| | Clone the repo | Published Docker image |
|---|---|---|
| What you get | the full source, built on this machine | a prebuilt image, pulled |
| Time to a running server | the image build adds about 5 to 10 minutes on a cold cache | about a minute to pull |
| Files you need | the whole repository | three files: `deploy/docker-compose.yml`, `config/memory.yml`, `.env.example` |
| Update | `git pull`, then `docker compose ... up -d --build` | `docker compose ... pull`, then `up -d` |
| Best for | someone who wants to read or change the code | someone who just wants it running |

**Recommend** the clone path unless the human says they only want it running, or you are told the image is public. The published image lives at `ghcr.io/maximem-ai/maximem_synap_core` (tags `latest` and release tags such as `0.1.0`). Until the maintainers make that package public, pulling it needs a registry login (`docker login ghcr.io` with a token that has `read:packages`); if a pull answers `unauthorized`, use the clone path and say why.

## The clone path

```bash
git clone https://github.com/maximem-ai/maximem_synap_core <folder>
cd <folder>
cp -n .env.example .env && chmod 600 .env      # -n: never replace an existing .env
```

Then follow `model-setup.md`, `scoping-mode.md` and `start-and-verify.md`. Compose commands are always `docker compose -f deploy/docker-compose.yml ...` from the repository root.

## The published-image path

```bash
mkdir -p <folder>/deploy <folder>/config && cd <folder>
BASE=https://raw.githubusercontent.com/maximem-ai/maximem_synap_core/main
# download a file ONLY if it is missing: a kept install already has its own config and settings
[ -f deploy/docker-compose.yml ] || curl -fsSL -o deploy/docker-compose.yml $BASE/deploy/docker-compose.yml
[ -f config/memory.yml ]         || curl -fsSL -o config/memory.yml        $BASE/config/memory.yml
[ -f .env.example ]              || curl -fsSL -o .env.example             $BASE/.env.example
cp -n .env.example .env && chmod 600 .env
# a business app (strict) also needs the sample, copied over config/memory.yml BEFORE the first start:
#   mkdir -p config/samples && curl -fsSL -o config/samples/business-app.yml $BASE/config/samples/business-app.yml
```

Every Compose command for this folder needs the two variables, in the same command:

```bash
SYNAP_IMAGE=ghcr.io/maximem-ai/maximem_synap_core:latest SYNAP_PULL_POLICY=always \
  docker compose -f deploy/docker-compose.yml up -d
```

Without `SYNAP_PULL_POLICY=always` Compose looks for a Dockerfile to build from and finds none. Compose reads these two from the shell, not from `.env`. To stay on one release, name its tag (`:0.1.0`) instead of `latest`. `config/memory.yml` is required: without it the server falls back to defaults whose scoping mode is `strict`, and the first call with only a user id fails.

## Choosing the folder

Ask where it should live, and recommend a folder of its own outside any project repository (for example `~/synap-core`), so a `git add -A` in the human's project never picks up `.env` or `originals/`. The repository's `.gitignore` already ignores `.env`, `originals/` and the generated config files.
