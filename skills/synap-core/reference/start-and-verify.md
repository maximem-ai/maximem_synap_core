# Start it, wait for it, verify it

All commands run from the Synap Core folder (the one that contains `deploy/`). In this file `scripts/x.py` means `<skill>/scripts/x.py`. **Each shell call starts fresh**, so prefixes do not carry over: put them in the same command, every time. On **Linux** `SYNAP_USER=$(id -u):$(id -g)`; on the **published-image path** `SYNAP_IMAGE=ghcr.io/maximem-ai/maximem_synap_core:<tag> SYNAP_PULL_POLICY=always`.

## Before you start

1. `python3 scripts/preflight.py` (the machine) and `python3 scripts/check_env.py` (the `.env`). Do not start with an [ERROR].
2. The scoping mode is chosen (`scoping-mode.md`) and `config/memory.yml` is the one you want.
3. Tell the human what the first start will do and how long it takes, and get a yes:
   - builds the image (clone path) or pulls it (image path): about 5 to 10 minutes on a cold cache, about a minute for a pull; in all, 10 to 25 minutes on the clone path when the models must be downloaded;
   - generates the database passwords, once;
   - downloads the embedding model (about 2.3 GiB) once; at a typical home connection this is the long part, up to 15 minutes. **Do not let the machine sleep.** A sleep or a dropped connection stops the download; run the same `up -d` again and it continues from where it stopped;
   - creates the database tables.
   If the models volume already exists from an earlier install, there is no download and the start takes about half a minute.

## Start

```bash
docker compose -f deploy/docker-compose.yml up -d > up.log 2>&1 &
python3 <skill>/scripts/wait_healthy.py > wait.out 2>&1 &     # same prefixes as the line above, if any
```

On the first run `up -d` does not return until `first-start` has finished, because the server waits for it, and it prints very little. So run both in the BACKGROUND and read the files every minute or so: `wait.out` prints a line when something changes plus the download progress, and ends with `RESULT:` (exit 0 healthy, 1 a service failed or keeps restarting, 2 timed out). Read `up.log` too: a build error or a port already in use shows up there, not in `wait.out`.

or by hand: `docker compose -f deploy/docker-compose.yml logs --tail 20 first-start` (it reports how far each large file is every ten seconds), and `docker compose -f deploy/docker-compose.yml ps`.

## Ready means

`docker compose -f deploy/docker-compose.yml ps` shows `synap` as `healthy`, and `curl http://127.0.0.1:8000/health` answers `{"status":"ok","edition":"core"}`. Until the models are in place `/health` answers 503 `not_ready`; that is not an error.

The first memory can take up to a minute while the models load into memory.

## Verify (do not skip)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install maximem-synap            # 0.6.13 or newer
python scripts/verify_core.py > verify.out 2>&1; echo "exit=$?"
```

Paste the WHOLE of `verify.out` to the human, unedited, `[WARN]` lines included (explain every WARN), ending with `RESULT:` and `EXIT CODE:`. It writes a real memory, recalls it, checks that two users stay apart, checks forget on an empty user, checks the browser is read-only, and deletes its own test memories. Never report success the script did not print. A starter-config WARN is expected on a first install.

## Open the memory browser

`http://127.0.0.1:8000/browser/`: memories, entities, requests and the active config, read-only, no login (if `SYNAP_API_TOKEN` is set it asks for the token). Tell the human about it.

## Commands to leave with the human

(On Linux each needs `SYNAP_USER=$(id -u):$(id -g)` in front, and on the image path the two `SYNAP_IMAGE` variables. Tell them, or give them a shell alias.)

```bash
docker compose -f deploy/docker-compose.yml ps            # state
docker compose -f deploy/docker-compose.yml logs -f synap # server log
docker compose -f deploy/docker-compose.yml stop          # pause, keep everything
docker compose -f deploy/docker-compose.yml up -d         # start again, or pick up a .env change
docker compose -f deploy/docker-compose.yml down          # remove containers, KEEP data
# down -v removes the data too: see uninstall-and-reset.md
```
