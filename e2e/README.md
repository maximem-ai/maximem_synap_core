# End-to-end tests

Checks that a fresh Synap Core install works, through the real SDK and the real Docker stack. Not a
benchmark: it asks whether the thing you cloned starts, stores, recalls, keeps users apart, survives a
restart and refuses callers without a token.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r e2e/requirements.txt
cp .env.example .env        # fill in SYNAP_LLM_PROVIDER, SYNAP_LLM_MODEL and the key line, as in the quickstart
cd e2e && pytest
```

Needs Docker, port 8000 free, and a model: the one in your `.env` is called a few dozen times. A run takes
about 2 minutes when the models are already downloaded, and about 15 more for the 2.3 GB first download.

The suite copies the tracked files of this repository to a temporary folder, writes a `.env` there from
yours (dropping any Jev, token and Synap Cloud key), and starts its own Compose project, so it never touches
your `.env`, `config/`, `originals/` or another Core stack. It removes only its own project at the end.

| Variable | Use |
|---|---|
| `E2E_ENV_FILE` | a filled-in env file other than `./.env` |
| `E2E_SEED_MODELS_VOLUME` | a Docker volume that already holds the models (for example `synap-core_models`), to skip the download |
| `E2E_KEEP=1` | leave the stack and the temporary folder in place, to look at them |

## What it checks

| # | Check |
|---|---|
| 1 | The server is up and says it is Core |
| 2 | The setup containers (secrets, schema, first start) exit 0 |
| 3 | Only the API port is published, and only on 127.0.0.1 |
| 4 | Three distinct generated passwords, in no file of the repository |
| 5 | The missing-token warning is logged, and no ERROR line |
| 6 | The SDK connects with no key; Core reports its scoping mode |
| 7 | A memory is stored and recalled |
| 8 | One user cannot see another user's memories |
| 9 | A request that does not match the scoping mode fails loudly |
| 10 | Text that is too short is skipped, and the status says why |
| 11 | An uploaded file is read |
| 12 | A conversation is recorded and compacted |
| 13 | One memory can be deleted by id |
| 14 | Forget answers "nothing matches" for an unknown user and "needs a Jev key" without one |
| 15 | Cloud-only calls say "not available in the open-source server" |
| 16 | The MACA plugin without a Synap Cloud key says it is not configured |
| 17 | The memory browser shows memories and refuses every write |
| 18 | Data and passwords survive `down` and `up` |
| 19 | With `SYNAP_API_TOKEN` set, callers without it are refused and the SDK with it works |

Not covered: Linux and Windows (run it there), the first-start download (use no seed volume), and a
business-app (`strict`) install.
