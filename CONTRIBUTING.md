# Contributing to Synap Core

Thanks for your interest. Please read the next section before writing any code, because how this repository accepts changes is probably not what you expect.

## How this repository works

The engine under `synap/` is maintained by the Maximem team. Pull requests are welcome for everything around it: `deploy/`, `config/`, `e2e/`, `skills/`, `Dockerfile`, `requirements.txt`, `.env.example`, `.github/`, and the files at the root (README, LICENSE, NOTICE, SECURITY.md and this file).

> **Open an issue before changing `synap/`.** When a maintainer labels it `approved`, a pull request for that issue is welcome. A pull request that changes `synap/` without an approved issue is closed.

So instead:

| You want to | Do this |
|---|---|
| Report a bug, or a gap in the engine | [Open an issue](https://github.com/maximem-ai/maximem_synap_core/issues/new/choose). A failing test or a diff attached to the issue helps |
| Ask how something works, or suggest an idea | [Discussions](https://github.com/maximem-ai/maximem_synap_core/discussions), and the [docs](https://docs.maximem.ai/core/quickstart) |
| Report a security problem | Privately, see [SECURITY.md](SECURITY.md) |
| Improve the setup (`deploy/`, `config/`, `e2e/`, `skills/`, the Dockerfile, the docs files) | Open a pull request |
| Change the engine in `synap/` | Open an issue first. Once it is labelled `approved`, send a pull request for it |
| Read, run or debug the code against your own app | [Working with the code locally](#working-with-the-code-locally) below |

Accepted issues are implemented by the maintainers, or by the contributor an approved issue is assigned to, and ship in the next release of the published image.

**Every pull request runs a check** ([`.github/workflows/check.yml`](.github/workflows/check.yml)), with no secret and no model key, so it also runs for a pull request from a fork. It checks that the Compose file is valid (every service caps its log, every store has a healthcheck and the server waits for it, no store publishes a port), runs the setup skill's tests, and builds the image. It does **not** run the end-to-end suite, which needs a model key: run it yourself for a change to `deploy/`, `config/`, the `Dockerfile` or `.env.example`, as described below. The unit tests under `synap/tests` are run by the maintainers.

## Reporting an issue

A good issue includes:

- The commit or release you run, and whether you use `docker compose` from this repository, the published image, or your own build
- Your operating system and Docker version
- Your model provider and model name (never the key)
- The SDK and its version, if the problem is in a call
- What you expected, what happened, and the exact steps
- The output of `docker compose -f deploy/docker-compose.yml logs synap`, with anything private removed

Please do not paste API keys, tokens, or memories that hold personal data into an issue.

## Working with the code locally

### Repository layout

```
deploy/                   docker-compose.yml: the server and the four stores
config/                   memory.yml (the starter) and samples/ (the business-app config)
e2e/                      the end-to-end suite
skills/synap-core/        the setup skill for coding agents
synap/
  cloud/                  the engine: API, ingestion, retrieval, compaction, the memory browser
  providers/              vector, graph, storage and LLM adapters
  contracts/              the wire contract
  db/                     the baseline schema and its runner
  boot/                   the first-start download of the models
  tests/                  the unit tests that ship
Dockerfile  requirements.txt  .env.example
```

### Run it

Follow the [quickstart](https://docs.maximem.ai/core/quickstart), or the three options in the README. To run your edits to the engine, build from source: `docker compose -f deploy/docker-compose.yml up -d --build`. Read the server log with `docker compose -f deploy/docker-compose.yml logs -f synap`, and look at what it stored in the memory browser at `http://127.0.0.1:8000/browser/`.

### Running the tests

The end-to-end suite starts its own copy of the stack under its own Compose project and checks the install through the real SDK. It takes about two minutes once the models are downloaded, and it removes only what it created:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r e2e/requirements.txt
cd e2e && pytest
```

The setup skill's own tests need no Docker, no server and no model: `pip install pytest && python -m pytest skills/synap-core/tests`.

It needs Docker, port 8000 free, and a model in your `.env`. See [e2e/README.md](e2e/README.md). Run it before you open a pull request that touches `deploy/`, `config/`, the `Dockerfile` or `.env.example`.

### Style

- Shell and Compose changes: keep every command copy-pasteable from a fresh clone, and say what a command deletes before it does.
- Python (the e2e suite and the skill's scripts): the standard library where you can, type hints on public functions, and no secret ever printed.
- Docs and READMEs: plain words, SDK examples rather than raw HTTP calls, no marketing claims you cannot reproduce.

## Working against Synap Core

If you write code on top of Synap Core, these are the rules the first-party integrations follow.

**Use only the public SDK.** `pip install maximem-synap` or `npm install @maximem/synap-js-sdk`, with `SYNAP_BASE_URL=http://127.0.0.1:8000` and no API key. Internal modules change without notice.

**Call it from your backend.** Synap Core has no login by default; keep it off the public network and set `SYNAP_API_TOKEN` before you expose it ([SECURITY.md](SECURITY.md)).

**Never crash the host application.** Wrap SDK calls, log failures, and degrade to no-memory behaviour rather than raising into user code.

**Cloud-only features are out of scope here.** Credits, delete by scope, the live stream and the dashboard answer "not available in the open-source server" on purpose.

Framework integrations are developed in the [SDK repository](https://github.com/maximem-ai/maximem_synap_sdk); see its CONTRIBUTING for how to propose one.

## Sign your commits

We use the [Developer Certificate of Origin](https://developercertificate.org/), not a CLA. Add a sign-off line to each commit:

```bash
git commit -s -m "Describe the change"
```

This adds `Signed-off-by: Your Name <you@example.com>` and states that you have the right to submit the change under the Apache License 2.0. The email must be the one on the commit. A check on every pull request fails when a commit has no matching sign-off; to fix it, run `git commit --amend -s` for the last commit, or `git rebase --signoff main` for several, then push again.

## Pull request checklist

- The change is small, does one thing, and the description says why.
- A change to `synap/` fixes an issue labelled `approved`.
- Every commit is signed off.
- No secrets, keys, personal data or internal addresses anywhere in the diff.
- If it changes `deploy/`, `config/` or `.env.example`, the end-to-end suite passes and the README still tells the truth.

## Code of conduct

Be respectful and constructive. We are building something useful together. The full text is in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Questions

[Documentation](https://docs.maximem.ai) · [Synap Cloud](https://synap.maximem.ai) · [Issues](https://github.com/maximem-ai/maximem_synap_core/issues) · [Discussions](https://github.com/maximem-ai/maximem_synap_core/discussions)
