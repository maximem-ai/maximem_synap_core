# Choose and configure the model

> **Commands here are written for the clone path on a Mac.** Put the prefixes from `SKILL.md` Conventions in front of every `docker compose` command: `SYNAP_USER=$(id -u):$(id -g)` on Linux, and `SYNAP_IMAGE=... SYNAP_PULL_POLICY=always` on the published-image path. In this file `scripts/x.py` means `<skill>/scripts/x.py`, run from the Synap Core folder.


Synap Core uses **one provider and one model for every step** (reading conversations, the graph, deciding whether a new fact replaces an old one, compaction, images). It ships no default and will not start until the human chooses. The human decides; you recommend (see `ask-rules.md`).

## The lines in `.env`

```text
SYNAP_LLM_PROVIDER=<openai | anthropic | ollama | openai-compatible>
SYNAP_LLM_MODEL=<the model name, exactly as the provider names it>
```

plus the key line for that provider:

| Provider | Also set | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | |
| `anthropic` | `ANTHROPIC_API_KEY` | |
| `openai-compatible` | `SYNAP_LLM_BASE_URL` and `SYNAP_LLM_API_KEY` | any server that speaks the OpenAI chat API: Gemini, a hosted open-weights provider, vLLM, LM Studio, a company gateway |
| `ollama` | nothing (no key); `OLLAMA_BASE_URL` only if Ollama is not at the default address | see below |

**Google Gemini** goes through `openai-compatible`:

```text
SYNAP_LLM_PROVIDER=openai-compatible
SYNAP_LLM_MODEL=<a Gemini model name>
SYNAP_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
SYNAP_LLM_API_KEY=<the Gemini key>
```

Gemini 2.5 and 3 models spend part of the output budget on thinking; Synap Core accounts for that, so no extra setting is needed.

## How to write the lines without ever seeing the key

The key is the human's secret. **Never ask for it in chat and never print it.** Use the script:

```bash
# non-secret lines: write them with the script, never by editing the file yourself
python3 scripts/set_env.py SYNAP_LLM_PROVIDER=openai SYNAP_LLM_MODEL=<model>

# the key: ask first "was this agent started from the terminal where you exported it?".
# If yes (they ran  export OPENAI_API_KEY=...  in the terminal that launched you), copy it across without displaying it:
python3 scripts/set_env.py --from-env OPENAI_API_KEY
python3 scripts/set_env.py --from-env SYNAP_LLM_API_KEY=GEMINI_API_KEY   # .env name = shell variable name
```

If the answer is no, or the variable is not set (the script says so), do not retry: tell the human exactly which line to fill in and where (`<folder>/.env`, the line `OPENAI_API_KEY=`), tell them to save the file, and wait for them to say it is done. Then run `python3 scripts/check_env.py`; it shows names and lengths, never values.

## Optional lines

| Line | When |
|---|---|
| `SYNAP_LLM_CONTEXT_WINDOW` | the model is not a well-known hosted one (a local model, or a new one). Compaction sizes its calls from the window; without this it assumes 128000 and logs one warning at start. Set it to the model's real window in tokens. A value under 4000 is refused. |
| `SYNAP_LLM_MODEL_INGESTION`, `_GRAPH`, `_KNOWLEDGE_UPDATE`, `_COMPACTION` | a different, usually cheaper, model for one step. The provider stays the same. Not for a first setup. |
| `SYNAP_LLM_MODEL_VISION` | a model that can read images. If unset the main model is used; if that model cannot read images, every image read fails. The server logs "No vision model is configured" at start when it cannot tell; that is only a warning. |
| `EMBEDDING_PROVIDER` | leave as `sentence-transformers`. Embeddings run inside the server. |
| `JEV_API_KEY` | forget by description (`production.md`) |

After any change to `.env`, run `docker compose -f deploy/docker-compose.yml up -d` again so the server picks it up.

## Ollama (a model on the human's machine)

1. Install Ollama and `ollama pull <model>`. If images must be read, also pull a model with vision.
2. **Mac and Windows (Docker Desktop):** nothing to do; the container reaches Ollama at `http://host.docker.internal:11434`.
3. **Linux:** Ollama listens on `127.0.0.1` only by default, which the container cannot reach. Recommend binding it to the Docker bridge address (the `docker0` address, often `172.17.0.1`), for example `OLLAMA_HOST=172.17.0.1:11434`, which keeps it off the network; `0.0.0.0:11434` also works but exposes it to everything that can reach the machine. Changing Ollama's service environment needs administrator rights and is outside Synap Core's own project: the human makes this change, you only tell them what and why.
4. Check from the host: `curl http://localhost:11434/api/tags` lists the model.
5. `.env`: `SYNAP_LLM_PROVIDER=ollama`, `SYNAP_LLM_MODEL=<the exact name you pulled>`; optionally `OLLAMA_NUM_CTX` (default 16384) and `SYNAP_LLM_CONTEXT_WINDOW` set to the same size.

Be honest about quality: a small local model extracts and summarises less well than a large hosted one, and accuracy on local models has not been measured. A reply that is not valid JSON is skipped (a missed update, never a deleted memory).

## What it costs

Every message takes several model calls: the memory types are grouped into a few extraction calls, the graph takes its own (about four extraction calls per message with the graph on), and a separate call decides whether a new memory replaces an old one, for each pair of similar memories. Compaction and image reads are more. Embeddings cost nothing per call. Tell the human this when they choose a model, and that a config from the MACA plugin asks for smaller chunks (512 tokens), which means more calls.
