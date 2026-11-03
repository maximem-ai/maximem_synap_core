# Benchmarks

Every number in the root [README](../README.md) comes from a run of the open
[eval harness](https://github.com/maximem-ai/memory_and_context_eval_harness)
against a Synap Core server. The run, its prompts and its per-question records are in
[`results/longmemeval/`](results/longmemeval/). The category breakdown is in [RESULTS.md](RESULTS.md)
and the exact settings are in [METHODOLOGY.md](METHODOLOGY.md).

| Benchmark | Synap Core | Run |
|---|---|---|
| LongMemEval, all 500 questions | 83.6% (418 / 500) | 2 Nov 2026 |
| LongMemEval, answerable questions | 83.0% (390 / 470) | 2 Nov 2026 |

## Settings used for the published result

- LongMemEval: `LongMemEval_S`, all 500 questions. Answer model `gemini-3.8-flash`, judge `gpt-4o` (binary yes / no using LongMemEval's per-question-type judge prompts)
- Synap Core: extraction with `gemini-3.1-pro-preview` through the `openai-compatible` provider, `accurate` retrieval, 50 results, sample config as shipped
- Answering prompt: [`results/longmemeval/LME_answering_prompt.md`](results/longmemeval/LME_answering_prompt.md)

## Check the score without a model call

```bash
python3 - <<'EOF'
import json
recs = json.load(open("benchmarks/results/longmemeval/question_records.json"))
ok = lambda r: not r["judge"]["error"] and r["judge"]["judge_raw"].strip().lower().startswith("yes")
print(sum(map(ok, recs)), "/", len(recs))   # 418 / 500
EOF
```

## Re-run it

There is no one-line bench command. You run the harness and point its Synap provider at your local server.

```bash
git clone https://github.com/maximem-ai/memory_and_context_eval_harness.git
cd memory_and_context_eval_harness
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # set the answer-model and judge keys
python scripts/download_datasets.py --variant s
python -m runner.server  # http://localhost:8766
```

Start Synap Core first (see the root README), set `SYNAP_BASE_URL=http://127.0.0.1:8000`, leave the Synap API key empty, choose `gemini-3.8-flash` as the answer model and `gpt-4o` as the judge, and run LongMemEval. Expect a score within normal run-to-run variance, not an exact match. See [METHODOLOGY.md](METHODOLOGY.md).
