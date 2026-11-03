# Methodology

This document says how the [Synap Core results](RESULTS.md) were produced. Memory benchmarks are easy to misreport: the same system can move 10 to 30 points depending on the answer model, the judge, the extraction model and the prompt. We state our choices so they can be checked.

## Benchmark

### LongMemEval ([Wu et al., ICLR 2025](https://arxiv.org/abs/2410.10813))

500 human-curated questions over long, multi-session chat histories (about 115K tokens in `LongMemEval_S`). It probes information extraction, multi-session reasoning, temporal reasoning, knowledge updates and preference recall, plus abstention. Primary metric: LLM-as-a-judge accuracy.

| Category | n |
|---|---:|
| multi-session | 133 |
| temporal-reasoning | 133 |
| knowledge-update | 78 |
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |

30 of the 500 are abstention questions (the question id ends in `_abs`).

## Our configuration

| Variable | Value |
|---|---|
| System | Synap Core, started with `docker compose up` from this repository, `config/memory.yml` as shipped (`equals_customer`) |
| SDK | `maximem-synap` 0.6.12, keyless local mode |
| Extraction model | `gemini-3.1-pro-preview` through the `openai-compatible` provider |
| Embeddings | the server's own pinned model |
| Retrieval | `accurate` mode, 50 results |
| Answer-generation model | `gemini-3.8-flash` |
| Judge model | `gpt-4o` |
| Judge protocol | Binary yes / no against the gold answer, using LongMemEval's per-question-type judge prompts (`evaluate_qa.py`), upstream commit `9e0b455f4ef0e2ab8f2e582289761153549043fc` |
| Answering prompt | [`results/longmemeval/LME_answering_prompt.md`](results/longmemeval/LME_answering_prompt.md), run with source-evidence chunks on, memory metadata off, plus a `## Current date` section holding the question date |
| Isolation | One Synap user per question haystack, 500 users |
| Run date | 2 November 2026 |
| Scope | Full 500-question set, all 6 categories |

## Scoring

A question counts as correct when its `judge.judge_raw` starts with "yes" (any case) and the record has no error. That gives 418 / 500. A question that errored counts as wrong. One question did: `8752c811` returned no retrieval result on four attempts.

The answerable view leaves out the 30 abstention questions (390 / 470).

## Comparing with other published numbers

Synap Cloud's published LongMemEval run uses the same answer model and judge as this run, but it is a different system and the settings differ in ways that move the score: Cloud's run used memory metadata on, and its extraction model and prompts are Cloud's. This run used metadata off and the Core server's plain prompts. Do not read the two numbers as one system measured twice. The Cloud run is in [eval_benchmark_runs_output](https://github.com/maximem-ai/eval_benchmark_runs_output).

Other vendors' scores depend on their own answer model, judge and scope. Compare only when those match.

## Not measured

- Recall@20 (the retrieval metric) has not been run on Synap Core.
- One run is one sample. Extraction, answer and judge models are not deterministic, so a re-run will land near this number, not on it.

## Reproducing

1. Clone the harness: [`maximem-ai/memory_and_context_eval_harness`](https://github.com/maximem-ai/memory_and_context_eval_harness).
2. Download `LongMemEval_S`.
3. Start Synap Core as in the [README](../README.md), with the extraction model above.
4. Point the harness's Synap provider at `http://127.0.0.1:8000` with no API key, ingest, then run the answer and judge passes with `gemini-3.8-flash` and `gpt-4o`.
5. Score all 500 questions.

The run is published in [`results/longmemeval/`](results/longmemeval/):

- `question_records.json`: one record per question (500), with the question, gold answer, the memories Synap retrieved, the generated answer, and the judge verdict.
- `manifest.json`: dataset variant, models, run date, prompt settings and system settings.
- `LME_answering_prompt.md`: the prompt the answer model received.
