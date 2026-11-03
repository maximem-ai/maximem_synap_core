# Synap Core Benchmark Results: Category Breakdown

Synap Core scores on LongMemEval, broken down by category. This is the open-source server, run from the Compose file in this repository with the sample config as shipped.
Headline: **LongMemEval 83.6%** (all 500 questions) and **LongMemEval 83.0%** (answerable questions).

## Methodology

- **LongMemEval:** answer model `gemini-3.8-flash`, judge `gpt-4o` (binary yes / no), run on 2 November 2026.
- **LongMemEval:** full 500-question set, `LongMemEval_S`.
- **Extraction:** Synap Core extracted memories with `gemini-3.1-pro-preview` through its `openai-compatible` provider. Synap Core has no default model, so this is the model we chose for the run.
- **Retrieval:** `accurate` mode, 50 results, no re-ranking or LLM-written queries.
- **Scoring:** a question counts as correct when its `judge.judge_raw` starts with "yes" (any case) and the record has no error. One question errored (see below) and counts as wrong.

---

## LongMemEval: Overall 83.6%

Full benchmark: 500 questions across 6 categories.

| Category | Synap Core | Correct / n |
|---|---|---|
| single-session-user | **97.1%** | 68 / 70 |
| single-session-assistant | 92.9% | 52 / 56 |
| knowledge-update | 91.0% | 71 / 78 |
| single-session-preference | 90.0% | 27 / 30 |
| temporal-reasoning | 78.2% | 104 / 133 |
| multi-session | 72.2% | 96 / 133 |
| **Overall** | **83.6%** | **418 / 500** |

---

## LongMemEval: Answerable questions 83.0%

30 abstention questions are excluded, which leaves 470. The 30 abstention questions alone: 28 / 30 correct (93.3%).

| Category | Synap Core | Correct / n |
|---|---|---|
| single-session-user | **96.9%** | 62 / 64 |
| single-session-assistant | 92.9% | 52 / 56 |
| knowledge-update | 90.3% | 65 / 72 |
| single-session-preference | 90.0% | 27 / 30 |
| temporal-reasoning | 77.2% | 98 / 127 |
| multi-session | 71.1% | 86 / 121 |
| **Overall** | **83.0%** | **390 / 470** |

Per-question records for this run (question, gold answer, retrieved memories, generated
answer, judge verdict) are in [`results/longmemeval/question_records.json`](results/longmemeval/question_records.json).
Run details are in [`results/longmemeval/manifest.json`](results/longmemeval/manifest.json).

**The one error.** Question `8752c811` (single-session-assistant) returned no retrieval result on four attempts and was not answered. It is in the records with its error, and it counts as wrong in every table above.

---

## Not run

- **LongMemEval retrieval (Recall@20):** not run on Synap Core.

---

### Scope

- LongMemEval question counts: multi-session 133, temporal-reasoning 133,
  knowledge-update 78, single-session-user 70, single-session-assistant 56,
  single-session-preference 30.
