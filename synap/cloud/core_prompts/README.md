# Core prompt pack

The prompts Synap Core ships, one folder per prompt. Cloud prompts are not here and never ship.

| Folder | Origin |
|---|---|
| `merged_extraction` | written new |
| `graph_extraction` | written new |
| `person_and_company` | written new |
| `knowledge_update` | existing draft, unchanged |
| `user_profile` | existing draft, unchanged |
| `compaction_v3` | written new |
| `image_description` | existing draft, unchanged |
| `forget_judge` | written new |
| `episodes_and_temporal` | existing drafts, unchanged |
| `date_guidance` | existing draft, two fields renamed, one sentence added for an undated text |
| `coalesce_merge` | existing drafts, unchanged |
| `extraction_blocks` | two short header texts written after reading the Cloud versions. Not yet reviewed |

## Conventions

- Placeholders are written `[[NAME]]`. The knowledge-update prompt keeps its draft's `{name}` fields.
- `[[IF X]]` ... `[[END IF]]` marks a block the loader keeps only when `X` is given.
- `meta.json` holds the hash of the Cloud text each folder was written against. `synap/tests/core_prompts` fails when that Cloud text changes, so the Core text gets reviewed.

## Not done here

- Not run through a model. Accuracy is tuned at the end, once Core runs.
- Wired in by the loader (`synap/cloud/shared/prompt_switch.py`) and the Core file behind each interface (`prompts/core/*.py`).
- `person_and_company/customer_graph.txt` is the company rule in the graph prompt: its paragraph takes the place of the body of the first `[[IF CUSTOMER_NAME]]` block in `graph_extraction/system.txt`, so the rule is stated once and keeps its three rules (exact name, one entity, no entity without a "we"). Neither file is edited.
