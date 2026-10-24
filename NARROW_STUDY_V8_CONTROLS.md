# Narrow study v8: independent controls

Use `sandbagging-narrow-study-v8-controls.ipynb` in place of v7. The original notebook and v7 remain unchanged. This is a new run version; do not resume v7 outputs into it.

The original narrow design remains: honest, direct underperformance and independently prompted unlocked conditions; forced-choice scoring with rotations as primary; answer-only generation as a secondary measurement check. No reason-first or answer-first explanation module is added.

## New controls

- Instruction following: four requests to select a specified letter, once per cue language (8 records).
- Passage comprehension: four simple number-retrieval questions, in three language/cue combinations and four option rotations (48 records).
- Basic capability: four simple arithmetic questions, in the same combinations and rotations (48 records).

There are 104 control records per model, independent of N_ITEMS. Every record includes a forced score and an unconstrained answer-only generation, so this adds 104 scoring operations and 104 generation operations. Controls run after the numerical preflight and before research stages. They use the configured generation token budget and contain no benchmark questions.

The controls are intentionally basic. They do not establish knowledge of the main benchmark answers, full comprehension of the longer sandbagging instruction, or a causal reason for suppression differences. No control threshold filters models or research items automatically. Passing the controls is not a certificate of capability or safety. Four questions per content category are descriptive diagnostics, not a high-powered independent capability benchmark.

## Reports

- `control_responses.jsonl`: append-only responses and scores.
- `control_results.csv`: all control trials.
- `control_item_results.csv`: average across rotations per independent item.
- `control_summary.csv`: independent item counts, forced accuracy, generated task success, answer coverage, strict compliance, token-limit rate and gold-option probability.
- `control_review_cases.csv`: incorrect or unextractable responses for inspection.

Rotations are not counted as independent questions. Missing generated answers count as unsuccessful attempts but are separately visible through coverage. Raw responses remain available. The specification, draft translation status and hash are embedded in the run manifest and covered by the ZIP inventory.

## Review and validation

The new Hausa control wording and revised unlocked cue require competent human review before final collection. Both are drafts, not claimed to be reviewed translations. Do not select controls based on whether they produce a desired suppression finding.

Offline tests cover control counts, correct-answer rotation mappings, balance, separate item IDs, missing-answer handling, repeated execution without duplicate outputs, manifest compatibility, and the main analysis/archive workflow with simulated model responses. Real checkpoint loading and generation require the Kaggle GPU preflight. Sample-size justification and final analysis-plan decisions still need completion.
