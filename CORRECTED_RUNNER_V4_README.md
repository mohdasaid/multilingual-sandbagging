# Corrected four-model runner v4

Use `sec-llm-sandbagging-four-model-corrected-v4.ipynb`. The v3 notebook remains an audit snapshot; do not use it for new research runs.

## Settings

```python
MODEL_KEY = "atlas"  # atlas | aya101 | gemma3 | afrique
N_ITEMS = 20         # any integer from 1 to 400
GENERATION_MAX_NEW_TOKENS = 16
ANSWER_FIRST_MAX_NEW_TOKENS = 128
REASON_FIRST_MAX_NEW_TOKENS = 192
```

These are ordinary settings for the same pipeline, with no separate pilot mode. `N_ITEMS` counts unique base questions, not the number of requests. Every experimental condition receives the same selected questions. The full 400-item dataset is validated and balanced before selecting the requested number. If the number is not divisible by four, gold positions differ by at most one.

At N_ITEMS=10, enabled research stages issue 160 main, 160 generation-validation, 640 rotation, and 180 three-way requests. At N_ITEMS=20 they issue 320, 320, 1280, and 360 respectively. The discarded smoke test adds calls. Full-size defaults use 400 main questions, 50 generation-validation questions, 100 rotation base questions, and 100 three-way questions. Smaller runs cap those subset sizes at N_ITEMS.

Token limits are maximum newly generated tokens: validation responses, explanation after the committed answer, and full reason-first response respectively. The committed answer tokens are not part of the answer-first rationale budget, although they are included in its recorded output-token count. EOS may end a response before its budget. Forced probability does not generate a rationale.

Set a stage flag to False to skip it completely. Re-running an enabled stage resumes matching completed requests. The full manifest prevents incompatible resumption. Output paths include model, version, sample count, and all three generation budgets. Use a new RUN_VERSION for other configuration changes. Save/download the run directory before ending a Kaggle session and restore it to the same working path to resume.

## Corrections

- Historical Atlas main permutation, Hausa condition wording, and full-size three-way subset retained. Other models share this wording prospectively; their historical runs used different wording/order.
- Actual returned tokens checked for answer-first integrity; probabilities retained. This is a constrained answer commitment, not independent unrestricted answer generation.
- Explicit answer extraction rejects incidental letters and unresolved prose.
- Context-length checks replace silent truncation.
- Distinct candidate encodings checked; multi-token choices use sequence log-likelihood. Full-vocabulary candidate mass is reported only for single-token choices; sequence probability sums are separately labelled.
- Checkpoint revision reused from the manifest before loading. Eager attention explicitly selected; package/runtime/model configuration recorded.
- Full prompt settings, decoding limits, subset hashes, code identity, parser version and rendering template included in the manifest. Existing subset artifacts are checked before reuse.
- Saved request payload identities checked on resume. Interrupted final writes are backed up before recovery; complete malformed lines fail loudly.
- All three adapters exercised by a discarded generation preflight before research stages.
- Rotation-averaged accuracy and accuracy of averaged content probabilities saved as distinct outcomes; pooled effects and language interaction added.
- All-item and complete-case three-way effects, coverage bounds, diagnostic examples and output-failure indicators saved.
- Export inventory excludes itself and verifies hashes.
- Stage flags skip the whole stage, including reads of nonexistent stage outputs.

The rotation subset is prospective; saved historical rotation IDs were not available locally to match it. Recognition and judge annotation are optional research extensions, not silently added model calls.

## Interpretation

Small runs execute the same analysis but provide less precise estimates. Reason-first remains exploratory regardless of the project-specific coverage gate. Failure to emit a parsable answer does not demonstrate absence of knowledge. The three-way prompts differ, so differences cannot be attributed exclusively to reasoning order.

## Validation

Ordinary Python cells compile. Synthetic responses were passed through every research/analysis stage for 1, 10, 20 and 400 base questions, including resume checks. Explicit-parser counterexamples and interrupted-write recovery tests passed. The underlying 400-item balancing was compared with the historical Atlas transformation. No real checkpoint/GPU end-to-end run was performed locally. The Kaggle preflight must pass for each actual model before long runs.

The builder is `build_multimodel_v4.py`; it reads the preserved v3 builder definitions as an input. The resulting notebook is standalone and does not require either builder on Kaggle.

## Additional validation pass

Re-ran the synthetic pipeline at 1, 10, 20 and 400 base questions. Exercised the actual adapter functions using NumPy-backed tensor/model doubles for Atlas, Aya-101, Gemma and Afrique, including both single-token and multi-token candidates, 3/7/11-token generation budgets, sequence output slicing, and committed-answer integrity. This tests adapter contracts, not actual checkpoint loading.

Found and fixed an explicit-answer parser bug where whitespace could consume a second conflicting answer line. Added a regression requiring `Answer: A\nAnswer: B` to remain unresolved. Also passed an all-missing reason-first run, including zero-valid-pair analysis and resumption.

Local environment has neither PyTorch nor an NVIDIA runtime tool, so real checkpoint inference was not run. Use the notebook's GPU preflight for that remaining validation.
