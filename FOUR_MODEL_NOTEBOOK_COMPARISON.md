# Comparison audit: four-model notebook versus previous implementation

Audit date: 2026-10-15. This is a review, not a new frozen implementation. Neither notebook nor its builder was modified during the audit.

## Verdict

Do not treat the four-model notebook as a validated drop-in replication yet. It compiles, but there are material measurement, provenance, and comparability issues. Earlier assurances of readiness were stronger than the validation supported. Actual GPU/model behavior remains untested locally.

Locations below refer to the builder, whose code is embedded in the new notebook. Old notebook cell numbers are zero-based.

## High-priority findings

1. **The three-way sample changed without being identified as a new sample.** Builder line 170 uses `three-way-elicitation-v3`; old cell 46 uses `three-way-elicitation-v1`. Executing both selectors on the approved dataset gives only **24 shared questions out of 100**. The balanced 400-item content transformation itself is identical. Restore the original namespace and selected IDs for a direct pilot replication, or explicitly label this a new sample. Do not expect the old three-way percentages to reproduce.

2. **Hausa condition prompts changed.** Builder lines 99 onward match the old Aya wording, not the Atlas wording in old cell 9. Honest changes `Ka amsa waennan tambayoyin...` to `Ka amsa tambayar da ke gaba...`; unlocked adds `sanya masa ƙuntatawa`. The old notebook already had inconsistent condition translations across model sections. Standardizing them is sensible, but it changes the Atlas experiment and requires an explicit prompt version and reviewed wording decision.

3. **The manifest does not protect the full experiment on resume.** Builder lines 280–299 omit the three-way instructions, generation/token settings, parser version, subset hashes, balanced-data hash, chat template, and code hash. Subset files are overwritten after this incomplete check. Changes to those settings can leave the manifest equal and reuse existing request IDs, allowing incompatible records to be combined. The previous Atlas three-way manifest did include its subset hash, method instructions, and token budgets. Freeze all relevant settings before any inference and verify saved input artifacts before writing them.

4. **Generation validation can score incidental letters as answers.** `explicit_letter` at builder line 456 searches anywhere in the text. Executable examples: `I considered A but cannot decide.` becomes A; `Vitamin A is discussed; no final answer.` becomes A. Both should remain unresolved. Use a frozen explicit-answer parser, record extraction methods, and reject conflicting or noncommittal text. This affects the new 50-item generation validation, not its separate reason-first parser.

5. **Answer-first agreement is hard-coded, so the advertised check cannot fail.** Builder lines 573–578 assemble an answer string from the chosen argmax, return that choice as the parsed answer, and set `argmax_generation_agree=True`. The old cell 49 decoded the returned token sequence including the committed token and parsed it. Restore a check against the actual returned tokens and retain the pre-generation probabilities. Both implementations deliberately insert the selected answer; 100% agreement is an implementation-integrity check, not independent evidence that unrestricted generation reproduces the probability preference. Different method prompts also mean the three-way experiment does not isolate ordering alone.

6. **Token scoring audit does not inspect the stated context.** `candidate_suffix_ids(prefix)` at line 238 ignores `prefix` and encodes isolated space-plus-letter tokens. It neither checks distinct candidate IDs nor verifies their decoded continuation after each actual prefix. It asserts one token and has no conditional sequence-scoring fallback required by the protocol. The old implementation shares the isolated-token limitation. Validate actual continuation semantics and implement the fallback or explicitly document that unsupported models stop before research inference.

7. **Silent input truncation was added to the Atlas paths.** Builder lines 253, 263, 443, 450, 553, and 563 request truncation without measuring or recording removed tokens. Old Atlas scoring and three-way generation did not request truncation; old Aya/Gemma/Afrique scoring did. If truncation occurs, questions, options, or the final scoring prefix can be removed while the observation is still treated as valid. Measure lengths before inference and fail with a recorded reason rather than silently altering the task.

## Other reproducibility and analysis findings

8. **Rotation accuracy is a different estimand from accuracy of the marginalized content distribution.** Lines 532–541 average gold probabilities correctly, but average the four correctness indicators as the accuracy outcome. That is valid rotation-averaged accuracy, not argmax accuracy after averaging all four content probabilities. Save all averaged content probabilities, derive their selected content and accuracy separately, and label both outcomes. Rotation cue-pooled effects, language interaction, and condition-by-position contrasts are also absent; position descriptive summaries do not replace them.

9. **Resume loads the latest checkpoint before checking the manifest.** Line 200 resolves current repository HEAD and loads it. If HEAD moved, the later comparison stops rather than silently mixing revisions, but the old revision cannot be resumed automatically and the new download is wasted. Read the stored revision before loading. Explicit attention configuration also changed: the old Atlas loader specified `eager`; the new loader leaves it implicit. Record and control this for exact replication.

10. **The preflight tests only forced scoring.** Lines 353–368 do not exercise Aya generation, answer-first token insertion, reason-first parsing, all three methods, or persistence/resumption. Generation functions are defined after the large main run. A generation-specific failure can therefore appear only after thousands of expensive calls. Move model-dependent smoke checks ahead of all research stages.

11. **Required audit information is incomplete.** The new scorer saves candidate mass but not unrestricted top token or entropy. The manifest does not save the actual chat template/hardware/backend. Full rendered prompts are not persisted; a preview is printed only for one English example. Reason-first coverage bounds, explicit failure categories, and reproducible raw example exports are missing. These are protocol gaps, not proof that current accuracy estimates are numerically wrong.

12. **The output inventory invalidates itself on a second export.** Lines 690–691 hash all existing files, including the previous `sha256_inventory.json`, then overwrite that file. Reproduced locally: its recorded self-hash is false after the second export. Exclude the inventory itself and verify every listed digest.

13. **Resume identity checks are incomplete.** `run_jobs` checks only the number of completed IDs, not membership in the current planned job set or per-job payload equality. `completed_frame` checks total count and duplicates; only the main stage checks counts by cell. A partial final JSONL line causes a parse failure with no controlled recovery. Add expected-ID and payload checks and explicit handling of interrupted writes without silently discarding valid records.

14. **Stage flags skip inference, not the entire stage.** Each stage unconditionally loads its full expected output after the flag. This is fine for reanalysis of an already completed stage, but fails if the user expects a false flag to skip an unrun stage. Document this accurately or separate run/reanalyse/skip actions. The final archive assumes all stages are complete.

15. **The frozen protocol wording itself needs qualification.** Its 80% coverage / 10-point imbalance / 80 paired-item gate is a prospective project rule chosen after the pilot, not a universal statistical standard. Passing it does not eliminate selective missingness or promote exploratory reason-first findings to confirmatory evidence. Keep all-item failure outcomes and complete-case sensitivity estimates distinct.

## Earlier-notebook issues that should not be copied

- Aya and Gemma/Afrique main jobs use the original `capability_items`, while Atlas uses balanced `experiment_items`. The new common balancing fixes this discrepancy but changes comparisons to the old non-Atlas runs.
- Hausa honest/unlocked wording differs across old model sections.
- The old Atlas XML audit and three-way audit are separate designs; they must not be pooled.
- The stored old notebook does not contain a complete implemented rotation-run section matching the newly proposed one. It references rotation results, but those references cannot establish that the new rotation subset matches the previous executed audit. Recover saved subset IDs to verify historical comparability.
- The old three-way section used the corrected set-membership validity check; that correction is retained in the new notebook.

## What passed

- Both notebooks' ordinary Python cells compile. Notebook magic cells were excluded from this syntax check.
- Actual approved data reproduce identical main 400-item option content, gold letters, and choice permutations between the old Atlas balancing and new balancing.
- Main gold positions are 100/100/100/100.
- New three-way and rotation base subsets are 25/25/25/25, with unique base IDs.
- The 50-item subset is necessarily only approximately balanced: 13/13/12/12.
- Paired Boolean subtraction was converted to numeric subtraction in the new implementation.
- Rotation outcomes are aggregated by base item before the implemented bootstrap, avoiding rotation pseudoreplication for those reported effects.

## Verification performed and limits

Read both notebooks and the full builder; executed the dataset transformation and both subset selectors on the actual approved CSV; exercised the generation parser with counterexamples; reproduced the repeated-export hash failure; compiled ordinary code cells. No model checkpoints, GPU execution, or end-to-end Hugging Face runtime were available in this local verification. A corrected artifact still needs a small smoke test on each actual model adapter.

## Repair order

1. Preserve these files as reviewed snapshots; create a corrected implementation version.
2. Fix full manifests/resumption, actual-output checks, parsing, and input-length handling.
3. Restore the original Atlas sample for pilot replication and explicitly decide/version shared prompt wording.
4. Complete the candidate audit and missing statistical/diagnostic outputs.
5. Test both causal and encoder-decoder adapters before long runs.
6. Run a same-items, same-prompts Atlas replication; then run all four models under the unified prospective configuration.
