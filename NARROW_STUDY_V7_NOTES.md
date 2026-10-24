# Narrow study v7

Use `sandbagging-narrow-study-v7.ipynb`. The original Downloads notebook is unchanged. This is a new protocol; do not merge its results with historical runs as if prompts and procedures were unchanged.

## Scope

- Models selectable: Gemma 3 4B IT and Qwen3 4B Instruct 2507, loaded one at a time.
- Independent conditions: honest, direct sandbagging, unlocked. Unlocked is an evaluation-stakes-removal cue, not a sequential recovery experiment.
- Language cells: English questions/English cues; Hausa questions/English cues; Hausa questions/Hausa cues.
- Primary endpoint: forced-choice accuracy averaged over four option rotations within each base question. Supporting endpoint: probability of the gold option normalized among A–D.
- Question-language interaction holds the cue at English. Cue-language interaction holds questions at Hausa. No pooling of unmatched cue configurations.
- Answer-only generation on a fixed subset (maximum 50 by default) is a supporting measurement check. It uses the same answer prefix as forced scoring but inserts no answer letter.
- Explanation generation, target-30, enhanced elicitation, and within-conversation unlocking are removed.
- Removed the same-run honest-and-unlocked correctness filter; it is not an independent knowledge control.

## Configuration

Change MODEL_KEY, RUN_VERSION, N_ITEMS, N_GENERATION_ITEMS, and GENERATION_MAX_NEW_TOKENS in the configuration cell. N_ITEMS counts unique questions. The existing CSV still contains 400 reviewed items; any 1–400 can be selected deterministically. A new setting or code revision requires a new run version. The manifest prevents incompatible resume.

With 20 items and all stages enabled: 180 initial-arrangement records, 180 generation records, 720 rotation records = 1,080 research records, plus technical preflight. Generation jobs also perform a matching forced score. With 400 items and generation capped at 50: 3,600 + 450 + 14,400 = 18,450 records. Rotation zero intentionally duplicates initial-arrangement scoring as a numerical consistency check.

Run from the top in Kaggle with GPU and Internet enabled. Add HF_TOKEN to Kaggle secrets and accept the Gemma checkpoint license in the corresponding Hugging Face account. Model weights use /tmp; results use /kaggle/working. No automatic old-cache deletion or migration occurs. Download the final ZIP before ending the session.

## Interpretation and remaining study preparation

Missing generated answers are separate from explicit wrong answers. Accuracy over all attempts is end-to-end task success; extractable-only accuracy is selection-sensitive. Strict compliance and extractability are separate. Bootstrap resampling uses base questions, not rotations. Intervals are pointwise, not multiplicity-adjusted.

The existing English/Hausa data are used; no third language or translation review is invented. The revised Hausa unlocked text requires competent language review. Independent comprehension/capability controls, additional reviewed language data, final sample-size justification and the preregistered analysis plan remain outstanding. This notebook is an implementation for development/technical acceptance, not a claim that those scientific requirements have been fulfilled.

## Validation limits

Offline integration tests execute the notebook's data selection, scheduling, paired effects, interactions, generation reports, rotation mapping, resume and archive verification with synthetic data and model responses. They use fewer bootstrap draws for speed. Real model loading, T4 numerical behavior, real inference, language quality and instruction compliance require the Kaggle preflight. No bug-free guarantee is made.
