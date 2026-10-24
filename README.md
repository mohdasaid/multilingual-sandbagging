# Multilingual sandbagging evaluation research

Work in progress investigating how English/Hausa evaluation instructions, elicitation methods and answer formatting affect measurements of instructed underperformance.

This repository contains development notebooks, parsers, validation code and preliminary audit artifacts. It is **not a claim of spontaneous deception, a completed confirmatory study, or a bug-free pipeline**. Historical protocols and results are retained for transparency; findings from different versions should not be combined without checking their manifests.

## Project map

- `sandbagging-narrow-study-v11-four-models.ipynb`: newest numbered local study notebook in this snapshot; inspect its own configuration and protocol before execution.
- `sec-llm-sandbagging-four-model-corrected-v5.ipynb`: earlier four-model pipeline used in the initial development work.
- `output/gemini-sandbagging/`: separate generated-answer API experiment and a Gemini 3.5 replacement definitions file. The original notebook is preserved; it does not automatically incorporate that replacement. See the notes below.
- `multilingual-evaluation-pilot/`: dataset preparation and historical pilot materials.
- `notebook-validation/`: development validation scripts. Some reference original local paths and need path configuration; this snapshot does not claim a portable CI suite.
- `output/`: preliminary analyses, audit scripts and saved result artifacts. Filenames such as `corrected` denote development versions, not independent validation guarantees.
- Root protocol and comparison documents: historical design records, whose version labels must be matched to corresponding runs.
- `STAGING_INVENTORY.json`: hashes of staged research files. Notebook execution outputs were removed during packaging; original local notebooks were preserved.

## Scientific interpretation

Distinguish probability scoring from generated answers, missing/blocked outputs from incorrect answers, and instruction-following failures from lack of knowledge. Cyclic rotations are repeated measurements of the same question, not independent samples. In the early local answer-first procedure, the selected answer was supplied before explanation generation; API answer-first generation is a different procedure.

The initial 20-item runs are development checks. They are not sufficient evidence for broad model/language generalization. Further protocol validation, translation review and prespecified analyses remain necessary.

## Gemini experiment notes

Provide API credentials through a hidden prompt or an environment variable, never committed source. Google quota limits are project/model specific. The observed project limit was 5 requests/minute and 20/day; use 13-second spacing and stop/resume at quota errors. Keep a persistent results directory.

The packaged original runner defaults to Gemini 2.5. `replacement_functions_gemini35.py` adds a Gemini 3.5 configuration using provider-default thinking; corresponding setup metadata and model configuration must also be updated. Do not mix models or scientific configurations in one results folder. Live backend compatibility and availability have not been guaranteed. No API calls were made while preparing this repository.

## Data and attribution

Question data derives from CohereLabs/Global-MMLU with local Hausa review and choice alignment. Preserve source attribution and consult the dataset's license and manifest before reuse:
https://huggingface.co/datasets/CohereLabs/Global-MMLU

Research files are included without assigning a new blanket license. Third-party data/model terms continue to apply. No model weights are included. Reviewer and dataset metadata should be checked before making this repository public.

## Reproducibility status

This is a source-and-artifact snapshot, not a freshly executed replication. Individual experiments record their own dependencies, configuration, seeds and model identities where available. Local paths in historical builders/audits may need adaptation. The full dataset of remote Kaggle outputs and the latest Gemini responses is not automatically available here; preserve and add their original manifests when importing them.
