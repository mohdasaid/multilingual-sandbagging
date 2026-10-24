# Answer extraction correction v5

For new inference, use `sec-llm-sandbagging-four-model-corrected-v5.ipynb`. MODEL_KEY, N_ITEMS and the three generation-token budgets remain editable. New runs have v5 output directories and a manifest with parser/code hashes.

For existing saved outputs, use `sandbagging-reanalyse-saved-responses-v5.ipynb`. Set SOURCE to a saved run directory or result ZIP and ANALYSIS_OUTPUT to a new folder. Run its four cells; no model loading, Hugging Face secret or GPU is required. The notebook embeds the full analysis implementation and requires only NumPy and pandas.

The authoritative generation reports are in `answer-analysis-v5/` in new runs. Original/basic stage outputs remain available. Analysis tables preserve original raw fields and add:

- Strict-format compliance, separately from answer extraction.
- Initial and explicit final choices, unambiguous extracted answer, ambiguity and conflict flags.
- Initial accuracy/agreement separately from unambiguous-response accuracy.
- Reason-first ordering: answer_before_reason, answer_only, text_before_answer or unresolved. Text-before-answer verifies placement only, not rationale quality.
- All-item, extractable-pair and reason-first-order-pair effects.
- Separate reason-first coverage and fixed 80-pair requirements. These are not applied to forced-probability or answer-first rows. Reason-first remains exploratory even if both pass.

The parser accepts explicit English/Hausa markers, bare letters, leading punctuated selections and a limited set of explicit final-answer phrases. It rejects incidental letters, multi-option lists without a commitment, unresolved alternatives and conflicting selections. No subject-matter inference or gold answer is used to extract choices. This remains a conservative parser; unresolved responses are exported for review rather than guessed.

Your 20-question archive was reanalysed without modifying its contents. Generation validation now has 319/320 unambiguous extractable responses; one contains conflicting A/B choices. All 320 initial answers agree with forced probability. Reason-first and constrained answer-first reports separately expose conflicts and ordering failures. Existing forced-choice and rotation results were not changed.

Validation: parser regressions (including Hausa sentence openings and ordinary lowercase words), strict compliance, response ordering, gold-blind extraction, idempotent analysis, raw ZIP preservation and notebook compilation passed. A 10-item synthetic pipeline also passed. These changes do not require checkpoint inference to reanalyse existing text.
