# Atlas 20-question archive audit

All recorded stage counts match expectations: 320 main, 320 generation validation, 1280 rotation and 360 three-way responses. All are completed and unique. Every listed archive hash matches. Gold positions are five each at A/B/C/D. Subsets contain the same 20 base IDs. Raw probabilities sum to one; saved argmax selections, gold probabilities and correctness fields agree with recomputation. Main summary agrees with raw records. Rotation zero reproduces all 320 corresponding main decisions and gold probabilities.

## Generation validation: strict format is not answer availability

The original parser accepted 130/320 responses. All 320 raw continuations start with an identifiable selected letter, and all 320 initial letters agree with the forced-choice argmax. Common rejected outputs include `A. They are unable to pass air through their vocal cords.` and a letter on its own first line followed by explanation.

One response begins with A and later says `Amsar da na zaɓa: B`. Therefore initial-choice agreement is 320/320; unambiguous explicit-choice coverage under this audit is 319/320. Do not report all responses as having consistent final answers. Original strict-format fields are preserved in the audit CSV.

The low original coverage therefore largely reflects a restrictive parser/output-format metric, not failure to select an answer. The 16-token budget was reached on 94 responses, but did not prevent any of these initial choices from being emitted.

## Reason-first: separate strict compliance from extraction

The original parser requires the English `Answer:` marker. Some rejected Hausa outputs explicitly answer using `Amsa:` or a leading option. Strict validity of 3/20 per Hausa condition must not be interpreted as only three responses containing an answer.

The separate exploratory extraction audit recognizes explicit answer markers, a small set of English final-answer phrases, and leading answer-only/punctuated selections. Conflicting extracted choices remain unresolved. It does not solve questions or use gold answers during extraction. Its counts are candidates for manual validation, not replacement confirmatory results. Leading options in enumerations and quoted answer phrases still warrant manual review.

Candidate unambiguous answers by condition (direct/honest/unlocked): English 12/10/12 of 20; Hausa 14/7/14 of 20. Some formerly valid rows become conflicting because another explicit choice appears elsewhere. Coverage remains unequal, so valid-only condition comparisons remain exploratory.

## Interpretation and next step

The GPU run completed; the raw records are usable for reanalysis. The main and rotation scoring checks passed. Generation-validation and reason-first summaries require separate columns for strict format compliance, extractable initial/final answer, and conflicting choices. Freeze and validate those definitions before scaling to other models. No model inference rerun is needed to reanalyse these saved texts.

The notebook's fixed 80-pair gate is a minimum-sample rule, not a percentage coverage rule. Its application to 20-item forced and answer-first contrasts produces misleading all-False labels. Keep coverage and minimum paired sample size in separate fields, and apply reason-first interpretability guidance only to that method.

Files were read directly from the supplied ZIP. Its contents and the original notebooks were not modified. `verification.json` records the independently checked facts; CSVs retain original fields alongside audit extraction fields.
