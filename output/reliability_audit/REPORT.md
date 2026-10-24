# Saved-activation audit and robustness checks

Completed 27 September 2026. Original activation ZIPs and existing paper tables were not changed.

## Findings

1. **No numerical discrepancy was found in the audited geometry.** An implementation that does not import the project analysis helpers reproduced all 93 available native reliability rows and their intervals, and all 183 cross-arm rows (cosines, heuristic ceilings, gaps, and intervals). Missing/undefined cross-arm values also matched. The three unavailable native configurations were checked for agreement on layer eligibility.
   Maximum native estimate/interval error: 3.33e-16. Maximum cross-arm error: 6.66e-16. Selection scores and layer choices also matched (allowing float32 accumulation tolerance).
2. **The last-token language ordering survives topic-split sensitivity.** Across 20 predefined seeds per model, English > Hausa > Yoruba held in 77/77 complete comparisons. Three of the 80 comparisons lacked an eligible Yoruba layer. The seeds include the original seed and are overlapping reuses of the same dataset, not independent replications.
3. **Mean pooling is less consistent.** English > Hausa > Yoruba held in 56/65 complete comparisons; 15 of 80 comparisons were incomplete. Do not describe the strict three-language ordering as universal across readouts.
4. **Mean-difference directions had higher agreement than probe weights in all 23 eligible original-split configurations when both used the same 20 half-partitions.** Approximate, unadjusted paired jackknife intervals excluded zero in 17/23. This is conditional on the selected layer, fixed regularization/scaling, and split schedule. It does not establish universal estimator superiority. Layer selection itself optimizes mean-difference reliability on separate selection topics.
5. **Exact estimates and layer eligibility can be sensitive to topic allocation.** For example, E2B Yoruba last-token reliability ranged from -0.117 to 0.275 over 18 eligible splits; two splits had no eligible layer. AfroLlama Yoruba mean pooling was eligible in 14/20 splits, although it was unavailable in the original split. The 12B English mean-pooling original value (0.473) was the minimum among these 20 splits (median 0.717). These findings should qualify any emphasis on a single split or particular layer.

## Language-ordering sensitivity at threshold 0.15

| Model | Readout | Ordered / complete | Missing comparisons |
|---|---|---:|---:|
| 12B | last | 20/20 | 0/20 |
| 12B | mean | 16/18 | 2/20 |
| AfroLlama | last | 20/20 | 0/20 |
| AfroLlama | mean | 9/12 | 8/20 |
| E2B | last | 18/18 | 2/20 |
| E2B | mean | 15/17 | 3/20 |
| E4B | last | 19/19 | 1/20 |
| E4B | mean | 16/18 | 2/20 |

## Reliability across 20 topic allocations

Medians and ranges below include eligible splits only. Missing cases are explicitly counted and must not be treated as zero reliability. Ranges describe sensitivity, not confidence intervals.

| Model | Language | Readout | Eligible / 20 | Median | Min–max | Distinct layers |
|---|---|---|---:|---:|---:|---:|
| 12B | eng | last | 20/20 | 0.830 | 0.792–0.853 | 4 |
| 12B | eng | mean | 20/20 | 0.717 | 0.473–0.771 | 9 |
| 12B | hau | last | 20/20 | 0.601 | 0.427–0.658 | 9 |
| 12B | hau | mean | 19/20 | 0.494 | 0.432–0.638 | 8 |
| 12B | yor | last | 20/20 | 0.157 | -0.087–0.292 | 13 |
| 12B | yor | mean | 19/20 | 0.304 | -0.071–0.644 | 12 |
| AfroLlama | eng | last | 20/20 | 0.747 | 0.720–0.779 | 4 |
| AfroLlama | eng | mean | 20/20 | 0.478 | 0.184–0.505 | 3 |
| AfroLlama | hau | last | 20/20 | 0.484 | 0.376–0.589 | 5 |
| AfroLlama | hau | mean | 18/20 | 0.408 | 0.304–0.467 | 8 |
| AfroLlama | yor | last | 20/20 | 0.284 | 0.066–0.358 | 10 |
| AfroLlama | yor | mean | 14/20 | 0.322 | 0.137–0.480 | 3 |
| E2B | eng | last | 20/20 | 0.864 | 0.826–0.892 | 5 |
| E2B | eng | mean | 20/20 | 0.753 | 0.716–0.798 | 4 |
| E2B | hau | last | 20/20 | 0.563 | 0.309–0.703 | 6 |
| E2B | hau | mean | 19/20 | 0.432 | 0.255–0.524 | 9 |
| E2B | yor | last | 18/20 | 0.099 | -0.117–0.275 | 12 |
| E2B | yor | mean | 18/20 | 0.208 | -0.067–0.458 | 10 |
| E4B | eng | last | 20/20 | 0.884 | 0.841–0.905 | 4 |
| E4B | eng | mean | 20/20 | 0.814 | 0.783–0.836 | 2 |
| E4B | hau | last | 20/20 | 0.659 | 0.496–0.762 | 3 |
| E4B | hau | mean | 19/20 | 0.592 | 0.403–0.658 | 9 |
| E4B | yor | last | 19/20 | 0.387 | -0.021–0.508 | 6 |
| E4B | yor | mean | 19/20 | 0.399 | 0.166–0.575 | 13 |

## What was checked

- Cached native files: pair-ID uniqueness, 100 pairs, finite activations; cross-arm subject-label agreement.
- Independent reconstruction of selection scores, length-overlap scores, eligible-layer decisions, native split-half agreement, and subject-delete-one intervals at all four thresholds.
- Independent reconstruction of every existing cross-arm cosine, attenuation heuristic, gap, and associated interval. Numerical agreement does not validate the attenuation assumptions.
- Independent refitting reproduced all 23 same-layer held-out probe accuracies, and probe-weight agreement agreed within the specified numerical tolerance. This audit did not independently rerun the separate probe-layer cross-validation search or its accuracy bootstrap.
- Twenty predetermined seeds (2026–2045): 10 selection topics, 15 evaluation topics, 100 half-partitions, thresholds 0.10/0.15/0.20/0.25. No best seed was selected. This reran mean-difference layer selection, not probe layer selection, on every allocation.
- Matched estimator comparison: original selection split and threshold 0.15; exactly the same first 20 evaluation half-partitions for both estimators. Each subject was then deleted and both probe halves retrained to approximate uncertainty in the paired difference. Entire topics and their contrast pairs stayed together.
- All fitting completed with convergence warnings treated as errors.

## Limits and confidence in the code

These checks increase confidence that the published calculations match the saved activations. They are not a proof that the pipeline is bug-free. This is a second implementation by the same assistant, not an external human code review. It shares NumPy, SciPy and scikit-learn dependencies with the project. It does not verify model weights, tokenizer behavior, extraction on a real GPU, or correspondence of every raw text to its saved activation.

The paired estimator intervals are approximate, conditional, unadjusted for multiple comparisons, and have not received a dedicated probe-weight coverage simulation. Previous geometry-only coverage simulations do not validate this new estimator-difference interval. There are only 15 evaluation topics per split, and writer/topic confounding remains. Repeated allocations overlap heavily and must not be pooled as independent observations or used to compute a standard error across seeds.

The findings do not establish a causal tokenizer effect, generalization to other concepts/languages/writers, or successful steering. Sample-size curves and additional length-adjusted analyses were not part of these three prioritized checks.

## Suggested paper changes

- Add the repeated-selection table or a concise figure in the appendix and report failed eligibility alongside valid estimates.
- State that the last-token ordering is robust across the tested topic allocations, while exact estimates, selected layers, and mean-pooling ordering are sensitive.
- Add the matched estimator comparison as a descriptive result with conditional intervals and fixed regularization/scaling explicitly stated.
- Retain the original split as primary and label this analysis as a retrospective robustness check. Do not replace the primary split with one giving a stronger result.
- Keep causal fertility claims and universal pooling/estimator claims out of the conclusions.

## Files

`independent_audit.csv`, `cross_arm_audit.csv`, `repeated_selection.csv`, `robustness_summary.csv`, `language_ordering.csv`, and `matched_estimators.csv` contain the numerical results. The two Python scripts reproduce the computations from the local ZIPs; their path constants identify the inputs. Completion records and environment information accompany them.
