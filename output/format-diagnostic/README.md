# Final format and answer-mapping diagnostic
Copy the **entire code cell** below into a NEW cell at the bottom of the already-loaded v8 research notebook. Run that cell only. Do not run this notebook as a fresh Kaggle session: it intentionally reuses the loaded model and helper functions.

Run once with Gemma 4B loaded and once with Qwen 4B loaded. It uses eight existing control items, four rotations, three language/cue combinations and three output procedures, with 64 tokens throughout. This means **288 generated responses for Qwen**; Gemma adds **32 English answer-mapping checks**, for **320 responses**. The two prefixed procedures also perform 192 forced-scoring operations. This is development work, not a main-study rerun.

The procedures are the current language-specific prefix, a shared English prefix, and a shared `FINAL: X` marker without a supplied prefix. The identical English-cue prefix arms deliberately check repeatability. Gemma additionally receives arithmetic options with numeric-answer output, and the correct numeric value with a request to select its letter.

The new Hausa final-marker instruction is a draft requiring human review. The final-marker parser is isolated; your main notebook's parser is not changed. Strict compliance, extractability and accuracy are separate. Numeric prose is retained for human review rather than guessed from incidental numbers.

Outputs go into a new timestamped diagnostic folder and ZIP under `/kaggle/working`. Your main results remain untouched. If interrupted, partial outputs are retained; rerunning creates a separate diagnostic folder. No model reload, dependency installation or quantization change occurs.
