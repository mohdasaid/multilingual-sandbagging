# Gemma 20-item archive audit

All 50 inventory hashes match. Completed unique records: main 320, generation validation 320, rotation 1280, three-way 360. Generation log retains one NameError attempt, followed by a completed retry. All stored option probability vectors in scored stages are finite and normalized. Main gold probabilities, argmax labels and correctness agree with the saved vectors.

Rotation-averaged cue-pooled accuracy suppression: English -0.3625 (reported bootstrap CI -0.54375 to -0.18125); Hausa -0.0875 (-0.2625 to 0.08125). Hausa-minus-English interaction +0.275 (0.075 to 0.48125). These are exploratory 20-item findings; rotations do not create independent questions.

English reason-first accuracy honest 0.55, direct 0.25, unlocked 0.55, with all 60 responses extractable and order eligible. Hausa reason-first all-item accuracy 0.30, 0.20, 0.25 respectively; extractable counts 19,17,18 out of 20.

Parser limitation: 57 previously unextractable generation-validation outputs are exactly a single Markdown-bold letter. Narrow recovery gives 308/320 extractable responses, versus 251 originally. This audit does not alter raw files or frozen summaries. A separate initial-choice diagnostic identifies 319 initial choices, 316 matching main forced choices; it is not a final-answer adjudication. See generation_markdown_audit.csv.

Reproducibility limitation: run_manifest.json labels quantization only as prequantized checkpoint; it does not explicitly record the effective FP32 layer computation and nonquantized parameter dtypes. Preserve the executed notebook and add a separate runtime configuration record before releasing the session. A descriptive run name alone does not verify runtime precision.
