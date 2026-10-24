# Gemini sandbagging API test

Open `gemini-sandbagging-api.ipynb` in Colab or Kaggle (CPU is sufficient). Upload the included `balanced_items.jsonl` next to the notebook, or point DATA_PATH to one of your previous archive ZIPs. Run cells in order. Enter your Google AI Studio API key only at the hidden input prompt.

Configuration: N_ITEMS=10; rotations=[0]; answer_only; token budget=32; at most 10 API call starts/minute; 160 attempts maximum per execution. Select 20 by changing N_ITEMS. More than 20 requires a larger reviewed dataset. Optional generated explanation methods and four rotations are available. Changing scientific settings requires a new OUT directory; rerunning unchanged settings resumes saved responses. Use persistent storage to survive notebook-session resets. Do not run concurrent workers.

Choose a supported model available under your project's free tier: gemini-2.5-flash-lite (default) or gemini-2.5-flash. The notebook explicitly disables thinking for these model configurations. Other model families need separately verified configurations. Billing status and quotas are controlled by Google, not this notebook. A 429 stops immediately; rerun after quota availability returns. Errors are not scored as answers. No automatic retry loop consumes quota.

The instructions come from your previous archived run. Reviewed Hausa fields are used, not unreviewed option mappings. Parser v6 is embedded. All responses, blocks, completion reasons, reported model versions and scientific settings are saved. No fallback to A, invented gold probability, or local argmax comparison is used. Answer-first here is freely generated and differs from the injected-answer local experiment. Structured output enforcement is not used because format-following is measured.

The analysis averages rotations within base questions before bootstrapping. Partial runs show only descriptive summaries; complete runs yield paired accuracy effects. Unextractable/blocked model responses count as unsuccessful outputs, with separate coverage. Such failures are not evidence that a model lacks knowledge. Conservative parser misses remain visible for human review. Reason-first order does not establish explanation correctness. This notebook does not reproduce the entire local-logit pipeline or its language interaction analysis.

Local verification: six offline test cases passed, including data selection/rotation, parser edge cases, quota stop/resume/rate spacing, configuration mismatch protection, partial-run withholding, question-clustered effects, and compilation of notebook code cells. No live API call was made. Free access/model availability must be verified on your account.

References:
- https://ai.google.dev/api/generate-content
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/rate-limits
- https://ai.google.dev/gemini-api/docs/generate-content/thinking
