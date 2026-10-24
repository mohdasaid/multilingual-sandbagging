# Reanalyse saved outputs only: no model loading or inference.
analysis_dir = OUT / "answer-analysis-v6"
reanalyse_saved_run(OUT, analysis_dir)
for stage in ["generation_validation", "threeway"]:
    path = analysis_dir / f"{stage}_summary.csv"
    if path.exists():
        print(stage)
        display(pd.read_csv(path))
print("Versioned answer reports:", analysis_dir)
