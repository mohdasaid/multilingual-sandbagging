"""Build the fixed 400-item confirmatory review pool from pinned Global-MMLU files."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "confirmatory-data"
SEED = 42
REVISION = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"
DEVELOPMENT_COUNT = 20
PILOT_COUNT = 100
CONFIRMATORY_COUNT = 400


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    english_path = DATA / "en-test.parquet"
    hausa_path = DATA / "ha-test.parquet"

    english = pd.read_parquet(english_path)
    hausa = pd.read_parquet(hausa_path)

    for language, frame in (("en", english), ("ha", hausa)):
        assert frame["sample_id"].notna().all(), f"Missing {language} sample IDs"
        assert frame["sample_id"].is_unique, f"Duplicate {language} sample IDs"

    paired = english.merge(
        hausa,
        on="sample_id",
        suffixes=("_en", "_ha"),
        validate="one_to_one",
    )
    assert len(paired) == len(english) == len(hausa)
    assert paired["subject_en"].eq(paired["subject_ha"]).all()
    assert paired["answer_en"].eq(paired["answer_ha"]).all()
    assert paired["answer_en"].isin(list("ABCD")).all()

    ids = sorted(paired["sample_id"].tolist())
    random.Random(SEED).shuffle(ids)
    development_ids = ids[:DEVELOPMENT_COUNT]
    pilot_ids = ids[DEVELOPMENT_COUNT : DEVELOPMENT_COUNT + PILOT_COUNT]
    confirmatory_ids = ids[
        DEVELOPMENT_COUNT + PILOT_COUNT :
        DEVELOPMENT_COUNT + PILOT_COUNT + CONFIRMATORY_COUNT
    ]

    assert len(set(development_ids) & set(confirmatory_ids)) == 0
    assert len(set(pilot_ids) & set(confirmatory_ids)) == 0
    assert len(confirmatory_ids) == CONFIRMATORY_COUNT

    columns = ["sample_id", "subject_en", "subject_category_en"]
    for field in ["question", "option_a", "option_b", "option_c", "option_d"]:
        columns.extend([f"{field}_en", f"{field}_ha"])
    columns += ["answer_en", "cultural_sensitivity_label_en"]

    indexed = paired.set_index("sample_id")
    review = indexed.loc[confirmatory_ids].reset_index()[columns].copy()

    for field in ["question", "option_a", "option_b", "option_c", "option_d"]:
        review[f"{field}_ha_reviewed"] = review[f"{field}_ha"]

    review["review_status"] = "pending"
    review["review_notes"] = ""
    review["reviewer"] = ""
    review["review_date"] = ""

    csv_path = OUT / "confirmatory_review_400.csv"
    ids_path = OUT / "confirmatory_ids_400.csv"
    review.to_csv(csv_path, index=False)
    review[["sample_id", "subject_en", "subject_category_en"]].to_csv(
        ids_path, index=False
    )

    category_counts = review["subject_category_en"].value_counts().sort_index()
    answer_counts = review["answer_en"].value_counts().sort_index()
    manifest = {
        "dataset": "CohereLabs/Global-MMLU",
        "revision": REVISION,
        "seed": SEED,
        "source_files": {
            "en": {"path": str(english_path), "sha256": sha256(english_path)},
            "ha": {"path": str(hausa_path), "sha256": sha256(hausa_path)},
        },
        "matched_source_items": len(paired),
        "selection": (
            "Sort all matched sample IDs, shuffle once with Python random.Random(42), "
            "reserve positions 0:20 for development and 20:120 for the prior pilot, "
            "then select positions 120:520 for confirmation."
        ),
        "confirmatory_items": len(review),
        "overlap_with_original_development": 0,
        "overlap_with_original_pilot": 0,
        "category_counts": category_counts.to_dict(),
        "answer_counts": answer_counts.to_dict(),
        "review_status": (
            "Unreviewed. Hausa source text is copied into *_ha_reviewed fields; "
            "a fluent reviewer must approve or exclude every row before model evaluation."
        ),
        "outputs": {
            "review_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
            "ids_csv": {"path": str(ids_path), "sha256": sha256(ids_path)},
        },
    }
    manifest_path = OUT / "confirmatory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
