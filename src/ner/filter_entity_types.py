"""Filter ner_features.csv down to a subset of predicted_entity_type values, before
deduplication -- so the dropped types' candidates never get a chance to win a span-overlap
conflict during dedup either (unlike filtering after dedup, which would only remove rows,
not let a previously-losing candidate of a kept type win the slot instead).

Motivating use case (see docs/running.md SS2, "match_entities" ablation): hipe2020_fr's
GLiNER2 extraction used all 5 HIPE types (PERS/LOC/ORG/TIME/PROD), but letemps_fr's own
GLiNER2 extraction only ever asked for PERS/LOC/ORG (its own labels.json), since that's all
its gold annotation covers. Training the reliability model on hipe2020_fr's full 5-type
candidate set means it learns from TIME/PROD's very different base rate/difficulty, which
letemps_fr can never exhibit -- filtering hipe2020_fr's raw candidates down to PERS/LOC/ORG
*before* dedup produces a hipe-trained model whose input distribution actually matches what
letemps_fr looks like, isolating whether that mismatch (rather than the OCR/domain shift
itself) is what breaks cross-dataset calibration transfer.

Usage:
    python src/ner/filter_entity_types.py --keep-types PERS LOC ORG \
        --ner-features data/hipe2020_fr/gliner/data_baseline/ner_features.csv \
        --out data/hipe2020_fr/gliner/data_baseline/match_entities/ner_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_NER_FEATURES = REPO_ROOT / "data" / "data_baseline" / "ner_features.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "data_baseline" / "ner_features_filtered.csv"


def filter_entity_types(candidates_df: pd.DataFrame, keep_types: list[str]) -> pd.DataFrame:
    return candidates_df[candidates_df["predicted_entity_type"].isin(keep_types)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="ner_features.csv (span-level candidates, pre-dedup)")
    parser.add_argument("--keep-types", nargs="+", required=True, help="predicted_entity_type values to keep, e.g. PERS LOC ORG")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Filtered candidates output CSV path")
    args = parser.parse_args()

    print("=== Step 1: Load ner_features.csv ===")
    print(f"Loading {args.ner_features}")
    candidates_df = pd.read_csv(args.ner_features)
    print(f"{len(candidates_df)} candidates loaded")
    print(candidates_df["predicted_entity_type"].value_counts().to_string())

    print(f"\n=== Step 2: Keep only {args.keep_types} ===")
    filtered_df = filter_entity_types(candidates_df, args.keep_types)
    print(f"{len(filtered_df)} candidates kept ({len(candidates_df) - len(filtered_df)} removed)")
    print(filtered_df["predicted_entity_type"].value_counts().to_string())

    print("\n=== Step 3: Save filtered candidates ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
