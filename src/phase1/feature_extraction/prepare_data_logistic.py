"""Combine deduplicate_ner_features.csv, ocr_features.csv, context_features.csv, and
reliability_score (from gliner/label_reliability.py's output) into one row-per-candidate
table, with split (train/val/test) attached -- this is the
ready-to-use input for training and evaluating the B3 logistic regression model
(train_b3_logistic_regression.py / calibrate_ner_confidence.py), so those scripts don't
each have to re-join the feature files and re-derive the label themselves.

Join semantics: the three feature CSVs are one row per candidate, in the same order
(verified below: identical document_id/sentence_id/start_token_id/end_token_id across all
three) -- so they're joined positionally, no key merge needed. This duplicates
analyze_ocr_context_features.py's load_candidates rather than importing it, so this script
has no dependency on that file.

reliability_score: read directly from --label-reliability (default:
label_reliability_type_only.csv, see gliner/label_reliability.py --mode type_only) and
merged in on (document_id, sentence_id, start_token_id, end_token_id) -- not recomputed
here, since that file is already the authoritative, already-computed source for it.

split: each candidate's document-level split, from the train data CSV's own "split"
column (assigned by preprocessing_data.py from which official HIPE-2022 file -- train/
dev/test -- the document came from).

Usage:
    python src/phase1/feature_extraction/prepare_data_logistic.py
    python src/phase1/feature_extraction/prepare_data_logistic.py --label-reliability data/label_reliability_span_type.csv --out data/logistic_regression_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA
from ner.gliner.deduplicate_ner_features import DEFAULT_OUT as DEFAULT_NER_FEATURES
from ner.label_reliability import default_out_path as default_label_reliability_path
from phase1.feature_extraction.extract_ocr_features import DEFAULT_OUT as DEFAULT_OCR_FEATURES
from phase1.feature_extraction.extract_context_features import DEFAULT_OUT as DEFAULT_CONTEXT_FEATURES

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_baseline"
DEFAULT_LABEL_RELIABILITY = default_label_reliability_path("type_only")
DEFAULT_OUT = DATA_DIR / "logistic_regression_data.csv"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]


def join_candidate_features(ner_df: pd.DataFrame, ocr_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    """Join deduplicated NER candidates with their OCR span evidence and context
    evidence into one row-per-candidate table, positionally. Raises if the three aren't
    row-aligned (same document_id/sentence_id/start_token_id/end_token_id in the same
    order) -- e.g. if ocr_features.csv/context_features.csv were generated from a
    different deduplicate_ner_features.csv than the one passed in here."""
    if not ner_df[KEY_COLS].equals(ocr_df[KEY_COLS]) or not ner_df[KEY_COLS].equals(context_df[KEY_COLS]):
        raise ValueError("deduplicate_ner_features.csv, ocr_features.csv, and context_features.csv are not row-aligned")

    ocr_only = ocr_df.drop(columns=KEY_COLS + ["span_text"])
    context_only = context_df.drop(columns=KEY_COLS + ["sentence_ocr_mean"])
    return pd.concat([ner_df, ocr_only, context_only], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV (for the document->split map)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="Deduplicated NER candidates CSV (see gliner/deduplicate_ner_features.py)")
    parser.add_argument("--ocr-features", default=str(DEFAULT_OCR_FEATURES), help="ocr_features.csv (see extract_ocr_features.py)")
    parser.add_argument("--context-features", default=str(DEFAULT_CONTEXT_FEATURES), help="context_features.csv (see extract_context_features.py)")
    parser.add_argument("--label-reliability", default=str(DEFAULT_LABEL_RELIABILITY), help="label_reliability_*.csv (see gliner/label_reliability.py)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    args = parser.parse_args()

    print("=== Step 1: Load train data and build the document->split map ===")
    print(f"Loading {args.load_data}")
    data_df = pd.read_csv(args.load_data, dtype={"TOKEN": str, "MISC": str},
        # pandas' default NA-string sentinels ("NA", "null", "nan", ...) would otherwise
        # silently corrupt a genuine OCR token whose text happens to collide with one of
        # them (confirmed: one real token in hipe2020_fr is literally "NA") into a float
        # NaN despite the dtype=str hint above -- dtype coercion happens AFTER NA
        # detection, so it can't prevent this. keep_default_na=False turns that off
        # entirely, and na_values restores it only for the two genuinely-numeric columns
        # that still need a blank cell to become NaN.
        keep_default_na=False, na_values={"sentence_ocr_mean": [""], "document_ocr_mean": [""], "dictionary_score": [""]})
    doc_to_split = data_df.drop_duplicates("document_id").set_index("document_id")["split"].to_dict()
    print(f"{len(doc_to_split)} documents")

    print("=== Step 2: Load and join deduplicated NER + OCR + context features ===")
    print(f"Loading {args.ner_features}, {args.ocr_features}, {args.context_features}")
    ner_df = pd.read_csv(args.ner_features)
    ocr_df = pd.read_csv(args.ocr_features)
    context_df = pd.read_csv(args.context_features)
    candidates_df = join_candidate_features(ner_df, ocr_df, context_df)
    print(f"{len(candidates_df)} candidates joined")

    print("=== Step 3: Merge in reliability_score ===")
    print(f"Loading {args.label_reliability}")
    reliability_df = pd.read_csv(args.label_reliability)[KEY_COLS + ["reliability_score"]]
    before = len(candidates_df)
    candidates_df = candidates_df.merge(reliability_df, on=KEY_COLS, how="left")
    if candidates_df["reliability_score"].isna().any():
        n_missing = int(candidates_df["reliability_score"].isna().sum())
        raise ValueError(f"{n_missing} candidate(s) had no matching row in {args.label_reliability} -- is it stale relative to --ner-features?")
    assert len(candidates_df) == before, "merge changed row count -- --label-reliability isn't uniquely keyed by " + str(KEY_COLS)
    n_reliable = int(candidates_df["reliability_score"].sum())
    print(f"{n_reliable} / {len(candidates_df)} candidates reliable ({n_reliable / len(candidates_df):.4%})")

    print("=== Step 4: Attach document-level split ===")
    candidates_df["split"] = candidates_df["document_id"].map(doc_to_split)
    print(candidates_df["split"].value_counts().to_string())

    print("=== Step 5: Drop sentence_chunked (kept in ner_features.csv upstream, but not used as a B3/MLP feature) ===")
    candidates_df = candidates_df.drop(columns=["sentence_chunked"])

    print("=== Step 6: Save prepared logistic-regression data ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
