"""Analyze OCR span evidence (ocr_features.csv) and context evidence (context_features.csv)
against gold NE-COARSE-LIT labels, and print reliability tables -- no plots here, see
plot_ocr_quality_distributions.py for the token-level OCR-quality plots (dictionary_score
counts, document_ocr_mean/sentence_ocr_mean distributions).

Per docs/phase1_manual.md SS3, a candidate (a predicted span + type from ner_features.csv)
is "reliable" (label_reliable = 1) iff it exactly matches a gold entity: same
start_token_id/end_token_id and the same type as NE-COARSE-LIT, closed into spans. This
script builds that gold-match label and reports whether OCR quality -- of the span
itself, or of the text around it -- predicts reliability.

ner_features.csv, ocr_features.csv, and context_features.csv are one row per candidate,
in the same order (verified: identical document_id/sentence_id/start_token_id/end_token_id
across all three) -- so they're joined positionally, no key merge needed.

Reliability tables (printed, no plot):
    1. Does the span's own OCR quality (ocr_correct) predict reliability?
    2. Does span_low_conf_word_fraction (finer-grained than ocr_correct) predict it?
    3. Does the OCR quality of the surrounding context (10 tokens each side) predict it,
       even though the context tokens aren't part of the span at all?

gold_type, build_gold_spans, and label_reliability now live in
gliner/label_reliability.py (imported below and re-exported from here, so
existing callers' imports of them from this module keep working unchanged); load_candidates
stays here. train_b3_logistic_regression.py, calibrate_ner_confidence.py,
plot_b3_weights.py, and analyze_data_splits.py all reuse these -- this module (plus
label_reliability.py) is the shared gold-matching entry point, not reimplemented per caller.

Usage:
    python src/analysis/analyze_ocr_context_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gliner.label_reliability import build_gold_spans, gold_type, label_reliability  # noqa: F401

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_TRAIN_DATA = DATA_DIR / "hipe2020_train_fr_train_data.csv"
DEFAULT_NER_FEATURES = DATA_DIR / "ner_features.csv"
DEFAULT_OCR_FEATURES = DATA_DIR / "ocr_features.csv"
DEFAULT_CONTEXT_FEATURES = DATA_DIR / "context_features.csv"


def load_candidates(ner_path: Path, ocr_path: Path, context_path: Path) -> pd.DataFrame:
    ner_df = pd.read_csv(ner_path)
    ocr_df = pd.read_csv(ocr_path)
    context_df = pd.read_csv(context_path)

    key_cols = ["document_id", "sentence_id", "start_token_id", "end_token_id"]
    if not ner_df[key_cols].equals(ocr_df[key_cols]) or not ner_df[key_cols].equals(context_df[key_cols]):
        raise ValueError("ner_features.csv, ocr_features.csv, and context_features.csv are not row-aligned")

    ocr_only = ocr_df.drop(columns=key_cols + ["span_text"])
    context_only = context_df.drop(columns=key_cols + ["sentence_ocr_mean"])
    return pd.concat([ner_df, ocr_only, context_only], axis=1)


def reliability_by_bucket(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    grouped = df.groupby(bucket_col, observed=True)["label_reliable"]
    return pd.DataFrame({"n_candidates": grouped.size(), "reliability_rate": grouped.mean()}).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (gold labels)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="ner_features.csv")
    parser.add_argument("--ocr-features", default=str(DEFAULT_OCR_FEATURES), help="ocr_features.csv")
    parser.add_argument("--context-features", default=str(DEFAULT_CONTEXT_FEATURES), help="context_features.csv")
    args = parser.parse_args()

    print("=== Step 1: Load train data and close gold spans ===")
    train_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    train_df["token_id"] = train_df["token_id"].astype(int)
    gold_spans = build_gold_spans(train_df)
    print(f"{len(gold_spans)} gold entity spans")

    print("=== Step 2: Load and join candidate feature tables ===")
    print(f"Loading {args.ner_features}, {args.ocr_features}, {args.context_features}")
    candidates_df = load_candidates(Path(args.ner_features), Path(args.ocr_features), Path(args.context_features))
    print(f"{len(candidates_df)} candidates")

    print("=== Step 3: Label each candidate reliable/unreliable against gold ===")
    candidates_df["label_reliable"] = label_reliability(candidates_df, gold_spans)
    overall_rate = candidates_df["label_reliable"].mean()
    print(f"Overall reliability rate: {overall_rate:.4%} ({candidates_df['label_reliable'].sum()} / {len(candidates_df)})")

    print("=== Step 4: Reliability vs span OCR correctness (ocr_correct) ===")
    ocr_correct_df = candidates_df.dropna(subset=["ocr_correct"]).copy()
    ocr_correct_df["ocr_correct"] = ocr_correct_df["ocr_correct"].map({True: "Correct", False: "Has OCR error", "True": "Correct", "False": "Has OCR error"})
    ocr_summary = reliability_by_bucket(ocr_correct_df, "ocr_correct")
    print(ocr_summary.to_string(index=False))

    print("=== Step 5: Reliability vs span_low_conf_word_fraction (finer-grained) ===")
    frac_df = candidates_df.dropna(subset=["span_low_conf_word_fraction"]).copy()
    bins = [-0.01, 0.0, 0.34, 0.67, 1.0]
    bin_labels = ["0% (all known)", "1-34%", "35-67%", "68-100%"]
    frac_df["low_conf_bucket"] = pd.cut(frac_df["span_low_conf_word_fraction"], bins=bins, labels=bin_labels)
    frac_summary = reliability_by_bucket(frac_df, "low_conf_bucket")
    print(frac_summary.to_string(index=False))

    print("=== Step 6: Reliability vs surrounding context OCR quality (10 tokens each side) ===")
    context_df = candidates_df.dropna(subset=["context_low_conf_word_fraction_10"]).copy()
    context_df["context_bucket"] = pd.cut(
        context_df["context_low_conf_word_fraction_10"], bins=bins, labels=bin_labels
    )
    context_summary = reliability_by_bucket(context_df, "context_bucket")
    print(context_summary.to_string(index=False))

    print("=== Done ===")


if __name__ == "__main__":
    main()
