"""Fit Phase 1 baseline B3 (logistic regression over manual features) on
logistic_regression_data.csv (produced by feature_extraction/prepare_data_logistic.py,
which already joins deduplicated NER + OCR + context features and attaches
reliability_score + split) -- score every candidate, and save both the calibrated
scores and a weights plot.

Reuses build_feature_matrix/fit_b3_model from train_b3_logistic_regression.py (same
feature set and imputation rules -- see that module's docstring) and plot_weights from
plot_b3_weights.py, but loads directly from the already-prepared CSV rather than
re-joining deduplicate_ner_features.csv/ocr_features.csv/context_features.csv/
label_reliability.csv itself, so this script has no dependency on
analyze_ocr_context_features.py.

Fit split: expert_train (same as train_b3_logistic_regression.py).

Output:
    logistic_regression.csv -- document_id, sentence_id, start_token_id, end_token_id,
        split, ner_score, calibrated_score (one row per candidate, every split) -- same
        shape as platt_scaling.csv; the join-key + calibrated_score columns are exactly
        what modeling/plot_reliability_diagram.py's --logistic-score expects, split and
        ner_score are extra.
    logistic_regression_weights.png -- standardized coefficients, one bar per feature,
        sorted by |coefficient| descending (see plot_b3_weights.plot_weights).

Usage:
    python src/modeling/logistic_regression.py
    python src/modeling/logistic_regression.py --data data/logistic_regression_data.csv --out data/logistic_regression.csv --figures-dir figures/modeling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction.prepare_data_logistic import DEFAULT_OUT as DEFAULT_DATA
from plot_b3_weights import plot_weights
from train_b3_logistic_regression import build_feature_matrix, fit_b3_model

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "logistic_regression.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "modeling"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="logistic_regression_data.csv (see feature_extraction/prepare_data_logistic.py)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path (calibrated_score for every candidate)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the weights plot into")
    parser.add_argument("--top-n", type=int, default=None, help="Only plot the N largest-|coefficient| features (default: all)")
    args = parser.parse_args()

    print("=== Step 1: Load logistic_regression_data.csv ===")
    print(f"Loading {args.data}")
    candidates_df = pd.read_csv(args.data)
    print(f"{len(candidates_df)} candidates")
    print(candidates_df["split"].value_counts().to_string())

    print("=== Step 2: Build B3 feature matrix (expert_train medians + missing-indicator set) ===")
    train_mask = candidates_df["split"] == "expert_train"
    X_train, fit_stats = build_feature_matrix(candidates_df[train_mask])
    y_train = candidates_df.loc[train_mask, "reliability_score"].astype(int)
    print(f"{X_train.shape[1]} features, {len(X_train)} expert_train candidates")

    print("=== Step 3: Fit B3 logistic regression on expert_train ===")
    model = fit_b3_model(X_train, y_train)
    coefs = pd.Series(model.named_steps["logreg"].coef_[0], index=X_train.columns)
    print("Top coefficients (standardized scale, sorted by |coefficient|):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).to_string())

    print("=== Step 4: Score every candidate (all splits) ===")
    X_all, _ = build_feature_matrix(candidates_df, fit_stats=fit_stats)
    candidates_df["calibrated_score"] = model.predict_proba(X_all)[:, 1]
    for split, group in candidates_df.groupby("split"):
        print(f"{split}: {len(group)} candidates, mean calibrated_score {group['calibrated_score'].mean():.4f}")

    print("=== Step 5: Save logistic_regression.csv ===")
    out_df = candidates_df[KEY_COLS + ["split", "ner_score", "calibrated_score"]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("=== Step 6: Plot weights ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weights_path = figures_dir / "logistic_regression_weights.png"
    plot_weights(coefs, weights_path, top_n=args.top_n)
    print(f"Saved {weights_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
