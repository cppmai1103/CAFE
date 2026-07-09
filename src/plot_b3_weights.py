"""Plot Phase 1 baseline B3's fitted logistic regression weights (coefficients), one bar
per feature -- see train_b3_logistic_regression.py for what B3 is and how it's fit.

Refits B3 on the expert_train split (same as train_b3_logistic_regression.py -- fitting
itself is cheap; the slow part is loading and gold-matching candidates, see that script's
own docstring) and plots every feature's standardized coefficient as a horizontal bar,
sorted by |coefficient| descending so the most influential features are at the top.
Coefficients are on the STANDARDIZED feature scale (StandardScaler is the first step of
B3's fitted Pipeline), so they're directly comparable across features regardless of each
feature's original units/range -- a coefficient of 0.5 means "one standard deviation of
this feature" shifts the predicted logit by 0.5, for every feature alike. One-hot columns
(predicted_entity_type_*) and *_missing indicator columns are plotted the same way as any
other feature.

Sign: positive (green) pushes the predicted probability of label_reliable=1 up; negative
(red) pushes it down.

Output: figures/b3_logistic_regression_weights.png

Usage:
    python plot_b3_weights.py
    python plot_b3_weights.py --top-n 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_ocr_context_features import DEFAULT_CONTEXT_FEATURES, DEFAULT_NER_FEATURES, DEFAULT_OCR_FEATURES, DEFAULT_TRAIN_DATA
from train_b3_logistic_regression import build_feature_matrix, fit_b3_model, load_labeled_candidates

DEFAULT_FIGURES_DIR = Path(__file__).parent.parent / "figures"

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def plot_weights(coefs: pd.Series, out_path: Path, top_n: int | None = None) -> None:
    """coefs: feature name -> standardized coefficient, any order. Plots a horizontal bar
    per feature, sorted by |coefficient| descending (largest at top), colored by sign."""
    ordered = coefs.reindex(coefs.abs().sort_values(ascending=True).index)
    if top_n is not None:
        ordered = ordered.iloc[-top_n:]

    colors = [STATUS_GOOD if v >= 0 else STATUS_CRITICAL for v in ordered]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(ordered))), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    bars = ax.barh(ordered.index, ordered.values, color=colors, alpha=0.9)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in ordered.values], fontsize=8, color=MUTED_INK, padding=3)

    ax.axvline(0, color=PRIMARY_INK, linewidth=0.8)
    ax.set_xlabel("Standardized coefficient (logit scale)", color=PRIMARY_INK)
    ax.set_title("B3 logistic regression weights", color=PRIMARY_INK)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (gold labels, split)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="ner_features.csv")
    parser.add_argument("--ocr-features", default=str(DEFAULT_OCR_FEATURES), help="ocr_features.csv")
    parser.add_argument("--context-features", default=str(DEFAULT_CONTEXT_FEATURES), help="context_features.csv")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    parser.add_argument(
        "--top-n", type=int, default=None, help="Only plot the N largest-|coefficient| features (default: all)"
    )
    args = parser.parse_args()

    print("=== Step 1: Load train data, gold spans, and joined candidate features ===")
    candidates_df = load_labeled_candidates(args.train_data, args.ner_features, args.ocr_features, args.context_features)

    print("=== Step 2: Build B3 feature matrix and fit on expert_train ===")
    expert_train_df = candidates_df[candidates_df["split"] == "expert_train"]
    X_train, _ = build_feature_matrix(expert_train_df)
    y_train = expert_train_df["label_reliable"].astype(int)
    model = fit_b3_model(X_train, y_train)
    coefs = pd.Series(model.named_steps["logreg"].coef_[0], index=X_train.columns)
    print(f"Fit on {len(X_train)} expert_train candidates, {len(coefs)} features")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).to_string())

    print("=== Step 3: Plot weights ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "b3_logistic_regression_weights.png"
    plot_weights(coefs, out_path, top_n=args.top_n)
    print(f"Saved {out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
