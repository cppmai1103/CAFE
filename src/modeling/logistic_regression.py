"""Fit Phase 1 baseline B3 (logistic regression over manual features) on
logistic_regression_data.csv (produced by feature_extraction/prepare_data_logistic.py,
which already joins deduplicated NER + OCR + context features and attaches
reliability_score + split) -- score every candidate, and save both the calibrated
scores and a weights plot.

build_feature_matrix/fit_b3_model/plot_weights are inlined here (reconstructed from the
now-deleted train_b3_logistic_regression.py / plot_b3_weights.py -- see docs/phase1_manual_features.md
SS4.1-4.3 for the feature groups) rather than imported, since this script has no
dependency on the raw-joining path (analyze_ocr_context_features.py) those files needed --
it loads directly from the already-prepared logistic_regression_data.csv.

Split roles (document-level train/val/test, see preprocessing/preprocessing_data.py):
    train -- B3's coefficients are fit here (imputation medians and the missing-indicator
        column set are also derived from train only, then reused as-is on every other
        split -- see build_feature_matrix's docstring).
    val   -- watched every epoch to catch overfitting and pick the best epoch (early
        stopping); never used to fit.
    test  -- not touched until the final scoring pass in Step 5.

Fit iteratively (training_curve.fit_logistic_with_curve) so a train/val loss curve can be
tracked and early-stopped -- unlike B1's 2-parameter fit, B3 has ~20+ parameters (one per
feature, plus one-hot columns) fit on only the train split, so overfitting is a real risk
here, not just a formality.

Output:
    logistic_regression.csv -- document_id, sentence_id, start_token_id, end_token_id,
        split, ner_score, calibrated_score (one row per candidate, every split) -- same
        shape as platt_scaling.csv; the join-key + calibrated_score columns are exactly
        what modeling/plot_reliability_diagram.py's --logistic-score expects, split and
        ner_score are extra.
    logistic_regression_weights.png -- standardized coefficients, one bar per feature,
        sorted by |coefficient| descending.
    logistic_regression_track_training.png -- train vs. val log loss per epoch, with the
        best (early-stopped) epoch marked.

Usage:
    python src/modeling/logistic_regression.py
    python src/modeling/logistic_regression.py --data data/data_baseline/logistic_regression_data.csv --out data/data_baseline/logistic_regression.csv --figures-dir figures/modeling/train_tracking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction.prepare_data_logistic import DEFAULT_OUT as DEFAULT_DATA
from gliner.extract_ner_features import LABELS
from training_curve import fit_logistic_with_curve, plot_training_curve

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_OUT = DATA_DIR / "logistic_regression.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "modeling" / "train_tracking"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]

# Feature groups (docs/phase1_manual_features.md SS4.1-4.3). top1_top2_type_margin and
# type_entropy (also SS4.1) aren't included -- GLiNER2 scores each entity type as an
# independent sigmoid, so there's no shared softmax distribution across types to compute
# a margin or entropy from.
NUMERIC_FEATURES = (
    "ner_score", "span_length_tokens", "span_length_characters",
    "span_ocr_mean", "span_low_conf_word_fraction", "span_first_word_ocr", "span_last_word_ocr",
    "sentence_ocr_mean", "document_ocr_mean",
    "left_context_ocr_mean_10", "right_context_ocr_mean_10", "context_ocr_min_10",
    "context_low_conf_word_fraction_10", "sentence_length", "context_window_length",
)
BOOLEAN_FEATURES = ("ocr_correct", "sentence_chunked")
CATEGORICAL_FEATURES = ("predicted_entity_type",)

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def _as_float_bool(series: pd.Series) -> pd.Series:
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})


def build_feature_matrix(candidates_df: pd.DataFrame, fit_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Build B3's feature matrix from logistic_regression_data.csv. If `fit_stats` is
    None, both the imputation medians AND which features get a `_missing` indicator
    column are derived from candidates_df itself (the train-split call site).
    Otherwise the given `fit_stats` (computed on train) are reused as-is on every
    other split -- medians, so no other split's own distribution leaks into its
    imputation, and *which* features get a `_missing` column, so a feature that happens
    to have zero NaNs in one split (e.g. test) but not another (e.g. train) can't
    silently change the output's column count/order and break the fitted model's
    predict_proba."""
    out = pd.DataFrame(index=candidates_df.index)

    if fit_stats is None:
        medians = {feat: candidates_df[feat].median() for feat in NUMERIC_FEATURES}
        missing_features = [feat for feat in NUMERIC_FEATURES if candidates_df[feat].isna().any()]
        fit_stats = {"medians": medians, "missing_features": missing_features}

    for feat in NUMERIC_FEATURES:
        values = candidates_df[feat]
        if feat in fit_stats["missing_features"]:
            out[f"{feat}_missing"] = values.isna().astype(float)
        out[feat] = values.fillna(fit_stats["medians"][feat])

    for feat in BOOLEAN_FEATURES:
        out[feat] = _as_float_bool(candidates_df[feat]).fillna(0.0)

    for feat in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(pd.Categorical(candidates_df[feat], categories=LABELS), prefix=feat, dtype=float)
        dummies.index = candidates_df.index
        out = pd.concat([out, dummies], axis=1)

    return out, fit_stats


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


def fit_b3_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
    max_epochs: int = 200, patience: int = 15,
) -> tuple[StandardScaler, LogisticRegression, list[float], list[float], int]:
    """Fits StandardScaler on X_train only, then a LogisticRegression fit iteratively
    (training_curve.fit_logistic_with_curve) on the scaled features, early-stopped on
    (X_val, y_val). Returns (scaler, model, train_losses, val_losses, best_epoch)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1, warm_start=True)
    model, train_losses, val_losses, best_epoch = fit_logistic_with_curve(
        model, X_train_scaled, y_train, X_val_scaled, y_val,
        max_epochs=max_epochs, patience=patience,
        desc="Fitting B3 logistic regression (train/val)",
    )
    return scaler, model, train_losses, val_losses, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="logistic_regression_data.csv (see feature_extraction/prepare_data_logistic.py)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path (calibrated_score for every candidate)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the weights/training-curve plots into")
    parser.add_argument("--top-n", type=int, default=None, help="Only plot the N largest-|coefficient| features (default: all)")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max training epochs before stopping regardless of val loss")
    parser.add_argument("--patience", type=int, default=15, help="Stop early after this many epochs with no val-loss improvement")
    args = parser.parse_args()

    print("=== Step 1: Load logistic_regression_data.csv ===")
    print(f"Loading {args.data}")
    candidates_df = pd.read_csv(args.data)
    print(f"{len(candidates_df)} candidates")
    print(candidates_df["split"].value_counts().to_string())

    print("=== Step 2: Build B3 feature matrix (train medians + missing-indicator set) ===")
    train_mask = candidates_df["split"] == "train"
    val_mask = candidates_df["split"] == "val"
    X_train, fit_stats = build_feature_matrix(candidates_df[train_mask])
    y_train = candidates_df.loc[train_mask, "reliability_score"].astype(int)
    X_val, _ = build_feature_matrix(candidates_df[val_mask], fit_stats=fit_stats)
    y_val = candidates_df.loc[val_mask, "reliability_score"].astype(int)
    print(f"{X_train.shape[1]} features, {len(X_train)} train candidates, {len(X_val)} val candidates")

    print("=== Step 3: Fit B3 logistic regression on train, early-stop on val ===")
    scaler, model, train_losses, val_losses, best_epoch = fit_b3_model(
        X_train, y_train, X_val, y_val, max_epochs=args.max_epochs, patience=args.patience,
    )
    coefs = pd.Series(model.coef_[0], index=X_train.columns)
    print("Top coefficients (standardized scale, sorted by |coefficient|):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).to_string())

    print("=== Step 4: Score every candidate (all splits) ===")
    X_all, _ = build_feature_matrix(candidates_df, fit_stats=fit_stats)
    candidates_df["calibrated_score"] = model.predict_proba(scaler.transform(X_all))[:, 1]
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

    print("=== Step 7: Plot train/val training curve ===")
    curve_out_path = figures_dir / "logistic_regression_track_training.png"
    plot_training_curve(
        train_losses, val_losses, best_epoch,
        "B3 logistic regression: train vs val loss", curve_out_path,
    )
    print(f"Saved {curve_out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
