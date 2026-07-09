"""Compares Phase 1 baselines B0/B1/B3 (docs/phase1_manual.md SS11) against NE-COARSE-LIT
gold: B0 is GLiNER2's own raw confidence (ner_score), B1 is B0 after Platt scaling
(sigmoid calibration, SS10), and B3 is a logistic regression over all implemented manual
features (see train_b3_logistic_regression.py). B2 (the adaptive gate) and B4-B7 aren't
here -- they need trained experts/gate, which don't exist in this repo yet.

Fit and evaluation use separate Phase 1 splits (docs/phase1_manual.md SS6.1), assigned
per document by preprocessing_data.py's assign_splits and joined here via document_id:
    - B1's Platt scaling is fit ONLY on the "calibration" split, per SS10 ("calibrate ...
      on the calibration split").
    - B3's logistic regression is fit ONLY on the "expert_train" split (SS6.1:
      "expert_train: for training experts" -- B3 is trained like an expert would be, just
      over the combined feature set).
    - Metrics (Brier score, ECE) and both plots are computed on the held-out "test" split
      ("test: final evaluation only", SS6.1) -- never on data any baseline was fit on, so
      the reported numbers reflect generalization, not memorization.

Ground truth (label_reliable): NE-COARSE-LIT is closed into gold spans and each
candidate's (document_id, start_token_id, end_token_id, predicted_entity_type) is
exact-matched against them -- same construction as analyze_ocr_context_features.py
(reused here via train_b3_logistic_regression.load_labeled_candidates, not
reimplemented).

Platt scaling fits calibrated_r = sigmoid(B0_coef + B1_coef * ner_score) via a 1-feature
logistic regression of label_reliable on ner_score. Plots (filenames carry the baseline
numbers they cover, so it's clear at a glance which is which):
    1. b0_ner_score_distribution.png -- distribution of the raw ner_score itself (all
       candidates, reliable vs not).
    2. b0_b1_platt_scaling_fit.png -- the fitted B0->B1 sigmoid curve, with the test
       split's empirical per-bin accuracy overlaid so the fit's held-out generalization
       can be checked by eye.
    3. b0_b1_b3_metrics_bar.png -- bar chart of Brier score and ECE (test split) for B0,
       B1, and B3, side by side.
    4. b0_b1_b3_reliability_diagram.png -- a reliability diagram (test split) comparing
       B0, B1, and B3.

Caching (reliable_score.csv): fitting B1 and B3 and scoring all ~520K candidates is the
slow part of this script; the metrics/plots that follow are cheap. So the fit+score step
writes its output -- one row per candidate: document_id, sentence_id, start_token_id,
end_token_id, split, label_reliable, ner_score (B0), platt_calibrated_score (B1),
b3_score (B3) -- to --reliable-scores (default data/reliable_score.csv), plus the two
Platt coefficients to a small *_meta.json sidecar next to it (needed to redraw the
sigmoid-fit plot's curve without refitting). On the next run, if that CSV already exists,
it's loaded directly and the loading/gold-matching/fitting/scoring steps are skipped
entirely -- pass --force-recompute to redo them anyway (e.g. after regenerating
ner/ocr/context features upstream, or after changing what gets computed here).

Usage:
    python calibrate_ner_confidence.py
    python calibrate_ner_confidence.py --force-recompute
    python calibrate_ner_confidence.py --figures-dir /tmp/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from analyze_ocr_context_features import DEFAULT_CONTEXT_FEATURES, DEFAULT_NER_FEATURES, DEFAULT_OCR_FEATURES, DEFAULT_TRAIN_DATA
from train_b3_logistic_regression import build_feature_matrix, fit_b3_model, load_labeled_candidates

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RELIABLE_SCORES = DATA_DIR / "reliable_score.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent / "figures"

CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_RED = "#e34948"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"

BASELINE_COLORS = {
    "B0_raw_ner_score": CATEGORICAL_RED,
    "B1_platt_calibrated": CATEGORICAL_BLUE,
    "B3_logistic_regression": STATUS_GOOD,
}
BASELINE_LABELS = {
    "B0_raw_ner_score": "B0: raw ner_score",
    "B1_platt_calibrated": "B1: Platt-calibrated",
    "B3_logistic_regression": "B3: logistic regression (manual features)",
}


def fit_platt_scaling(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, LogisticRegression]:
    """Platt scaling is exactly a 1-feature logistic regression of the binary label on the
    raw score: calibrated = sigmoid(B0 + B1 * score)."""
    model = LogisticRegression(solver="lbfgs")
    model.fit(scores.reshape(-1, 1), labels)
    b0 = float(model.intercept_[0])
    b1 = float(model.coef_[0][0])
    return b0, b1, model


def plot_score_distribution(candidates_df: pd.DataFrame, out_path: Path) -> None:
    """Histogram of GLiNER2's raw ner_score across every candidate, split into reliable
    (matches gold) vs not, on a log-scale y-axis -- the two classes differ by ~2 orders of
    magnitude in count (label_reliable is ~1% of candidates), which a linear y-axis would
    flatten into invisibility."""
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    bins = np.linspace(0, 1, 31)
    not_reliable = candidates_df.loc[~candidates_df["label_reliable"], "ner_score"]
    reliable = candidates_df.loc[candidates_df["label_reliable"], "ner_score"]

    ax.hist(not_reliable, bins=bins, color=STATUS_CRITICAL, alpha=0.75, label=f"Not reliable (n={len(not_reliable):,})")
    ax.hist(reliable, bins=bins, color=STATUS_GOOD, alpha=0.85, label=f"Reliable (n={len(reliable):,})")

    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Raw ner_score", color=PRIMARY_INK)
    ax.set_ylabel("Candidate count (log scale)", color=PRIMARY_INK)
    ax.set_title("Distribution of GLiNER2's raw confidence score (ner_score)", color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "avg_confidence": np.nan, "accuracy": np.nan})
            continue
        avg_conf = confidences[in_bin].mean()
        acc = correct[in_bin].mean()
        ece += (count / n) * abs(avg_conf - acc)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": count, "avg_confidence": avg_conf, "accuracy": acc})
    return ece, pd.DataFrame(rows)


def plot_sigmoid_fit(b0: float, b1: float, bin_stats: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    x = np.linspace(0, 1, 200)
    y = 1 / (1 + np.exp(-(b0 + b1 * x)))
    ax.plot(x, y, color=CATEGORICAL_BLUE, linewidth=2, label="Fitted Platt curve")

    valid = bin_stats.dropna(subset=["avg_confidence"])
    ax.scatter(
        valid["avg_confidence"], valid["accuracy"], s=valid["count"] / valid["count"].max() * 300 + 20,
        color=CATEGORICAL_RED, alpha=0.85, zorder=3, label="Empirical accuracy per bin (size = # candidates)",
    )

    ax.text(
        0.03, 0.95, f"calibrated = sigmoid({b0:.3f} + {b1:.3f} × ner_score)",
        transform=ax.transAxes, fontsize=9, color=SECONDARY_INK, va="top",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.05, valid["accuracy"].max() * 1.3 if len(valid) else 0.05))
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_xlabel("Raw ner_score", color=PRIMARY_INK)
    ax.set_ylabel("Calibrated probability / empirical accuracy", color=PRIMARY_INK)
    ax.set_title("Platt scaling fit vs held-out test-split accuracy", color=PRIMARY_INK)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram(series: list[tuple[pd.DataFrame, str, str]], out_path: Path) -> None:
    """series: list of (bins_df, color, label) -- bins_df from expected_calibration_error,
    one per baseline being compared (B0/B1/B3). Any number of series is fine."""
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED_INK, label="Perfect calibration")

    for bins_df, color, label in series:
        valid = bins_df.dropna(subset=["avg_confidence"])
        ax.plot(valid["avg_confidence"], valid["accuracy"], color=color, linewidth=1, alpha=0.6)
        ax.scatter(
            valid["avg_confidence"], valid["accuracy"], s=valid["count"] / valid["count"].max() * 300 + 20,
            color=color, alpha=0.85, label=label,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability", color=PRIMARY_INK)
    ax.set_ylabel("Empirical accuracy (label_reliable rate)", color=PRIMARY_INK)
    ax.set_title("Reliability diagram (test split): B0 vs B1 vs B3", color=PRIMARY_INK)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: one group per metric (Brier score, ECE), one bar per baseline
    (B0/B1/B3) within each group -- same color scheme as the reliability diagram, so the
    two plots read as a matched pair."""
    baselines = [c for c in metrics_df.columns if c != "metric"]
    metrics = metrics_df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(baselines)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for i, baseline in enumerate(baselines):
        values = metrics_df[baseline].to_numpy()
        offset = (i - (len(baselines) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width,
            color=BASELINE_COLORS.get(baseline, MUTED_INK), label=BASELINE_LABELS.get(baseline, baseline), alpha=0.9,
        )
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], fontsize=8, color=MUTED_INK, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value (lower is better)", color=PRIMARY_INK)
    ax.set_title("Test-split metrics: B0 vs B1 vs B3", color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def meta_path_for(reliable_scores_path: Path) -> Path:
    return reliable_scores_path.with_name(reliable_scores_path.stem + "_meta.json")


def compute_reliable_scores(args) -> tuple[pd.DataFrame, float, float]:
    """The slow path: load candidates, fit B1 on calibration and B3 on expert_train, then
    score every candidate (all splits, not just test -- so the saved CSV is a complete,
    reusable artifact) with all three baselines. Returns (scored_df, platt_b0, platt_b1)."""
    print("=== Step 1: Load train data, gold spans, and joined candidate features ===")
    candidates_df = load_labeled_candidates(args.train_data, args.ner_features, args.ocr_features, args.context_features)
    n_reliable = int(candidates_df["label_reliable"].sum())
    print(f"{len(candidates_df)} candidates; {n_reliable} reliable ({n_reliable / len(candidates_df):.4%})")

    print("=== Step 2: Split into expert_train (fit B3) / calibration (fit B0->B1) ===")
    expert_train_df = candidates_df[candidates_df["split"] == "expert_train"]
    calibration_df = candidates_df[candidates_df["split"] == "calibration"]
    print(f"{len(expert_train_df)} candidates in expert_train -- B3 is fit on these only")
    print(f"{len(calibration_df)} candidates in calibration -- B1's Platt scaling is fit on these only")

    calibration_scores = calibration_df["ner_score"].to_numpy()
    calibration_labels = calibration_df["label_reliable"].to_numpy().astype(int)

    print("=== Step 3: Fit Platt scaling B0 -> B1 (sigmoid calibration) on the calibration split ===")
    b0, b1, model = fit_platt_scaling(calibration_scores, calibration_labels)
    print(f"B0 (intercept) = {b0:.4f}")
    print(f"B1 (coefficient) = {b1:.4f}")
    print(f"calibrated_r = sigmoid({b0:.4f} + {b1:.4f} * ner_score)")

    print("=== Step 4: Fit B3 (logistic regression over manual features) on expert_train ===")
    X_train, fit_stats = build_feature_matrix(expert_train_df)
    y_train = expert_train_df["label_reliable"].astype(int)
    b3_model = fit_b3_model(X_train, y_train)
    print(f"B3 fit on {len(X_train)} expert_train candidates, {X_train.shape[1]} features")

    print("=== Step 5: Score every candidate (all splits) with B0/B1/B3 ===")
    X_all, _ = build_feature_matrix(candidates_df, fit_stats=fit_stats)
    scored_df = candidates_df[
        ["document_id", "sentence_id", "start_token_id", "end_token_id", "split", "label_reliable", "ner_score"]
    ].copy()
    scored_df["platt_calibrated_score"] = model.predict_proba(candidates_df[["ner_score"]].to_numpy())[:, 1]
    scored_df["b3_score"] = b3_model.predict_proba(X_all)[:, 1]

    return scored_df, b0, b1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (gold labels, split)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="ner_features.csv")
    parser.add_argument("--ocr-features", default=str(DEFAULT_OCR_FEATURES), help="ocr_features.csv (needed for B3)")
    parser.add_argument("--context-features", default=str(DEFAULT_CONTEXT_FEATURES), help="context_features.csv (needed for B3)")
    parser.add_argument(
        "--reliable-scores",
        default=str(DEFAULT_RELIABLE_SCORES),
        help="Cached per-candidate B0/B1/B3 scores; loaded if present instead of refitting",
    )
    parser.add_argument(
        "--force-recompute", action="store_true", help="Refit B1/B3 and rescore even if --reliable-scores already exists"
    )
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save plots into")
    args = parser.parse_args()

    reliable_scores_path = Path(args.reliable_scores)
    meta_path = meta_path_for(reliable_scores_path)

    if reliable_scores_path.exists() and meta_path.exists() and not args.force_recompute:
        print(f"=== Loading cached reliable scores from {reliable_scores_path} (--force-recompute to refit) ===")
        scored_df = pd.read_csv(reliable_scores_path)
        with open(meta_path) as f:
            meta = json.load(f)
        b0, b1 = meta["platt_b0"], meta["platt_b1"]
        print(f"{len(scored_df)} candidates loaded; calibrated_r = sigmoid({b0:.4f} + {b1:.4f} * ner_score)")
    else:
        scored_df, b0, b1 = compute_reliable_scores(args)
        reliable_scores_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(reliable_scores_path, index=False)
        with open(meta_path, "w") as f:
            json.dump({"platt_b0": b0, "platt_b1": b1}, f)
        print(f"Saved {reliable_scores_path} and {meta_path}")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Plot distribution of raw ner_score (all candidates) ===")
    plot_score_distribution(scored_df, figures_dir / "b0_ner_score_distribution.png")
    print(f"Saved {figures_dir / 'b0_ner_score_distribution.png'}")

    test_df = scored_df[scored_df["split"] == "test"]
    print(f"{len(test_df)} candidates in test -- all metrics/plots below are computed on these only")
    test_scores = test_df["ner_score"].to_numpy()
    test_calibrated = test_df["platt_calibrated_score"].to_numpy()
    test_b3 = test_df["b3_score"].to_numpy()
    test_labels = test_df["label_reliable"].to_numpy().astype(int)

    print("=== Held-out metrics on the test split -- B0 vs B1 vs B3 (Brier score, ECE) ===")
    raw_ece, raw_bins = expected_calibration_error(test_scores, test_labels)
    cal_ece, cal_bins = expected_calibration_error(test_calibrated, test_labels)
    b3_ece, b3_bins = expected_calibration_error(test_b3, test_labels)
    metrics_df = pd.DataFrame(
        {
            "metric": ["Brier score", "ECE"],
            "B0_raw_ner_score": [brier_score_loss(test_labels, test_scores), raw_ece],
            "B1_platt_calibrated": [brier_score_loss(test_labels, test_calibrated), cal_ece],
            "B3_logistic_regression": [brier_score_loss(test_labels, test_b3), b3_ece],
        }
    )
    print(metrics_df.to_string(index=False))

    print("=== Plot the fitted B0->B1 sigmoid curve against test-split empirical bins ===")
    plot_sigmoid_fit(b0, b1, raw_bins, figures_dir / "b0_b1_platt_scaling_fit.png")
    print(f"Saved {figures_dir / 'b0_b1_platt_scaling_fit.png'}")

    print("=== Plot metrics bar chart (test split: B0 vs B1 vs B3) ===")
    plot_metrics_bar(metrics_df, figures_dir / "b0_b1_b3_metrics_bar.png")
    print(f"Saved {figures_dir / 'b0_b1_b3_metrics_bar.png'}")

    print("=== Plot reliability diagram (test split: B0 vs B1 vs B3) ===")
    plot_reliability_diagram(
        [
            (raw_bins, CATEGORICAL_RED, BASELINE_LABELS["B0_raw_ner_score"]),
            (cal_bins, CATEGORICAL_BLUE, BASELINE_LABELS["B1_platt_calibrated"]),
            (b3_bins, STATUS_GOOD, BASELINE_LABELS["B3_logistic_regression"]),
        ],
        figures_dir / "b0_b1_b3_reliability_diagram.png",
    )
    print(f"Saved {figures_dir / 'b0_b1_b3_reliability_diagram.png'}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
