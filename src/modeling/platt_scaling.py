"""Fit Platt scaling (sigmoid calibration) on ner_score, then score every candidate (all
splits) with the fitted model.

Split roles (document-level train/val/test, see preprocessing/preprocessing_data.py):
    train -- Platt scaling's 2 parameters (b0, b1) are fit here.
    val   -- watched every epoch to catch overfitting and pick the best epoch (early
        stopping); never used to fit.
    test  -- only used for the final held-out fit-quality plot.

Input: --label-reliability (default: label_reliability_type_only.csv, see
gliner/label_reliability.py) for ner_score + reliability_score, joined with
--train-data's document-level split assignment.

Platt scaling is exactly a 1-feature logistic regression of the binary label
(reliability_score) on the raw score (ner_score): calibrated_score = sigmoid(b0 + b1 *
ner_score). Fit iteratively (training_curve.fit_logistic_with_curve) so a train/val loss
curve can be tracked and early-stopped, even though a 2-parameter model is unlikely to
overfit in practice -- see that module's docstring for why sklearn needs the warm_start
trick to expose per-epoch loss at all.

Output:
    platt_scaling.csv -- document_id, sentence_id, start_token_id, end_token_id, split,
        ner_score, calibrated_score (one row per candidate, every split) -- the join-key +
        calibrated_score columns are exactly the shape
        modeling/plot_reliability_diagram.py's --platt-scaling-score expects; split and
        ner_score are extra (not used by that loader) but kept for anyone inspecting the
        file directly.
    platt_scaling_fit.png -- the fitted sigmoid curve, with the test split's empirical
        per-bin accuracy overlaid so the fit's held-out generalization can be checked by
        eye (fitting happens on train, this plot never uses train data).
    platt_scaling_track_training.png -- train vs. val log loss per epoch, with the best
        (early-stopped) epoch marked.

Usage:
    python src/modeling/platt_scaling.py
    python src/modeling/platt_scaling.py --out data/data_baseline/platt_scaling.csv --figures-dir figures/modeling/train_tracking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gliner.label_reliability import default_out_path as default_label_reliability_path
from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_TRAIN_DATA
from metrics import expected_calibration_error
from training_curve import fit_logistic_with_curve, plot_training_curve

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_LABEL_RELIABILITY = default_label_reliability_path("type_only")
DEFAULT_OUT = DATA_DIR / "platt_scaling.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "modeling" / "train_tracking"

CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_RED = "#e34948"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]


def fit_platt_scaling(
    train_scores: np.ndarray, train_labels: np.ndarray, val_scores: np.ndarray, val_labels: np.ndarray,
    max_epochs: int = 200, patience: int = 15,
) -> tuple[float, float, LogisticRegression, list[float], list[float], int]:
    """Platt scaling is exactly a 1-feature logistic regression of the binary label on the
    raw score: calibrated = sigmoid(B0 + B1 * score). Fit on (train_scores, train_labels),
    early-stopped on (val_scores, val_labels) -- see training_curve.fit_logistic_with_curve."""
    model = LogisticRegression(solver="lbfgs", warm_start=True, max_iter=1)
    model, train_losses, val_losses, best_epoch = fit_logistic_with_curve(
        model,
        train_scores.reshape(-1, 1), train_labels,
        val_scores.reshape(-1, 1), val_labels,
        max_epochs=max_epochs, patience=patience,
        desc="Fitting Platt scaling (train/val)",
    )
    b0 = float(model.intercept_[0])
    b1 = float(model.coef_[0][0])
    return b0, b1, model, train_losses, val_losses, best_epoch


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--label-reliability", default=str(DEFAULT_LABEL_RELIABILITY),
        help="CSV with join keys + ner_score + reliability_score (see gliner/label_reliability.py)",
    )
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (for the document-level split)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path (calibrated_score for every candidate)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the fit plot into")
    args = parser.parse_args()

    print("=== Step 1: Load label_reliability and attach document-level split ===")
    print(f"Loading {args.label_reliability}")
    candidates_df = pd.read_csv(args.label_reliability)
    print(f"{len(candidates_df)} candidates loaded")

    print(f"Loading {args.train_data}")
    train_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    doc_to_split = train_df.drop_duplicates("document_id").set_index("document_id")["split"].to_dict()
    candidates_df["split"] = candidates_df["document_id"].map(doc_to_split)
    print(candidates_df["split"].value_counts().to_string())

    print("=== Step 2: Split train / val ===")
    train_df = candidates_df[candidates_df["split"] == "train"]
    train_scores = train_df["ner_score"].to_numpy()
    train_labels = train_df["reliability_score"].to_numpy().astype(int)
    val_df = candidates_df[candidates_df["split"] == "val"]
    val_scores = val_df["ner_score"].to_numpy()
    val_labels = val_df["reliability_score"].to_numpy().astype(int)
    print(f"{len(train_df)} train candidates, {len(val_df)} val candidates")

    print("=== Step 3: Fit Platt scaling on train, early-stop on val ===")
    b0, b1, model, train_losses, val_losses, best_epoch = fit_platt_scaling(
        train_scores, train_labels, val_scores, val_labels,
    )
    print(f"calibrated_score = sigmoid({b0:.4f} + {b1:.4f} * ner_score)")

    print("=== Step 4: Score every candidate (all splits) ===")
    candidates_df["calibrated_score"] = model.predict_proba(candidates_df[["ner_score"]].to_numpy())[:, 1]

    print("=== Step 5: Save platt_scaling.csv ===")
    out_df = candidates_df[KEY_COLS + ["split", "ner_score", "calibrated_score"]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("=== Step 6: Plot fit vs held-out test-split accuracy ===")
    test_df = candidates_df[candidates_df["split"] == "test"]
    test_scores = test_df["ner_score"].to_numpy()
    test_labels = test_df["reliability_score"].to_numpy().astype(int)
    _, test_bins = expected_calibration_error(test_scores, test_labels)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fit_out_path = figures_dir / "platt_scaling_fit.png"
    plot_sigmoid_fit(b0, b1, test_bins, fit_out_path)
    print(f"Saved {fit_out_path}")

    print("=== Step 7: Plot train/val training curve ===")
    curve_out_path = figures_dir / "platt_scaling_track_training.png"
    plot_training_curve(
        train_losses, val_losses, best_epoch,
        "Platt scaling: train vs val loss", curve_out_path,
    )
    print(f"Saved {curve_out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
