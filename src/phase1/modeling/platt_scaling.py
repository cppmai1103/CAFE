"""Fit Platt scaling (sigmoid calibration) on ner_score, then score every candidate (all
splits) with the fitted model.

Split roles (document-level train/val/test, see preprocessing/preprocessing_data.py):
    train -- Platt scaling's 2 parameters (b0, b1) are fit here.
    val   -- watched every epoch to catch overfitting and pick the best epoch (early
        stopping); never used to fit.
    test  -- only used for the final held-out fit-quality plot.

Input: --label-reliability (default: label_reliability_type_only.csv, see
gliner/label_reliability.py) for ner_score + reliability_score, joined with
--load-data's document-level split assignment.

Platt scaling is exactly a 1-feature logistic regression of the binary label
(reliability_score) on the raw score (ner_score): calibrated_score = sigmoid(b0 + b1 *
ner_score). Fit iteratively (training_curve.fit_logistic_with_curve) so a train/val loss
curve can be tracked and early-stopped, even though a 2-parameter model is unlikely to
overfit in practice -- see that module's docstring for why sklearn needs the warm_start
trick to expose per-epoch loss at all.

Output:
    data_baseline/test_results/platt_scaling.csv -- document_id, sentence_id,
        start_token_id, end_token_id, split, ner_score, calibrated_score, test split only
        (docs/pipeline.md SS1: "test: final evaluation only" -- same convention
        phase2/phase2_simple/phase2_expert's evaluate.py already default to; kept in its
        own test_results/ subfolder, name unchanged, rather than a _test filename suffix)
        -- the join-key + calibrated_score columns are exactly the shape
        modeling/plot_reliability_diagram.py's --platt-scaling-score expects; split and
        ner_score are extra (not used by that loader) but kept for anyone inspecting the
        file directly. plot_reliability_diagram.py re-derives each candidate's split from
        --load-data rather than trusting this file's own split column, so a test-only
        file merges in cleanly (see that module's load_and_merge docstring).
    platt_scaling_fit.png -- the fitted sigmoid curve, with the test split's empirical
        per-bin accuracy overlaid so the fit's held-out generalization can be checked by
        eye (fitting happens on train, this plot never uses train data).
    platt_scaling_track_training.png -- train vs. val log loss per epoch, with the best
        (early-stopped) epoch marked.
    checkpoints/baseline/platt_scaling.pt -- {"b0", "b1", "model"} (the fitted sklearn
        LogisticRegression, so calibrated_score can be recomputed on new ner_score values
        without refitting -- see save_checkpoint()/load_model() below).

Usage:
    python src/phase1/modeling/platt_scaling.py
    python src/phase1/modeling/platt_scaling.py --out data/data_baseline/test_results/platt_scaling.csv --figures-dir figures/modeling/train_tracking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ner.label_reliability import default_out_path as default_label_reliability_path
from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA
from phase1.modeling.metrics import expected_calibration_error
from phase1.modeling.training_curve import fit_logistic_with_curve, plot_training_curve

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_baseline"
DEFAULT_LABEL_RELIABILITY = default_label_reliability_path("type_only")
TEST_RESULTS_DIR = DATA_DIR / "test_results"
DEFAULT_OUT = TEST_RESULTS_DIR / "platt_scaling.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "figures" / "modeling" / "train_tracking"
CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent / "checkpoints" / "baseline"
DEFAULT_CHECKPOINT_OUT = CHECKPOINTS_DIR / "platt_scaling.pt"

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


def save_checkpoint(b0: float, b1: float, model: LogisticRegression, path: str | Path) -> None:
    """b0/b1 are redundant with model (they're just model.intercept_[0]/model.coef_[0][0],
    already printed at Step 3) but kept top-level for quick inspection without unpickling
    the sklearn object. torch.save is used purely for a checkpoints/ layout/naming
    convention consistent with the Phase 2 models (see phase2_simple/model.py) -- torch's
    pickle-based serializer handles a plain dict of Python/sklearn objects fine, no tensor
    conversion needed since Platt scaling has no PyTorch model of its own."""
    torch.save({"b0": b0, "b1": b1, "model": model}, path)


def load_model(path: str | Path) -> tuple[float, float, LogisticRegression]:
    checkpoint = torch.load(path, weights_only=False)
    return checkpoint["b0"], checkpoint["b1"], checkpoint["model"]


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
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV (for the document-level split)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path (calibrated_score for every candidate)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the fit plot into")
    parser.add_argument("--checkpoint-out", default=str(DEFAULT_CHECKPOINT_OUT), help="Checkpoint output path (see save_checkpoint())")
    parser.add_argument(
        "--checkpoint-in", default=None,
        help="Score-only mode: load an already-fitted checkpoint (e.g. from a different dataset's train split) "
        "instead of fitting one here. Skips Steps 2-4 (train/val split, fit, save) and Step 8's train/val curve "
        "plot (there's no fit to plot) -- everything else runs the same, scoring --label-reliability/--load-data "
        "with the loaded model.",
    )
    args = parser.parse_args()

    print("=== Step 1: Load label_reliability and attach document-level split ===")
    print(f"Loading {args.label_reliability}")
    candidates_df = pd.read_csv(args.label_reliability)
    print(f"{len(candidates_df)} candidates loaded")

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
    candidates_df["split"] = candidates_df["document_id"].map(doc_to_split)
    print(candidates_df["split"].value_counts().to_string())

    train_losses = val_losses = best_epoch = None
    if args.checkpoint_in:
        print(f"=== Steps 2-4 skipped: loading already-fitted checkpoint {args.checkpoint_in} ===")
        b0, b1, model = load_model(args.checkpoint_in)
        print(f"calibrated_score = sigmoid({b0:.4f} + {b1:.4f} * ner_score)")
    else:
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

        print("=== Step 4: Save checkpoint ===")
        checkpoint_out_path = Path(args.checkpoint_out)
        checkpoint_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(b0, b1, model, checkpoint_out_path)
        print(f"Saved {checkpoint_out_path}")

    print("=== Step 5: Score every candidate (all splits) ===")
    candidates_df["calibrated_score"] = model.predict_proba(candidates_df[["ner_score"]].to_numpy())[:, 1]

    print("=== Step 6: Save platt_scaling.csv to test_results/ (test split only) ===")
    out_df = candidates_df.loc[candidates_df["split"] == "test", KEY_COLS + ["split", "ner_score", "calibrated_score"]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("=== Step 7: Plot fit vs held-out test-split accuracy ===")
    test_df = candidates_df[candidates_df["split"] == "test"]
    test_scores = test_df["ner_score"].to_numpy()
    test_labels = test_df["reliability_score"].to_numpy().astype(int)
    _, test_bins = expected_calibration_error(test_scores, test_labels)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fit_out_path = figures_dir / "platt_scaling_fit.png"
    plot_sigmoid_fit(b0, b1, test_bins, fit_out_path)
    print(f"Saved {fit_out_path}")

    if train_losses is not None:
        print("=== Step 8: Plot train/val training curve ===")
        curve_out_path = figures_dir / "platt_scaling_track_training.png"
        plot_training_curve(
            train_losses, val_losses, best_epoch,
            "Platt scaling: train vs val loss", curve_out_path,
        )
        print(f"Saved {curve_out_path}")
    else:
        print("=== Step 8 skipped: no fit was run (--checkpoint-in), no train/val curve to plot ===")

    print("=== Done ===")


if __name__ == "__main__":
    main()
