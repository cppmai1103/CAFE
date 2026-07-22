"""Fit a small MLP baseline over the same manual features B3 (logistic_regression.py)
uses, on logistic_regression_data.csv -- score every candidate, comparable to B3 via the
same plot_reliability_diagram.py comparison.

Same feature matrix as B3 (build_feature_matrix, imported from logistic_regression.py --
see that module for the exact feature groups and imputation rules) and the same
StandardScaler-then-fit shape, but the model itself is a 2-layer MLP instead of a linear
model:

    Linear(d, hidden_dim) -> ReLU -> Dropout(dropout) -> Linear(hidden_dim, 1)

(the same expert-MLP shape spec'd in docs/phase1_manual_features.md's shared expert-gate formula,
applied here over the full concatenated feature vector rather than one evidence group).

Why PyTorch and not sklearn's MLPClassifier: sklearn has no dropout support at all, and
training_curve.py's fit_logistic_with_curve already had to fake per-epoch checkpoints for
sklearn's LogisticRegression via a warm_start/max_iter=1 trick (see that module's
docstring) since sklearn's fit() doesn't expose per-iteration state. An MLP is more prone
to overfitting than a ~20-parameter linear model, so real epochs and real dropout (active
during the training forward pass, off during eval) matter more here, not less -- PyTorch
gives both natively. torch is already provisioned by script.sh (installed for GLiNER2),
so this adds no new dependency.

Split roles (same as B1/B3 -- document-level train/val/test, see
preprocessing/preprocessing_data.py):
    train -- the MLP's weights are fit here (StandardScaler and the feature matrix's
        imputation medians/missing-indicator set are also derived from train only, then
        reused as-is on every other split).
    val   -- real forward passes (dropout off) every epoch, used only for early stopping;
        never backpropagated through.
    test  -- not touched until the final scoring pass.

Training loss (used for the gradient step) is computed in train mode (dropout active);
the train_loss logged and plotted is a separate eval-mode (dropout off) forward pass over
the same data, so the train/val curve compares two numbers on equal footing instead of
letting dropout noise make the train curve look artificially jumpy.

No class-balancing is applied (matches B3's rationale, see logistic_regression.py) -- the
model's whole purpose is to produce a probability that means what it says.

Output:
    data_baseline/test_results/mlp_baseline.csv -- document_id, sentence_id,
        start_token_id, end_token_id, split, ner_score, calibrated_score, test split only
        (docs/pipeline.md SS1: "test: final evaluation only"; kept in its own
        test_results/ subfolder, name unchanged, rather than a _test filename suffix) --
        same shape as platt_scaling.csv/logistic_regression.csv; ready for
        plot_reliability_diagram.py's --mlp-score.
    mlp_baseline_track_training.png -- train vs. val log loss per epoch, with the best
        (early-stopped) epoch marked.
    checkpoints/baseline/mlp_baseline.pt -- {"state_dict", "config" (MLPBaseline's
        constructor kwargs), "scaler", "fit_stats", "feature_names"}, everything
        build_feature_matrix + scaler.transform + MLPBaseline.forward need to reproduce
        calibrated_score on new candidates without refitting (see
        save_checkpoint()/load_model() below, same shape as phase2_simple/model.py's).

Usage:
    python src/phase1/modeling/mlp_baseline.py
    python src/phase1/modeling/mlp_baseline.py --hidden-dim 32 --dropout 0.1 --max-epochs 100 --patience 10
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase1.feature_extraction.prepare_data_logistic import DEFAULT_OUT as DEFAULT_DATA
from phase1.modeling.logistic_regression import build_feature_matrix
from phase1.modeling.training_curve import plot_training_curve

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_baseline"
TEST_RESULTS_DIR = DATA_DIR / "test_results"
DEFAULT_OUT = TEST_RESULTS_DIR / "mlp_baseline.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "figures" / "modeling" / "train_tracking"
CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent / "checkpoints" / "baseline"
DEFAULT_CHECKPOINT_OUT = CHECKPOINTS_DIR / "mlp_baseline.pt"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits, shape (batch,)

    def config(self) -> dict:
        return {"input_dim": self.net[0].in_features, "hidden_dim": self.net[0].out_features, "dropout": self.net[2].p}


def save_checkpoint(
    model: MLPBaseline, scaler: StandardScaler, fit_stats: dict, feature_names: list[str], path: str | Path,
) -> None:
    """scaler + fit_stats (build_feature_matrix's train-derived imputation medians and
    missing-indicator column set) + feature_names (column order scaler/model were fit on)
    are all required to reproduce calibrated_score on new candidates -- without them, a
    freshly-loaded model can't be fed a matching feature matrix. Same
    {"state_dict", "config"} shape as phase2_simple/model.py's save_checkpoint, plus the
    3 extra fields above (phase2_simple has no scaler/imputation step of its own, since
    its input is raw text, not manual features)."""
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config(), "scaler": scaler, "fit_stats": fit_stats, "feature_names": feature_names},
        path,
    )


def load_model(path: str | Path, device: str = "cpu") -> tuple[MLPBaseline, StandardScaler, dict, list[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MLPBaseline(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["scaler"], checkpoint["fit_stats"], checkpoint["feature_names"]


def fit_mlp_with_curve(
    model: nn.Module,
    X_train, y_train, X_val, y_val,
    max_epochs: int = 200, patience: int = 10, lr: float = 1e-3, weight_decay: float = 0.0,
    device: str = "cpu",
) -> tuple[nn.Module, list[float], list[float], int]:
    """Real per-epoch training loop (unlike training_curve.fit_logistic_with_curve's
    warm_start fake) -- one full-batch gradient step per epoch, train/val loss logged in
    eval mode (dropout off) after each step. Keeps a deep copy of the state_dict from
    whichever epoch had the lowest val loss so far, and restores it if training stops
    early (patience epochs with no val-loss improvement) or hits max_epochs."""
    model = model.to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    epochs_without_improvement = 0

    progress = tqdm(range(1, max_epochs + 1), desc="Fitting MLP baseline (train/val)", unit="epoch")
    for epoch in progress:
        model.train()
        optimizer.zero_grad()
        loss_fn(model(X_train_t), y_train_t).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train_t), y_train_t).item()
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        progress.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", best_epoch=best_epoch)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}: no val-loss improvement for {patience} epochs")
                break

    print(f"Best epoch: {best_epoch} (train_loss={train_losses[best_epoch - 1]:.4f}, val_loss={best_val_loss:.4f})")
    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="logistic_regression_data.csv (see feature_extraction/prepare_data_logistic.py)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path (calibrated_score for every candidate)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the training-curve plot into")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate after the hidden layer's ReLU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 weight decay")
    parser.add_argument(
        "--max-epochs", type=int, default=2000,
        help="Max training epochs before stopping regardless of val loss -- full-batch Adam needs far more steps "
        "than B1/B3's L-BFGS-based fit_logistic_with_curve to converge on this data (~1100-1200 in practice)",
    )
    parser.add_argument("--patience", type=int, default=10, help="Stop early after this many epochs with no val-loss improvement")
    parser.add_argument("--seed", type=int, default=42, help="torch random seed (weight init, dropout masks)")
    parser.add_argument("--checkpoint-out", default=str(DEFAULT_CHECKPOINT_OUT), help="Checkpoint output path (see save_checkpoint())")
    parser.add_argument(
        "--checkpoint-in", default=None,
        help="Score-only mode: load an already-fitted checkpoint (e.g. from a different dataset's train split) "
        "instead of fitting one here. Skips Steps 2-5 (feature matrix on train/val, standardize, fit, save) and "
        "Step 8's train/val curve plot (there's no fit to plot) -- everything else runs the same, scoring --data "
        "with the loaded model/scaler/fit_stats/feature_names.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load logistic_regression_data.csv ===")
    print(f"Loading {args.data}")
    candidates_df = pd.read_csv(args.data)
    print(f"{len(candidates_df)} candidates")
    print(candidates_df["split"].value_counts().to_string())

    train_losses = val_losses = best_epoch = None
    if args.checkpoint_in:
        print(f"=== Steps 2-5 skipped: loading already-fitted checkpoint {args.checkpoint_in} ===")
        model, scaler, fit_stats, feature_names = load_model(args.checkpoint_in, device=device)
    else:
        print("=== Step 2: Build feature matrix (train medians + missing-indicator set) ===")
        train_mask = candidates_df["split"] == "train"
        val_mask = candidates_df["split"] == "val"
        X_train_df, fit_stats = build_feature_matrix(candidates_df[train_mask])
        y_train = candidates_df.loc[train_mask, "reliability_score"].astype(int)
        X_val_df, _ = build_feature_matrix(candidates_df[val_mask], fit_stats=fit_stats)
        y_val = candidates_df.loc[val_mask, "reliability_score"].astype(int)
        print(f"{X_train_df.shape[1]} features, {len(X_train_df)} train candidates, {len(X_val_df)} val candidates")

        print("=== Step 3: Standardize features (fit on train only) ===")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df).astype("float32")
        X_val = scaler.transform(X_val_df).astype("float32")

        print("=== Step 4: Fit MLP on train, early-stop on val ===")
        model = MLPBaseline(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
        model, train_losses, val_losses, best_epoch = fit_mlp_with_curve(
            model, X_train, y_train.to_numpy(), X_val, y_val.to_numpy(),
            max_epochs=args.max_epochs, patience=args.patience, lr=args.lr, weight_decay=args.weight_decay,
            device=device,
        )
        feature_names = list(X_train_df.columns)

        print("=== Step 5: Save checkpoint ===")
        checkpoint_out_path = Path(args.checkpoint_out)
        checkpoint_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, scaler, fit_stats, feature_names, checkpoint_out_path)
        print(f"Saved {checkpoint_out_path}")

    print("=== Step 6: Score every candidate (all splits) ===")
    X_all_df, _ = build_feature_matrix(candidates_df, fit_stats=fit_stats)
    X_all_df = X_all_df.reindex(columns=feature_names, fill_value=0.0)  # match the checkpoint's fitted column order, esp. important in --checkpoint-in mode on a different dataset
    X_all = scaler.transform(X_all_df).astype("float32")
    with torch.no_grad():
        logits = model(torch.tensor(X_all, dtype=torch.float32, device=device))
        candidates_df["calibrated_score"] = torch.sigmoid(logits).cpu().numpy()
    for split, group in candidates_df.groupby("split"):
        print(f"{split}: {len(group)} candidates, mean calibrated_score {group['calibrated_score'].mean():.4f}")

    print("=== Step 7: Save mlp_baseline.csv to test_results/ (test split only) ===")
    out_df = candidates_df.loc[candidates_df["split"] == "test", KEY_COLS + ["split", "ner_score", "calibrated_score"]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    if train_losses is not None:
        print("=== Step 8: Plot train/val training curve ===")
        figures_dir = Path(args.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        curve_out_path = figures_dir / "mlp_baseline_track_training.png"
        plot_training_curve(
            train_losses, val_losses, best_epoch,
            "MLP baseline: train vs val loss", curve_out_path,
        )
        print(f"Saved {curve_out_path}")
    else:
        print("=== Step 8 skipped: no fit was run (--checkpoint-in), no train/val curve to plot ===")

    print("=== Done ===")


if __name__ == "__main__":
    main()
