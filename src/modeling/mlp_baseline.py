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
    mlp_baseline.csv -- document_id, sentence_id, start_token_id, end_token_id, split,
        ner_score, calibrated_score (one row per candidate, every split) -- same shape as
        platt_scaling.csv/logistic_regression.csv; ready for
        plot_reliability_diagram.py's --mlp-score.
    mlp_baseline_track_training.png -- train vs. val log loss per epoch, with the best
        (early-stopped) epoch marked.

Usage:
    python src/modeling/mlp_baseline.py
    python src/modeling/mlp_baseline.py --hidden-dim 32 --dropout 0.1 --max-epochs 100 --patience 10
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction.prepare_data_logistic import DEFAULT_OUT as DEFAULT_DATA
from logistic_regression import build_feature_matrix
from training_curve import plot_training_curve

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_OUT = DATA_DIR / "mlp_baseline.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "modeling" / "train_tracking"

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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load logistic_regression_data.csv ===")
    print(f"Loading {args.data}")
    candidates_df = pd.read_csv(args.data)
    print(f"{len(candidates_df)} candidates")
    print(candidates_df["split"].value_counts().to_string())

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

    print("=== Step 5: Score every candidate (all splits) ===")
    X_all_df, _ = build_feature_matrix(candidates_df, fit_stats=fit_stats)
    X_all = scaler.transform(X_all_df).astype("float32")
    with torch.no_grad():
        logits = model(torch.tensor(X_all, dtype=torch.float32, device=device))
        candidates_df["calibrated_score"] = torch.sigmoid(logits).cpu().numpy()
    for split, group in candidates_df.groupby("split"):
        print(f"{split}: {len(group)} candidates, mean calibrated_score {group['calibrated_score'].mean():.4f}")

    print("=== Step 6: Save mlp_baseline.csv ===")
    out_df = candidates_df[KEY_COLS + ["split", "ner_score", "calibrated_score"]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("=== Step 7: Plot train/val training curve ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    curve_out_path = figures_dir / "mlp_baseline_track_training.png"
    plot_training_curve(
        train_losses, val_losses, best_epoch,
        "MLP baseline: train vs val loss", curve_out_path,
    )
    print(f"Saved {curve_out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
