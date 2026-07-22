"""Shared train/val loss-curve fitting for the sklearn-based Phase 1 baselines (B1 Platt
scaling, B3 logistic regression) -- both are just LogisticRegression with a different
feature set, so both overfitting-control needs are identical: fit on train, watch loss on
val (never on test), stop when val loss stops improving.

sklearn's LogisticRegression has no built-in per-iteration callback, so
fit_logistic_with_curve fakes one: it calls .fit() repeatedly with warm_start=True and
max_iter=1, so each call resumes from the previous call's coefficients and advances the
solver by roughly one step. After each step it records log loss on both X_train and
X_val, and keeps a copy of the coefficients from whichever step had the lowest val loss
so far. If val loss hasn't improved for `patience` steps in a row, training stops early.
The returned model's coefficients are the BEST step's, not the last step's -- so a model
that started overfitting after step 40 doesn't get returned as its step-200 self.
"""

from __future__ import annotations

import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tqdm import tqdm

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_RED = "#e34948"


def fit_logistic_with_curve(
    model: LogisticRegression,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
    patience: int = 15,
    desc: str = "Fitting",
) -> tuple[LogisticRegression, list[float], list[float], int]:
    """`model` must already be constructed with warm_start=True, max_iter=1 (the caller
    owns hyperparameters like C/solver). Returns (model restored to its best epoch,
    train_losses, val_losses, best_epoch) -- train_losses/val_losses have one entry per
    epoch actually run (shorter than max_epochs if early stopping fired)."""
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    epochs_without_improvement = 0

    progress = tqdm(range(1, max_epochs + 1), desc=desc, unit="epoch")
    for epoch in progress:
        with warnings.catch_warnings():
            # max_iter=1 by design (one step per epoch) -- lbfgs "didn't converge" every
            # single epoch is expected noise here, not a real convergence problem.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        train_loss = log_loss(y_train, model.predict_proba(X_train)[:, 1])
        val_loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        progress.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", best_epoch=best_epoch)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.__dict__)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}: no val-loss improvement for {patience} epochs")
                break

    print(f"Best epoch: {best_epoch} (train_loss={train_losses[best_epoch - 1]:.4f}, val_loss={best_val_loss:.4f})")
    model.__dict__.update(best_state)
    return model, train_losses, val_losses, best_epoch


def plot_training_curve(
    train_losses: list[float], val_losses: list[float], best_epoch: int, title: str, out_path,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.plot(epochs, train_losses, color=CATEGORICAL_BLUE, linewidth=2, label="Train loss")
    ax.plot(epochs, val_losses, color=CATEGORICAL_RED, linewidth=2, label="Val loss")
    ax.axvline(best_epoch, color=MUTED_INK, linestyle="--", linewidth=1.2, label=f"Best epoch ({best_epoch})")

    ax.set_xlabel("Epoch", color=PRIMARY_INK)
    ax.set_ylabel("Log loss (binary cross-entropy)", color=PRIMARY_INK)
    ax.set_title(title, color=PRIMARY_INK)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)
