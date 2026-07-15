"""Diagnostics for phase2_expert's latent MoE head (docs/phase2_learned_features.md SS25/26). Two
independent questions, from two different sources:

1. Gate usage -- from a scores CSV (evaluate.py, which records the gate's per-candidate
   alpha_0..alpha_{K-1} weights alongside calibrated_score):
   a. Distribution -- for each expert, what's the spread of alpha values it receives
      across candidates? A violin plot per expert
      (figures/phase2_expert/expert_alpha_distribution.png). If one expert's alpha sits
      near 0 for almost every candidate, the gate has effectively stopped using it (dead
      expert); if all K distributions look identical, the gate isn't discriminating
      between experts at all.
   b. Pairwise correlation -- Pearson correlation between every pair of experts' alpha_k
      columns across candidates (figures/phase2_expert/expert_alpha_correlation.png/.csv).
      High positive correlation means the gate raises/lowers two experts together (it
      isn't using them to distinguish different candidates). Note the K alpha values sum
      to 1 per candidate (softmax), which mechanically induces some negative correlation
      even among genuinely independent experts -- a real caveat of this metric, not a bug.

2. Expert specialization -- from the CHECKPOINT itself (not the scores CSV): pairwise
   cosine similarity between each pair of experts' own LEARNED PARAMETERS (every weight +
   bias in each Expert_k flattened into one vector). This is a different question from
   (1) -- two experts can have highly correlated alpha (always used together) while still
   computing very different functions of v_c, or vice versa. At random init, K
   high-dimensional random vectors are already close to orthogonal (~0 cosine similarity)
   purely by chance; if training pulls two experts' parameter vectors close to 1.0, they've
   converged to (near-)identical functions -- effectively wasted capacity, since the MoE
   then behaves like a single expert wearing two hats. Saved to
   figures/phase2_expert/expert_parameter_similarity.png/.csv.

Usage:
    python src/phase2_expert/analyze_experts.py
    python src/phase2_expert/analyze_experts.py --scores data_phase2_expert/xlm-roberta_experts_scores.csv --checkpoint checkpoints/phase2_expert/xlm-roberta_experts.pt
    python src/phase2_expert/analyze_experts.py --split ""   # every candidate, not just test
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2_expert.model import load_model
from phase2_expert.train import DEFAULT_CHECKPOINT_OUT

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SCORES = REPO_ROOT / "data" / "data_phase2_expert" / "camembert_experts_scores.csv"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "phase2_expert"

# Same visual language as src/modeling/plot_reliability_diagram.py, reused for consistency
# across the project's figures.
CATEGORICAL_RED = "#e34948"
CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_ORANGE = "#e8871e"
CATEGORICAL_PURPLE = "#8e5cd9"
STATUS_GOOD = "#0ca30c"
EXTRA_COLOR_PALETTE = ["#1a9e96", "#a56a3a", "#d64d9a", "#5b6b73", "#8a8a2f", "#3f5fbf"]
EXPERT_COLORS = [CATEGORICAL_BLUE, CATEGORICAL_ORANGE, STATUS_GOOD, CATEGORICAL_PURPLE, CATEGORICAL_RED, *EXTRA_COLOR_PALETTE]
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"

ALPHA_COL_RE = re.compile(r"^alpha_(\d+)$")


def find_alpha_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted((c for c in df.columns if ALPHA_COL_RE.match(c)), key=lambda c: int(ALPHA_COL_RE.match(c).group(1)))
    if not cols:
        raise ValueError(
            "No alpha_<k> columns found in this scores CSV -- it must come from evaluate.py "
            "(this folder's version, which records the gate's per-candidate weights)."
        )
    return cols


def _style_axes(ax) -> None:
    ax.set_facecolor(CHART_SURFACE)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)


def _plot_heatmap(matrix: pd.DataFrame, title: str, cbar_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * len(matrix) + 2, 1.1 * len(matrix) + 1.5), facecolor=CHART_SURFACE)
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)

    labels = list(matrix.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=PRIMARY_INK)
    ax.set_yticklabels(labels, color=PRIMARY_INK)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix.iloc[i, j]
            text_color = "white" if abs(value) > 0.6 else PRIMARY_INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title(title, color=PRIMARY_INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=MUTED_INK)
    cbar.set_label(cbar_label, color=PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_distribution(df: pd.DataFrame, alpha_cols: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.6 * len(alpha_cols) + 2, 5), facecolor=CHART_SURFACE)
    data = [df[col].to_numpy() for col in alpha_cols]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        color = EXPERT_COLORS[i % len(EXPERT_COLORS)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[key].set_color(PRIMARY_INK)
        parts[key].set_linewidth(1.0)

    means = [d.mean() for d in data]
    for i, m in enumerate(means):
        ax.annotate(f"mean={m:.3f}", (i + 1, m), xytext=(6, 0), textcoords="offset points", fontsize=8, color=PRIMARY_INK, va="center")

    ax.axhline(1.0 / len(alpha_cols), color=MUTED_INK, linewidth=1.0, linestyle="--", zorder=1, label=f"uniform (1/{len(alpha_cols)})")
    ax.set_xticks(range(1, len(alpha_cols) + 1))
    ax.set_xticklabels([f"Expert {c.split('_')[1]}" for c in alpha_cols], color=PRIMARY_INK)
    ax.set_ylabel("Gate weight (alpha)", color=PRIMARY_INK)
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-expert gate weight distribution ({len(df)} candidates)", color=PRIMARY_INK)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def expert_parameter_similarity(model) -> pd.DataFrame:
    """Cosine similarity between each pair of experts' own learned parameters (every
    weight/bias in Expert_k flattened into one vector) -- a static, per-checkpoint
    measure of whether the experts computed genuinely different functions, independent of
    how the gate happens to route candidates (see module docstring, part 2)."""
    vectors = [torch.cat([p.detach().flatten() for p in expert.parameters()]) for expert in model.experts]
    labels = [f"E{k}" for k in range(len(vectors))]
    sim = torch.zeros(len(vectors), len(vectors))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            sim[i, j] = torch.nn.functional.cosine_similarity(vectors[i], vectors[j], dim=0)
    return pd.DataFrame(sim.numpy(), index=labels, columns=labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default=str(DEFAULT_SCORES), help="Scores CSV from this folder's evaluate.py (must have alpha_<k> columns) -- used for the gate-usage plots")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_OUT), help="Trained checkpoint (see train.py) -- used for the parameter-similarity plot")
    parser.add_argument("--split", default="test", help="Filter to this value in the scores CSV's own 'split' column (default: test); pass \"\" for every candidate")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plots into")
    args = parser.parse_args()

    print(f"=== Step 1: Load {args.scores} ===")
    df = pd.read_csv(args.scores)
    print(f"{len(df)} candidates")
    if args.split:
        df = df[df["split"] == args.split]
        print(f"=== Step 2: Filter to split={args.split!r} -- {len(df)} candidates remain ===")
    else:
        print("=== Step 2: No --split given, using every candidate ===")

    alpha_cols = find_alpha_columns(df)
    print(f"Found {len(alpha_cols)} experts: {alpha_cols}")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 3: Per-expert gate-usage summary ===")
    summary = df[alpha_cols].agg(["mean", "std", "min", "max"]).T
    summary.index.name = "expert"
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    # argmax_share: fraction of candidates where this expert has the single highest alpha
    # -- a second usage signal alongside the mean (a low-mean expert could still "win"
    # outright on a small, distinct subset of candidates).
    argmax_counts = df[alpha_cols].to_numpy().argmax(axis=1)
    for k, col in enumerate(alpha_cols):
        share = float((argmax_counts == k).mean())
        print(f"  {col}: top expert on {share:.1%} of candidates")

    print("=== Step 4: Plot per-expert alpha distribution ===")
    dist_path = figures_dir / "expert_alpha_distribution.png"
    plot_alpha_distribution(df, alpha_cols, dist_path)
    print(f"Saved {dist_path}")

    print("=== Step 5: Gate-usage correlation between experts (alpha_k columns, Pearson) ===")
    alpha_corr = df[alpha_cols].corr()
    alpha_corr.columns = alpha_corr.index = [f"E{c.split('_')[1]}" for c in alpha_cols]
    print(alpha_corr.to_string(float_format=lambda x: f"{x:.3f}"))
    alpha_corr_csv_path = figures_dir / "expert_alpha_correlation.csv"
    alpha_corr.to_csv(alpha_corr_csv_path)
    print(f"Saved {alpha_corr_csv_path}")
    alpha_corr_png_path = figures_dir / "expert_alpha_correlation.png"
    _plot_heatmap(alpha_corr, "Pairwise gate-usage correlation between experts (alpha)", "Pearson correlation", alpha_corr_png_path)
    print(f"Saved {alpha_corr_png_path}")
    print("Note: the K alpha values sum to 1 per candidate (softmax), which mechanically "
          "induces some negative correlation even among genuinely independent experts.")

    print(f"=== Step 6: Load {args.checkpoint} for parameter-level similarity ===")
    model = load_model(args.checkpoint)
    print(f"variant={model.variant_name()} num_experts={model.num_experts}")

    print("=== Step 7: Pairwise similarity between experts' LEARNED PARAMETERS (cosine) ===")
    param_sim = expert_parameter_similarity(model)
    print(param_sim.to_string(float_format=lambda x: f"{x:.3f}"))
    param_sim_csv_path = figures_dir / "expert_parameter_similarity.csv"
    param_sim.to_csv(param_sim_csv_path)
    print(f"Saved {param_sim_csv_path}")
    param_sim_png_path = figures_dir / "expert_parameter_similarity.png"
    _plot_heatmap(param_sim, "Pairwise cosine similarity between experts' learned weights", "Cosine similarity", param_sim_png_path)
    print(f"Saved {param_sim_png_path}")
    print("Note: ~0 is the random-init baseline (independent high-dim random vectors are "
          "already near-orthogonal) -- values pulled toward 1.0 after training mean two "
          "experts converged to near-identical functions (wasted capacity), not that they "
          "specialize on the same candidates (that's what Step 5's alpha correlation measures).")

    print("=== Done ===")


if __name__ == "__main__":
    main()
