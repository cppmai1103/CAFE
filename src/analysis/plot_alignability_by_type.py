"""Plot alignability broken down by entity type (PERS/LOC/ORG/TIME/PROD): of all gold
NE-COARSE-LIT tokens of a given type, how many does Gliner correctly predict as that same
type? Same bullet-bar style as analyze_gliner_mismatches.py's plot_alignability (pale bar
= n_tagged, solid bar = n_matched), but grouped by entity type instead of by gold column
-- so it's visible at a glance which types Gliner tracks well vs poorly, complementing
that script's per-column view.

Input: the token-format CSV (see gliner/ner_features_to_token_format.py) -- same file
analyze_gliner_mismatches.py uses, reusing its load_and_normalize/LABELS/color palette
rather than reimplementing type normalization.

Output filename bakes in --threshold (e.g. alignability_by_type_threshold0.5.png, the
"_by_type" distinguishing it from analyze_gliner_mismatches.py's own
alignability_threshold0.5.png, its per-gold-column breakdown), matching
ner_features_to_token_format.py's own thresholded output naming, since the alignability
numbers are entirely a function of which threshold's token-format file was used -- pass
--threshold to match whatever --token-format was actually generated with.

Usage:
    python src/analysis/plot_alignability_by_type.py
    python src/analysis/plot_alignability_by_type.py --threshold 0.3 --token-format data/hipe2020_train_fr_gliner_token_format_threshold0.3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_gliner_mismatches import (
    BLUE_SEQUENTIAL,
    CATEGORICAL,
    CHART_SURFACE,
    GRIDLINE,
    LABELS,
    MUTED_INK,
    PRIMARY_INK,
    load_and_normalize,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "ner_analysis"
DEFAULT_THRESHOLD = 0.5


def default_token_format_path(threshold: float) -> Path:
    return DATA_DIR / f"hipe2020_train_fr_gliner_token_format_threshold{threshold}.csv"


def default_out_path(figures_dir: Path, threshold: float) -> Path:
    return figures_dir / f"alignability_by_type_threshold{threshold}.png"


def compute_type_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """For each entity type in LABELS: how many gold tokens are that type (n_tagged), and
    how many does Gliner correctly predict as that same type (n_matched)?"""
    rows = []
    for label in LABELS:
        tagged = df["gold_type"] == label
        n_tagged = int(tagged.sum())
        n_matched = int((tagged & (df["pred_type"] == label)).sum())
        match_rate = n_matched / n_tagged if n_tagged else 0.0
        rows.append({"type": label, "n_tagged": n_tagged, "n_matched": n_matched, "match_rate": match_rate})
    return pd.DataFrame(rows)


def plot_alignability_by_type(align_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    y = range(len(align_df))
    ax.barh(y, align_df["n_tagged"], color=BLUE_SEQUENTIAL[1], label="Gold tokens of this type (n_tagged)")
    ax.barh(y, align_df["n_matched"], color=CATEGORICAL["blue"], label="...and Gliner predicts correctly (n_matched)")

    for yi, (n_tagged, n_matched, rate) in enumerate(zip(align_df["n_tagged"], align_df["n_matched"], align_df["match_rate"])):
        ax.text(
            n_tagged + max(align_df["n_tagged"]) * 0.02, yi, f"{n_matched:,} / {n_tagged:,}  ({rate:.1%})",
            va="center", ha="left", fontsize=9, color=MUTED_INK,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(align_df["type"], color=PRIMARY_INK)
    ax.set_xlabel("Gold tokens of this type", color=PRIMARY_INK)
    ax.set_title("Alignability by entity type: gold vs Gliner", color=PRIMARY_INK)
    ax.set_xlim(0, align_df["n_tagged"].max() * 1.35)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.invert_yaxis()
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Threshold the token-format CSV was generated with -- only used to name the default --token-format/output paths (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--token-format", default=None, help="Token-format CSV (default: matches --threshold's filename)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()
    token_format_path = Path(args.token_format) if args.token_format is not None else default_token_format_path(args.threshold)

    print("=== Step 1: Load and normalize token-format data ===")
    print(f"Loading {token_format_path}")
    df = load_and_normalize(token_format_path)
    print(f"{len(df)} tokens loaded")

    print("=== Step 2: Compute alignability by entity type ===")
    align_df = compute_type_alignment(df)
    print(align_df.to_string(index=False))

    print("=== Step 3: Plot ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = default_out_path(figures_dir, args.threshold)
    plot_alignability_by_type(align_df, out_path)
    print(f"Saved {out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
