"""Plot the distribution of GLiNER2's raw ner_score, split into reliable vs not, from
label_reliability.csv (see gliner/label_reliability.py) -- split out of
calibrate_ner_confidence.py, which used to draw this as a side effect of fitting B0/B1/B3
even though the plot itself only needs ner_score + reliability_score, not any of the
fitted/calibrated scores.

Output: ner_score_distribution.png -- histogram of ner_score for reliable
(reliability_score=1) vs not-reliable (reliability_score=0) candidates, log-scale y-axis
-- the two classes differ by roughly an order of magnitude in count, which a linear
y-axis would flatten into invisibility.

Usage:
    python src/analysis/plot_ner_score_distribution.py
    python src/analysis/plot_ner_score_distribution.py --label-reliability data/label_reliability.csv --figures-dir figures/ner_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_LABEL_RELIABILITY = DATA_DIR / "label_reliability_type_only.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "ner_analysis"

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def plot_score_distribution(candidates_df: pd.DataFrame, out_path: Path) -> None:
    """Histogram of GLiNER2's raw ner_score across every candidate, split into reliable
    (reliability_score=1) vs not, on a log-scale y-axis."""
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    bins = np.linspace(0, 1, 31)
    is_reliable = candidates_df["reliability_score"].astype(bool)
    not_reliable = candidates_df.loc[~is_reliable, "ner_score"]
    reliable = candidates_df.loc[is_reliable, "ner_score"]

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--label-reliability", default=str(DEFAULT_LABEL_RELIABILITY), help="label_reliability.csv (see gliner/label_reliability.py)"
    )
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()

    print("=== Step 1: Load label_reliability.csv ===")
    print(f"Loading {args.label_reliability}")
    candidates_df = pd.read_csv(args.label_reliability)
    print(f"{len(candidates_df)} candidates loaded")

    print("=== Step 2: Plot ner_score distribution (reliable vs not) ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "ner_score_distribution.png"
    plot_score_distribution(candidates_df, out_path)
    print(f"Saved {out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
