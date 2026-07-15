"""Plot token-level OCR-quality signals from the train data CSV -- split out of
analyze_ocr_context_features.py, which covers the candidate-level reliability tables
instead (no plots, see that module's docstring). dictionary_score/sentence_ocr_mean/
document_ocr_mean are computed by preprocessing/ocr_dictionary_check.py.

Plots:
    1. How many tokens does the OCR-QA bloom filter mark known (True) / unknown (False) /
       not-applicable-punctuation (None)?                          -> bar chart
    2. Distribution of document_ocr_mean, one value per document.  -> histogram
    3. Distribution of sentence_ocr_mean, one value per sentence.  -> histogram

Usage:
    python src/analysis/plot_ocr_quality_distributions.py
    python src/analysis/plot_ocr_quality_distributions.py --figures-dir /tmp/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_TRAIN_DATA = DATA_DIR / "hipe2020_train_fr_train_data.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "data_analysis"

CATEGORICAL_BLUE = "#2a78d6"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def plot_dictionary_score_counts(train_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of how many tokens the OCR-QA bloom filter marked known (True) / unknown
    (False) / not-applicable-punctuation (None)."""
    dictionary_score = train_df["dictionary_score"]
    labels = ["Known (True)", "Unknown (False)", "N/A -- punctuation (None)"]
    values = [int((dictionary_score == True).sum()), int((dictionary_score == False).sum()), int(dictionary_score.isna().sum())]
    total = sum(values)
    colors = [STATUS_GOOD, STATUS_CRITICAL, MUTED_INK]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{v:,} ({v / total:.1%})" for v in values], fontsize=9, color=MUTED_INK, padding=3)

    ax.set_ylabel("Token count", color=PRIMARY_INK)
    ax.set_title("Token-level dictionary_score (OCR-QA bloom filter)", color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_ocr_mean_distribution(values: pd.Series, out_path: Path, title: str, xlabel: str) -> None:
    # Auto-range to where the data actually lives -- these means cluster tight near 1.0,
    # so a fixed (0, 1) range would waste most of the chart on empty space.
    lo, hi = values.min(), values.max()
    pad = max((hi - lo) * 0.05, 0.005)
    x_min, x_max = max(0.0, lo - pad), min(1.0, hi + pad)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.hist(values, bins=30, range=(x_min, x_max), color=CATEGORICAL_BLUE, edgecolor=CHART_SURFACE, linewidth=0.5)

    ax.set_ylabel("Count", color=PRIMARY_INK)
    ax.set_xlabel(xlabel, color=PRIMARY_INK)
    ax.set_title(title, color=PRIMARY_INK)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save plots into")
    args = parser.parse_args()

    print("=== Step 1: Load train data ===")
    train_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    print(f"{len(train_df)} tokens")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 2: Token-level dictionary_score counts (True/False/None) ===")
    dictionary_score = train_df["dictionary_score"]
    n_known = int((dictionary_score == True).sum())
    n_unknown = int((dictionary_score == False).sum())
    n_na = int(dictionary_score.isna().sum())
    n_total = n_known + n_unknown + n_na
    print(
        f"Known (True): {n_known:,} ({n_known / n_total:.2%})  "
        f"Unknown (False): {n_unknown:,} ({n_unknown / n_total:.2%})  "
        f"N/A (None): {n_na:,} ({n_na / n_total:.2%})"
    )
    plot_dictionary_score_counts(train_df, figures_dir / "dictionary_score_counts.png")
    print(f"Saved {figures_dir / 'dictionary_score_counts.png'}")

    print("=== Step 3: Distribution of document_ocr_mean (one value per document) ===")
    document_means = train_df.drop_duplicates("document_id")["document_ocr_mean"].dropna()
    print(document_means.describe())
    plot_ocr_mean_distribution(
        document_means, figures_dir / "document_ocr_mean_distribution.png",
        "Distribution of document_ocr_mean (one value per document)", "document_ocr_mean",
    )
    print(f"Saved {figures_dir / 'document_ocr_mean_distribution.png'}")

    print("=== Step 4: Distribution of sentence_ocr_mean (one value per sentence) ===")
    sentence_means = train_df.drop_duplicates(["document_id", "sentence_id"])["sentence_ocr_mean"].dropna()
    print(sentence_means.describe())
    plot_ocr_mean_distribution(
        sentence_means, figures_dir / "sentence_ocr_mean_distribution.png",
        "Distribution of sentence_ocr_mean (one value per sentence)", "sentence_ocr_mean",
    )
    print(f"Saved {figures_dir / 'sentence_ocr_mean_distribution.png'}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
