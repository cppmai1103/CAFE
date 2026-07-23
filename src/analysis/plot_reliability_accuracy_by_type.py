"""Plot the fraction of predicted NER candidates that are reliable (reliability_score=1),
grouped by predicted_entity_type, plus an "ALL" bar aggregating every candidate -- from
label_reliability.csv (see ner/label_reliability.py). Model-agnostic, works on any model's
label_reliability.csv (GLiNER2's own or historical-ner-baseline's) and any --mode
(span_type/type_only/fuzzy) it was generated with.

This answers "of everything the model predicted as type X, how much of it was actually
correct?" -- i.e. per-type precision under whichever label_reliability --mode was used to
decide "correct", not the raw per-token type-agreement precision that
analyze_ner_mismatches.py plots.

Output: reliability_accuracy_by_type.png -- bar chart of reliable-fraction per type (+ALL),
annotated with the fraction and the underlying reliable/total candidate counts.

Usage:
    python src/analysis/plot_reliability_accuracy_by_type.py
    python src/analysis/plot_reliability_accuracy_by_type.py --label-reliability data/hipe2020_fr/historical_ner/data_baseline/label_reliability_span_level_fuzzy.csv --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/all_set
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_LABEL_RELIABILITY = DATA_DIR / "label_reliability_span_level_fuzzy.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "ner_analysis"

ALL_LABEL = "ALL"
BAR_TYPE = "#2a78d6"
BAR_ALL = "#4a3aa7"
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def accuracy_by_type(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """One row per predicted_entity_type (sorted by descending candidate count) plus a
    trailing ALL row, each with the fraction of candidates that are reliable
    (reliability_score=1) and the underlying (reliable, total) counts."""
    is_reliable = candidates_df["reliability_score"].astype(bool)

    per_type = candidates_df.groupby("predicted_entity_type").agg(
        reliable=("reliability_score", lambda s: int(s.astype(bool).sum())),
        total=("reliability_score", "size"),
    )
    per_type = per_type.sort_values("total", ascending=False)
    per_type["fraction"] = per_type["reliable"] / per_type["total"]

    overall = pd.DataFrame(
        {"reliable": [int(is_reliable.sum())], "total": [len(candidates_df)]}, index=[ALL_LABEL]
    )
    overall["fraction"] = overall["reliable"] / overall["total"]

    return pd.concat([per_type, overall])


def plot_accuracy_by_type(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    colors = [BAR_ALL if idx == ALL_LABEL else BAR_TYPE for idx in summary_df.index]
    bars = ax.bar(summary_df.index, summary_df["fraction"], color=colors, zorder=3)

    for bar, (_, row) in zip(bars, summary_df.iterrows()):
        ax.annotate(
            f"{row['fraction']:.2f}\n(n={int(row['reliable']):,}/{int(row['total']):,})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color=MUTED_INK,
        )

    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Fraction of predicted candidates that are reliable", color=PRIMARY_INK)
    ax.set_title("Reliability accuracy per predicted entity type", color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--label-reliability", default=str(DEFAULT_LABEL_RELIABILITY), help="label_reliability.csv (see ner/label_reliability.py)"
    )
    parser.add_argument(
        "--load-data", default=str(DEFAULT_LOAD_DATA),
        help="Token-level data CSV (for the --split filter; label_reliability.csv has no split column of its own)",
    )
    parser.add_argument(
        "--split", default="",
        help="Filter to this document-level split before plotting (train/val/test); pass \"\" to use every "
        "candidate (default: every candidate, i.e. all data)",
    )
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()

    print("=== Step 1: Load label_reliability.csv ===")
    print(f"Loading {args.label_reliability}")
    candidates_df = pd.read_csv(args.label_reliability)
    print(f"{len(candidates_df)} candidates loaded")

    if args.split:
        print(f"=== Step 1b: Filter to split={args.split!r} ===")
        print(f"Loading {args.load_data}")
        data_df = pd.read_csv(args.load_data, dtype={"TOKEN": str, "MISC": str},
        # See plot_ner_score_distribution.py for why keep_default_na=False is required here
        # (a real hipe2020_fr token's text is literally "NA").
        keep_default_na=False, na_values={"sentence_ocr_mean": [""], "document_ocr_mean": [""], "dictionary_score": [""]})
        doc_to_split = data_df.drop_duplicates("document_id").set_index("document_id")["split"].to_dict()
        candidates_df = candidates_df[candidates_df["document_id"].map(doc_to_split) == args.split]
        print(f"{len(candidates_df)} candidates remain")

    print("=== Step 2: Compute reliability accuracy per type ===")
    summary_df = accuracy_by_type(candidates_df)
    print(summary_df)

    print("=== Step 3: Plot reliability accuracy per type ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "reliability_accuracy_by_type.png"
    plot_accuracy_by_type(summary_df, out_path)
    print(f"Saved {out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
