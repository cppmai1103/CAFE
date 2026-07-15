"""Summarize dataset composition -- document counts, gold entity counts, and entity type
breakdown -- for the whole dataset and for each of the document-level train/val/test
splits, assigned by preprocessing_data.py's assign_splits and carried in train data's
"split" column.

Gold entities are NE-COARSE-LIT closed into spans (analyze_ocr_context_features.py's
build_gold_spans, reused here, not reimplemented), each carrying a normalized type in
{PERS, LOC, ORG, TIME, PROD}. A gold entity's split is its document's split.

Two plots, "All" (the whole dataset) shown alongside the three splits for reference:
    1. Documents per split               -> bar chart
    2. Entity type breakdown per split    -> stacked bar chart

Gold entity counts per split are still printed to console (Step 4) even though the
"Gold entities per split" bar chart was dropped -- the stacked breakdown's bar totals
cover the same total-count information.

Usage:
    python src/analysis/analyze_data_splits.py
    python src/analysis/analyze_data_splits.py --figures-dir /tmp/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_ocr_context_features import DEFAULT_TRAIN_DATA, build_gold_spans

DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "data_analysis"

SPLITS = ["train", "val", "test"]
CATEGORIES = ["All"] + SPLITS
LABELS = ["PERS", "LOC", "ORG", "TIME", "PROD"]

CATEGORICAL = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300", "violet": "#4a3aa7"}
TYPE_COLORS = {"PERS": CATEGORICAL["blue"], "LOC": CATEGORICAL["aqua"], "ORG": CATEGORICAL["yellow"], "TIME": CATEGORICAL["green"], "PROD": CATEGORICAL["violet"]}
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def build_gold_entities_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """One row per gold entity: document_id, start_token_id, end_token_id, entity_type."""
    gold_spans = build_gold_spans(train_df)
    rows = [
        {"document_id": doc_id, "start_token_id": start, "end_token_id": end, "entity_type": entity_type}
        for (doc_id, start, end), entity_type in gold_spans.items()
    ]
    return pd.DataFrame(rows, columns=["document_id", "start_token_id", "end_token_id", "entity_type"])


def counts_per_category(per_doc_counts: pd.Series, doc_to_split: dict) -> pd.Series:
    """Given a per-document_id count (e.g. entities per document), sum it per split, plus
    an "All" total, in CATEGORIES order."""
    by_split = per_doc_counts.groupby(per_doc_counts.index.map(doc_to_split)).sum()
    out = {"All": int(per_doc_counts.sum())}
    for split in SPLITS:
        out[split] = int(by_split.get(split, 0))
    return pd.Series(out).reindex(CATEGORIES)


def plot_count_bar(counts: pd.Series, out_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    colors = [MUTED_INK] + [CATEGORICAL["blue"]] * len(SPLITS)
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], fontsize=9, color=MUTED_INK, padding=3)

    ax.set_ylabel(ylabel, color=PRIMARY_INK)
    ax.set_title(title, color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_type_breakdown(type_counts: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar: one bar per category (All + 4 splits), segments = entity type."""
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    bottoms = pd.Series(0, index=type_counts.index)
    for entity_type in LABELS:
        values = type_counts[entity_type]
        ax.bar(type_counts.index, values, bottom=bottoms, color=TYPE_COLORS[entity_type], label=entity_type)
        bottoms = bottoms + values

    for category, total in zip(type_counts.index, bottoms):
        ax.text(category, total + max(bottoms) * 0.01, f"{int(total):,}", ha="center", va="bottom", fontsize=9, color=MUTED_INK)

    ax.set_ylabel("Gold entity count", color=PRIMARY_INK)
    ax.set_title("Entity type breakdown per split", color=PRIMARY_INK)
    ax.set_ylim(0, max(bottoms) * 1.15)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (has document_id, split, NE-COARSE-LIT)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save plots into")
    args = parser.parse_args()

    print("=== Step 1: Load train data ===")
    train_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    train_df["token_id"] = train_df["token_id"].astype(int)
    doc_to_split = train_df.drop_duplicates("document_id").set_index("document_id")["split"].to_dict()
    print(f"{train_df['document_id'].nunique()} documents, splits: {train_df.drop_duplicates('document_id')['split'].value_counts().to_dict()}")

    print("=== Step 2: Close NE-COARSE-LIT gold spans ===")
    entities_df = build_gold_entities_df(train_df)
    print(f"{len(entities_df)} gold entities")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 3: Documents per split ===")
    doc_counts = pd.Series(1, index=train_df.drop_duplicates("document_id")["document_id"])
    doc_counts_by_category = counts_per_category(doc_counts, doc_to_split)
    print(doc_counts_by_category.to_string())
    plot_count_bar(doc_counts_by_category, figures_dir / "documents_per_split.png", "Documents per split", "Document count")
    print(f"Saved {figures_dir / 'documents_per_split.png'}")

    print("=== Step 4: Gold entities per split ===")
    entity_counts = entities_df.groupby("document_id").size()
    entity_counts_by_category = counts_per_category(entity_counts, doc_to_split)
    print(entity_counts_by_category.to_string())

    print("=== Step 5: Entity type breakdown per split ===")
    entities_df["split"] = entities_df["document_id"].map(doc_to_split)
    type_by_split = entities_df.groupby(["split", "entity_type"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    type_counts = pd.DataFrame({"All": entities_df["entity_type"].value_counts().reindex(LABELS, fill_value=0)}).T
    type_counts = pd.concat([type_counts, type_by_split.reindex(SPLITS, fill_value=0)])
    print(type_counts.to_string())
    plot_type_breakdown(type_counts, figures_dir / "entity_type_breakdown_per_split.png")
    print(f"Saved {figures_dir / 'entity_type_breakdown_per_split.png'}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
