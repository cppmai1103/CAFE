"""Plot two token-level confusion matrices (gold NE-COARSE-LIT vs Gliner prediction) side
by side in one figure: one for tokens the OCR-QA bloom filter marked known
(dictionary_score == True) and one for tokens it marked unknown (dictionary_score ==
False) -- so it's visible at a glance whether GLiNER agrees with gold less often on
OCR-garbled (unknown) tokens than on clean (known) ones.

Input: the token-format CSV (see gliner/ner_features_to_token_format.py), which already
carries dictionary_score per token (see preprocessing/ocr_dictionary_check.py) alongside
NE-COARSE-LIT (gold) and Gliner (predicted). Tokens where dictionary_score is neither
True nor False (punctuation, not scoreable) are grouped into the "known" bucket rather
than dropped, so the two matrices' token counts always sum back to the input's total --
only dictionary_score == False (an actual OCR-QA "unknown word" flag) is split out into
the "unknown" bucket; everything else (True, or N/A punctuation) is "known".

Reuses bare_type/LABELS/ALL_LABELS/compute_confusion_matrix from
analyze_gliner_mismatches.py rather than reimplementing the type-normalization logic.

Output: confusion_matrix_by_dictionary_score.png -- two heatmaps (unknown left, known
right), each with its own independent color scale (their token counts differ by ~68x, so
a shared scale would flatten the smaller panel's own pattern to near-invisibility) --
compare the two on relative pattern, not absolute color intensity across panels.

Usage:
    python src/analysis/plot_confusion_matrix_by_dictionary_score.py
    python src/analysis/plot_confusion_matrix_by_dictionary_score.py --token-format /tmp/smoke_token_format.csv --figures-dir /tmp/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analyze_gliner_mismatches import (
    ALL_LABELS,
    BLUE_SEQUENTIAL,
    DEFAULT_TOKEN_FORMAT,
    MUTED_INK,
    PRIMARY_INK,
    bare_type,
    compute_confusion_matrix,
)

CHART_SURFACE = "#fcfcfb"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "ner_analysis"


def load_and_normalize(token_format_path: Path) -> pd.DataFrame:
    df = pd.read_csv(token_format_path, dtype=str)
    df["gold_type"] = df["NE-COARSE-LIT"].apply(bare_type)
    df["pred_type"] = df["Gliner"].apply(bare_type)
    return df


def plot_confusion_matrix_on_ax(cm: pd.DataFrame, ax, title: str) -> None:
    blue_cmap = LinearSegmentedColormap.from_list("blue_sequential", BLUE_SEQUENTIAL)
    ax.set_facecolor(CHART_SURFACE)
    values = cm.to_numpy()

    # The (O, O) cell (tokens correctly left untagged) dwarfs every other cell and would
    # flatten the color scale; it's excluded from the color mapping (shown as a fixed
    # dark cell) but its true count is still printed as a label. vmax is this panel's own
    # max (excluding O/O) -- independent per panel, since the known/unknown token counts
    # differ by ~68x and a shared scale would flatten the smaller panel to near-invisible.
    color_values = values.copy()
    color_values[-1, -1] = 0
    vmax = color_values.max() if color_values.max() > 0 else 1
    im = ax.imshow(color_values, cmap=blue_cmap, vmin=0, vmax=vmax)
    ax.add_patch(plt.Rectangle((len(ALL_LABELS) - 1.5, len(ALL_LABELS) - 1.5), 1, 1, facecolor=BLUE_SEQUENTIAL[-1], edgecolor="none"))

    ax.set_xticks(range(len(ALL_LABELS)))
    ax.set_yticks(range(len(ALL_LABELS)))
    ax.set_xticklabels(ALL_LABELS, color=PRIMARY_INK)
    ax.set_yticklabels(ALL_LABELS, color=PRIMARY_INK)
    ax.set_xlabel("Gliner prediction", color=PRIMARY_INK)
    ax.set_ylabel("Gold (NE-COARSE-LIT)", color=PRIMARY_INK)
    ax.set_title(title, color=PRIMARY_INK)

    last = len(ALL_LABELS) - 1
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            is_o_o_cell = i == last and j == last
            color = "white" if is_o_o_cell or values[i, j] > vmax * 0.5 else PRIMARY_INK
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center", color=color, fontsize=9)

    return im


def plot_confusion_matrices_by_dictionary_score(cm_unknown: pd.DataFrame, cm_known: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=CHART_SURFACE)

    # Independent color scale per panel (see plot_confusion_matrix_on_ax) -- known/unknown
    # token counts differ by ~68x, so a shared scale would flatten the smaller panel's
    # own hotspots into near-invisibility, defeating the point of comparing the two.
    im_unknown = plot_confusion_matrix_on_ax(cm_unknown, axes[0], f"Unknown (dictionary_score=False, n={int(cm_unknown.to_numpy().sum()):,})")
    im_known = plot_confusion_matrix_on_ax(cm_known, axes[1], f"Known (dictionary_score=True or N/A, n={int(cm_known.to_numpy().sum()):,})")

    fig.colorbar(im_unknown, ax=axes[0], label="token count", shrink=0.85)
    fig.colorbar(im_known, ax=axes[1], label="token count", shrink=0.85)
    fig.suptitle("Token-level confusion matrix: gold vs Gliner, by OCR dictionary_score", color=PRIMARY_INK)
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-format", default=str(DEFAULT_TOKEN_FORMAT), help="Token-format CSV (see gliner/ner_features_to_token_format.py)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()

    print("=== Step 1: Load and normalize token-format data ===")
    print(f"Loading {args.token_format}")
    df = load_and_normalize(Path(args.token_format))
    print(f"{len(df)} tokens loaded")

    print("=== Step 2: Split by dictionary_score (NaN/punctuation counted as known) ===")
    unknown_df = df[df["dictionary_score"] == "False"]
    known_df = df[df["dictionary_score"] != "False"]
    print(f"{len(known_df)} known tokens (True + punctuation/NaN), {len(unknown_df)} unknown tokens -- {len(known_df) + len(unknown_df)} total")

    print("=== Step 3: Compute confusion matrices ===")
    cm_known = compute_confusion_matrix(known_df)
    cm_unknown = compute_confusion_matrix(unknown_df)
    print("--- Unknown (dictionary_score=False) ---")
    print(cm_unknown)
    print("--- Known (dictionary_score=True or N/A) ---")
    print(cm_known)

    print("=== Step 4: Plot side by side ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "confusion_matrix_by_dictionary_score.png"
    plot_confusion_matrices_by_dictionary_score(cm_unknown, cm_known, out_path)
    print(f"Saved {out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
