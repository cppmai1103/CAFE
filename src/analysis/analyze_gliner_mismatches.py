"""Analyze GLiNER2 vs gold NE-COARSE-LIT mismatches on the token-format CSV (see
ner_features_to_token_format.py), and plot the results.

Three questions, one plot each:
    1. Where does GLiNER agree/disagree with gold, per type?  -> confusion matrix heatmap
    2. How good is each type's precision/recall/F1?            -> grouped bar chart
    3. Of the tokens each of the six gold NE-* columns (NE-COARSE-LIT, NE-COARSE-METO,
       NE-FINE-LIT, NE-FINE-METO, NE-FINE-COMP, NE-NESTED) tags as an entity, how many
       does GLiNER agree with?                                  -> bullet-style bar per column

Types are normalized to GLiNER's flat scheme (PERS/LOC/ORG/TIME/PROD) before comparing:
NE-COARSE-LIT tags like "B-pers.ind" collapse to "PERS", any fine/component subtype
outside the five in-scope types (e.g. "comp.title") collapses to "OTHER" (treated as
non-matching, since GLiNER never predicts it).

Usage:
    python src/analysis/analyze_gliner_mismatches.py
    python src/analysis/analyze_gliner_mismatches.py --token-format /tmp/smoke_token_format.csv --figures-dir /tmp/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_TOKEN_FORMAT = DATA_DIR / "hipe2020_train_fr_gliner_token_format_threshold0.5.csv"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "ner_analysis"

LABELS = ["PERS", "LOC", "ORG", "TIME", "PROD"]
ALL_LABELS = LABELS + ["O"]
OTHER_GOLD_COLUMNS = ["NE-COARSE-METO", "NE-FINE-LIT", "NE-FINE-METO", "NE-FINE-COMP", "NE-NESTED"]

# HIPE's coarse bare types map 1:1 onto LABELS; anything else (e.g. "comp.title") is
# out of GLiNER's label space and can never match a prediction.
_TYPE_MAP = {"pers": "PERS", "loc": "LOC", "org": "ORG", "time": "TIME", "prod": "PROD"}

# Project's validated reference palette (docs/... shares this with evaluate_ner_metrics.ipynb).
BLUE_SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#104281", "#0d366b"]
CATEGORICAL = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def bare_type(tag: str) -> str:
    """Normalize a HIPE/Gliner IOB2 tag ("O", "B-pers.ind", "I-LOC", ...) to one of
    LABELS, "O", or "OTHER" (an out-of-scope subtype, e.g. a NE-FINE-COMP component tag)."""
    if pd.isna(tag) or tag == "O":
        return "O"
    raw_type = tag.split("-", 1)[1]
    coarse = raw_type.split(".", 1)[0].lower()
    return _TYPE_MAP.get(coarse, "OTHER")


def load_and_normalize(token_format_path: Path) -> pd.DataFrame:
    df = pd.read_csv(token_format_path, dtype=str)

    tqdm.pandas(desc="Normalizing NE-COARSE-LIT (gold)", unit="token")
    df["gold_type"] = df["NE-COARSE-LIT"].progress_apply(bare_type)

    tqdm.pandas(desc="Normalizing Gliner (predicted)", unit="token")
    df["pred_type"] = df["Gliner"].progress_apply(bare_type)

    for column in OTHER_GOLD_COLUMNS:
        tqdm.pandas(desc=f"Normalizing {column}", unit="token")
        df[column + "_type"] = df[column].progress_apply(bare_type)

    return df


def compute_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cm = pd.crosstab(df["gold_type"], df["pred_type"])
    return cm.reindex(index=ALL_LABELS, columns=ALL_LABELS, fill_value=0)


def compute_prf(df: pd.DataFrame) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        df["gold_type"], df["pred_type"], labels=LABELS, zero_division=0
    )
    return pd.DataFrame(
        {"type": LABELS, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )


def compute_error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    tp = df[(df["gold_type"] != "O") & (df["gold_type"] == df["pred_type"])]["gold_type"].value_counts()
    fn = df[(df["gold_type"] != "O") & (df["pred_type"] == "O")]["gold_type"].value_counts()
    fp = df[(df["gold_type"] == "O") & (df["pred_type"] != "O")]["pred_type"].value_counts()
    confused = df[(df["gold_type"] != "O") & (df["pred_type"] != "O") & (df["gold_type"] != df["pred_type"])]
    confused_as_gold = confused["gold_type"].value_counts()
    return pd.DataFrame(
        {
            "TP": tp.reindex(LABELS, fill_value=0),
            "FN_missed": fn.reindex(LABELS, fill_value=0),
            "FP_spurious": fp.reindex(LABELS, fill_value=0),
            "type_confused": confused_as_gold.reindex(LABELS, fill_value=0),
        }
    )


def compute_column_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """For each of the six gold NE-* columns (NE-COARSE-LIT itself plus the other five):
    how many tokens does that column actually tag as an entity (n_tagged, i.e. not "O"),
    and of those, how many does Gliner agree with (n_matched)? Restricting to n_tagged
    avoids the trivial-agreement problem of counting O/O as a "match" -- since ~87% of
    tokens are O in every column, a plain whole-dataset agreement rate is ~85-88% for
    every column regardless of how well it actually tracks Gliner."""
    rows = []
    for column in ["NE-COARSE-LIT"] + OTHER_GOLD_COLUMNS:
        type_col = "gold_type" if column == "NE-COARSE-LIT" else column + "_type"
        tagged = df[type_col] != "O"
        n_tagged = int(tagged.sum())
        n_matched = int((tagged & (df[type_col] == df["pred_type"])).sum())
        match_rate = n_matched / n_tagged if n_tagged else 0.0
        rows.append({"column": column, "n_tagged": n_tagged, "n_matched": n_matched, "match_rate": match_rate})
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm: pd.DataFrame, out_path: Path) -> None:
    blue_cmap = LinearSegmentedColormap.from_list("blue_sequential", BLUE_SEQUENTIAL)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)
    values = cm.to_numpy()

    # The (O, O) cell (tokens correctly left untagged) dwarfs every other cell and would
    # flatten the color scale; it's excluded from the color mapping (shown as a fixed
    # dark cell) but its true count is still printed as a label.
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
    ax.set_title("Token-level confusion matrix: gold vs Gliner", color=PRIMARY_INK)

    last = len(ALL_LABELS) - 1
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            is_o_o_cell = i == last and j == last
            color = "white" if is_o_o_cell or values[i, j] > vmax * 0.5 else PRIMARY_INK
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="token count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_prf(prf_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    colors = [CATEGORICAL["blue"], CATEGORICAL["aqua"], CATEGORICAL["yellow"]]

    x = range(len(prf_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offsets = [xi + (i - 1) * width for xi in x]
        bars = ax.bar(offsets, prf_df[metric], width=width, label=metric.capitalize(), color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED_INK, padding=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(prf_df["type"], color=PRIMARY_INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color=PRIMARY_INK)
    ax.set_title("Precision / Recall / F1 per entity type (token-level)", color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_alignability(align_df: pd.DataFrame, n_gliner_entities: int, out_path: Path) -> None:
    """Bullet-style bar per column: a pale full-length bar for n_tagged (how many tokens
    this column marks as an entity at all), overlaid with a solid bar for n_matched (how
    many of those Gliner agrees with) -- so the chart shows coverage and match rate
    together instead of a single number that both O/O agreement and real matches feed."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    y = range(len(align_df))
    ax.barh(y, align_df["n_tagged"], color=BLUE_SEQUENTIAL[1], label="Tagged as an entity (n_tagged)")
    ax.barh(y, align_df["n_matched"], color=CATEGORICAL["blue"], label="...and matches Gliner (n_matched)")

    for yi, (n_tagged, n_matched, rate) in enumerate(zip(align_df["n_tagged"], align_df["n_matched"], align_df["match_rate"])):
        ax.text(n_tagged + max(align_df["n_tagged"]) * 0.02, yi, f"{n_matched:,} / {n_tagged:,}  ({rate:.1%})",
                va="center", ha="left", fontsize=8, color=MUTED_INK)

    ax.axvline(
        n_gliner_entities, color=CATEGORICAL["red"], linestyle="--", linewidth=1.5,
        label=f"Total entities Gliner found ({n_gliner_entities:,})",
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(align_df["column"], color=PRIMARY_INK)
    ax.set_xlabel("Tokens tagged as an entity by this column", color=PRIMARY_INK)
    ax.set_title("Of the tokens each gold column tags as an entity, how many match Gliner?", color=PRIMARY_INK)
    ax.set_xlim(0, max(align_df["n_tagged"].max(), n_gliner_entities) * 1.35)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.invert_yaxis()
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-format", default=str(DEFAULT_TOKEN_FORMAT), help="Token-format CSV (see ner_features_to_token_format.py)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save plots into")
    args = parser.parse_args()

    print("=== Step 1: Load and normalize token-format data ===")
    print(f"Loading {args.token_format}")
    df = load_and_normalize(Path(args.token_format))
    print(f"{df.shape[0]} tokens loaded")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 2: Confusion matrix (gold vs Gliner) ===")
    cm = compute_confusion_matrix(df)
    print(cm)
    plot_confusion_matrix(cm, figures_dir / "confusion_matrix_threshold0.5.png")
    print(f"Saved {figures_dir / 'confusion_matrix_threshold0.5.png'}")

    print("=== Step 3: Precision / recall / F1 per type ===")
    prf_df = compute_prf(df)
    print(prf_df.to_string(index=False))
    plot_prf(prf_df, figures_dir / "precision_recall_f1_threshold0.5.png")
    print(f"Saved {figures_dir / 'precision_recall_f1_threshold0.5.png'}")

    print("=== Step 4: TP / FN / FP / type-confusion breakdown per type ===")
    breakdown_df = compute_error_breakdown(df)
    print(breakdown_df)

    print("=== Step 5: Of each gold column's tagged tokens, how many match Gliner? ===")
    align_df = compute_column_alignment(df)
    n_gliner_entities = int((df["pred_type"] != "O").sum())
    print(align_df.to_string(index=False))
    print(f"Total entities Gliner found: {n_gliner_entities:,}")
    plot_alignability(align_df, n_gliner_entities, figures_dir / "alignability_threshold0.5.png")
    print(f"Saved {figures_dir / 'alignability_threshold0.5.png'}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
