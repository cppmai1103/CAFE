"""Plot a reliability diagram, a matching Brier score/ECE/MCE/AUROC/E-AURC metrics bar
chart, an ROC curve, a risk-coverage curve, and a bins_<labels>.csv table: ner_score (raw,
--raw-score), B1 (Platt scaling), B3 (logistic regression), the MLP baseline
(mlp_baseline.py), and/or the Phase 2 model (frozen encoder + MLP head,
checkpoints/phase2/<variant_name()>.pt, e.g. mbert_mlp.pt -- scored via phase2/base/evaluate.py)
are each drawn only if
explicitly requested -- raw is no longer included by default; pass --raw-score to add it,
same as every other score needs its own flag. At least one of --raw-score / the four
--*-score flags / --extra-score must be given, or there's nothing to plot. The reliability
diagram and bins table are calibration-only (per-bin computation from
metrics.expected_calibration_error/maximum_calibration_error_from_bins).
The ROC curve and risk-coverage curve are discrimination-only (metrics.roc_curve/
metrics.risk_coverage_curve) -- do higher scores rank reliable candidates above
unreliable ones, regardless of whether the score value itself is a calibrated
probability -- and their scalar summaries (AUROC, E-AURC) are also included in the
metrics bar chart alongside Brier/ECE/MCE. See metrics.py's module docstring for the
calibration-vs-discrimination distinction.

bins_<labels>.csv columns: bin (the score range, e.g. "0.7-0.8"), true (empirical
reliability rate among that bin's candidates), raw (raw score's own average there, always
present regardless of --raw-score -- see below), and platt_scaling/logistic (that same
group of candidates' average under each other score, if given) -- candidates are grouped
into bins by the RAW score only, so every column in a row describes the exact same set of
candidates (build_bins_table). Each score column also gets a matching delta_<label> =
<label> - true: positive means overconfident in that bin, negative means underconfident.
Binning is always anchored to ner_score even when --raw-score isn't passed (raw just
won't appear as its own plotted line/legend entry in that case) -- build_bins_table's
whole methodology depends on a fixed reference score to bin by.

--platt-scaling-score / --logistic-score / --mlp-score / --camembert-mlp-score each take a
path to a CSV with the join keys (document_id, sentence_id, start_token_id, end_token_id) plus
a column literally named calibrated_score (see gliner/label_reliability.py's
OUTPUT_COLUMNS shape for the join keys). --extra-score LABEL=PATH (repeatable) adds any
number of further named series in the same shape, cycling through EXTRA_COLOR_PALETTE for
color -- meant for comparing phase2/train.py's --no-* ablation variants side by side,
which don't each get their own fixed flag.

--label-reliability (default: label_reliability_type_only.csv, see
gliner/label_reliability.py) supplies the join keys, the raw score (ner_score), and the
ground truth (reliability_score).

A reliability diagram bins candidates by predicted confidence and plots each bin's
empirical accuracy (x) -- i.e. the fraction of candidates in that bin that are actually
reliable -- against its mean predicted confidence (y). Perfect calibration is the y=x
diagonal; a curve above it means the score is underconfident in that range, below means
overconfident.

Splitting by train/val/test: label_reliability_type_only.csv has no split column
of its own (it's computed over every candidate) -- pass --load-data + --split (default:
test, per docs/phase1_manual_features.md's "test: final evaluation only") to filter to one
split via the document-level split that data carries; pass --split "" to skip filtering
and use every candidate.

Usage:
    python src/phase1/modeling/plot_reliability_diagram.py --raw-score                  # raw only
    python src/phase1/modeling/plot_reliability_diagram.py --raw-score --platt-scaling-score data/data_baseline/test_results/platt_scaling.csv
    python src/phase1/modeling/plot_reliability_diagram.py --logistic-score data/data_baseline/test_results/logistic_regression.csv   # raw omitted on purpose
    python src/phase1/modeling/plot_reliability_diagram.py --raw-score --platt-scaling-score data/data_baseline/test_results/platt_scaling.csv --logistic-score data/data_baseline/test_results/logistic_regression.csv --mlp-score data/data_baseline/test_results/mlp_baseline.csv --camembert-mlp-score data_phase2/phase2_scores.csv
    python src/phase1/modeling/plot_reliability_diagram.py --camembert-mlp-score data_phase2/camembert_mlp_scores.csv --extra-score without_ner_score=data_phase2/camembert_mlp_without_ner_score_scores.csv --extra-score without_type=data_phase2/camembert_mlp_without_type_scores.csv --figures-dir figures/ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ner.label_reliability import default_out_path as default_label_reliability_path
from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA
from phase1.modeling.metrics import auroc, brier_score_loss, excess_aurc, expected_calibration_error, maximum_calibration_error_from_bins, risk_coverage_curve, roc_curve

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_baseline"
DEFAULT_LABEL_RELIABILITY = default_label_reliability_path("type_only")
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "figures" / "modeling"

CATEGORICAL_RED = "#e34948"
CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_ORANGE = "#e8871e"
CATEGORICAL_PURPLE = "#8e5cd9"
STATUS_GOOD = "#0ca30c"

# Rotating palette for --extra-score series (e.g. ablation variants) that don't have
# their own fixed color -- distinct from the 5 fixed colors above.
EXTRA_COLOR_PALETTE = ["#1a9e96", "#a56a3a", "#d64d9a", "#5b6b73", "#8a8a2f", "#3f5fbf"]
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]
DATASET_NAME = "hipe2020_fr"  # the only dataset this project's pipeline runs on

# (CLI arg attr, column name to give it after merging, display color, legend label)
RAW_LABEL = "raw"
PLATT_LABEL = "platt_scaling"
LOGISTIC_LABEL = "logistic"
MLP_LABEL = "mlp"
CAMEMBERT_MLP_LABEL = "camembert_mlp"

DISPLAY_LABELS = {
    RAW_LABEL: "raw ner_score",
    PLATT_LABEL: "platt-calibrated",
    LOGISTIC_LABEL: "logistic regression",
    MLP_LABEL: "MLP baseline",
    # CAMEMBERT_MLP_LABEL is just this flag's fixed internal key/color slot for "phase2's
    # full model", whichever encoder that currently is (see phase2/base/model.py's
    # DEFAULT_ENCODER_NAME) -- this default text is a fallback only; callers should pass
    # --camembert-mlp-label <variant_name()> (e.g. "mbert_mlp") so the legend matches the
    # label the same scores CSV gets everywhere else it's plotted.
    CAMEMBERT_MLP_LABEL: "Phase 2 (full model)",
    # Cross-encoder comparison (--extra-score labels used in script.sh) -- not tied to a
    # dedicated CLI flag like the ones above, just nicer legend text for these labels.
    "mbert": "mBERT + MLP",
    "mdeberta_v3": "mDeBERTa-v3 + MLP",
    "multilingual_e5": "multilingual-E5 + MLP",
    "xlm_roberta": "XLM-RoBERTa + MLP",
    "camembert_simple_mlp": "CamemBERT + MLP (simple/marker-prompt, pool=one)",
    "camembert_simple_mlp_average": "CamemBERT + MLP (simple/marker-prompt, pool=average)",
    "camembert_mlp_without_dict_flag": "CamemBERT + MLP (no dict flag)",
}


def load_and_merge(label_reliability_path: Path, score_paths: dict[str, Path]) -> tuple[pd.DataFrame, list[str]]:
    """Join the base raw-score/label file with whichever of score_paths (label -> CSV
    path) were actually given. Returns (merged_df, labels_present) -- labels_present is
    the subset of score_paths' keys that were merged in, giving the column
    "<label>_calibrated_score" in merged_df for each. A left merge, deliberately NOT
    validated for completeness here -- a score file covering only one split (e.g.
    phase2/evaluate.py, which only scores --split test by default, since scoring every
    candidate through a transformer is far more expensive than the sklearn/tabular
    baselines) is expected to leave NaNs for every other split's candidates. Call
    validate_scores_present AFTER filtering to the split actually being plotted."""
    base_df = pd.read_csv(label_reliability_path)

    labels_present = []
    for label, path in score_paths.items():
        labels_present.append(label)
        score_df = pd.read_csv(path)[KEY_COLS + ["calibrated_score"]].rename(
            columns={"calibrated_score": f"{label}_calibrated_score"}
        )
        before = len(base_df)
        base_df = base_df.merge(score_df, on=KEY_COLS, how="left")
        assert len(base_df) == before, f"merge with {path} changed row count -- it isn't uniquely keyed by {KEY_COLS}"

    return base_df, labels_present


def validate_scores_present(df: pd.DataFrame, labels_present: list[str], score_paths: dict[str, Path]) -> None:
    """Raises if any candidate actually being plotted (i.e. already filtered to the
    requested --split) is missing a score for one of labels_present -- run this AFTER
    the --split filter, not before, so a score file that only covers one split (see
    load_and_merge) isn't penalized for splits that were filtered out anyway."""
    for label in labels_present:
        col = f"{label}_calibrated_score"
        if df[col].isna().any():
            n_missing = int(df[col].isna().sum())
            raise ValueError(
                f"{n_missing} candidate(s) being plotted have no matching row in {score_paths[label]} -- "
                f"is it stale, or does it only cover a different --split?"
            )


def default_out_path(figures_dir: Path, labels: list[str]) -> Path:
    return figures_dir / f"reliability_diagram_{'_'.join(labels)}.png"


def default_metrics_bar_out_path(figures_dir: Path, labels: list[str]) -> Path:
    return figures_dir / f"metrics_bar_{'_'.join(labels)}.png"


def default_bins_table_out_path(figures_dir: Path, labels: list[str]) -> Path:
    return figures_dir / f"bins_{'_'.join(labels)}.csv"


def default_roc_curve_out_path(figures_dir: Path, labels: list[str]) -> Path:
    return figures_dir / f"roc_curve_{'_'.join(labels)}.png"


def default_risk_coverage_out_path(figures_dir: Path, labels: list[str]) -> Path:
    return figures_dir / f"risk_coverage_{'_'.join(labels)}.png"


def build_bins_table(
    raw_scores: np.ndarray, labels_arr: np.ndarray, other_scores: dict[str, np.ndarray], n_bins: int = 10
) -> pd.DataFrame:
    """One row per bin over the raw score's range (fixed 0.0-1.0, n_bins equal-width
    bins, same edges/inclusivity convention as metrics.expected_calibration_error) --
    "true" is the empirical accuracy (actual reliability rate) among that bin's
    candidates, "raw" is the raw score's own average there, and each of other_scores is
    that SAME group of candidates' average under that other score -- so every column in
    a row describes the exact same set of candidates, partitioned by the raw score.
    delta_<label> = <label> - true for every score column (raw included): positive means
    that score is overconfident in this bin, negative means underconfident -- the same
    per-bin gap ECE/MCE are built from, just signed and per score rather than averaged
    or maxed away."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (raw_scores > lo) & (raw_scores <= hi) if i > 0 else (raw_scores >= lo) & (raw_scores <= hi)
        count = int(in_bin.sum())
        row = {"bin": f"{lo:.1f}-{hi:.1f}", "count": count}
        true = float(labels_arr[in_bin].mean()) if count else np.nan
        row["true"] = true

        raw_avg = float(raw_scores[in_bin].mean()) if count else np.nan
        row[RAW_LABEL] = raw_avg
        row[f"delta_{RAW_LABEL}"] = raw_avg - true if count else np.nan

        for label, scores in other_scores.items():
            avg = float(scores[in_bin].mean()) if count else np.nan
            row[label] = avg
            row[f"delta_{label}"] = avg - true if count else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_reliability_diagram(series: list[tuple[pd.DataFrame, str, str]], out_path: Path, title_suffix: str = "") -> None:
    """series: list of (bins_df, color, label) -- bins_df from
    metrics.expected_calibration_error, one per score being compared. Any number of
    series is fine."""
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED_INK, label="Perfect calibration")

    for bins_df, color, label in series:
        valid = bins_df.dropna(subset=["avg_confidence"])
        ax.plot(valid["avg_confidence"], valid["accuracy"], color=color, linewidth=1, alpha=0.6)
        ax.scatter(
            valid["avg_confidence"], valid["accuracy"], s=valid["count"] / valid["count"].max() * 300 + 20,
            color=color, alpha=0.85, label=label,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability", color=PRIMARY_INK)
    ax.set_ylabel("Empirical accuracy (reliability_score rate)", color=PRIMARY_INK)
    ax.set_title(f"Reliability diagram{title_suffix}", color=PRIMARY_INK)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, label_colors: dict[str, str], out_path: Path, title_suffix: str = "") -> None:
    """Grouped bar chart: one group per metric (Brier score, ECE, MCE, AUROC, E-AURC),
    one bar per score present (raw, plus platt_scaling/logistic if given) within each
    group. Direction of "better" varies by metric: lower is better for Brier/ECE/MCE/
    E-AURC, higher is better for AUROC -- see metrics.py's module docstring."""
    labels = [c for c in metrics_df.columns if c != "metric"]
    metrics = metrics_df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for i, label in enumerate(labels):
        values = metrics_df[label].to_numpy()
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width,
            color=label_colors.get(label, MUTED_INK), label=DISPLAY_LABELS.get(label, label), alpha=0.9,
        )
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], fontsize=6, color=MUTED_INK, padding=2, rotation=90)

    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)  # headroom for the rotated bar-top labels
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value", color=PRIMARY_INK)
    ax.set_title(f"Metrics{title_suffix}", color=PRIMARY_INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(series: list[tuple[np.ndarray, np.ndarray, str, str]], out_path: Path, title_suffix: str = "") -> None:
    """series: list of (labels_arr, scores, color, label) -- one ROC curve per score
    being compared (metrics.roc_curve), annotated with its AUROC (metrics.auroc)."""
    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED_INK, label="Random (AUROC=0.500)")
    for labels_arr, scores, color, label in series:
        fpr, tpr, _ = roc_curve(labels_arr, scores)
        auc = auroc(scores, labels_arr)
        ax.plot(fpr, tpr, color=color, linewidth=1.5, label=f"{DISPLAY_LABELS.get(label, label)} (AUROC={auc:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate", color=PRIMARY_INK)
    ax.set_ylabel("True positive rate", color=PRIMARY_INK)
    ax.set_title(f"ROC curve{title_suffix}", color=PRIMARY_INK)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage_curve(series: list[tuple[pd.DataFrame, float, str, str]], out_path: Path, title_suffix: str = "") -> None:
    """series: list of (rc_df, e_aurc, color, label) -- one risk-coverage curve per score
    being compared (metrics.risk_coverage_curve), annotated with its E-AURC
    (metrics.excess_aurc)."""
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    for rc_df, e_aurc, color, label in series:
        ax.plot(rc_df["coverage"], rc_df["risk"], color=color, linewidth=1.5, label=f"{DISPLAY_LABELS.get(label, label)} (E-AURC={e_aurc:.4f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage (fraction of most-confident candidates kept)", color=PRIMARY_INK)
    ax.set_ylabel("Risk (error rate among kept candidates)", color=PRIMARY_INK)
    ax.set_title(f"Risk-coverage curve{title_suffix}", color=PRIMARY_INK)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=PRIMARY_INK, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--label-reliability", default=str(DEFAULT_LABEL_RELIABILITY),
        help="CSV with join keys + ner_score (raw) + reliability_score (see gliner/label_reliability.py)",
    )
    parser.add_argument(
        "--raw-score", action="store_true",
        help="Include raw ner_score as a plotted baseline -- previously always included by default, now opt-in like every other score",
    )
    parser.add_argument(
        "--platt-scaling-score", default=None, metavar="PATH",
        help="CSV with join keys + a calibrated_score column (B1, Platt scaling) -- omit to not draw this line",
    )
    parser.add_argument(
        "--logistic-score", default=None, metavar="PATH",
        help="CSV with join keys + a calibrated_score column (B3, logistic regression) -- omit to not draw this line",
    )
    parser.add_argument(
        "--mlp-score", default=None, metavar="PATH",
        help="CSV with join keys + a calibrated_score column (MLP baseline, mlp_baseline.py) -- omit to not draw this line",
    )
    parser.add_argument(
        "--camembert-mlp-score", default=None, metavar="PATH",
        help="CSV with join keys + a calibrated_score column (phase2's full model, phase2/base/evaluate.py) -- omit to not draw this line",
    )
    parser.add_argument(
        "--camembert-mlp-label", default=DISPLAY_LABELS[CAMEMBERT_MLP_LABEL],
        help="Legend text for --camembert-mlp-score -- pass the actual variant name (e.g. \"mbert_mlp\", from "
        "phase2.base.model.variant_name()) so it matches the label the same scores CSV gets everywhere else "
        f"it's plotted (default: {DISPLAY_LABELS[CAMEMBERT_MLP_LABEL]!r})",
    )
    parser.add_argument(
        "--extra-score", action="append", default=None, metavar="LABEL=PATH",
        help="Additional named score CSV, repeatable -- e.g. --extra-score without_ner_score=data_phase2/camembert_mlp_without_ner_score_scores.csv. "
        "For comparing ablation variants (phase2/train.py --no-*) that don't have their own fixed flag.",
    )
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV (for the --split filter)")
    parser.add_argument("--split", default="test", help="Filter to this document-level split before plotting; pass \"\" to use every candidate (default: test)")
    parser.add_argument("--out", default=None, help="Output PNG path (default: figures/reliability_diagram_<labels>.png)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into (ignored if --out is given)")
    args = parser.parse_args()

    score_paths = {}
    if args.platt_scaling_score:
        score_paths[PLATT_LABEL] = Path(args.platt_scaling_score)
    if args.logistic_score:
        score_paths[LOGISTIC_LABEL] = Path(args.logistic_score)
    if args.mlp_score:
        score_paths[MLP_LABEL] = Path(args.mlp_score)
    if args.camembert_mlp_score:
        score_paths[CAMEMBERT_MLP_LABEL] = Path(args.camembert_mlp_score)
        DISPLAY_LABELS[CAMEMBERT_MLP_LABEL] = args.camembert_mlp_label
    extra_labels = []
    for item in args.extra_score or []:
        label, sep, path = item.partition("=")
        if not sep:
            parser.error(f"--extra-score must be LABEL=PATH, got {item!r}")
        score_paths[label] = Path(path)
        extra_labels.append(label)

    if not args.raw_score and not score_paths:
        parser.error("nothing to plot -- pass --raw-score and/or one of --platt-scaling-score/--logistic-score/--mlp-score/--camembert-mlp-score/--extra-score")

    print(f"=== Step 1: Load {args.label_reliability} and merge in {list(score_paths)} ===")
    candidates_df, labels_present = load_and_merge(Path(args.label_reliability), score_paths)
    included = ([RAW_LABEL] if args.raw_score else []) + labels_present
    print(f"{len(candidates_df)} candidates joined; scores present: {included}")

    if args.split:
        print(f"=== Step 2: Filter to split={args.split!r} ===")
        print(f"Loading {args.load_data}")
        data_df = pd.read_csv(args.load_data, dtype={"TOKEN": str, "MISC": str},
        # pandas' default NA-string sentinels ("NA", "null", "nan", ...) would otherwise
        # silently corrupt a genuine OCR token whose text happens to collide with one of
        # them (confirmed: one real token in hipe2020_fr is literally "NA") into a float
        # NaN despite the dtype=str hint above -- dtype coercion happens AFTER NA
        # detection, so it can't prevent this. keep_default_na=False turns that off
        # entirely, and na_values restores it only for the two genuinely-numeric columns
        # that still need a blank cell to become NaN.
        keep_default_na=False, na_values={"sentence_ocr_mean": [""], "document_ocr_mean": [""], "dictionary_score": [""]})
        doc_to_split = data_df.drop_duplicates("document_id").set_index("document_id")["split"].to_dict()
        candidates_df = candidates_df[candidates_df["document_id"].map(doc_to_split) == args.split]
        print(f"{len(candidates_df)} candidates remain")
    else:
        print("=== Step 2: No --split given, using every candidate ===")

    validate_scores_present(candidates_df, labels_present, score_paths)

    labels_arr = candidates_df["reliability_score"].to_numpy().astype(int)

    print("=== Step 3: Compute ECE bins and Brier score / ECE / MCE per score ===")
    label_colors = {
        RAW_LABEL: CATEGORICAL_RED, PLATT_LABEL: CATEGORICAL_BLUE, LOGISTIC_LABEL: STATUS_GOOD,
        MLP_LABEL: CATEGORICAL_ORANGE, CAMEMBERT_MLP_LABEL: CATEGORICAL_PURPLE,
    }
    for i, label in enumerate(extra_labels):
        label_colors[label] = EXTRA_COLOR_PALETTE[i % len(EXTRA_COLOR_PALETTE)]
    score_columns = {RAW_LABEL: "ner_score"} if args.raw_score else {}
    score_columns.update({label: f"{label}_calibrated_score" for label in labels_present})

    series = []
    roc_series = []
    rc_series = []
    metrics_by_metric: dict[str, list[float]] = {"Brier score": [], "ECE": [], "MCE": [], "AUROC": [], "E-AURC": []}
    for label, col in score_columns.items():
        scores = candidates_df[col].to_numpy()
        ece, bins_df = expected_calibration_error(scores, labels_arr)
        mce = maximum_calibration_error_from_bins(bins_df)
        brier = brier_score_loss(labels_arr, scores)
        auc = auroc(scores, labels_arr)
        e_aurc = excess_aurc(scores, labels_arr)
        rc_df = risk_coverage_curve(scores, labels_arr)
        series.append((bins_df, label_colors[label], label))
        roc_series.append((labels_arr, scores, label_colors[label], label))
        rc_series.append((rc_df, e_aurc, label_colors[label], label))
        metrics_by_metric["Brier score"].append(brier)
        metrics_by_metric["ECE"].append(ece)
        metrics_by_metric["MCE"].append(mce)
        metrics_by_metric["AUROC"].append(auc)
        metrics_by_metric["E-AURC"].append(e_aurc)
        print(f"{label} ({col}): {bins_df['count'].sum()} candidates across {len(bins_df)} bins -- Brier={brier:.4f} ECE={ece:.4f} MCE={mce:.4f} AUROC={auc:.4f} E-AURC={e_aurc:.4f}")

    metrics_df = pd.DataFrame({"metric": list(metrics_by_metric)})
    for i, label in enumerate(score_columns):
        metrics_df[label] = [metrics_by_metric[m][i] for m in metrics_by_metric]

    all_labels = list(score_columns)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f" on {DATASET_NAME} {args.split} set" if args.split else f" on {DATASET_NAME} (all splits)"

    print("=== Step 4: Plot reliability diagram ===")
    reliability_out_path = Path(args.out) if args.out is not None else default_out_path(figures_dir, all_labels)
    plot_reliability_diagram(series, reliability_out_path, title_suffix)
    print(f"Saved {reliability_out_path}")

    print("=== Step 5: Plot metrics bar chart (Brier score, ECE, MCE, AUROC, E-AURC) ===")
    metrics_bar_out_path = default_metrics_bar_out_path(figures_dir, all_labels)
    plot_metrics_bar(metrics_df, label_colors, metrics_bar_out_path, title_suffix)
    print(f"Saved {metrics_bar_out_path}")

    print("=== Step 6: Plot ROC curve ===")
    roc_out_path = default_roc_curve_out_path(figures_dir, all_labels)
    plot_roc_curve(roc_series, roc_out_path, title_suffix)
    print(f"Saved {roc_out_path}")

    print("=== Step 7: Plot risk-coverage curve ===")
    rc_out_path = default_risk_coverage_out_path(figures_dir, all_labels)
    plot_risk_coverage_curve(rc_series, rc_out_path, title_suffix)
    print(f"Saved {rc_out_path}")

    print(f"=== Step 8: Save bins table (bin, true, {', '.join(all_labels)}) ===")
    other_scores = {label: candidates_df[score_columns[label]].to_numpy() for label in labels_present}
    # Bins are always anchored to raw ner_score (build_bins_table's own methodology --
    # "candidates are grouped into bins by the RAW score only", see its docstring), even
    # if --raw-score wasn't passed and raw isn't itself one of the plotted series above.
    bins_table = build_bins_table(candidates_df["ner_score"].to_numpy(), labels_arr, other_scores)
    print(bins_table.to_string(index=False))
    bins_table_out_path = default_bins_table_out_path(figures_dir, all_labels)
    bins_table.to_csv(bins_table_out_path, index=False)
    print(f"Saved {bins_table_out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
