"""Compare GLiNER2's phase2 camembert_mlp model against historical-ner-baseline's own
phase2 camembert_mlp model (see extract_ner_features.py in this folder), plus each
model's raw ner_score and each model's phase2_expert (K=4 latent-expert MoE head) result.

Why this can't just shell out to src/phase1/modeling/plot_reliability_diagram.py the way
phase2/simple/compare.py and phase2/expert/compare.py do: that script's load_and_merge()
joins every score CSV onto ONE shared --label-reliability base by (document_id,
sentence_id, start_token_id, end_token_id), and validate_scores_present() then requires
every candidate being plotted to have a matching row in every score file. That's fine for
ablations/encoder swaps, which all score the SAME GLiNER2 candidate pool -- but GLiNER2
and historical-ner-baseline are two different NER models that find two different sets of
spans over the same text, so their candidate keys barely overlap at all. Forcing them
through one shared join would either crash validate_scores_present or silently compare
almost nothing.

Instead, this script computes each model's calibration/discrimination numbers
independently -- its own scores CSV joined only with its own label_reliability_type_only.csv,
never mixed with the other model's candidates -- then overlays the two independently
computed series on one figure. It imports (not subprocesses) plot_reliability_diagram.py's
plot_reliability_diagram/plot_metrics_bar/plot_roc_curve/plot_risk_coverage_curve directly:
those four functions already take pre-computed bins_df/metrics_df/series inputs, not the
raw merged dataframe, so no shared-key assumption leaks in through them -- only
load_and_merge/validate_scores_present (which this script deliberately doesn't call) would
have been a problem.

Usage:
    python src/ner/historical_ner/compare.py
    python src/ner/historical_ner/compare.py --split ""  # every candidate, no split filter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase1.modeling import plot_reliability_diagram as prd  # noqa: E402 (path setup above must run first)
from phase1.modeling.metrics import (  # noqa: E402
    auroc, brier_score_loss, excess_aurc, expected_calibration_error, maximum_calibration_error_from_bins,
    risk_coverage_curve,
)

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]

DEFAULT_GLINER_SCORE = REPO_ROOT / "data" / "data_phase2" / "camembert_mlp_scores.csv"
DEFAULT_GLINER_LABELS = REPO_ROOT / "data" / "data_baseline" / "label_reliability_type_only.csv"
GLINER_LABEL = "gliner"
GLINER_RAW_LABEL = "gliner_raw"

DEFAULT_GLINER_EXPERTS_SCORE = REPO_ROOT / "data" / "data_phase2_expert" / "camembert_experts_scores.csv"
GLINER_EXPERTS_LABEL = "gliner_experts"

DEFAULT_GLINER_SIMPLE_SCORE = REPO_ROOT / "data" / "data_phase2_simple" / "camembert_simple_mlp_average_scores.csv"
GLINER_SIMPLE_LABEL = "gliner_simple"

DEFAULT_HISTORICAL_SCORE = Path(__file__).parent / "data" / "camembert_mlp_scores.csv"
DEFAULT_HISTORICAL_LABELS = Path(__file__).parent / "data" / "label_reliability_type_only.csv"
HISTORICAL_LABEL = "historical_ner"
HISTORICAL_RAW_LABEL = "historical_ner_raw"

DEFAULT_HISTORICAL_EXPERTS_SCORE = Path(__file__).parent / "data" / "camembert_experts_scores.csv"
HISTORICAL_EXPERTS_LABEL = "historical_ner_experts"

DEFAULT_HISTORICAL_SIMPLE_SCORE = Path(__file__).parent / "data" / "camembert_simple_mlp_average_scores.csv"
HISTORICAL_SIMPLE_LABEL = "historical_ner_simple"

DEFAULT_FIGURES_DIR = Path(__file__).parent / "figures" / "phase2"

# Nicer legend/title text for these labels -- merged into prd.DISPLAY_LABELS so the
# imported plotting functions pick them up automatically.
DISPLAY_LABELS = {
    GLINER_LABEL: "GLiNER2 + CamemBERT MLP",
    GLINER_RAW_LABEL: "GLiNER2 raw ner_score",
    GLINER_EXPERTS_LABEL: "GLiNER2 + CamemBERT MoE experts",
    GLINER_SIMPLE_LABEL: "GLiNER2 + CamemBERT MLP (simple/marker-prompt, pool=average)",
    HISTORICAL_LABEL: "historical-ner-baseline + CamemBERT MLP",
    HISTORICAL_RAW_LABEL: "historical-ner-baseline raw ner_score",
    HISTORICAL_EXPERTS_LABEL: "historical-ner-baseline + CamemBERT MoE experts",
    HISTORICAL_SIMPLE_LABEL: "historical-ner-baseline + CamemBERT MLP (simple/marker-prompt, pool=average)",
}


def load_scores_and_labels(score_path: Path, labels_path: Path, split: str) -> pd.DataFrame:
    """Join one model's own scores CSV (document_id/sentence_id/start_token_id/
    end_token_id/split/ner_score/calibrated_score -- see phase2/evaluate.py) with that
    SAME model's own label_reliability_type_only.csv (adds reliability_score, renamed
    label_reliable) -- both keyed on this one model's own candidate pool, never merged
    against the other model's."""
    scores_df = pd.read_csv(score_path)
    if split:
        scores_df = scores_df[scores_df["split"] == split]
    labels_df = pd.read_csv(labels_path)[KEY_COLS + ["reliability_score"]].rename(
        columns={"reliability_score": "label_reliable"}
    )
    merged = scores_df.merge(labels_df, on=KEY_COLS, how="left")
    n_missing = int(merged["label_reliable"].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} candidate(s) in {score_path} have no matching row in {labels_path} -- stale files?")
    return merged


def compute_all_metrics(df: pd.DataFrame, score_col: str = "calibrated_score") -> dict:
    """Every number/curve plot_reliability_diagram.py's four plots need, computed once
    per model from that model's own (scores, labels) pair. score_col lets the same
    function compute either the phase2 model's calibrated_score or the raw ner_score
    already carried in every scores CSV (see phase2/evaluate.py's KEY_COLS)."""
    labels_arr = df["label_reliable"].to_numpy()
    scores = df[score_col].to_numpy()
    ece, bins_df = expected_calibration_error(scores, labels_arr)
    return {
        "n": len(df),
        "labels_arr": labels_arr,
        "scores": scores,
        "brier": brier_score_loss(labels_arr, scores),
        "ece": ece,
        "mce": maximum_calibration_error_from_bins(bins_df),
        "auroc": auroc(scores, labels_arr),
        "e_aurc": excess_aurc(scores, labels_arr),
        "bins_df": bins_df,
        "rc_df": risk_coverage_curve(scores, labels_arr),
    }


def plot_group(group_name: str, labels: list[str], metrics_by_label: dict, colors: dict, figures_dir: Path) -> None:
    """Runs plot_reliability_diagram.py's four plots + bins CSVs for exactly this subset
    of labels, into its own figures_dir/group_name/ subfolder -- kept separate from other
    groups so each figure only shows the comparison it's meant for (see module docstring:
    default-vs-raw is one question, architecture-vs-architecture is a different one, and
    cramming both into one 8-line plot makes neither easy to read)."""
    group_dir = figures_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Group {group_name!r}: {labels} ---")

    series = [(metrics_by_label[l]["bins_df"], colors[l], l) for l in labels]
    out_path = group_dir / f"reliability_diagram_{'_'.join(labels)}.png"
    prd.plot_reliability_diagram(series, out_path)
    print(f"Saved {out_path}")

    metrics_df = pd.DataFrame({
        "metric": ["Brier score", "ECE", "MCE", "AUROC", "E-AURC"],
        **{l: [metrics_by_label[l][k] for k in ("brier", "ece", "mce", "auroc", "e_aurc")] for l in labels},
    })
    out_path = group_dir / f"metrics_bar_{'_'.join(labels)}.png"
    prd.plot_metrics_bar(metrics_df, colors, out_path)
    print(f"Saved {out_path}")

    series = [(metrics_by_label[l]["labels_arr"], metrics_by_label[l]["scores"], colors[l], l) for l in labels]
    out_path = group_dir / f"roc_curve_{'_'.join(labels)}.png"
    prd.plot_roc_curve(series, out_path)
    print(f"Saved {out_path}")

    series = [(metrics_by_label[l]["rc_df"], metrics_by_label[l]["e_aurc"], colors[l], l) for l in labels]
    out_path = group_dir / f"risk_coverage_{'_'.join(labels)}.png"
    prd.plot_risk_coverage_curve(series, out_path)
    print(f"Saved {out_path}")

    for l in labels:
        bins_out = group_dir / f"bins_{l}.csv"
        metrics_by_label[l]["bins_df"].to_csv(bins_out, index=False)
        print(f"Saved {bins_out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gliner-score", default=str(DEFAULT_GLINER_SCORE), help="GLiNER2's phase2 camembert_mlp scores CSV (see phase2/evaluate.py)")
    parser.add_argument("--gliner-labels", default=str(DEFAULT_GLINER_LABELS), help="GLiNER2's label_reliability_type_only.csv")
    parser.add_argument("--gliner-experts-score", default=str(DEFAULT_GLINER_EXPERTS_SCORE), help="GLiNER2's phase2_expert camembert_experts scores CSV (see phase2_expert/evaluate.py)")
    parser.add_argument("--gliner-simple-score", default=str(DEFAULT_GLINER_SIMPLE_SCORE), help="GLiNER2's phase2_simple camembert_simple_mlp_average scores CSV (see phase2_simple/evaluate.py)")
    parser.add_argument("--historical-score", default=str(DEFAULT_HISTORICAL_SCORE), help="historical-ner-baseline's phase2 camembert_mlp scores CSV")
    parser.add_argument("--historical-labels", default=str(DEFAULT_HISTORICAL_LABELS), help="historical-ner-baseline's label_reliability_type_only.csv")
    parser.add_argument("--historical-experts-score", default=str(DEFAULT_HISTORICAL_EXPERTS_SCORE), help="historical-ner-baseline's phase2_expert camembert_experts scores CSV")
    parser.add_argument("--historical-simple-score", default=str(DEFAULT_HISTORICAL_SIMPLE_SCORE), help="historical-ner-baseline's phase2_simple camembert_simple_mlp_average scores CSV")
    parser.add_argument("--split", default="test", help="Filter each model's own scores CSV by its own split column (default: test); pass \"\" to use every candidate")
    parser.add_argument("--no-raw", action="store_true", help="Omit each model's raw ner_score series (included by default)")
    parser.add_argument("--no-experts", action="store_true", help="Omit each model's phase2_expert (MoE) series (included by default)")
    parser.add_argument("--no-simple", action="store_true", help="Omit each model's phase2_simple (marker-prompt, pool=average) series (included by default)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save plots into")
    args = parser.parse_args()

    print("=== Step 1: Load + join each model's own scores with its own labels (two independent candidate pools -- see module docstring) ===")
    gliner_df = load_scores_and_labels(Path(args.gliner_score), Path(args.gliner_labels), args.split)
    historical_df = load_scores_and_labels(Path(args.historical_score), Path(args.historical_labels), args.split)
    print(f"gliner: {len(gliner_df)} candidates, historical_ner: {len(historical_df)} candidates")

    print("=== Step 2: Compute each model's own calibration/discrimination metrics (calibrated_score, raw ner_score unless --no-raw, phase2_expert unless --no-experts, phase2_simple unless --no-simple) ===")
    metrics_by_label = {
        GLINER_LABEL: compute_all_metrics(gliner_df),
        HISTORICAL_LABEL: compute_all_metrics(historical_df),
    }
    if not args.no_raw:
        metrics_by_label[GLINER_RAW_LABEL] = compute_all_metrics(gliner_df, score_col="ner_score")
        metrics_by_label[HISTORICAL_RAW_LABEL] = compute_all_metrics(historical_df, score_col="ner_score")
    if not args.no_experts:
        gliner_experts_df = load_scores_and_labels(Path(args.gliner_experts_score), Path(args.gliner_labels), args.split)
        historical_experts_df = load_scores_and_labels(Path(args.historical_experts_score), Path(args.historical_labels), args.split)
        metrics_by_label[GLINER_EXPERTS_LABEL] = compute_all_metrics(gliner_experts_df)
        metrics_by_label[HISTORICAL_EXPERTS_LABEL] = compute_all_metrics(historical_experts_df)
    if not args.no_simple:
        gliner_simple_df = load_scores_and_labels(Path(args.gliner_simple_score), Path(args.gliner_labels), args.split)
        historical_simple_df = load_scores_and_labels(Path(args.historical_simple_score), Path(args.historical_labels), args.split)
        metrics_by_label[GLINER_SIMPLE_LABEL] = compute_all_metrics(gliner_simple_df)
        metrics_by_label[HISTORICAL_SIMPLE_LABEL] = compute_all_metrics(historical_simple_df)
    for label, m in metrics_by_label.items():
        print(f"{label}: n={m['n']} Brier={m['brier']:.4f} ECE={m['ece']:.4f} MCE={m['mce']:.4f} AUROC={m['auroc']:.4f} E-AURC={m['e_aurc']:.4f}")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        GLINER_RAW_LABEL: prd.CATEGORICAL_ORANGE,
        GLINER_LABEL: prd.CATEGORICAL_BLUE,
        GLINER_EXPERTS_LABEL: prd.CATEGORICAL_PURPLE,
        GLINER_SIMPLE_LABEL: "#5b6b73",
        HISTORICAL_RAW_LABEL: "#1a9e96",
        HISTORICAL_LABEL: prd.CATEGORICAL_RED,
        HISTORICAL_EXPERTS_LABEL: "#a56a3a",
        HISTORICAL_SIMPLE_LABEL: "#d64d9a",
    }
    prd.DISPLAY_LABELS.update(DISPLAY_LABELS)

    # Two separate comparisons, two separate subfolders -- each answers one question
    # instead of cramming everything into one crowded 8-line plot (see module docstring
    # and plot_group's docstring). Each label list is filtered down to whatever actually
    # got computed above, so --no-raw/--no-experts/--no-simple degrade gracefully instead
    # of KeyError-ing.
    print("=== Group 1: default model (calibrated) vs raw ner_score, both NER sources ===")
    default_vs_raw_labels = [l for l in (GLINER_RAW_LABEL, GLINER_LABEL, HISTORICAL_RAW_LABEL, HISTORICAL_LABEL) if l in metrics_by_label]
    plot_group("compare_default_vs_raw", default_vs_raw_labels, metrics_by_label, colors, figures_dir)

    print("=== Group 2: three architectures (simple / default MLP / MoE experts), historical-ner-baseline only ===")
    architectures_labels = [
        l for l in (HISTORICAL_SIMPLE_LABEL, HISTORICAL_LABEL, HISTORICAL_EXPERTS_LABEL)
        if l in metrics_by_label
    ]
    plot_group("compare_architectures", architectures_labels, metrics_by_label, colors, figures_dir)

    print("=== Done ===")


if __name__ == "__main__":
    main()
