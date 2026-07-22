"""Thin wrapper around src/phase1/modeling/plot_reliability_diagram.py (the project's one
generic plotting script, reused by every baseline/ablation/encoder comparison so far --
see docs/pipeline.md SS4) preset for phase2_simple's 4-way comparison: both
type_confidence_pool modes (see model.py) against two phase2 references. Shells out
rather than reimplementing any plotting logic, so phase2_simple never drifts from how
every other score CSV in this project gets turned into a reliability diagram/metrics
bar/ROC/risk-coverage figure.

Two phase2 references, both included by default (their DEFAULT_*_LABEL is derived from
phase2/phase2_simple's own variant_name(), so it always names whichever encoder is
currently DEFAULT_ENCODER_NAME there, e.g. "mbert_mlp_without_dict_flag" -- never a
hand-typed string that can drift out of sync with model.py's actual naming):
  - <full>_without_dict_flag -- the apples-to-apples baseline: phase2_simple has no
    dictionary-quality side-channel at all (its marker block only ever states
    [Entity]/[Type]/[Confidence], never a dict-flag equivalent), so this is the fairest
    single comparison.
  - <full> model (full model, --full-score/--full-label) -- included too so you can see
    where phase2_simple lands relative to phase2's best full-featured model, not just the
    matched-features ablation.

Usage:
    python src/phase2/simple/compare.py
    python src/phase2/simple/compare.py --baseline-score data_phase2/test_results/camembert_mlp_without_ner_score_scores.csv --baseline-label camembert_mlp_without_ner_score
    python src/phase2/simple/compare.py --simple-average-score data_phase2_simple/test_results/xlm-roberta_simple_mlp_average_scores.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase2.base.model import variant_name as phase2_variant_name
from phase2.simple.model import variant_name as phase2_simple_variant_name

PLOT_SCRIPT = REPO_ROOT / "src" / "phase1" / "modeling" / "plot_reliability_diagram.py"
DATA_PHASE2_TEST_RESULTS_DIR = REPO_ROOT / "data" / "data_phase2" / "test_results"
DATA_PHASE2_SIMPLE_TEST_RESULTS_DIR = REPO_ROOT / "data" / "data_phase2_simple" / "test_results"

# Labels are computed from phase2/phase2_simple's own variant_name() (not hand-typed
# strings) so they can never drift out of sync with what train.py/evaluate.py actually
# name a checkpoint/scores CSV -- see model.py's variant_name() docstring in each folder.
DEFAULT_FULL_LABEL = phase2_variant_name()
DEFAULT_FULL_SCORE = DATA_PHASE2_TEST_RESULTS_DIR / f"{DEFAULT_FULL_LABEL}_scores.csv"
DEFAULT_BASELINE_LABEL = phase2_variant_name(use_dict_flag=False)
DEFAULT_BASELINE_SCORE = DATA_PHASE2_TEST_RESULTS_DIR / f"{DEFAULT_BASELINE_LABEL}_scores.csv"
DEFAULT_SIMPLE_ONE_LABEL = phase2_simple_variant_name()
DEFAULT_SIMPLE_ONE_SCORE = DATA_PHASE2_SIMPLE_TEST_RESULTS_DIR / f"{DEFAULT_SIMPLE_ONE_LABEL}_scores.csv"
DEFAULT_SIMPLE_AVERAGE_LABEL = phase2_simple_variant_name(type_confidence_pool="average")
DEFAULT_SIMPLE_AVERAGE_SCORE = DATA_PHASE2_SIMPLE_TEST_RESULTS_DIR / f"{DEFAULT_SIMPLE_AVERAGE_LABEL}_scores.csv"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "phase2_simple"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-score", default=str(DEFAULT_FULL_SCORE), help=f"phase2's full model scores CSV, default variant {DEFAULT_FULL_LABEL!r} (see phase2/evaluate.py)")
    parser.add_argument("--full-label", default=DEFAULT_FULL_LABEL, help="Legend label for --full-score")
    parser.add_argument("--baseline-score", default=str(DEFAULT_BASELINE_SCORE), help=f"phase2 ablation scores CSV to compare against (default: {DEFAULT_BASELINE_LABEL!r}, the fair apples-to-apples ablation -- see module docstring)")
    parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL, help="Legend label for --baseline-score")
    parser.add_argument("--simple-one-score", default=str(DEFAULT_SIMPLE_ONE_SCORE), help="phase2_simple scores CSV, type_confidence_pool='one' (marker-token) mode")
    parser.add_argument("--simple-one-label", default=DEFAULT_SIMPLE_ONE_LABEL, help="Legend label for --simple-one-score")
    parser.add_argument("--simple-average-score", default=str(DEFAULT_SIMPLE_AVERAGE_SCORE), help="phase2_simple scores CSV, type_confidence_pool='average' (value mean-pool) mode")
    parser.add_argument("--simple-average-label", default=DEFAULT_SIMPLE_AVERAGE_LABEL, help="Legend label for --simple-average-score")
    parser.add_argument("--split", default="test", help="Passed through to plot_reliability_diagram.py (default: test)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()

    cmd = [
        sys.executable, str(PLOT_SCRIPT),
        "--extra-score", f"{args.full_label}={args.full_score}",
        "--extra-score", f"{args.baseline_label}={args.baseline_score}",
        "--extra-score", f"{args.simple_one_label}={args.simple_one_score}",
        "--extra-score", f"{args.simple_average_label}={args.simple_average_score}",
        "--split", args.split,
        "--figures-dir", args.figures_dir,
    ]
    print(f"=== Running: {' '.join(cmd)} ===")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
