"""Thin wrapper around src/phase1/modeling/plot_reliability_diagram.py (the project's one
generic plotting script, reused by every baseline/ablation/encoder comparison so far --
see docs/pipeline.md SS4) preset for phase2_expert's one comparison: this folder's MoE
model against phase2's full model. Shells out rather than reimplementing any plotting
logic -- DEFAULT_EXPERTS_LABEL is passed as a plain --extra-score label and shown as-is
in the legend, and DEFAULT_FULL_LABEL is passed through as --camembert-mlp-label so the
full model's legend text always matches its actual variant name (e.g. "mbert_mlp"), same
as every other comparison plot names it (see phase2/simple/compare.py).

Usage:
    python src/phase2/expert/compare.py
    python src/phase2/expert/compare.py --experts-score data_phase2_expert/test_results/xlm-roberta_experts_scores.csv --experts-label xlm_roberta_experts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase2.base.model import variant_name as phase2_variant_name
from phase2.expert.model import variant_name as phase2_expert_variant_name

PLOT_SCRIPT = REPO_ROOT / "src" / "phase1" / "modeling" / "plot_reliability_diagram.py"

# Labels are computed from phase2/phase2_expert's own variant_name() (not hand-typed
# strings) so they can never drift out of sync with what train.py/evaluate.py actually
# name a checkpoint/scores CSV -- see model.py's variant_name() docstring in each folder.
DEFAULT_FULL_LABEL = phase2_variant_name()
DEFAULT_FULL_SCORE = REPO_ROOT / "data" / "data_phase2" / "test_results" / f"{DEFAULT_FULL_LABEL}_scores.csv"
DEFAULT_EXPERTS_LABEL = phase2_expert_variant_name()
DEFAULT_EXPERTS_SCORE = REPO_ROOT / "data" / "data_phase2_expert" / "test_results" / f"{DEFAULT_EXPERTS_LABEL}_scores.csv"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "phase2_expert"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-score", default=str(DEFAULT_FULL_SCORE), help=f"phase2's full model scores CSV, default variant {DEFAULT_FULL_LABEL!r} (see phase2/evaluate.py)")
    parser.add_argument("--full-label", default=DEFAULT_FULL_LABEL, help="Legend label for --full-score")
    parser.add_argument("--experts-score", default=str(DEFAULT_EXPERTS_SCORE), help="phase2_expert's scores CSV (see evaluate.py in this folder)")
    parser.add_argument("--experts-label", default=DEFAULT_EXPERTS_LABEL, help="Legend label for --experts-score")
    parser.add_argument("--split", default="test", help="Passed through to plot_reliability_diagram.py (default: test)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()

    cmd = [
        sys.executable, str(PLOT_SCRIPT),
        "--camembert-mlp-score", args.full_score,
        "--camembert-mlp-label", args.full_label,
        "--extra-score", f"{args.experts_label}={args.experts_score}",
        "--split", args.split,
        "--figures-dir", args.figures_dir,
    ]
    print(f"=== Running: {' '.join(cmd)} ===")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
