"""Thin wrapper around src/modeling/plot_reliability_diagram.py (the project's one
generic plotting script, reused by every baseline/ablation/encoder comparison so far --
see docs/pipeline.md SS4) preset for phase2_expert's one comparison: this folder's MoE
model against phase2's full camembert_mlp model. Shells out rather than reimplementing
any plotting logic, and does NOT modify plot_reliability_diagram.py itself (no
DISPLAY_LABELS edit either) -- "camembert_experts" is passed as a plain --extra-score
label and shown as-is in the legend.

Usage:
    python src/phase2_expert/compare.py
    python src/phase2_expert/compare.py --experts-score data_phase2_expert/xlm-roberta_experts_scores.csv --experts-label xlm_roberta_experts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
PLOT_SCRIPT = REPO_ROOT / "src" / "modeling" / "plot_reliability_diagram.py"

DEFAULT_FULL_SCORE = REPO_ROOT / "data" / "data_phase2" / "camembert_mlp_scores.csv"
DEFAULT_EXPERTS_SCORE = REPO_ROOT / "data" / "data_phase2_expert" / "camembert_experts_scores.csv"
DEFAULT_EXPERTS_LABEL = "camembert_experts"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures" / "phase2_expert"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-score", default=str(DEFAULT_FULL_SCORE), help="phase2's full camembert_mlp scores CSV (see phase2/evaluate.py)")
    parser.add_argument("--experts-score", default=str(DEFAULT_EXPERTS_SCORE), help="phase2_expert's scores CSV (see evaluate.py in this folder)")
    parser.add_argument("--experts-label", default=DEFAULT_EXPERTS_LABEL, help="Legend label for --experts-score")
    parser.add_argument("--split", default="test", help="Passed through to plot_reliability_diagram.py (default: test)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the plot into")
    args = parser.parse_args()

    cmd = [
        sys.executable, str(PLOT_SCRIPT),
        "--camembert-mlp-score", args.full_score,
        "--extra-score", f"{args.experts_label}={args.experts_score}",
        "--split", args.split,
        "--figures-dir", args.figures_dir,
    ]
    print(f"=== Running: {' '.join(cmd)} ===")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
