#!/bin/bash

# CPU-only counterpart of script.sh -- for running the same active pipeline steps
# directly on a node with no GPU (e.g. a login/head node), no SLURM submission
# involved (run with `bash script_cpu.sh`, not `sbatch`). torch auto-detects
# cuda.is_available()==False and every phase2 script (evaluate.py, train.py) already
# falls back to CPU on its own -- this file just skips the SBATCH directives and
# nvidia-smi/GPU checks that don't apply here.
#
# Mirrors whichever steps are currently active (uncommented) in script.sh -- keep the
# two in sync. build_candidate_windows.py/tokenize_windows.py/train.py are commented
# out here too since checkpoints/hipe2020_fr/gliner/phase2/mbert_mlp.pt and
# data/hipe2020_fr/gliner/data_phase2/phase2_candidate_windows.jsonl already exist.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"
echo "Working directory: $(pwd)"

ENVIRONMENT_NAME="cafe"

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate ${ENVIRONMENT_NAME}

python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

################################################################################
## hipe2020_fr / gliner -- Phase 2 base model: evaluate + plot (CPU)
################################################################################

NER_BASE=data/hipe2020_fr/gliner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv
PHASE1_OUT=data/hipe2020_fr/gliner/data_baseline
PHASE2_OUT=data/hipe2020_fr/gliner/data_phase2
PHASE2_FIGS=figures/hipe2020_fr/gliner/phase2
PHASE2_CKPT=checkpoints/hipe2020_fr/gliner/phase2

# --- Phase 2: candidate windows (shared by base/expert/simple, only needs building once) ---
python src/phase2/base/build_candidate_windows.py \
  --load-data $DATA_SRC \
  --label-reliability $NER_BASE/label_reliability_type_only.csv \
  --out $PHASE2_OUT/phase2_candidate_windows.jsonl

# Verify tokenization/truncation before training -- optional, train.py's own Dataset
# already tokenizes+verifies on the fly, this is just a standalone stats/sanity pass.
python src/phase2/base/tokenize_windows.py \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl

# --- Phase 2 base model: frozen mBERT (DEFAULT_ENCODER_NAME) + side embeddings + simple pooling + MLP head ---
python src/phase2/base/train.py \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --out $PHASE2_CKPT/mbert_mlp.pt \
  --figures-dir $PHASE2_FIGS/train_tracking

python src/phase2/base/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --split test \
  --out $PHASE2_OUT/test_results/mbert_mlp_scores.csv

python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $NER_BASE/label_reliability_type_only.csv \
  --load-data $DATA_SRC \
  --platt-scaling-score $PHASE1_OUT/test_results/platt_scaling.csv \
  --logistic-score $PHASE1_OUT/test_results/logistic_regression.csv \
  --mlp-score $PHASE1_OUT/test_results/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp \
  --figures-dir $PHASE2_FIGS

python src/phase2/base/check.py --windows $PHASE2_OUT/phase2_candidate_windows.jsonl                # first 5, no split filter
python src/phase2/base/check.py --windows $PHASE2_OUT/phase2_candidate_windows.jsonl --n 3 --split test
