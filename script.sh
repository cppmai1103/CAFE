#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=01:00:00
#SBATCH --job-name=test
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=0
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="cafe"

# Pin CWD to the directory `sbatch` was run from. Don't use $0 here -- on this
# cluster SLURM stages the submitted script into a per-job spool dir
# (/var/spool/slurm/d/job<ID>/) and runs it from there, so $0 resolves to that
# spool copy, not to confidence-aware/. $SLURM_SUBMIT_DIR is set by SLURM itself
# to the real submission directory and isn't affected by the staging.
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"


module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh


if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  echo "Env '${ENVIRONMENT_NAME}' not found, creating it with python=${PYTHON_VERSION}"
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
else
  echo "Env '${ENVIRONMENT_NAME}' already exists, skipping creation"
fi
conda activate ${ENVIRONMENT_NAME}

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r src/requirements.txt

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

################################################################################
## hipe2020_fr / gliner -- Phase 1 (manual features + B0/B1/B3/MLP baselines)
## and Phase 2 base model (frozen mBERT + MLP head), all inputs/outputs
## redirected under data/hipe2020_fr/gliner/ so they never collide with another
## dataset/NER-source combo's own Phase 1/2 artifacts.
################################################################################

NER_BASE=data/hipe2020_fr/gliner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv
PHASE1_OUT=data/hipe2020_fr/gliner/data_baseline
PHASE1_FIGS=figures/hipe2020_fr/gliner/modeling
PHASE1_CKPT=checkpoints/hipe2020_fr/gliner/phase1
PHASE2_OUT=data/hipe2020_fr/gliner/data_phase2
PHASE2_FIGS=figures/hipe2020_fr/gliner/phase2
PHASE2_CKPT=checkpoints/hipe2020_fr/gliner/phase2

## --- Phase 1: manual feature extraction ---
# python src/phase1/feature_extraction/extract_ocr_features.py \
#   --load-data $DATA_SRC \
#   --ner-features $NER_BASE/deduplicate_ner_features.csv \
#   --out $PHASE1_OUT/ocr_features.csv

# python src/phase1/feature_extraction/extract_context_features.py \
#   --load-data $DATA_SRC \
#   --ner-features $NER_BASE/deduplicate_ner_features.csv \
#   --out $PHASE1_OUT/context_features.csv

# python src/phase1/feature_extraction/prepare_data_logistic.py \
#   --load-data $DATA_SRC \
#   --ner-features $NER_BASE/deduplicate_ner_features.csv \
#   --ocr-features $PHASE1_OUT/ocr_features.csv \
#   --context-features $PHASE1_OUT/context_features.csv \
#   --label-reliability $NER_BASE/label_reliability_type_only.csv \
#   --out $PHASE1_OUT/logistic_regression_data.csv

## --- Phase 2: candidate windows (shared by base/expert/simple, only needs building once) ---
# python src/phase2/base/build_candidate_windows.py \
#   --load-data $DATA_SRC \
#   --label-reliability $NER_BASE/label_reliability_type_only.csv \
#   --out $PHASE2_OUT/phase2_candidate_windows.jsonl

# Verify tokenization/truncation before training -- optional, train.py's own Dataset
# already tokenizes+verifies on the fly, this is just a standalone stats/sanity pass.
# python src/phase2/base/tokenize_windows.py \
#   --windows $PHASE2_OUT/phase2_candidate_windows.jsonl

## --- Phase 2 base model: frozen mBERT (DEFAULT_ENCODER_NAME) + side embeddings + simple pooling + MLP head ---
# python src/phase2/base/train.py \
#   --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
#   --out $PHASE2_CKPT/mbert_mlp.pt \
#   --figures-dir $PHASE2_FIGS/train_tracking

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