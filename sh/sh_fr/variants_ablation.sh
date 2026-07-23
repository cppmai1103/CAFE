#!/bin/bash
set -euo pipefail

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="cafe"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
echo "Working directory: $(pwd)"

# SLURM auto-saves a job's stdout/stderr to job-%j.out/job-%j.err; running locally instead
# (no sbatch), so do the same by hand -- tee keeps both streams live in the terminal while
# also saving them, mirroring SLURM's own after-the-fact log files.
LOG_DIR=logs
mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
exec > >(tee "$LOG_DIR/${RUN_ID}.out") 2> >(tee "$LOG_DIR/${RUN_ID}.err" >&2)
echo "Logging stdout to $LOG_DIR/${RUN_ID}.out, stderr to $LOG_DIR/${RUN_ID}.err"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  echo "Env '${ENVIRONMENT_NAME}' not found, creating it with python=${PYTHON_VERSION}"
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
else
  echo "Env '${ENVIRONMENT_NAME}' already exists, skipping creation"
fi
conda activate ${ENVIRONMENT_NAME}

# pip install torch --index-url https://download.pytorch.org/whl/cu124
# pip install -r src/requirements.txt

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

################################################################################
## hipe2020_fr / gliner -- model-variants ablation: Phase 1 baselines (B0 raw,
## B1 Platt, B3 logistic, MLP) vs Phase 2's three architectures --
##   base   -- frozen encoder + side embeddings (type/ner-score/dict-flag/target-flag)
##             + simple pooling + one MLP head (src/phase2/base/, already trained by
##             baseline_phase2base.sh, reused here rather than retrained).
##   simple -- same frozen encoder, but type/confidence are written directly into the
##             token sequence as text markers instead of side embeddings -- no new
##             trainable embeddings, classifier head only (src/phase2/simple/).
##   expert -- same base backbone, but the MLP head is replaced by a K=4 latent
##             mixture-of-experts head with a load-balancing auxiliary loss
##             (src/phase2/expert/).
## simple/expert both reuse base's candidate windows JSONL directly (same
## tokenizer/dataset pipeline for expert; simple's own Phase2SimpleWindowDataset
## reads the same file) -- no separate windows-build step for either.
##
## LEVEL matches baseline_phase2base.sh's own default/override convention
## (span_level_fuzzy | word_level_type_only) -- run this only after
## baseline_phase2base.sh has already produced that LEVEL's Phase 1 baseline
## scores, Phase 2 base checkpoint+scores, and candidate windows; this script does
## not rebuild any of them.
################################################################################

LEVEL=${LEVEL:-span_level_fuzzy}   # span_level_fuzzy | word_level_type_only
TRAIN_TEST_TAG=train_hipe_test_hipe   # trained on hipe2020_fr, evaluated on hipe2020_fr's own test split

NER_BASE=data/hipe2020_fr/gliner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv
LABEL_RELIABILITY=$NER_BASE/label_reliability_${LEVEL}.csv

PHASE1_OUT=$NER_BASE/level_ablation/$LEVEL
PHASE1_SCORES=$PHASE1_OUT/test_results/$TRAIN_TEST_TAG

PHASE2_OUT=data/hipe2020_fr/gliner/data_phase2/level_ablation/$LEVEL
WINDOWS=$PHASE2_OUT/phase2_candidate_windows.jsonl
BASE_SCORE=$PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv

VARIANTS_CKPT=checkpoints/hipe2020_fr/gliner/phase2/variants_ablation/$TRAIN_TEST_TAG
VARIANTS_OUT=data/hipe2020_fr/gliner/data_phase2/variants_ablation/$TRAIN_TEST_TAG
VARIANTS_FIGS=figures/hipe2020_fr/gliner/phase2/variants_ablation/$TRAIN_TEST_TAG

for f in "$WINDOWS" "$PHASE1_SCORES/platt_scaling.csv" "$PHASE1_SCORES/logistic_regression.csv" "$PHASE1_SCORES/mlp_baseline.csv" "$BASE_SCORE"; do
  if [ ! -f "$f" ]; then
    echo "Missing $f -- run baseline_phase2base.sh (LEVEL=$LEVEL) first." >&2
    exit 1
  fi
done

# --- Phase 2 simple: train + evaluate (type-confidence-pool=one, the default) ---
python src/phase2/simple/train.py \
  --windows $WINDOWS \
  --out $VARIANTS_CKPT/mbert_simple_mlp.pt \
  --figures-dir $VARIANTS_FIGS/train_tracking

python src/phase2/simple/evaluate.py \
  --checkpoint $VARIANTS_CKPT/mbert_simple_mlp.pt \
  --windows $WINDOWS \
  --split test \
  --out $VARIANTS_OUT/mbert_simple_mlp_scores.csv

# --- Phase 2 expert: train + evaluate (K=4 experts, lambda-balance=0.01, the defaults) ---
python src/phase2/expert/train.py \
  --windows $WINDOWS \
  --out $VARIANTS_CKPT/mbert_experts.pt \
  --figures-dir $VARIANTS_FIGS/train_tracking

python src/phase2/expert/evaluate.py \
  --checkpoint $VARIANTS_CKPT/mbert_experts.pt \
  --windows $WINDOWS \
  --split test \
  --out $VARIANTS_OUT/mbert_experts_scores.csv

# --- Compare every baseline + all 3 Phase 2 architectures in one plot ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $PHASE1_SCORES/platt_scaling.csv \
  --logistic-score $PHASE1_SCORES/logistic_regression.csv \
  --mlp-score $PHASE1_SCORES/mlp_baseline.csv \
  --camembert-mlp-score $BASE_SCORE \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$VARIANTS_OUT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$VARIANTS_OUT/mbert_experts_scores.csv \
  --figures-dir $VARIANTS_FIGS

################################################################################
## letemps_fr / gliner -- score-only cross-dataset check: reuses the hipe-trained
## Phase 2 simple/expert checkpoints above (no retraining), evaluated on
## letemps_fr's own test split, compared against the Phase 1 baselines' and
## Phase 2 base's already-existing letemps scores (from baseline_phase2base.sh's
## own letemps block).
################################################################################

TRAIN_TEST_TAG_LT=train_hipe_test_letemps

NER_BASE_LT=data/letemps_fr/gliner/data_baseline
DATA_SRC_LT=data/data_source/letemps/letemps_fr.csv
LABEL_RELIABILITY_LT=$NER_BASE_LT/label_reliability_${LEVEL}.csv

PHASE1_OUT_LT=$NER_BASE_LT/level_ablation/$LEVEL
PHASE1_SCORES_LT=$PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT

PHASE2_OUT_LT=data/letemps_fr/gliner/data_phase2/level_ablation/$LEVEL
WINDOWS_LT=$PHASE2_OUT_LT/phase2_candidate_windows.jsonl
BASE_SCORE_LT=$PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv

VARIANTS_OUT_LT=data/letemps_fr/gliner/data_phase2/variants_ablation/$TRAIN_TEST_TAG_LT
VARIANTS_FIGS_LT=figures/letemps_fr/gliner/phase2/variants_ablation/$TRAIN_TEST_TAG_LT

for f in "$WINDOWS_LT" "$PHASE1_SCORES_LT/platt_scaling.csv" "$PHASE1_SCORES_LT/logistic_regression.csv" "$PHASE1_SCORES_LT/mlp_baseline.csv" "$BASE_SCORE_LT" "$VARIANTS_CKPT/mbert_simple_mlp.pt" "$VARIANTS_CKPT/mbert_experts.pt"; do
  if [ ! -f "$f" ]; then
    echo "Missing $f -- run baseline_phase2base.sh (LEVEL=$LEVEL) and the hipe block above first." >&2
    exit 1
  fi
done

python src/phase2/simple/evaluate.py \
  --checkpoint $VARIANTS_CKPT/mbert_simple_mlp.pt \
  --windows $WINDOWS_LT \
  --split test \
  --out $VARIANTS_OUT_LT/mbert_simple_mlp_scores.csv

python src/phase2/expert/evaluate.py \
  --checkpoint $VARIANTS_CKPT/mbert_experts.pt \
  --windows $WINDOWS_LT \
  --split test \
  --out $VARIANTS_OUT_LT/mbert_experts_scores.csv

# --- Compare every baseline + all 3 Phase 2 architectures on letemps' test split (generalization check) ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --platt-scaling-score $PHASE1_SCORES_LT/platt_scaling.csv \
  --logistic-score $PHASE1_SCORES_LT/logistic_regression.csv \
  --mlp-score $PHASE1_SCORES_LT/mlp_baseline.csv \
  --camembert-mlp-score $BASE_SCORE_LT \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$VARIANTS_OUT_LT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$VARIANTS_OUT_LT/mbert_experts_scores.csv \
  --figures-dir $VARIANTS_FIGS_LT
