#!/bin/bash
set -euo pipefail

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="cafe"

cd "$(dirname "${BASH_SOURCE[0]}")"
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
## letemps_fr / gliner -- IN-DOMAIN variant of baseline_phase2base.sh: everything
## fit/trained on letemps_fr's own train split (early-stopped on letemps_fr's own
## val split where applicable), evaluated on letemps_fr's own test split -- unlike
## baseline_phase2base.sh's letemps block, which only SCORES letemps with hipe's
## checkpoints (--checkpoint-in / --checkpoint, no fitting), this one actually fits
## B1/B3/MLP and trains Phase 2 on letemps itself, for an apples-to-apples
## in-domain comparison against hipe2020_fr's own train_hipe_test_hipe numbers.
##
## Phase 1 manual features (ocr_features.csv/context_features.csv,
## level-independent) and Phase 2 candidate windows already exist from
## baseline_phase2base.sh's own letemps block -- reused here unchanged, not
## rebuilt (see the commented-out calls below each skipped step).
##
## LEVEL matches baseline_phase2base.sh's own default/override convention
## (span_level_fuzzy | word_level_type_only).
################################################################################

LEVEL=${LEVEL:-span_level_fuzzy}   # span_level_fuzzy | word_level_type_only
TRAIN_TEST_TAG=train_letemps_test_letemps   # trained on letemps_fr, evaluated on letemps_fr's own test split

NER_BASE=data/letemps_fr/gliner/data_baseline
DATA_SRC=data/data_source/letemps/letemps_fr.csv
LABEL_RELIABILITY=$NER_BASE/label_reliability_${LEVEL}.csv
PHASE1_OUT=$NER_BASE/level_ablation/$LEVEL
PHASE1_FIGS=figures/letemps_fr/gliner/modeling/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE1_CKPT=checkpoints/letemps_fr/gliner/phase1/level_ablation/$LEVEL
PHASE2_OUT=data/letemps_fr/gliner/data_phase2/level_ablation/$LEVEL
PHASE2_FIGS=figures/letemps_fr/gliner/phase2/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE2_CKPT=checkpoints/letemps_fr/gliner/phase2/level_ablation/$LEVEL

for f in "$NER_BASE/ocr_features.csv" "$NER_BASE/context_features.csv" "$PHASE1_OUT/logistic_regression_data.csv" "$PHASE2_OUT/phase2_candidate_windows.jsonl"; do
  if [ ! -f "$f" ]; then
    echo "Missing $f -- run baseline_phase2base.sh's letemps block (LEVEL=$LEVEL) first to build letemps_fr's own manual features/candidate windows." >&2
    exit 1
  fi
done

# --- Phase 1: manual feature extraction (level-independent, already built by
# baseline_phase2base.sh's letemps block -- uncomment to regenerate) ---
# python src/phase1/feature_extraction/extract_ocr_features.py \
#   --load-data $DATA_SRC \
#   --ner-features $NER_BASE/deduplicate_ner_features.csv \
#   --out $NER_BASE/ocr_features.csv

# python src/phase1/feature_extraction/extract_context_features.py \
#   --load-data $DATA_SRC \
#   --ner-features $NER_BASE/deduplicate_ner_features.csv \
#   --out $NER_BASE/context_features.csv

# python src/phase1/feature_extraction/prepare_data_logistic.py \
#   --load-data $DATA_SRC \
#   --ner-features $NER_BASE/deduplicate_ner_features.csv \
#   --ocr-features $NER_BASE/ocr_features.csv \
#   --context-features $NER_BASE/context_features.csv \
#   --label-reliability $LABEL_RELIABILITY \
#   --out $PHASE1_OUT/logistic_regression_data.csv

# --- Phase 1 baselines: FIT on letemps train, early-stop on letemps val (where applicable), evaluate on letemps test ---
python src/phase1/modeling/platt_scaling.py \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --out $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --figures-dir $PHASE1_FIGS \
  --checkpoint-out $PHASE1_CKPT/platt_scaling.pt

python src/phase1/modeling/logistic_regression.py \
  --data $PHASE1_OUT/logistic_regression_data.csv \
  --out $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --figures-dir $PHASE1_FIGS \
  --checkpoint-out $PHASE1_CKPT/logistic_regression.pt

python src/phase1/modeling/mlp_baseline.py \
  --data $PHASE1_OUT/logistic_regression_data.csv \
  --out $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --figures-dir $PHASE1_FIGS \
  --checkpoint-out $PHASE1_CKPT/mlp_baseline.pt

# --- Phase 2: candidate windows (already built by baseline_phase2base.sh's letemps block -- uncomment to regenerate) ---
# python src/phase2/base/build_candidate_windows.py \
#   --load-data $DATA_SRC \
#   --label-reliability $LABEL_RELIABILITY \
#   --out $PHASE2_OUT/phase2_candidate_windows.jsonl

# --- Phase 2 base model: TRAIN (not score-only) on letemps' own candidate windows ---
python src/phase2/base/train.py \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --out $PHASE2_CKPT/mbert_mlp.pt \
  --figures-dir $PHASE2_FIGS/train_tracking

python src/phase2/base/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --split test \
  --out $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv

# --- Compare all 5 scores on letemps' own test split ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --logistic-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --mlp-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp \
  --figures-dir $PHASE2_FIGS
