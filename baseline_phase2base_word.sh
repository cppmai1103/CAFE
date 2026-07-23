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
## hipe2020_fr / gliner -- WORD-LEVEL variant of baseline_phase2base.sh (span-level).
##
## Word-level candidates (ner/label_reliability.py --level word) are one row per TOKEN
## a GLiNER2 candidate span covers, reshaped so start_token_id=end_token_id=token_id
## and entity_text=that token's own text -- schema-identical to the span-level
## label_reliability.csv (OUTPUT_COLUMNS), so this ONE file doubles as both
## --ner-features (replacing deduplicate_ner_features.csv) and --label-reliability
## everywhere below; no separate word-level candidates file is needed. --mode is
## irrelevant at word level (always the type_only check applied to one token), hence
## "word_level_type_only" throughout rather than a $LEVEL variable.
##
## Verified this actually runs end-to-end before writing this script: extract_ocr/
## context_features.py's span-statistics math degrades correctly to a single token
## (sum/len over one element), prepare_data_logistic.py's KEY_COLS join and
## build_candidate_windows.py's window-building both work unmodified once fed this
## reshaped file, and build_candidate_windows.py's own sanity check (20 random windows)
## reconstructed every 1-token target span correctly. Two small upstream fixes were
## needed and are already applied: prepare_data_logistic.py now drops a pre-existing
## reliability_score column from --ner-features before merging (this file supplies it
## under both flags, which previously collided into reliability_score_x/_y), and its
## sentence_chunked drop is now errors="ignore" (that column is span-extraction metadata
## that label_reliability.py's word-level output never carried).
################################################################################

TRAIN_TEST_TAG=train_hipe_test_hipe

NER_BASE=data/hipe2020_fr/gliner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv
LABEL_RELIABILITY=$NER_BASE/label_reliability_word_level_type_only.csv
PHASE1_OUT=$NER_BASE/level_ablation/word_level_type_only
PHASE1_FIGS=figures/hipe2020_fr/gliner/modeling/level_ablation/word_level_type_only/$TRAIN_TEST_TAG
PHASE1_CKPT=checkpoints/hipe2020_fr/gliner/phase1/level_ablation/word_level_type_only
PHASE2_OUT=data/hipe2020_fr/gliner/data_phase2/level_ablation/word_level_type_only
PHASE2_FIGS=figures/hipe2020_fr/gliner/phase2/level_ablation/word_level_type_only/$TRAIN_TEST_TAG
PHASE2_CKPT=checkpoints/hipe2020_fr/gliner/phase2/level_ablation/word_level_type_only

# --- Explode deduplicate_ner_features.csv into the reshaped word-level candidates+label file ---
python src/ner/label_reliability.py \
  --load-data $DATA_SRC \
  --ner-features $NER_BASE/deduplicate_ner_features.csv \
  --out $LABEL_RELIABILITY \
  --level word

# --- Phase 1: manual feature extraction (word-level candidates = $LABEL_RELIABILITY itself) ---
python src/phase1/feature_extraction/extract_ocr_features.py \
  --load-data $DATA_SRC \
  --ner-features $LABEL_RELIABILITY \
  --out $PHASE1_OUT/ocr_features.csv

python src/phase1/feature_extraction/extract_context_features.py \
  --load-data $DATA_SRC \
  --ner-features $LABEL_RELIABILITY \
  --out $PHASE1_OUT/context_features.csv

python src/phase1/feature_extraction/prepare_data_logistic.py \
  --load-data $DATA_SRC \
  --ner-features $LABEL_RELIABILITY \
  --ocr-features $PHASE1_OUT/ocr_features.csv \
  --context-features $PHASE1_OUT/context_features.csv \
  --label-reliability $LABEL_RELIABILITY \
  --out $PHASE1_OUT/logistic_regression_data.csv

# --- Phase 1 baselines: fit on train, early-stop on val (where applicable), evaluate on test ---
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

# --- Phase 2: candidate windows (target span is now always the single word token) ---
python src/phase2/base/build_candidate_windows.py \
  --load-data $DATA_SRC \
  --label-reliability $LABEL_RELIABILITY \
  --out $PHASE2_OUT/phase2_candidate_windows.jsonl

# --- Phase 2 base model: frozen mBERT (DEFAULT_ENCODER_NAME) + side embeddings + simple pooling + MLP head ---
python src/phase2/base/train.py \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --out $PHASE2_CKPT/mbert_mlp.pt \
  --figures-dir $PHASE2_FIGS/train_tracking

python src/phase2/base/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --split test \
  --out $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv

python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --logistic-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --mlp-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp \
  --figures-dir $PHASE2_FIGS

################################################################################
## letemps_fr / gliner -- score-only cross-dataset generalization check, word-level.
## Same pattern as baseline_phase2base.sh's span-level version: reuses hipe2020_fr's
## already-trained checkpoints (no retraining), evaluates only on letemps_fr's test
## split (every Phase 1 script's own "test split only" output convention, plus
## phase2/base/evaluate.py's --split test).
################################################################################

TRAIN_TEST_TAG_LT=train_hipe_test_letemps

NER_BASE_LT=data/letemps_fr/gliner/data_baseline
DATA_SRC_LT=data/data_source/letemps/letemps_fr.csv
LABEL_RELIABILITY_LT=$NER_BASE_LT/label_reliability_word_level_type_only.csv
PHASE1_OUT_LT=$NER_BASE_LT/level_ablation/word_level_type_only
PHASE1_FIGS_LT=figures/letemps_fr/gliner/modeling/level_ablation/word_level_type_only/$TRAIN_TEST_TAG_LT
PHASE2_OUT_LT=data/letemps_fr/gliner/data_phase2/level_ablation/word_level_type_only
PHASE2_FIGS_LT=figures/letemps_fr/gliner/phase2/level_ablation/word_level_type_only/$TRAIN_TEST_TAG_LT

python src/ner/label_reliability.py \
  --load-data $DATA_SRC_LT \
  --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
  --out $LABEL_RELIABILITY_LT \
  --level word

python src/phase1/feature_extraction/extract_ocr_features.py \
  --load-data $DATA_SRC_LT \
  --ner-features $LABEL_RELIABILITY_LT \
  --out $PHASE1_OUT_LT/ocr_features.csv

python src/phase1/feature_extraction/extract_context_features.py \
  --load-data $DATA_SRC_LT \
  --ner-features $LABEL_RELIABILITY_LT \
  --out $PHASE1_OUT_LT/context_features.csv

python src/phase1/feature_extraction/prepare_data_logistic.py \
  --load-data $DATA_SRC_LT \
  --ner-features $LABEL_RELIABILITY_LT \
  --ocr-features $PHASE1_OUT_LT/ocr_features.csv \
  --context-features $PHASE1_OUT_LT/context_features.csv \
  --label-reliability $LABEL_RELIABILITY_LT \
  --out $PHASE1_OUT_LT/logistic_regression_data.csv

python src/phase2/base/build_candidate_windows.py \
  --load-data $DATA_SRC_LT \
  --label-reliability $LABEL_RELIABILITY_LT \
  --out $PHASE2_OUT_LT/phase2_candidate_windows.jsonl

python src/phase1/modeling/platt_scaling.py \
  --checkpoint-in $PHASE1_CKPT/platt_scaling.pt \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --out $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --figures-dir $PHASE1_FIGS_LT

python src/phase1/modeling/logistic_regression.py \
  --checkpoint-in $PHASE1_CKPT/logistic_regression.pt \
  --data $PHASE1_OUT_LT/logistic_regression_data.csv \
  --out $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --figures-dir $PHASE1_FIGS_LT

python src/phase1/modeling/mlp_baseline.py \
  --checkpoint-in $PHASE1_CKPT/mlp_baseline.pt \
  --data $PHASE1_OUT_LT/logistic_regression_data.csv \
  --out $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --figures-dir $PHASE1_FIGS_LT

python src/phase2/base/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
  --windows $PHASE2_OUT_LT/phase2_candidate_windows.jsonl \
  --split test \
  --out $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv

python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --platt-scaling-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --logistic-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --mlp-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp \
  --figures-dir $PHASE2_FIGS_LT
