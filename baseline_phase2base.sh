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
## hipe2020_fr / gliner -- Phase 1 (manual features + B0/B1/B3/MLP baselines)
## and Phase 2 base model (frozen mBERT + MLP head), all inputs/outputs
## redirected under data/hipe2020_fr/gliner/ so they never collide with another
## dataset/NER-source combo's own Phase 1/2 artifacts. Every baseline/model here
## fits (or, for Phase 2, trains) on hipe2020_fr's own train split, early-stops on
## val where applicable, and is evaluated on hipe2020_fr's own test split.
##
## LEVEL picks which label_reliability_<LEVEL>.csv is the ground-truth reliability
## target (span_level_fuzzy or word_level_type_only, see ner/label_reliability.py
## --level/--mode) -- override it on the command line to run the other level
## without editing this file, e.g. `LEVEL=word_level_type_only bash
## baseline_phase2base.sh`. Every label-dependent artifact (logistic_regression_data.csv,
## phase2 candidate windows, checkpoints, test_results, figures) lives under its own
## $LEVEL subfolder so the two runs never overwrite each other; ocr_features.csv/
## context_features.csv are level-independent (computed from OCR/position alone, not
## the label) and stay shared directly under $NER_BASE -- comment out those two
## extraction calls on a second LEVEL run, no need to redo them.
################################################################################

LEVEL=${LEVEL:-span_level_fuzzy}   # span_level_fuzzy | word_level_type_only
TRAIN_TEST_TAG=train_hipe_test_hipe   # trained on hipe2020_fr, evaluated on hipe2020_fr's own test split

NER_BASE=data/hipe2020_fr/gliner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv
LABEL_RELIABILITY=$NER_BASE/label_reliability_${LEVEL}.csv
PHASE1_OUT=$NER_BASE/level_ablation/$LEVEL
PHASE1_FIGS=figures/hipe2020_fr/gliner/modeling/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE1_CKPT=checkpoints/hipe2020_fr/gliner/phase1/level_ablation/$LEVEL
PHASE2_OUT=data/hipe2020_fr/gliner/data_phase2/level_ablation/$LEVEL
PHASE2_FIGS=figures/hipe2020_fr/gliner/phase2/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE2_CKPT=checkpoints/hipe2020_fr/gliner/phase2/level_ablation/$LEVEL

# # --- Phase 1: manual feature extraction (level-independent -- comment out on a second LEVEL run) ---
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

# # --- Phase 1 baselines: fit on train, early-stop on val (where applicable), evaluate on test ---
# python src/phase1/modeling/platt_scaling.py \
#   --label-reliability $LABEL_RELIABILITY \
#   --load-data $DATA_SRC \
#   --out $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
#   --figures-dir $PHASE1_FIGS \
#   --checkpoint-out $PHASE1_CKPT/platt_scaling.pt

# python src/phase1/modeling/logistic_regression.py \
#   --data $PHASE1_OUT/logistic_regression_data.csv \
#   --out $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
#   --figures-dir $PHASE1_FIGS \
#   --checkpoint-out $PHASE1_CKPT/logistic_regression.pt

# python src/phase1/modeling/mlp_baseline.py \
#   --data $PHASE1_OUT/logistic_regression_data.csv \
#   --out $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
#   --figures-dir $PHASE1_FIGS \
#   --checkpoint-out $PHASE1_CKPT/mlp_baseline.pt

# # --- Phase 2: candidate windows (shared by base/expert/simple, only needs building once) ---
# python src/phase2/base/build_candidate_windows.py \
#   --load-data $DATA_SRC \
#   --label-reliability $LABEL_RELIABILITY \
#   --out $PHASE2_OUT/phase2_candidate_windows.jsonl

# # Verify tokenization/truncation before training -- optional, train.py's own Dataset
# # already tokenizes+verifies on the fly, this is just a standalone stats/sanity pass.
# # python src/phase2/base/tokenize_windows.py \
# #   --windows $PHASE2_OUT/phase2_candidate_windows.jsonl

# # --- Phase 2 base model: frozen mBERT (DEFAULT_ENCODER_NAME) + side embeddings + simple pooling + MLP head ---
# python src/phase2/base/train.py \
#   --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
#   --out $PHASE2_CKPT/mbert_mlp.pt \
#   --figures-dir $PHASE2_FIGS/train_tracking

# python src/phase2/base/evaluate.py \
#   --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
#   --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
#   --split test \
#   --out $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv

# python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
#   --label-reliability $LABEL_RELIABILITY \
#   --load-data $DATA_SRC \
#   --platt-scaling-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
#   --logistic-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
#   --mlp-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
#   --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
#   --camembert-mlp-label mbert_mlp \
#   --figures-dir $PHASE2_FIGS

# --- Same 4 plots as above, faceted into one grid figure per plot kind (one panel per
# predicted_entity_type) instead of pooled -- see plot_reliability_diagram.py --facet-by-type.
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --logistic-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --mlp-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp \
  --facet-by-type PERS LOC ORG TIME PROD \
  --figures-dir "$PHASE2_FIGS/by_type"

# python src/phase2/base/check.py --windows $PHASE2_OUT/phase2_candidate_windows.jsonl                # first 5, no split filter
# python src/phase2/base/check.py --windows $PHASE2_OUT/phase2_candidate_windows.jsonl --n 3 --split test

################################################################################
## letemps_fr / gliner -- score-only cross-dataset generalization check: reuses
## hipe2020_fr's already-trained checkpoints (platt_scaling.pt, logistic_regression.pt,
## mlp_baseline.pt, mbert_mlp.pt) via each script's --checkpoint-in/--checkpoint
## flag (no retraining -- see src/phase1/modeling/{platt_scaling,logistic_regression,
## mlp_baseline}.py's --checkpoint-in mode; phase2/base/evaluate.py already worked
## this way). Every Phase 1 script's own "test split only" output convention (see
## each script's docstring) plus phase2/base/evaluate.py's --split test mean this
## evaluates only on letemps_fr's test-split documents, not its train/val ones.
## letemps_fr only has raw NER features + label_reliability so far (steps 2.x), so
## its own manual features (ocr/context/logistic_regression_data.csv, step 3.x) and
## candidate windows (step 5.x) are built here first -- those are genuinely new
## *inputs*, not new trained models.
################################################################################

TRAIN_TEST_TAG_LT=train_hipe_test_letemps   # hipe2020_fr's checkpoints, evaluated on letemps_fr's test split

NER_BASE_LT=data/letemps_fr/gliner/data_baseline
DATA_SRC_LT=data/data_source/letemps/letemps_fr.csv
LABEL_RELIABILITY_LT=$NER_BASE_LT/label_reliability_${LEVEL}.csv
PHASE1_OUT_LT=$NER_BASE_LT/level_ablation/$LEVEL
PHASE1_FIGS_LT=figures/letemps_fr/gliner/modeling/level_ablation/$LEVEL/$TRAIN_TEST_TAG_LT
PHASE2_OUT_LT=data/letemps_fr/gliner/data_phase2/level_ablation/$LEVEL
PHASE2_FIGS_LT=figures/letemps_fr/gliner/phase2/level_ablation/$LEVEL/$TRAIN_TEST_TAG_LT

## --- Build letemps_fr's own manual features (input: letemps_fr's raw NER output; output: ocr/context/logistic_regression_data.csv). ocr/context features are level-independent -- comment out on a second LEVEL run. ---
# python src/phase1/feature_extraction/extract_ocr_features.py \
#   --load-data $DATA_SRC_LT \
#   --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
#   --out $NER_BASE_LT/ocr_features.csv

# python src/phase1/feature_extraction/extract_context_features.py \
#   --load-data $DATA_SRC_LT \
#   --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
#   --out $NER_BASE_LT/context_features.csv

# python src/phase1/feature_extraction/prepare_data_logistic.py \
#   --load-data $DATA_SRC_LT \
#   --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
#   --ocr-features $NER_BASE_LT/ocr_features.csv \
#   --context-features $NER_BASE_LT/context_features.csv \
#   --label-reliability $LABEL_RELIABILITY_LT \
#   --out $PHASE1_OUT_LT/logistic_regression_data.csv

# ## --- Build letemps_fr's own candidate windows (input: letemps_fr's raw NER output; output: phase2_candidate_windows.jsonl) ---
# python src/phase2/base/build_candidate_windows.py \
#   --load-data $DATA_SRC_LT \
#   --label-reliability $LABEL_RELIABILITY_LT \
#   --out $PHASE2_OUT_LT/phase2_candidate_windows.jsonl

# ## --- Score letemps_fr with hipe2020_fr's already-trained checkpoints (no retraining) ---
# python src/phase1/modeling/platt_scaling.py \
#   --checkpoint-in $PHASE1_CKPT/platt_scaling.pt \
#   --label-reliability $LABEL_RELIABILITY_LT \
#   --load-data $DATA_SRC_LT \
#   --out $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
#   --figures-dir $PHASE1_FIGS_LT

# python src/phase1/modeling/logistic_regression.py \
#   --checkpoint-in $PHASE1_CKPT/logistic_regression.pt \
#   --data $PHASE1_OUT_LT/logistic_regression_data.csv \
#   --out $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
#   --figures-dir $PHASE1_FIGS_LT

# python src/phase1/modeling/mlp_baseline.py \
#   --checkpoint-in $PHASE1_CKPT/mlp_baseline.pt \
#   --data $PHASE1_OUT_LT/logistic_regression_data.csv \
#   --out $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
#   --figures-dir $PHASE1_FIGS_LT

# python src/phase2/base/evaluate.py \
#   --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
#   --windows $PHASE2_OUT_LT/phase2_candidate_windows.jsonl \
#   --split test \
#   --out $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv

# ## --- Compare, same conventions as hipe2020_fr's own comparison plot above ---
# python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
#   --label-reliability $LABEL_RELIABILITY_LT \
#   --load-data $DATA_SRC_LT \
#   --platt-scaling-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
#   --logistic-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
#   --mlp-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
#   --camembert-mlp-score $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv \
#   --camembert-mlp-label mbert_mlp \
#   --figures-dir $PHASE2_FIGS_LT

# --- Same 4 plots as above, faceted into one grid figure per plot kind (one panel per
# predicted_entity_type) instead of pooled -- letemps_fr's GLiNER2 extraction only used
# PER/LOC/ORG (see its own labels.json), so no TIME/PROD panel here.
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --platt-scaling-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --logistic-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --mlp-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp \
  --facet-by-type PERS LOC ORG \
  --figures-dir "$PHASE2_FIGS_LT/by_type"
