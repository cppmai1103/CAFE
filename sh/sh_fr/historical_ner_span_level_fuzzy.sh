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
## historical_ner, span_level_fuzzy ONLY -- merge of baseline_phase2base_historical_ner.sh
## + variants_ablation_historical_ner.sh into a single run (no word_level_type_only
## side -- LEVEL is hardcoded, not overridable, unlike the gliner scripts).
##
## Trains all 3 Phase 1 baselines (B1 Platt, B3 logistic, MLP -- B0 is just raw
## ner_score, no fitting needed) and all 3 Phase 2 architectures --
##   base   -- frozen encoder + side embeddings (type/ner-score/dict-flag/target-flag)
##             + simple pooling + one MLP head (src/phase2/base/)
##   simple -- same frozen encoder, but type/confidence are written directly into the
##             token sequence as text markers instead of side embeddings -- no new
##             trainable embeddings, classifier head only (src/phase2/simple/)
##   expert -- same base backbone, but the MLP head is replaced by a K=4 latent
##             mixture-of-experts head with a load-balancing auxiliary loss
##             (src/phase2/expert/)
## -- entirely on hipe2020_fr's own train split (early-stopped on its own val
## split), then evaluates all 6 scores on BOTH hipe2020_fr's own test split
## (in-domain) and letemps_fr's test split (cross-dataset generalization, scored
## with the hipe-trained checkpoints, no retraining -- letemps_fr's own manual
## features + candidate windows are new inputs, built here once).
##
## historical_ner's raw NER candidates (ner_features/deduplicate_ner_features/
## label_reliability_*.csv) already exist on disk for both datasets (built by
## data_prepare.sh), so this starts from Phase 1 manual feature extraction.
## historical_ner used the full 5-type HIPE-2022 tagset (PERS/LOC/ORG/TIME/PROD)
## for both hipe2020_fr and letemps_fr (unlike gliner, whose letemps_fr run only
## used PER/LOC/ORG) -- so both facet-by-type calls below list all 5 types.
################################################################################

LEVEL=span_level_fuzzy   # hardcoded -- this script is span_level_fuzzy only, no word_level_type_only side

## ---------------------------------------------------------------------------
## hipe2020_fr / historical_ner -- fit/train everything on hipe2020_fr's own
## train split, early-stop on val, evaluate on hipe2020_fr's own test split.
## ---------------------------------------------------------------------------

TRAIN_TEST_TAG=train_hipe_test_hipe

NER_BASE=data/hipe2020_fr/historical_ner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv
LABEL_RELIABILITY=$NER_BASE/label_reliability_${LEVEL}.csv
PHASE1_OUT=$NER_BASE/level_ablation/$LEVEL
PHASE1_FIGS=figures/hipe2020_fr/historical_ner/modeling/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE1_CKPT=checkpoints/hipe2020_fr/historical_ner/phase1/level_ablation/$LEVEL
PHASE2_OUT=data/hipe2020_fr/historical_ner/data_phase2/level_ablation/$LEVEL
PHASE2_FIGS=figures/hipe2020_fr/historical_ner/phase2/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE2_CKPT=checkpoints/hipe2020_fr/historical_ner/phase2/level_ablation/$LEVEL
WINDOWS=$PHASE2_OUT/phase2_candidate_windows.jsonl

VARIANTS_CKPT=checkpoints/hipe2020_fr/historical_ner/phase2/variants_ablation/$TRAIN_TEST_TAG
VARIANTS_OUT=data/hipe2020_fr/historical_ner/data_phase2/variants_ablation/$TRAIN_TEST_TAG
VARIANTS_FIGS=figures/hipe2020_fr/historical_ner/phase2/variants_ablation/$TRAIN_TEST_TAG

# --- Phase 1: manual feature extraction ---
python src/phase1/feature_extraction/extract_ocr_features.py \
  --load-data $DATA_SRC \
  --ner-features $NER_BASE/deduplicate_ner_features.csv \
  --out $NER_BASE/ocr_features.csv

python src/phase1/feature_extraction/extract_context_features.py \
  --load-data $DATA_SRC \
  --ner-features $NER_BASE/deduplicate_ner_features.csv \
  --out $NER_BASE/context_features.csv

python src/phase1/feature_extraction/prepare_data_logistic.py \
  --load-data $DATA_SRC \
  --ner-features $NER_BASE/deduplicate_ner_features.csv \
  --ocr-features $NER_BASE/ocr_features.csv \
  --context-features $NER_BASE/context_features.csv \
  --label-reliability $LABEL_RELIABILITY \
  --out $PHASE1_OUT/logistic_regression_data.csv

# --- Phase 1 baselines: fit on train, early-stop on val, evaluate on test ---
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

# --- Phase 2: candidate windows (shared by base/simple/expert) ---
python src/phase2/base/build_candidate_windows.py \
  --load-data $DATA_SRC \
  --label-reliability $LABEL_RELIABILITY \
  --out $WINDOWS

# --- Phase 2 base: frozen mBERT + side embeddings + simple pooling + MLP head ---
python src/phase2/base/train.py \
  --windows $WINDOWS \
  --out $PHASE2_CKPT/mbert_mlp.pt \
  --figures-dir $PHASE2_FIGS/train_tracking

python src/phase2/base/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
  --windows $WINDOWS \
  --split test \
  --out $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv

# --- Phase 2 simple: type/confidence as text markers instead of side embeddings ---
python src/phase2/simple/train.py \
  --windows $WINDOWS \
  --out $VARIANTS_CKPT/mbert_simple_mlp.pt \
  --figures-dir $VARIANTS_FIGS/train_tracking

python src/phase2/simple/evaluate.py \
  --checkpoint $VARIANTS_CKPT/mbert_simple_mlp.pt \
  --windows $WINDOWS \
  --split test \
  --out $VARIANTS_OUT/mbert_simple_mlp_scores.csv

# --- Phase 2 expert: K=4 latent mixture-of-experts head, lambda-balance=0.01 (defaults) ---
python src/phase2/expert/train.py \
  --windows $WINDOWS \
  --out $VARIANTS_CKPT/mbert_experts.pt \
  --figures-dir $VARIANTS_FIGS/train_tracking

python src/phase2/expert/evaluate.py \
  --checkpoint $VARIANTS_CKPT/mbert_experts.pt \
  --windows $WINDOWS \
  --split test \
  --out $VARIANTS_OUT/mbert_experts_scores.csv

# --- Compare all 6 scores (pooled) on hipe2020_fr's own test split ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --logistic-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --mlp-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$VARIANTS_OUT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$VARIANTS_OUT/mbert_experts_scores.csv \
  --figures-dir $VARIANTS_FIGS

# --- Same, faceted into one grid figure per plot kind (one panel per predicted_entity_type) ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --logistic-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --mlp-score $PHASE1_OUT/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$VARIANTS_OUT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$VARIANTS_OUT/mbert_experts_scores.csv \
  --facet-by-type PERS LOC ORG TIME PROD \
  --figures-dir "$VARIANTS_FIGS/by_type"

## ---------------------------------------------------------------------------
## letemps_fr / historical_ner -- score-only cross-dataset check: reuses every
## hipe-trained checkpoint above (platt_scaling.pt, logistic_regression.pt,
## mlp_baseline.pt, mbert_mlp.pt, mbert_simple_mlp.pt, mbert_experts.pt) via
## --checkpoint-in/--checkpoint (no retraining). letemps_fr's own manual features
## and candidate windows are new inputs, built here once.
## ---------------------------------------------------------------------------

TRAIN_TEST_TAG_LT=train_hipe_test_letemps

NER_BASE_LT=data/letemps_fr/historical_ner/data_baseline
DATA_SRC_LT=data/data_source/letemps/letemps_fr.csv
LABEL_RELIABILITY_LT=$NER_BASE_LT/label_reliability_${LEVEL}.csv
PHASE1_OUT_LT=$NER_BASE_LT/level_ablation/$LEVEL
PHASE1_FIGS_LT=figures/letemps_fr/historical_ner/modeling/level_ablation/$LEVEL/$TRAIN_TEST_TAG_LT
PHASE2_OUT_LT=data/letemps_fr/historical_ner/data_phase2/level_ablation/$LEVEL
PHASE2_FIGS_LT=figures/letemps_fr/historical_ner/phase2/level_ablation/$LEVEL/$TRAIN_TEST_TAG_LT
WINDOWS_LT=$PHASE2_OUT_LT/phase2_candidate_windows.jsonl

VARIANTS_OUT_LT=data/letemps_fr/historical_ner/data_phase2/variants_ablation/$TRAIN_TEST_TAG_LT
VARIANTS_FIGS_LT=figures/letemps_fr/historical_ner/phase2/variants_ablation/$TRAIN_TEST_TAG_LT

# --- Build letemps_fr's own manual features ---
python src/phase1/feature_extraction/extract_ocr_features.py \
  --load-data $DATA_SRC_LT \
  --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
  --out $NER_BASE_LT/ocr_features.csv

python src/phase1/feature_extraction/extract_context_features.py \
  --load-data $DATA_SRC_LT \
  --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
  --out $NER_BASE_LT/context_features.csv

python src/phase1/feature_extraction/prepare_data_logistic.py \
  --load-data $DATA_SRC_LT \
  --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
  --ocr-features $NER_BASE_LT/ocr_features.csv \
  --context-features $NER_BASE_LT/context_features.csv \
  --label-reliability $LABEL_RELIABILITY_LT \
  --out $PHASE1_OUT_LT/logistic_regression_data.csv

# --- Build letemps_fr's own candidate windows ---
python src/phase2/base/build_candidate_windows.py \
  --load-data $DATA_SRC_LT \
  --label-reliability $LABEL_RELIABILITY_LT \
  --out $WINDOWS_LT

# --- Score letemps_fr with every hipe-trained checkpoint above (no retraining) ---
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
  --windows $WINDOWS_LT \
  --split test \
  --out $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv

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

# --- Compare all 6 scores (pooled) on letemps_fr's test split (generalization check) ---
# --exclude-types-not-in-gold: letemps_fr's own gold annotation has NO time/prod entities
# at all (only pers/loc/org, confirmed against NE-COARSE-LIT directly) -- unlike hipe2020_fr,
# where historical_ner's TIME/PROD predictions are a genuine, scoreable model-quality
# question. On letemps a predicted TIME/PROD candidate can never be correct by
# construction (there's no matching gold category to be right against), so scoring them
# here isn't measuring calibration failure, it's measuring an unanswerable question --
# AUROC comes back undefined (all-negative) and Brier/ECE would unfairly drag down an
# otherwise-reasonable model on the 3 types letemps_fr actually judges.
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --platt-scaling-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --logistic-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --mlp-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$VARIANTS_OUT_LT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$VARIANTS_OUT_LT/mbert_experts_scores.csv \
  --exclude-types-not-in-gold \
  --figures-dir $VARIANTS_FIGS_LT

# --- Same, faceted -- PERS/LOC/ORG only (letemps_fr's actual gold scheme, see above; a
# TIME/PROD panel here would be 0 candidates after --exclude-types-not-in-gold) ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --platt-scaling-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --logistic-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --mlp-score $PHASE1_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$VARIANTS_OUT_LT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$VARIANTS_OUT_LT/mbert_experts_scores.csv \
  --exclude-types-not-in-gold \
  --facet-by-type PERS LOC ORG \
  --figures-dir "$VARIANTS_FIGS_LT/by_type"
