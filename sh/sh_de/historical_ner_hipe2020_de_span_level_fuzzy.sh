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
## hipe2020_de / historical_ner, span_level_fuzzy ONLY, IN-DOMAIN -- hipe2020_de has
## no German cross-dataset partner (letemps_fr is French), so unlike
## historical_ner_span_level_fuzzy.sh's fr+letemps pair there's no cross-dataset block
## here: everything below fits/trains on hipe2020_de's own train split, early-stops on
## its own val split, and evaluates on its own test split only.
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
##
## Requires data_prepare.sh's "hipe2020/de -- historical NER baseline" block to have
## finished first (raw extraction/dedup/label_reliability for this dataset/extractor
## pair) -- checked below rather than assumed, since that job may still be running.
## emanuelaboros/historical-ner-baseline's training-language coverage isn't documented
## in this repo (see data_prepare.sh's own caveat comment on that block) -- eyeball the
## ner_analysis figures from that block before trusting these numbers the way you would
## the French runs.
################################################################################

LEVEL=span_level_fuzzy   # hardcoded -- this script is span_level_fuzzy only, no word_level_type_only side
TRAIN_TEST_TAG=train_hipe2020de_test_hipe2020de   # trained on hipe2020_de, evaluated on hipe2020_de's own test split

NER_BASE=data/hipe2020_de/historical_ner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_de.csv
LABEL_RELIABILITY=$NER_BASE/label_reliability_${LEVEL}.csv
PHASE1_OUT=$NER_BASE/level_ablation/$LEVEL
PHASE1_FIGS=figures/hipe2020_de/historical_ner/modeling/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE1_CKPT=checkpoints/hipe2020_de/historical_ner/phase1/level_ablation/$LEVEL
PHASE2_OUT=data/hipe2020_de/historical_ner/data_phase2/level_ablation/$LEVEL
PHASE2_FIGS=figures/hipe2020_de/historical_ner/phase2/level_ablation/$LEVEL/$TRAIN_TEST_TAG
PHASE2_CKPT=checkpoints/hipe2020_de/historical_ner/phase2/level_ablation/$LEVEL
WINDOWS=$PHASE2_OUT/phase2_candidate_windows.jsonl

VARIANTS_CKPT=checkpoints/hipe2020_de/historical_ner/phase2/variants_ablation/$TRAIN_TEST_TAG
VARIANTS_OUT=data/hipe2020_de/historical_ner/data_phase2/variants_ablation/$TRAIN_TEST_TAG
VARIANTS_FIGS=figures/hipe2020_de/historical_ner/phase2/variants_ablation/$TRAIN_TEST_TAG

for f in "$NER_BASE/deduplicate_ner_features.csv" "$LABEL_RELIABILITY"; do
  if [ ! -f "$f" ]; then
    echo "Missing $f -- run data_prepare.sh's 'hipe2020/de -- historical NER baseline' block first (extraction/dedup/label_reliability for this dataset/extractor pair)." >&2
    exit 1
  fi
done

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

# --- Compare all 6 scores (pooled) on hipe2020_de's own test split ---
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
