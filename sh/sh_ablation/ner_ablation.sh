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
## hipe2020_fr / gliner -- NER-score ablation (docs/phase2_learned_features.md
## SS31 second dimension, see the old script.sh SS120-142 this replaces).
## Phase 2's full model feeds ScoreMLP [p, logit(p), 1-p] (--score-features full,
## the default -- already trained by baseline_phase2base.sh, reused here rather
## than retrained). This script trains+evaluates the other SCORE_FEATURES_CHOICES
## from src/phase2/base/model.py (p_only, logit_only, p_logit_only, binned -- a
## 10-bin ScoreEmb lookup instead of ScoreMLP) on the SAME candidate windows the
## full model already used, then plots all five together.
##
## LEVEL matches baseline_phase2base.sh's own default/override convention
## (span_level_fuzzy | word_level_type_only) -- run this only after
## baseline_phase2base.sh has already produced that LEVEL's "full" checkpoint and
## candidate windows; this script does not rebuild either.
################################################################################

LEVEL=${LEVEL:-span_level_fuzzy}   # span_level_fuzzy | word_level_type_only
TRAIN_TEST_TAG=train_hipe_test_hipe   # trained on hipe2020_fr, evaluated on hipe2020_fr's own test split

NER_BASE=data/hipe2020_fr/gliner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv
LABEL_RELIABILITY=$NER_BASE/label_reliability_${LEVEL}.csv
PHASE2_OUT=data/hipe2020_fr/gliner/data_phase2/level_ablation/$LEVEL
FULL_SCORE=$PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv

NER_ABLATION_CKPT=checkpoints/hipe2020_fr/gliner/phase2/ner_ablation/$TRAIN_TEST_TAG
NER_ABLATION_OUT=data/hipe2020_fr/gliner/data_phase2/ner_ablation/$TRAIN_TEST_TAG
NER_ABLATION_FIGS=figures/hipe2020_fr/gliner/phase2/ner_ablation/$TRAIN_TEST_TAG

if [ ! -f "$PHASE2_OUT/phase2_candidate_windows.jsonl" ]; then
  echo "Missing $PHASE2_OUT/phase2_candidate_windows.jsonl -- run baseline_phase2base.sh (LEVEL=$LEVEL) first." >&2
  exit 1
fi
if [ ! -f "$FULL_SCORE" ]; then
  echo "Missing $FULL_SCORE -- run baseline_phase2base.sh (LEVEL=$LEVEL) first to train/evaluate the full model." >&2
  exit 1
fi

# --- Train + evaluate each score_features ablation on the full model's already-built candidate windows ---
# For now: binned only, compared against full first. Add p_only/logit_only/p_logit_only
# back into this loop (and their --extra-score lines below) for the complete ablation.
for score_features in binned; do
  python src/phase2/base/train.py \
    --score-features "${score_features}" \
    --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
    --out $NER_ABLATION_CKPT/mbert_mlp_ner_${score_features}.pt \
    --figures-dir $NER_ABLATION_FIGS/train_tracking

  python src/phase2/base/evaluate.py \
    --checkpoint $NER_ABLATION_CKPT/mbert_mlp_ner_${score_features}.pt \
    --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
    --split test \
    --out $NER_ABLATION_OUT/mbert_mlp_ner_${score_features}_scores.csv
done

# --- Compare full (ScoreMLP [p, logit(p), 1-p]) against binned for now ---
python src/phase1/modeling/plot_reliability_diagram.py \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --camembert-mlp-score $FULL_SCORE \
  --camembert-mlp-label mbert_mlp_full \
  --extra-score binned=$NER_ABLATION_OUT/mbert_mlp_ner_binned_scores.csv \
  --figures-dir $NER_ABLATION_FIGS
  # --extra-score p_only=$NER_ABLATION_OUT/mbert_mlp_ner_p_only_scores.csv \
  # --extra-score logit_only=$NER_ABLATION_OUT/mbert_mlp_ner_logit_only_scores.csv \
  # --extra-score p_logit_only=$NER_ABLATION_OUT/mbert_mlp_ner_p_logit_only_scores.csv \

################################################################################
## letemps_fr / gliner -- score-only cross-dataset check: reuses the hipe-trained
## binned checkpoint above (no retraining), evaluated on letemps_fr's own test
## split, compared against the full model's already-existing letemps score
## (from baseline_phase2base.sh's own letemps block).
################################################################################

TRAIN_TEST_TAG_LT=train_hipe_test_letemps

NER_BASE_LT=data/letemps_fr/gliner/data_baseline
DATA_SRC_LT=data/data_source/letemps/letemps_fr.csv
LABEL_RELIABILITY_LT=$NER_BASE_LT/label_reliability_${LEVEL}.csv
PHASE2_OUT_LT=data/letemps_fr/gliner/data_phase2/level_ablation/$LEVEL
FULL_SCORE_LT=$PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv

NER_ABLATION_OUT_LT=data/letemps_fr/gliner/data_phase2/ner_ablation/$TRAIN_TEST_TAG_LT
NER_ABLATION_FIGS_LT=figures/letemps_fr/gliner/phase2/ner_ablation/$TRAIN_TEST_TAG_LT

if [ ! -f "$PHASE2_OUT_LT/phase2_candidate_windows.jsonl" ]; then
  echo "Missing $PHASE2_OUT_LT/phase2_candidate_windows.jsonl -- run baseline_phase2base.sh (LEVEL=$LEVEL) first." >&2
  exit 1
fi
if [ ! -f "$FULL_SCORE_LT" ]; then
  echo "Missing $FULL_SCORE_LT -- run baseline_phase2base.sh (LEVEL=$LEVEL) first to evaluate the full model on letemps." >&2
  exit 1
fi

python src/phase2/base/evaluate.py \
  --checkpoint $NER_ABLATION_CKPT/mbert_mlp_ner_binned.pt \
  --windows $PHASE2_OUT_LT/phase2_candidate_windows.jsonl \
  --split test \
  --out $NER_ABLATION_OUT_LT/mbert_mlp_ner_binned_scores.csv

# --- Compare full vs binned on letemps' test split (generalization check) ---
python src/phase1/modeling/plot_reliability_diagram.py \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --camembert-mlp-score $FULL_SCORE_LT \
  --camembert-mlp-label mbert_mlp_full \
  --extra-score binned=$NER_ABLATION_OUT_LT/mbert_mlp_ner_binned_scores.csv \
  --figures-dir $NER_ABLATION_FIGS_LT
