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
## hipe2020_fr / gliner -- "match_entities" ablation: retrain everything (Phase 1
## B1/B3/MLP + Phase 2 base/simple/expert) on hipe2020_fr's candidates restricted to
## PERS/LOC/ORG only, i.e. dropping TIME/PROD *before* deduplication -- so a
## TIME/PROD candidate can never win a span-overlap conflict against a kept-type
## candidate either, not just get filtered out after the fact.
##
## Motivation: letemps_fr's own GLiNER2 extraction only ever asked for PERS/LOC/ORG
## (its own labels.json has no TIME/PROD prompt at all), but hipe2020_fr's reliability
## model up to now was always trained on the full 5-type candidate set. That means the
## hipe-trained model's learned behavior is partly shaped by TIME/PROD's very different
## base rate/difficulty (see figures/.../by_type -- PROD in particular is almost never
## reliable), something letemps_fr can never exhibit. Restricting hipe2020_fr's training
## data to the 3 types letemps_fr actually has isolates whether *that* mismatch (rather
## than the OCR/domain shift itself) is what breaks cross-dataset calibration transfer
## (see the hipe->letemps Brier/MCE regression in variants_ablation.sh's own results).
##
## letemps_fr needs no candidate filtering of its own -- it's already PERS/LOC/ORG-only
## by construction -- but its label_reliability/logistic_regression_data.csv/candidate
## windows are still rebuilt fresh under their own match_entities/ folder (not reused
## from level_ablation/span_level_fuzzy) so this whole ablation stays self-contained;
## see the letemps_fr block below for why (content is identical either way, this is a
## folder-layout choice, not a data change).
################################################################################

KEEP_TYPES="PERS LOC ORG"
TRAIN_TEST_TAG=train_hipe_test_hipe   # trained on hipe2020_fr (3-type), evaluated on hipe2020_fr's own test split

NER_BASE=data/hipe2020_fr/gliner/data_baseline
DATA_SRC=data/data_source/hipe2020/hipe2020_fr.csv

MATCH_BASE=$NER_BASE/match_entities
LABEL_RELIABILITY=$MATCH_BASE/label_reliability_span_level_fuzzy.csv
LABELS_FILE=$MATCH_BASE/labels.json   # PERS/LOC/ORG only -- sizes Phase 2's TypeEmb to 3 rows instead of the default 5 (see model.py's entity_type_vocab)

PHASE1_FIGS=figures/hipe2020_fr/gliner/modeling/match_entities/$TRAIN_TEST_TAG
PHASE1_CKPT=checkpoints/hipe2020_fr/gliner/phase1/match_entities

PHASE2_OUT=data/hipe2020_fr/gliner/data_phase2/match_entities
PHASE2_FIGS=figures/hipe2020_fr/gliner/phase2/match_entities/$TRAIN_TEST_TAG
PHASE2_CKPT=checkpoints/hipe2020_fr/gliner/phase2/match_entities

# --- Step 0: filter hipe2020_fr's raw candidates down to PERS/LOC/ORG, before dedup ---
python src/ner/filter_entity_types.py \
  --ner-features $NER_BASE/ner_features.csv \
  --keep-types $KEEP_TYPES \
  --out $MATCH_BASE/ner_features.csv

# --- Step 1: dedup the filtered candidates (same greedy overlap resolution as usual) ---
python src/ner/gliner/deduplicate_ner_features.py \
  --ner-features $MATCH_BASE/ner_features.csv \
  --out $MATCH_BASE/deduplicate_ner_features.csv \
  --conflicts-out $MATCH_BASE/ner_overlap_conflicts.json

# --- Step 2: attach gold labels (span-level fuzzy, the project default) ---
python src/ner/label_reliability.py --mode fuzzy \
  --load-data $DATA_SRC \
  --ner-features $MATCH_BASE/deduplicate_ner_features.csv \
  --out $LABEL_RELIABILITY

# --- Step 3: Phase 1 manual features (OCR/context), rebuilt against the filtered candidate set ---
python src/phase1/feature_extraction/extract_ocr_features.py \
  --load-data $DATA_SRC \
  --ner-features $MATCH_BASE/deduplicate_ner_features.csv \
  --out $MATCH_BASE/ocr_features.csv

python src/phase1/feature_extraction/extract_context_features.py \
  --load-data $DATA_SRC \
  --ner-features $MATCH_BASE/deduplicate_ner_features.csv \
  --out $MATCH_BASE/context_features.csv

python src/phase1/feature_extraction/prepare_data_logistic.py \
  --load-data $DATA_SRC \
  --ner-features $MATCH_BASE/deduplicate_ner_features.csv \
  --ocr-features $MATCH_BASE/ocr_features.csv \
  --context-features $MATCH_BASE/context_features.csv \
  --label-reliability $LABEL_RELIABILITY \
  --out $MATCH_BASE/logistic_regression_data.csv

# --- Step 4: Phase 1 baselines -- fit on train (3-type only), early-stop on val, evaluate on test ---
python src/phase1/modeling/platt_scaling.py \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --out $MATCH_BASE/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --figures-dir $PHASE1_FIGS \
  --checkpoint-out $PHASE1_CKPT/platt_scaling.pt

python src/phase1/modeling/logistic_regression.py \
  --data $MATCH_BASE/logistic_regression_data.csv \
  --out $MATCH_BASE/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --figures-dir $PHASE1_FIGS \
  --checkpoint-out $PHASE1_CKPT/logistic_regression.pt

python src/phase1/modeling/mlp_baseline.py \
  --data $MATCH_BASE/logistic_regression_data.csv \
  --out $MATCH_BASE/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --figures-dir $PHASE1_FIGS \
  --checkpoint-out $PHASE1_CKPT/mlp_baseline.pt

# --- Step 5: Phase 2 candidate windows (shared by base/simple/expert) ---
python src/phase2/base/build_candidate_windows.py \
  --load-data $DATA_SRC \
  --label-reliability $LABEL_RELIABILITY \
  --out $PHASE2_OUT/phase2_candidate_windows.jsonl

# --- Step 6: Phase 2 base -- frozen mBERT + side embeddings + MLP head ---
python src/phase2/base/train.py \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --labels-file $LABELS_FILE \
  --out $PHASE2_CKPT/mbert_mlp.pt \
  --figures-dir $PHASE2_FIGS/train_tracking

python src/phase2/base/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --split test \
  --out $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv

# --- Step 7: Phase 2 simple -- type/confidence as text markers, no side embeddings ---
python src/phase2/simple/train.py \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --labels-file $LABELS_FILE \
  --out $PHASE2_CKPT/mbert_simple_mlp.pt \
  --figures-dir $PHASE2_FIGS/train_tracking

python src/phase2/simple/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_simple_mlp.pt \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --split test \
  --out $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_simple_mlp_scores.csv

# --- Step 8: Phase 2 expert -- K=4 mixture-of-experts head ---
python src/phase2/expert/train.py \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --labels-file $LABELS_FILE \
  --out $PHASE2_CKPT/mbert_experts.pt \
  --figures-dir $PHASE2_FIGS/train_tracking

python src/phase2/expert/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_experts.pt \
  --windows $PHASE2_OUT/phase2_candidate_windows.jsonl \
  --split test \
  --out $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_experts_scores.csv

# --- Step 9: compare every baseline + all 3 Phase 2 architectures, pooled + faceted (PERS/LOC/ORG only -- no TIME/PROD in this variant) ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $MATCH_BASE/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --logistic-score $MATCH_BASE/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --mlp-score $MATCH_BASE/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_experts_scores.csv \
  --figures-dir $PHASE2_FIGS

python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY \
  --load-data $DATA_SRC \
  --platt-scaling-score $MATCH_BASE/test_results/$TRAIN_TEST_TAG/platt_scaling.csv \
  --logistic-score $MATCH_BASE/test_results/$TRAIN_TEST_TAG/logistic_regression.csv \
  --mlp-score $MATCH_BASE/test_results/$TRAIN_TEST_TAG/mlp_baseline.csv \
  --camembert-mlp-score $PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$PHASE2_OUT/test_results/$TRAIN_TEST_TAG/mbert_experts_scores.csv \
  --facet-by-type PERS LOC ORG \
  --figures-dir "$PHASE2_FIGS/by_type"

################################################################################
## letemps_fr / gliner -- score-only cross-dataset check: reuses this script's
## match_entities checkpoints above (no retraining), evaluated on letemps_fr's own
## test split. letemps_fr needs no candidate filtering of its own (already
## PERS/LOC/ORG-only by construction -- its own labels.json never asked GLiNER2 for
## TIME/PROD), but its label_reliability/logistic_regression_data.csv/candidate
## windows are rebuilt here under their own match_entities/ folder (not reused from
## level_ablation/span_level_fuzzy) so this whole ablation stays fully self-contained
## and never reads out of the "normal" pipeline's folders -- content is identical
## either way (same deduplicate_ner_features.csv, same mode/level), this is purely a
## folder-layout separation. ocr_features.csv/context_features.csv are
## level-independent (SS3) and reused directly from $NER_BASE_LT since the candidate
## set itself didn't change.
################################################################################

TRAIN_TEST_TAG_LT=train_hipe_test_letemps

NER_BASE_LT=data/letemps_fr/gliner/data_baseline
DATA_SRC_LT=data/data_source/letemps/letemps_fr.csv

MATCH_OUT_LT=$NER_BASE_LT/match_entities
LABEL_RELIABILITY_LT=$MATCH_OUT_LT/label_reliability_span_level_fuzzy.csv

MATCH_PHASE2_OUT_LT=data/letemps_fr/gliner/data_phase2/match_entities
WINDOWS_LT=$MATCH_PHASE2_OUT_LT/phase2_candidate_windows.jsonl

PHASE1_FIGS_LT=figures/letemps_fr/gliner/modeling/match_entities/$TRAIN_TEST_TAG_LT
PHASE2_FIGS_LT=figures/letemps_fr/gliner/phase2/match_entities/$TRAIN_TEST_TAG_LT

for f in "$NER_BASE_LT/deduplicate_ner_features.csv" "$NER_BASE_LT/ocr_features.csv" "$NER_BASE_LT/context_features.csv" \
         "$PHASE1_CKPT/platt_scaling.pt" "$PHASE1_CKPT/logistic_regression.pt" "$PHASE1_CKPT/mlp_baseline.pt" \
         "$PHASE2_CKPT/mbert_mlp.pt" "$PHASE2_CKPT/mbert_simple_mlp.pt" "$PHASE2_CKPT/mbert_experts.pt"; do
  if [ ! -f "$f" ]; then
    echo "Missing $f -- run baseline_phase2base.sh first (for letemps_fr's shared NER/OCR/context artifacts) and the hipe block above (for this script's own checkpoints)." >&2
    exit 1
  fi
done

# --- Rebuild letemps_fr's label_reliability/logistic_regression_data.csv/candidate windows under match_entities/ ---
python src/ner/label_reliability.py --mode fuzzy \
  --load-data $DATA_SRC_LT \
  --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
  --out $LABEL_RELIABILITY_LT

python src/phase1/feature_extraction/prepare_data_logistic.py \
  --load-data $DATA_SRC_LT \
  --ner-features $NER_BASE_LT/deduplicate_ner_features.csv \
  --ocr-features $NER_BASE_LT/ocr_features.csv \
  --context-features $NER_BASE_LT/context_features.csv \
  --label-reliability $LABEL_RELIABILITY_LT \
  --out $MATCH_OUT_LT/logistic_regression_data.csv

python src/phase2/base/build_candidate_windows.py \
  --load-data $DATA_SRC_LT \
  --label-reliability $LABEL_RELIABILITY_LT \
  --out $WINDOWS_LT

python src/phase1/modeling/platt_scaling.py \
  --checkpoint-in $PHASE1_CKPT/platt_scaling.pt \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --out $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --figures-dir $PHASE1_FIGS_LT

python src/phase1/modeling/logistic_regression.py \
  --checkpoint-in $PHASE1_CKPT/logistic_regression.pt \
  --data $MATCH_OUT_LT/logistic_regression_data.csv \
  --out $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --figures-dir $PHASE1_FIGS_LT

python src/phase1/modeling/mlp_baseline.py \
  --checkpoint-in $PHASE1_CKPT/mlp_baseline.pt \
  --data $MATCH_OUT_LT/logistic_regression_data.csv \
  --out $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --figures-dir $PHASE1_FIGS_LT

python src/phase2/base/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_mlp.pt \
  --windows $WINDOWS_LT \
  --split test \
  --out $MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv

python src/phase2/simple/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_simple_mlp.pt \
  --windows $WINDOWS_LT \
  --split test \
  --out $MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_simple_mlp_scores.csv

python src/phase2/expert/evaluate.py \
  --checkpoint $PHASE2_CKPT/mbert_experts.pt \
  --windows $WINDOWS_LT \
  --split test \
  --out $MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_experts_scores.csv

# --- Compare, same conventions as the hipe2020_fr block above ---
python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --platt-scaling-score $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --logistic-score $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --mlp-score $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --camembert-mlp-score $MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_experts_scores.csv \
  --figures-dir $PHASE2_FIGS_LT

python src/phase1/modeling/plot_reliability_diagram.py --raw-score \
  --label-reliability $LABEL_RELIABILITY_LT \
  --load-data $DATA_SRC_LT \
  --platt-scaling-score $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/platt_scaling.csv \
  --logistic-score $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/logistic_regression.csv \
  --mlp-score $MATCH_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mlp_baseline.csv \
  --camembert-mlp-score $MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_mlp_scores.csv \
  --camembert-mlp-label mbert_mlp_base \
  --extra-score mbert_simple_mlp=$MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_simple_mlp_scores.csv \
  --extra-score mbert_experts=$MATCH_PHASE2_OUT_LT/test_results/$TRAIN_TEST_TAG_LT/mbert_experts_scores.csv \
  --facet-by-type PERS LOC ORG \
  --figures-dir "$PHASE2_FIGS_LT/by_type"
