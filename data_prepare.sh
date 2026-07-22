#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-a40
#SBATCH --time=02:00:00
#SBATCH --job-name=test
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
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

export HF_TOKEN=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

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

## preproces input data -- letemps/fr instead of hipe2020 (--train-url/--dev-url/--test-url
## sonar/de -- no train file at all (confirmed: only dev-de.tsv/test-de.tsv exist
## upstream), so --train-url "" skips that split entirely rather than falling back to
## hipe2020's train file (see preprocessing_data.py's load_train_data/skip logic). Gold
## tags are PER/LOC/ORG (same uppercase spelling as newseye, no HumanProd/TIME) -- already
## handled by label_reliability.py's _TYPE_MAP fix, no further code change needed.
# python src/preprocessing/preprocessing_data.py \
#   --train-url "" \
#   --dev-url https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/sonar/de/HIPE-2022-v2.1-sonar-dev-de.tsv \
#   --test-url https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/sonar/de/HIPE-2022-v2.1-sonar-test-de.tsv \
#   --language de \
#   --out data/data_source/sonar/sonar_de.csv
# python src/analysis/analyze_data_splits.py \
#   --load-data data/data_source/sonar/sonar_de.csv \
#   --labels-file data/data_source/sonar/sonar_de_labels.json \
#   --figures-dir figures/data_analysis/sonar_de
# python src/analysis/plot_ocr_quality_distributions.py \
#   --load-data data/data_source/sonar/sonar_de.csv \
#   --figures-dir figures/data_analysis/sonar_de

## extract NER features -- hipe2020/fr, every data output redirected off the flat
## data/data_baseline/ default into its own data/hipe2020_fr/data_baseline/gliner/ tree,
## since every dataset (hipe2020_fr/hipe2020_de/letemps_fr/newseye_*/sonar_de) sharing the
## same flat default filenames would silently overwrite each other's ner_features.csv etc.
## Figures redirected the same way into figures/ner_analysis/hipe2020_fr/.
# python src/ner/gliner/extract_ner_features.py \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv \
#   --labels-file data/data_source/hipe2020/hipe2020_fr_labels.json \
#   --out data/hipe2020_fr/gliner/data_baseline/ner_features.csv

# python src/ner/gliner/deduplicate_ner_features.py \
#   --ner-features data/hipe2020_fr/gliner/data_baseline/ner_features.csv \
#   --out data/hipe2020_fr/gliner/data_baseline/deduplicate_ner_features.csv \
#   --conflicts-out data/hipe2020_fr/gliner/data_baseline/ner_overlap_conflicts.json

# python src/ner/ner_features_to_token_format.py \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv \
#   --ner-features data/hipe2020_fr/gliner/data_baseline/deduplicate_ner_features.csv \
#   --out data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv

# python src/ner/label_reliability.py \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv \
#   --ner-features data/hipe2020_fr/gliner/data_baseline/deduplicate_ner_features.csv \
#   --out data/hipe2020_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --mode type_only

## Same 4 scripts, run once per split (all_set = every candidate, train, test) into their
## own figures-dir subfolder -- --split "" (all_set) is the default, passed explicitly
## below only for symmetry with the other two.
# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/all_set
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/hipe2020_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/all_set
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/all_set
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/all_set

# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/train
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/hipe2020_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/train
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/train
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/train

# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/test
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/hipe2020_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/test
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/test
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/test

# python src/ner/gliner/extract_ner_features.py \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv \
#   --labels-file data/data_source/hipe2020/hipe2020_de_labels.json \
#   --out data/hipe2020_de/gliner/data_baseline/ner_features.csv

# python src/ner/gliner/deduplicate_ner_features.py \
#   --ner-features data/hipe2020_de/gliner/data_baseline/ner_features.csv \
#   --out data/hipe2020_de/gliner/data_baseline/deduplicate_ner_features.csv \
#   --conflicts-out data/hipe2020_de/gliner/data_baseline/ner_overlap_conflicts.json

# python src/ner/ner_features_to_token_format.py \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv \
#   --ner-features data/hipe2020_de/gliner/data_baseline/deduplicate_ner_features.csv \
#   --out data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv

# python src/ner/label_reliability.py \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv \
#   --ner-features data/hipe2020_de/gliner/data_baseline/deduplicate_ner_features.csv \
#   --out data/hipe2020_de/gliner/data_baseline/label_reliability_type_only.csv \
#   --mode type_only

## Same 4 scripts, run once per split (all_set/train/test) into their own figures-dir
## subfolder -- see hipe2020_fr/gliner above for the same pattern.
# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/all_set
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/hipe2020_de/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/all_set
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/all_set
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split "" \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/all_set

# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/train
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/hipe2020_de/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/train
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/train
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split train \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/train

# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/test
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/hipe2020_de/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/test
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/test
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/hipe2020/hipe2020_de.csv --split test \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/test

## extract NER features -- letemps/fr, same redirection pattern as hipe2020_fr/hipe2020_de
## above (data/letemps_fr/gliner/data_baseline/ + figures/ner_analysis/letemps_fr/gliner).
# python src/ner/gliner/extract_ner_features.py \
#   --load-data data/data_source/letemps/letemps_fr.csv \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --out data/letemps_fr/gliner/data_baseline/ner_features.csv

# python src/ner/gliner/deduplicate_ner_features.py \
#   --ner-features data/letemps_fr/gliner/data_baseline/ner_features.csv \
#   --out data/letemps_fr/gliner/data_baseline/deduplicate_ner_features.csv \
#   --conflicts-out data/letemps_fr/gliner/data_baseline/ner_overlap_conflicts.json

# python src/ner/ner_features_to_token_format.py \
#   --load-data data/data_source/letemps/letemps_fr.csv \
#   --ner-features data/letemps_fr/gliner/data_baseline/deduplicate_ner_features.csv \
#   --out data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv

# python src/ner/label_reliability.py \
#   --load-data data/data_source/letemps/letemps_fr.csv \
#   --ner-features data/letemps_fr/gliner/data_baseline/deduplicate_ner_features.csv \
#   --out data/letemps_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --mode type_only

# ## Same 4 scripts, run once per split (all_set/train/test) into their own figures-dir
# ## subfolder -- see hipe2020_fr/gliner above for the same pattern.
# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split "" \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/all_set
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/letemps_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split "" \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/all_set
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split "" \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/all_set
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split "" \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/all_set

# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split train \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/train
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/letemps_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split train \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/train
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split train \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/train
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split train \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/train

# python src/analysis/analyze_ner_mismatches.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split test \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/test
# python src/analysis/plot_ner_score_distribution.py \
#   --label-reliability data/letemps_fr/gliner/data_baseline/label_reliability_type_only.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split test \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/test
# python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split test \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/test
# python src/analysis/plot_alignability_by_type.py \
#   --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#   --load-data data/data_source/letemps/letemps_fr.csv --split test \
#   --labels-file data/data_source/letemps/letemps_fr_labels.json \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/test


################ Historical NER baseline (for comparison)
## hipe2020/fr -- this model does single-label BIO tagging (one predicted type per span,
## chosen by the model itself), but its own tokenizer can still split one real word into
## multiple adjacent fragments that each get tagged separately (e.g. "Berlin" ->
## "BER"+"LIN", both landing on the same train-data token) -- rare (a few hundred out of
## ~10.8k candidates) but real, so this uses its own purpose-built dedup
## (src/ner/historical_ner/deduplicate_ner_features.py), not GLiNER2's discard-the-loser
## version: overlapping fragments are MERGED into one final span (highest ner_score wins
## type/score, entity_text is every fragment concatenated in file order) rather than
## silently truncated. ner_features_to_token_format.py/label_reliability.py below read
## this merged deduplicate_ner_features.csv, not the raw ner_features.csv directly --
## skipping dedup here would leave overlapping fragments in the input and corrupt the
## token-alignment merge downstream (confirmed the hard way).
# python src/ner/historical_ner/extract_ner_features.py \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv \
#   --out data/hipe2020_fr/historical_ner/data_baseline/ner_features.csv

python src/ner/historical_ner/deduplicate_ner_features.py \
  --ner-features data/hipe2020_fr/historical_ner/data_baseline/ner_features.csv \
  --out data/hipe2020_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
  --conflicts-out data/hipe2020_fr/historical_ner/data_baseline/ner_overlap_conflicts.json

python src/ner/ner_features_to_token_format.py \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv \
  --ner-features data/hipe2020_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
  --out data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv

python src/ner/label_reliability.py \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv \
  --ner-features data/hipe2020_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
  --out data/hipe2020_fr/historical_ner/data_baseline/label_reliability_type_only.csv \
  --mode type_only

## Same 4 scripts, run once per split (all_set/train/test) into their own figures-dir
## subfolder -- see hipe2020_fr/gliner above for the same pattern.
python src/analysis/analyze_ner_mismatches.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/all_set
python src/analysis/plot_ner_score_distribution.py \
  --label-reliability data/hipe2020_fr/historical_ner/data_baseline/label_reliability_type_only.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/all_set
python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/all_set
python src/analysis/plot_alignability_by_type.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "" \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/all_set

python src/analysis/analyze_ner_mismatches.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/train
python src/analysis/plot_ner_score_distribution.py \
  --label-reliability data/hipe2020_fr/historical_ner/data_baseline/label_reliability_type_only.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/train
python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/train
python src/analysis/plot_alignability_by_type.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split train \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/train

python src/analysis/analyze_ner_mismatches.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/test
python src/analysis/plot_ner_score_distribution.py \
  --label-reliability data/hipe2020_fr/historical_ner/data_baseline/label_reliability_type_only.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/test
python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/test
python src/analysis/plot_alignability_by_type.py \
  --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
  --load-data data/data_source/hipe2020/hipe2020_fr.csv --split test \
  --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/test