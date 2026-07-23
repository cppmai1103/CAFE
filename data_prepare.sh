#!/bin/bash
set -euo pipefail

PYTHON_VERSION=3.11
ENVIRONMENT_NAME="cafe"

cd "$(dirname "${BASH_SOURCE[0]}")"
echo "Working directory: $(pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  echo "Env '${ENVIRONMENT_NAME}' not found, creating it with python=${PYTHON_VERSION}"
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
else
  echo "Env '${ENVIRONMENT_NAME}' already exists, skipping creation"
fi
conda activate ${ENVIRONMENT_NAME}

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r src/requirements.txt

: "${HF_TOKEN:?set HF_TOKEN in your shell before running this script}"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

# ################ hipe2020/fr -- historical NER baseline
# ## This model does single-label BIO tagging (one predicted type per span, chosen by the
# ## model itself), but its own tokenizer can still split one real word into multiple
# ## adjacent fragments that each get tagged separately (e.g. "Berlin" -> "BER"+"LIN"), both
# ## landing on the same train-data token -- rare (a few hundred out of ~10.8k candidates)
# ## but real, so this uses its own purpose-built dedup
# ## (src/ner/historical_ner/deduplicate_ner_features.py), not GLiNER2's discard-the-loser
# ## version: overlapping fragments are MERGED into one final span (highest ner_score wins
# ## type/score, entity_text is every fragment concatenated in file order) rather than
# ## silently truncated. ner_features_to_token_format.py/label_reliability.py below read
# ## this merged deduplicate_ner_features.csv, not the raw ner_features.csv directly --
# ## skipping dedup here would leave overlapping fragments in the input and corrupt the
# ## token-alignment merge downstream. (ner_features.csv already exists on disk, so we
# ## start from dedup, not extraction.)
# python src/ner/historical_ner/deduplicate_ner_features.py \
#   --ner-features data/hipe2020_fr/historical_ner/data_baseline/ner_features.csv \
#   --out data/hipe2020_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
#   --conflicts-out data/hipe2020_fr/historical_ner/data_baseline/ner_overlap_conflicts.json

# python src/ner/ner_features_to_token_format.py \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv \
#   --ner-features data/hipe2020_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
#   --out data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv

# python src/ner/label_reliability.py \
#   --load-data data/data_source/hipe2020/hipe2020_fr.csv \
#   --ner-features data/hipe2020_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
#   --out data/hipe2020_fr/historical_ner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --mode fuzzy

# ## Same 4 analysis scripts, run once per split (all_set/train/test) into their own
# ## figures-dir subfolder.
# for split_arg in "" train test; do
#   split_dir="${split_arg:-all_set}"
#   figures_dir="figures/ner_analysis/hipe2020_fr/historical_ner/${split_dir}"

#   python src/analysis/analyze_ner_mismatches.py \
#     --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_ner_score_distribution.py \
#     --label-reliability data/hipe2020_fr/historical_ner/data_baseline/label_reliability_span_level_fuzzy.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#     --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_alignability_by_type.py \
#     --token-format data/hipe2020_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"
# done

# python src/analysis/plot_reliability_accuracy_by_type.py \
#   --label-reliability data/hipe2020_fr/historical_ner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --figures-dir figures/ner_analysis/hipe2020_fr/historical_ner/all_set

# ################ hipe2020/fr -- GLiNER2 baseline
# ## Uses GLiNER2's own dedup (src/ner/gliner/deduplicate_ner_features.py), not
# ## historical_ner's above -- it discards the losing span on overlap (highest ner_score
# ## wins) instead of merging fragments. (ner_features.csv already exists on disk, so we
# ## start from dedup, not extraction.)
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
#   --out data/hipe2020_fr/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --mode fuzzy

# for split_arg in "" train test; do
#   split_dir="${split_arg:-all_set}"
#   figures_dir="figures/ner_analysis/hipe2020_fr/gliner/${split_dir}"

#   python src/analysis/analyze_ner_mismatches.py \
#     --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_ner_score_distribution.py \
#     --label-reliability data/hipe2020_fr/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#     --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_alignability_by_type.py \
#     --token-format data/hipe2020_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"
# done

# python src/analysis/plot_reliability_accuracy_by_type.py \
#   --label-reliability data/hipe2020_fr/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --figures-dir figures/ner_analysis/hipe2020_fr/gliner/all_set

# ################ hipe2020/de -- GLiNER2 baseline
# ## (ner_features.csv already exists on disk, so we start from dedup, not extraction.)
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
#   --out data/hipe2020_de/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --mode fuzzy

# for split_arg in "" train test; do
#   split_dir="${split_arg:-all_set}"
#   figures_dir="figures/ner_analysis/hipe2020_de/gliner/${split_dir}"

#   python src/analysis/analyze_ner_mismatches.py \
#     --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_de.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_ner_score_distribution.py \
#     --label-reliability data/hipe2020_de/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#     --load-data data/data_source/hipe2020/hipe2020_de.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#     --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_de.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_alignability_by_type.py \
#     --token-format data/hipe2020_de/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/hipe2020/hipe2020_de.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"
# done

# python src/analysis/plot_reliability_accuracy_by_type.py \
#   --label-reliability data/hipe2020_de/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --figures-dir figures/ner_analysis/hipe2020_de/gliner/all_set

# ################ letemps/fr -- GLiNER2 baseline
# ## Gold tag set differs from hipe2020's default (src/ner/gliner/labels.json), so the 3
# ## analysis scripts that support --labels-file need it pointed at letemps_fr's own file.
# ## (ner_features.csv already exists on disk, so we start from dedup, not extraction.)
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
#   --out data/letemps_fr/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --mode fuzzy

# for split_arg in "" train test; do
#   split_dir="${split_arg:-all_set}"
#   figures_dir="figures/ner_analysis/letemps_fr/gliner/${split_dir}"

#   python src/analysis/analyze_ner_mismatches.py \
#     --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --labels-file data/data_source/letemps/letemps_fr_labels.json \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_ner_score_distribution.py \
#     --label-reliability data/letemps_fr/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#     --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --labels-file data/data_source/letemps/letemps_fr_labels.json \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_alignability_by_type.py \
#     --token-format data/letemps_fr/gliner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --labels-file data/data_source/letemps/letemps_fr_labels.json \
#     --figures-dir "${figures_dir}"
# done

# python src/analysis/plot_reliability_accuracy_by_type.py \
#   --label-reliability data/letemps_fr/gliner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --figures-dir figures/ner_analysis/letemps_fr/gliner/all_set

################ letemps/fr -- historical NER baseline
## Not run before (no ner_features.csv on disk yet for this dataset/extractor pair), so
## unlike the other blocks above this one starts from extraction, not dedup. Same French
## historical-ner-baseline model used for hipe2020_fr above -- letemps_fr is also French.
# python src/ner/historical_ner/extract_ner_features.py \
#   --load-data data/data_source/letemps/letemps_fr.csv \
#   --out data/letemps_fr/historical_ner/data_baseline/ner_features.csv

# ## Uses historical_ner's own dedup (merges overlapping fragments) rather than GLiNER2's
# ## discard-the-loser dedup -- see the hipe2020_fr/historical_ner block above for why.
# python src/ner/historical_ner/deduplicate_ner_features.py \
#   --ner-features data/letemps_fr/historical_ner/data_baseline/ner_features.csv \
#   --out data/letemps_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
#   --conflicts-out data/letemps_fr/historical_ner/data_baseline/ner_overlap_conflicts.json

# python src/ner/ner_features_to_token_format.py \
#   --load-data data/data_source/letemps/letemps_fr.csv \
#   --ner-features data/letemps_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
#   --out data/letemps_fr/historical_ner/data_baseline/token_format_threshold0.5.csv

# python src/ner/label_reliability.py \
#   --load-data data/data_source/letemps/letemps_fr.csv \
#   --ner-features data/letemps_fr/historical_ner/data_baseline/deduplicate_ner_features.csv \
#   --out data/letemps_fr/historical_ner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --mode fuzzy

# for split_arg in "" train test; do
#   split_dir="${split_arg:-all_set}"
#   figures_dir="figures/ner_analysis/letemps_fr/historical_ner/${split_dir}"

#   python src/analysis/analyze_ner_mismatches.py \
#     --token-format data/letemps_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --labels-file data/data_source/letemps/letemps_fr_labels.json \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_ner_score_distribution.py \
#     --label-reliability data/letemps_fr/historical_ner/data_baseline/label_reliability_span_level_fuzzy.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_confusion_matrix_by_dictionary_score.py \
#     --token-format data/letemps_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --labels-file data/data_source/letemps/letemps_fr_labels.json \
#     --figures-dir "${figures_dir}"

#   python src/analysis/plot_alignability_by_type.py \
#     --token-format data/letemps_fr/historical_ner/data_baseline/token_format_threshold0.5.csv \
#     --load-data data/data_source/letemps/letemps_fr.csv --split "${split_arg}" \
#     --labels-file data/data_source/letemps/letemps_fr_labels.json \
#     --figures-dir "${figures_dir}"
# done

# python src/analysis/plot_reliability_accuracy_by_type.py \
#   --label-reliability data/letemps_fr/historical_ner/data_baseline/label_reliability_span_level_fuzzy.csv \
#   --figures-dir figures/ner_analysis/letemps_fr/historical_ner/all_set

################ Reliability accuracy by type -- all 5 datasets
## Every label_reliability_span_level_fuzzy.csv above already exists on disk, so this just (re-)runs
## plot_reliability_accuracy_by_type.py across all 5 dataset/extractor pairs without
## touching dedup/token_format/label_reliability again.
for pair in \
  "hipe2020_fr:historical_ner" \
  "hipe2020_fr:gliner" \
  "hipe2020_de:gliner" \
  "letemps_fr:gliner" \
  "letemps_fr:historical_ner" \
; do
  dataset_dir="${pair%%:*}"
  extractor="${pair##*:}"
  python src/analysis/plot_reliability_accuracy_by_type.py \
    --label-reliability "data/${dataset_dir}/${extractor}/data_baseline/label_reliability_span_level_fuzzy.csv" \
    --figures-dir "figures/ner_analysis/${dataset_dir}/${extractor}/all_set"
done
