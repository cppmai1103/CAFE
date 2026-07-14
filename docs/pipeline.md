# Pipeline

Current end-to-end run order, matching `script.sh`. Each stage lists the scripts, their
inputs/outputs, and where they live in `src/`. See `docs/phase1_manual.md` for the
methodology behind splits/features/metrics; this doc just tracks what actually runs and
in what order.

## 0. Environment

`script.sh` provisions the `cafe` conda env (Python 3.11, `src/requirements.txt` +
torch/cu124) and runs everything below via `sbatch` on the cluster.

## 1. Preprocessing

| Script | Input | Output |
|---|---|---|
| `preprocessing/preprocessing_data.py` | HIPE-2022 (fr) TSV (downloaded) | `data/hipe2020_train_fr_train_data.csv` -- token-level, one row per token, with `document_id`/`sentence_id`/`token_id`/`split`/`dictionary_score`/`sentence_ocr_mean`/`document_ocr_mean` attached |
| `analysis/analyze_data_splits.py` | train data CSV | `figures/data_analysis/documents_per_split.png`, `entity_type_breakdown_per_split.png` |
| `analysis/plot_ocr_quality_distributions.py` | train data CSV | `figures/data_analysis/dictionary_score_counts.png`, `document_ocr_mean_distribution.png`, `sentence_ocr_mean_distribution.png` |

Splitting (`docs/phase1_manual.md` SS6.1) is per-document: `expert_train` 50%, `gate_train`
20%, `calibration` 10%, `test` 20% -- so no document's context leaks across splits.
`preprocessing/ocr_dictionary_check.py` is the bloom-filter dictionary-membership check
`preprocessing_data.py` calls into (not run standalone).

## 2. NER candidate extraction (GLiNER2)

| Script | Input | Output |
|---|---|---|
| `gliner/extract_ner_features.py` | train data CSV | `data/ner_features.csv` -- every (span, type) candidate GLiNER2 scores, one row each (no threshold, no dedup -- huge) |
| `gliner/deduplicate_ner_features.py` | `ner_features.csv` | `data/deduplicate_ner_features.csv` -- greedy per-sentence overlap resolution (keep highest `ner_score` first), sorted in document reading order; conflict report at `data/ner_overlap_conflicts.json` (gitignored, ~120MB) |
| `gliner/ner_features_to_token_format.py --threshold 0.5` | train data CSV + `deduplicate_ner_features.csv` | `data/hipe2020_train_fr_gliner_token_format_threshold0.5.csv` -- token-level, gold + GLiNER prediction side by side |
| `gliner/label_reliability.py --mode type_only` | train data CSV + `deduplicate_ner_features.csv` | `data/label_reliability_type_only.csv` -- adds ground-truth `reliability_score` per candidate (also supports `--mode span_type`) |

`label_reliability.py`'s two modes: `span_type` requires exact boundary + type match
against a gold entity; `type_only` only requires every token the candidate covers to have
the matching gold type (boundary-agnostic) -- `type_only` is what everything downstream
uses by default.

## 3. NER quality analysis (token-format CSV)

All read `hipe2020_train_fr_gliner_token_format_threshold0.5.csv`, output to
`figures/ner_analysis/`:

| Script | Output |
|---|---|
| `analysis/analyze_gliner_mismatches.py` | `confusion_matrix_threshold0.5.png`, `precision_recall_f1_threshold0.5.png`, `alignability_threshold0.5.png` |
| `analysis/plot_ner_score_distribution.py` | `ner_score_distribution.png` (reads `label_reliability_type_only.csv` instead) |
| `analysis/plot_confusion_matrix_by_dictionary_score.py` | `confusion_matrix_by_dictionary_score.png` |
| `analysis/plot_alignability_by_type.py` | `alignability_by_type_threshold0.5.png` |

## 4. Feature extraction (manual features, SS4.2/4.3)

| Script | Input | Output |
|---|---|---|
| `feature_extraction/extract_ocr_features.py` | train data CSV + `deduplicate_ner_features.csv` | `data/ocr_features.csv` -- span-level OCR evidence |
| `feature_extraction/extract_context_features.py` | train data CSV + `deduplicate_ner_features.csv` | `data/context_features.csv` -- context-window OCR evidence |
| `feature_extraction/prepare_data_logistic.py` | `deduplicate_ner_features.csv` + `ocr_features.csv` + `context_features.csv` + `label_reliability_type_only.csv` | `data/logistic_regression_data.csv` -- one joined row per candidate, `reliability_score` + `split` attached; ready-to-train input for B3 |

## 5. Modeling / calibration (B0/B1/B3)

| Script | Input | Output |
|---|---|---|
| `modeling/platt_scaling.py` | `label_reliability_type_only.csv` | `data/platt_scaling.csv` (B1: `calibrated_score` fit on `calibration` split, scored on every split), `figures/modeling/platt_scaling_fit.png` |
| `modeling/logistic_regression.py` | `logistic_regression_data.csv` | `data/logistic_regression.csv` (B3: `calibrated_score` fit on `expert_train` split, scored on every split), `figures/modeling/logistic_regression_weights.png` |
| `modeling/plot_reliability_diagram.py --platt-scaling-score data/platt_scaling.csv --logistic-score data/logistic_regression.csv` | `label_reliability_type_only.csv` + both scores above | see below |

B0 (raw `ner_score`) is always included; B1/B3 are drawn if their `--platt-scaling-score`/
`--logistic-score` CSV is given. Default `--split test` (final evaluation only, per
`docs/phase1_manual.md` SS6.1). Outputs, all in `figures/modeling/`:

- `reliability_diagram_<labels>.png` -- calibration curve (x = mean predicted probability, y = empirical accuracy)
- `metrics_bar_<labels>.png` -- Brier score / ECE / MCE / AUROC / E-AURC, one bar per score
- `roc_curve_<labels>.png` -- discrimination (TPR vs FPR)
- `risk_coverage_<labels>.png` -- discrimination (risk vs coverage)
- `bins_<labels>.csv` -- per-bin `true`/raw/platt_scaling/logistic + `delta_<label>` columns

Fit-split asymmetry is intentional (`docs/phase1_manual.md` SS6.1): B3 needs more data for
its ~20-parameter model (`expert_train`, 50%), B1 is a 2-parameter fit that saturates with
less (`calibration`, 10%) -- both are evaluated on the same held-out `test` split.

Metrics split into two families (see `modeling/metrics.py`'s module docstring):
**calibration** (Brier/ECE/MCE -- does the score's value match true probability) vs
**discrimination** (AUROC/AURC/E-AURC -- does the score rank reliable above unreliable,
independent of its value). Platt scaling is a monotonic transform of `ner_score`, so it
always has identical AUROC/E-AURC to raw -- only calibration metrics can show its effect.

## Not yet wired into `script.sh`

- `feature_extraction/extract_attribution_features.py`, `other/extract_pressmint_ocrqa.py`
- `train_b3_logistic_regression.py`/`plot_b3_weights.py`'s older path (loads features
  directly via `analysis/analyze_ocr_context_features.py` rather than the prepared
  `logistic_regression_data.csv`) -- superseded by `modeling/logistic_regression.py` for
  the main pipeline, kept for now, not deleted.

## Known open issue

`analysis/analyze_ocr_context_features.py`'s `DEFAULT_NER_FEATURES` still points at the
full `ner_features.csv` (530k rows) rather than `deduplicate_ner_features.csv` (112k rows)
-- would break `load_candidates()`'s row-alignment check if the old
`train_b3_logistic_regression.py`/`plot_b3_weights.py` path above is actually run. Deferred,
not yet fixed.
