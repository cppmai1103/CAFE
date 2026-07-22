# CAFE-TEI: Confidence-Aware NER Reliability Scoring

## 1. Project goal

CAFE-TEI estimates a per-candidate reliability score for named-entity candidates produced
by an NER model over OCR-derived historical newspapers (HIPE-2022). Raw NER confidence is
not enough on noisy OCR text -- a model can be confident about a wrong candidate when OCR
noise, broken lines, or historical spelling variation mislead it. The project trains a
second-stage model:

```text
NER candidate + context + OCR evidence -> reliability score r(c) in [0, 1]
```

The prediction unit is one NER candidate (span + predicted type + NER confidence + source
context), not a sentence or document. Example:

```json
{
  "candidate_id": "issue_1908_04_17_page_3_cand_0021",
  "span_text": "Raymond Poincar6",
  "predicted_type": "PERSON",
  "ner_score": 0.94,
  "calibrated_score": 0.72
}
```

Interpretation: the original NER model is confident, but because the OCR form is
suspicious, the reliability model estimates only 0.72 reliability.

## 2. Two NER sources, multiple datasets

The reliability-scoring pipeline (Phase 1 / Phase 2 below) is model-agnostic: it reads a
shared `document_id/sentence_id/start_token_id/end_token_id/predicted_entity_type/
ner_score` candidate schema regardless of which NER model produced it. Two are implemented
(`src/ner/`):

- **`src/ner/gliner/`** -- GLiNER2 (`fastino/gliner2-multi-v1`), scores every (span, type)
  pair independently (no joint softmax across types), so the same span can surface as
  multiple candidates under different types. Its own `deduplicate_ner_features.py`
  resolves this by greedily keeping the highest-`ner_score` candidate per sentence and
  discarding every span that overlaps it.
- **`src/ner/historical_ner/`** -- `emanuelaboros/historical-ner-baseline`, a single-label
  BIO tagger (HF `AutoModelForTokenClassification` + `token-classification` pipeline).
  Overlapping candidates are rare here but not impossible: the model's own tokenizer can
  split one real word into adjacent fragments that each get tagged separately (e.g.
  OCR "Berlin" -> "BER"+"LIN", both landing on the same train-data token). Its own
  `deduplicate_ner_features.py` therefore **merges** transitively-overlapping fragments
  into one final span (type/score from the highest-scoring fragment, `entity_text` is
  every fragment concatenated in file order) instead of discarding the losers.

Both feed into the same shared, model-agnostic code: `src/ner/ner_features_to_token_format.py`
(builds a token-level `NER` column) and `src/ner/label_reliability.py` (the `label_reliable`
ground-truth target). `src/ner/historical_ner/compare.py` overlays both models' calibration/
discrimination numbers on one figure.

The pipeline also runs across multiple HIPE-2022 datasets, each with its own gold tagset
declared in a `{TYPE: prompt wording}` JSON file (`--labels-file`, see
`src/ner/gliner/extract_ner_features.py`'s `load_label_map()`), so a dataset that doesn't
use the full 5-type scheme (e.g. letemps has no TIME/PROD) never carries always-zero
columns through the analysis plots:

| Dataset | Language(s) | Gold types | Labels file |
|---|---|---|---|
| hipe2020 | fr, de | PERS, LOC, ORG, TIME, PROD | `data/data_source/hipe2020/hipe2020_<lang>_labels.json` |
| letemps | fr | PERS, LOC, ORG | `data/data_source/letemps/letemps_fr_labels.json` |
| newseye | fr, de | PERS, LOC, ORG, PROD | `data/data_source/newseye/newseye_<lang>_labels.json` |
| sonar | de (no train split) | PERS, LOC, ORG | `data/data_source/sonar/sonar_de_labels.json` |

`src/preprocessing/preprocessing_data.py` loads a dataset's official HIPE-2022 train/dev/test
TSVs directly (`--train-url`/`--dev-url`/`--test-url`, defaulting from `--language`; pass
`""` to skip a split entirely -- needed for sonar, which ships no train file) into one
combined token-level CSV with a `split` column, so no document's context leaks across
splits and every downstream script's `--load-data` loads the same file and filters by
`split` for whatever it needs.

## 3. Two-phase design

| | Phase 1 | Phase 2 |
|---|---|---|
| Features | Manually engineered (OCR/dictionary evidence, context windows) | Learned from a frozen LM encoder's hidden states + raw metadata |
| Input | Aggregated per-candidate feature vector | Candidate-specific context window, target span marked |
| Models | B0 raw score, B1 Platt scaling, B3 logistic regression, MLP baseline | Frozen-encoder + MLP head (`phase2/base`), latent mixture-of-experts head (`phase2/expert`), text-marker alternative (`phase2/simple`) |
| Status | Implemented for hipe2020_fr (GLiNER2 candidates) | Implemented for hipe2020_fr (GLiNER2 candidates) |

Both phases share the same document-level train/val/test split (`preprocessing/
preprocessing_data.py`) and the same evaluation metrics, so results are directly
comparable. `val` is used for early stopping only (never fit on) by every model in both
phases; `test` is reserved for final reported numbers.

Phase 1's NER-extraction/dedup/token-format/label-reliability stages and the read-only
`src/analysis/` diagnostics are what's actually been run across the multi-dataset,
two-NER-model matrix described in SS2; Phase 1's calibration baselines (B0/B1/B3/MLP) and
all of Phase 2 are implemented but so far only exercised on hipe2020_fr + GLiNER2.

Deep-dive docs:
- `docs/phase1_manual_features.md` -- Phase 1 methodology (feature groups, expert/gate design)
- `docs/phase2_learned_features.md` -- Phase 2 design (candidate windows, encoder, MoE head)
- `docs/further_steps.md` -- open ideas / ablations not yet implemented
- `docs/running.md` -- step-by-step commands to run the pipeline (some paths predate the
  `src/gliner` -> `src/ner/`, `src/modeling` -> `src/phase1/modeling` reorg -- not yet updated)

## 4. Phase 1: manual features

Pipeline (`src/preprocessing/`, `src/ner/`, `src/phase1/feature_extraction/`, `src/phase1/modeling/`):

```text
HIPE-2022 TSV(s)
-> token-level CSV + document-level train/val/test split
-> NER candidate extraction (GLiNER2 or historical-ner-baseline) + dedup + gold-label alignment
-> manual OCR/context/dictionary feature extraction
-> B0 (raw ner_score) / B1 (Platt scaling) / B3 (logistic regression) / MLP baseline
```

All of B1/B3/MLP fit on `train`, early-stop on `val`, and are scored on every split;
`plot_reliability_diagram.py` produces the reliability diagram, metrics bar (Brier/ECE/MCE
calibration vs. AUROC/AURC/E-AURC discrimination), ROC curve, and risk-coverage curve.

## 5. Phase 2: learned features

Pipeline (`src/phase2/base/`):

```text
candidate window (target span marked)
+ token-level dictionary/OCR-quality flag embeddings
+ target-span flag embeddings
-> frozen CamemBERT encoder (gradients flow through, weights don't update)
-> simple pooling: concat(H_CLS, H_first_target, mean(H_target), H_last_target)
+ predicted-type embedding
+ NER-score embedding ([p, logit(p), 1-p] -> ScoreMLP)
-> MLP head -> reliability score
```

Trained/evaluated the same way as Phase 1 (fit on `train`, early-stop on `val`, report on
`test`). Two alternative architectures were built to compare against this base model, each
reusing the exact same candidate-windows JSONL from `phase2/base/build_candidate_windows.py`
(no separate data-build step of their own):

### `src/phase2/expert/` -- latent mixture-of-experts head

Same backbone as the base model above (frozen encoder, side embeddings, pooling, type/
score embeddings into `v_c`), but the single MLP head is replaced by K=4 latent experts
plus a softmax gate over `v_c`:

```text
v_c (same as base model)
-> z_k = Expert_k(v_c)         for k in 1..K, each a small MLP -> 1 logit
-> alpha = softmax(Gate(v_c))  -- v_c also decides how much to trust each expert
-> final_logit = sum_k alpha_k * z_k
-> reliability = sigmoid(final_logit)
```

Trained with a load-balancing auxiliary loss (`--lambda-balance`, default 0.01) to
discourage the gate from collapsing onto a single expert. `analyze_experts.py` diagnoses
gate usage (per-expert alpha distribution) and expert specialization (pairwise parameter
cosine similarity).

### `src/phase2/simple/` -- text-marker alternative

A lighter-weight alternative to the base model's embedding-injection approach: instead of
adding trainable dict/target/type/score embeddings into `inputs_embeds`, it writes the
candidate's type and confidence directly into the input as marker-tagged text, using only
the encoder's own pretrained vocabulary (no new embedding tables):

```text
text_a = window context (unmarked, target embedded naturally)
text_b = [Entity] <span words> [\Entity] [Type] <type word> [\Type]
         [Confidence] <confidence, e.g. "0.98"> [\Confidence]
input  = tokenizer(text_a, text_b)
H = FrozenEncoder(input)
v_c = concat(h_cls, h_span, h_sep, h_entity, h_type, h_confidence)
final_logit = MLP(v_c)
reliability = sigmoid(final_logit)
```

`h_span` mean-pools over the span's own subwords in `text_b`; `h_sep`/`h_entity`/`h_type`/
`h_confidence` each take a single marker token's hidden state (the "marker token as
feature" trick -- a bidirectional encoder's self-attention lets a tag's hidden state
summarize the value that follows it). `--type-confidence-pool {one,average}` controls
`h_type`/`h_confidence`: `one` (default) uses the tag token itself; `average` mean-pools
over the value's own subwords instead. Both pool modes are trained and compared.

Each folder's `compare.py` plots it against `phase2/base`'s model via
`phase1/modeling/plot_reliability_diagram.py --extra-score`.

## 6. Ablations

All under `src/phase2/base/`, varying `train.py` flags; results plotted per group in
`figures/ablation/<group>/`:

| Group | What varies | Flags |
|---|---|---|
| `embeddings` | Drop one metadata embedding at a time (NER score / type / dict flag / target flag) | `--no-ner-score`, `--no-type`, `--no-dict-flag`, `--no-target-flag` |
| `ner_score` | Keep NER-score info but simplify what the ScoreMLP sees | `--score-features {full,p_logit_only,logit_only,p_only,binned}` |
| `encoders` | Swap the frozen encoder (mBERT default, CamemBERT, mDeBERTa-v3, multilingual-e5, XLM-RoBERTa) | `--encoder-name ...` |

`ner_score` isolates whether the theoretically-redundant `1-p` term in `[p, logit(p), 1-p]`
matters empirically (`full` vs. `p_logit_only`), and whether `logit(p)` alone carries the
useful nonlinearity that raw `p` would otherwise force the head to relearn (`logit_only`/
`p_only`). See `src/phase2/base/model.py`'s module docstring for the full reasoning.

## 7. Shared evaluation metrics

`src/phase1/modeling/metrics.py` splits metrics into two families:

- **Calibration** -- Brier score, Expected Calibration Error (ECE), Maximum Calibration
  Error (MCE): does the score's value match the true empirical probability.
- **Discrimination** -- AUROC, AURC, E-AURC: does the score rank reliable candidates above
  unreliable ones, independent of its value. (Platt scaling is a monotonic transform of
  `ner_score`, so it never changes discrimination metrics, only calibration ones.)

Every model produces both a raw and a calibrated score per candidate, plus `label_reliable`
for evaluation. `plot_reliability_diagram.py` renders all four comparison figures (reliability
diagram, metrics bar, ROC curve, risk-coverage curve) filtered to `--split test` by default.

## 8. Repository layout

```text
confidence-aware/
├── src/                       pipeline code, run as `python src/<module>/<script>.py`
│   ├── preprocessing/         HIPE-2022 TSV(s) -> token-level CSV + train/val/test split
│   ├── ner/
│   │   ├── gliner/            GLiNER2 candidate extraction + its own (discard-loser) dedup
│   │   ├── historical_ner/    historical-ner-baseline extraction + its own (merge-fragments) dedup + compare.py
│   │   ├── ner_features_to_token_format.py   shared, model-agnostic -- builds the token-level "NER" column
│   │   └── label_reliability.py              shared, model-agnostic -- label_reliable ground truth
│   ├── phase1/
│   │   ├── feature_extraction/    manual OCR/context feature extraction
│   │   └── modeling/              B0/B1/B3/MLP baselines + shared metrics/plotting
│   ├── phase2/
│   │   ├── base/               frozen-encoder + MLP head model + ablations
│   │   ├── expert/              latent mixture-of-experts head + gate diagnostics
│   │   └── simple/              text-marker alternative architecture
│   └── analysis/                read-only diagnostics (data splits, OCR quality, NER mismatches by type/dictionary-score) -- accepts --split to scope to train/val/test
│
├── data/
│   ├── data_source/<dataset>/         <dataset>_<lang>.csv (all splits) + <dataset>_<lang>_labels.json per dataset (see SS2)
│   └── <dataset>_<lang>/<model>/data_baseline/   per dataset+model: ner_features.csv, deduplicate_ner_features.csv, token_format_threshold0.5.csv, label_reliability_type_only.csv
│
├── figures/
│   ├── data_analysis/<dataset>_<lang>/           split + OCR-quality sanity checks
│   ├── ner_analysis/<dataset>_<lang>/<model>/{all_set,train,test}/   confusion matrix, precision/recall/F1, alignability, ner_score distribution
│   ├── modeling/                Phase 1 baseline comparison plots (hipe2020_fr)
│   └── ablation/                embeddings/ ner_score/ encoders/ (see SS6 above)
│
├── docs/                       design docs -- see SS3 "Deep-dive docs" above
└── old_results/                prior exploratory runs -- not part of the current pipeline
```

## 9. Main claims the experiments support

```text
1. Raw NER confidence is not reliable enough for OCR-derived historical newspapers --
   calibrating it (B1) and comparing against learned models quantifies the gap.
2. Manually engineered evidence features (Phase 1) already improve over raw confidence.
3. Learned representations from a frozen LM encoder (Phase 2) can match or improve on
   manual feature engineering while requiring less hand-design.
4. Ablations isolate which evidence (NER score, entity type, OCR/dictionary flags, target
   span marking, choice of encoder) actually drives reliability estimation.
5. Findings generalize (or don't) across NER sources and datasets -- the same downstream
   pipeline runs unmodified over GLiNER2 or historical-ner-baseline candidates, and over
   hipe2020/letemps/newseye/sonar's differing tagsets and languages (see SS2).
```
