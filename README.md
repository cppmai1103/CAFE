# CAFE-TEI: Confidence-Aware NER Reliability Scoring

## 1. Project goal

CAFE-TEI estimates a per-candidate reliability score for named-entity candidates produced
by an off-the-shelf NER model (GLiNER2) over OCR-derived historical newspapers
(HIPE-2022, French). Raw NER confidence is not enough on noisy OCR text -- a model can be
confident about a wrong candidate when OCR noise, broken lines, or historical spelling
variation mislead it. The project trains a second-stage model:

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

## 2. Two-phase design

| | Phase 1 | Phase 2 |
|---|---|---|
| Features | Manually engineered (OCR/dictionary evidence, context windows) | Learned from a frozen LM encoder's hidden states + raw metadata |
| Input | Aggregated per-candidate feature vector | Candidate-specific context window, target span marked |
| Models | B0 raw score, B1 Platt scaling, B3 logistic regression, MLP baseline | Frozen-encoder + MLP head (`phase2`), latent mixture-of-experts head (`phase2_expert`), text-marker alternative (`phase2_simple`) |
| Status | **Implemented** | **Implemented** (base model, MoE variant, ablations, cross-encoder comparison) |

Both phases share the same document-level 70/10/20 train/val/test split (fixed seed,
`preprocessing/preprocessing_data.py`) and the same evaluation metrics, so results are
directly comparable. `val` is used for early stopping only (never fit on) by every model
in both phases; `test` is reserved for final reported numbers.

Deep-dive docs:
- `docs/pipeline.md` -- authoritative, up-to-date map of every stage/script/IO in run order (Phase 1)
- `docs/phase1_manual_features.md` -- Phase 1 methodology (feature groups, expert/gate design)
- `docs/phase2_learned_features.md` -- Phase 2 design (candidate windows, encoder, MoE head)
- `docs/further_steps.md` -- open ideas / ablations not yet implemented
- `docs/running.md` -- step-by-step commands to run the whole pipeline from raw data to figures

## 3. Phase 1: manual features

Pipeline (`src/preprocessing/`, `src/gliner/`, `src/feature_extraction/`, `src/modeling/`):

```text
HIPE-2022 TSV
-> token-level CSV + document-level train/val/test split
-> GLiNER2 candidate extraction + dedup + gold-label alignment
-> manual OCR/context/dictionary feature extraction
-> B0 (raw ner_score) / B1 (Platt scaling) / B3 (logistic regression) / MLP baseline
```

All of B1/B3/MLP fit on `train`, early-stop on `val`, and are scored on every split;
`plot_reliability_diagram.py` produces the reliability diagram, metrics bar (Brier/ECE/MCE
calibration vs. AUROC/AURC/E-AURC discrimination), ROC curve, and risk-coverage curve in
`figures/modeling/`. See `docs/pipeline.md` SS1-5 for the exact script/IO chain.

## 4. Phase 2: learned features

Pipeline (`src/phase2/`):

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
`test`). Two alternative architectures were built to compare against this base model:

- `src/phase2_expert/` -- swaps the MLP head for K=4 latent experts + a softmax gate
  (`final_logit = sum_k alpha_k * z_k`), with a load-balancing auxiliary loss to discourage
  gate collapse onto one expert. `analyze_experts.py` diagnoses gate usage (per-expert
  alpha distribution) and expert specialization (pairwise parameter cosine similarity).
- `src/phase2_simple/` -- writes type/confidence into the input as marker-tagged text
  instead of trainable side embeddings, using only the encoder's own pretrained
  vocabulary; a lighter-weight alternative to `phase2`'s embedding-injection approach.

Each folder's `compare.py` plots it against `phase2`'s base model via
`modeling/plot_reliability_diagram.py --extra-score`.

## 5. Ablations

All under `src/phase2/`, varying `train.py` flags; results plotted per group in
`figures/ablation/<group>/`:

| Group | What varies | Flags |
|---|---|---|
| `embeddings` | Drop one metadata embedding at a time (NER score / type / dict flag / target flag) | `--no-ner-score`, `--no-type`, `--no-dict-flag`, `--no-target-flag` |
| `ner_score` | Keep NER-score info but simplify what the ScoreMLP sees | `--score-features {full,p_logit_only,logit_only,p_only}` |
| `encoders` | Swap the frozen encoder (CamemBERT vs. mBERT, mDeBERTa-v3, multilingual-e5, XLM-RoBERTa) | `--encoder-name ...` |

`ner_score` isolates whether the theoretically-redundant `1-p` term in `[p, logit(p), 1-p]`
matters empirically (`full` vs. `p_logit_only`), and whether `logit(p)` alone carries the
useful nonlinearity that raw `p` would otherwise force the head to relearn (`logit_only`/
`p_only`). See `src/phase2/model.py`'s module docstring for the full reasoning.

## 6. Shared evaluation metrics

`src/modeling/metrics.py` splits metrics into two families:

- **Calibration** -- Brier score, Expected Calibration Error (ECE), Maximum Calibration
  Error (MCE): does the score's value match the true empirical probability.
- **Discrimination** -- AUROC, AURC, E-AURC: does the score rank reliable candidates above
  unreliable ones, independent of its value. (Platt scaling is a monotonic transform of
  `ner_score`, so it never changes discrimination metrics, only calibration ones.)

Every model produces both a raw and a calibrated score per candidate, plus `label_reliable`
for evaluation. `plot_reliability_diagram.py` renders all four comparison figures (reliability
diagram, metrics bar, ROC curve, risk-coverage curve) filtered to `--split test` by default.

## 7. Repository layout

```text
confidence-aware/
├── src/                      pipeline code, run as `python src/<module>/<script>.py`
│   ├── preprocessing/        HIPE-2022 TSV -> token-level CSV + train/val/test split
│   ├── gliner/                GLiNER2 candidate extraction, dedup, gold-label alignment
│   ├── feature_extraction/   Phase 1 manual OCR/context feature extraction
│   ├── modeling/              Phase 1 baselines (B0/B1/B3/MLP) + shared metrics/plotting
│   ├── phase2/                Phase 2 base model (frozen encoder + MLP head) + ablations
│   ├── phase2_expert/        Phase 2 latent mixture-of-experts head + gate diagnostics
│   ├── phase2_simple/        Phase 2 text-marker alternative architecture
│   ├── analysis/              read-only diagnostics (data splits, OCR quality, NER mismatches)
│   └── other/                 one-off scripts, not part of the main pipeline
│
├── data/
│   ├── data_baseline/        Phase 1 CSVs -- features, labels, baseline scores
│   ├── data_phase2/           Phase 2 candidate windows JSONL + base-model score CSVs
│   ├── data_phase2_expert/   phase2_expert score CSVs
│   └── data_phase2_simple/   phase2_simple score CSVs
│
├── checkpoints/
│   ├── phase2/                one .pt per phase2 variant / ablation / encoder
│   ├── phase2_expert/
│   └── phase2_simple/
│
├── figures/                  generated plots, mirroring src/ above
│   ├── data_analysis/         split + OCR-quality sanity checks (running.md step 1)
│   ├── ner_analysis/          GLiNER2 mismatch/confusion diagnostics (step 2)
│   ├── modeling/               Phase 1 baseline comparison plots
│   ├── ablation/               embeddings/ ner_score/ encoders/ (see SS5 above)
│   ├── phase2_simple/
│   ├── phase2_expert/
│   └── pipeline/               architecture/pipeline diagrams
│
├── docs/                     design docs -- see SS2 "Deep-dive docs" above
├── run/                      SLURM job logs from past runs
└── try1/                     prior exploratory run (notebooks, earlier data cut) -- not part of the current pipeline
```

## 8. Main claims the experiments support

```text
1. Raw NER confidence is not reliable enough for OCR-derived historical newspapers --
   calibrating it (B1) and comparing against learned models quantifies the gap.
2. Manually engineered evidence features (Phase 1) already improve over raw confidence.
3. Learned representations from a frozen LM encoder (Phase 2) can match or improve on
   manual feature engineering while requiring less hand-design.
4. Ablations isolate which evidence (NER score, entity type, OCR/dictionary flags, target
   span marking, choice of encoder) actually drives reliability estimation.
```
