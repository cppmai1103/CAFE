## Project

Confidence-aware NER/NEL reliability scoring for French historical newspaper OCR (the PressMint
project). This directory holds the current experiment: comparing two zero-shot NER models on the
HIPE-2020 (French) dev split and checking whether OCR word quality correlates with NER errors. It
is not a git repository — there's no package/build system, just a sequential chain of Jupyter
notebooks that read/write CSV checkpoints in this same directory.

## Environment

- Everything runs on the L3iCalcul SLURM cluster (GPU needed for the NER models). See
  `run_job_cpu.sh` (CPU jobs, `cpu-only` partition) and `run_job_notebook.sh` (Jupyter server on
  `gpu-a40`, port-forwarded back to a local browser) in the parent `workspace/` directory for the
  sbatch templates.
- Conda env is created/activated per `run_job_notebook.sh` (`module load Anaconda3`, then source
  the conda profile script, `conda activate notebook`). Each notebook installs its own deps in its
  first cell (`pip`/`%pip install`) — there's no shared requirements file.
- `nuextract3_ner.ipynb` needs a very recent `transformers` (NuExtract3 uses the `qwen3_5` arch);
  if you hit "unrecognized model type qwen3_5", install transformers from GitHub main.
- `gliner_ner.ipynb` is CPU-viable but much slower; NuExtract3 (4B, bf16) needs a real GPU
  (`gpu-a40`/`gpu-a6000`/`gpu-h100` — likely too tight on `gpu-2080ti`'s 11GB with activations).

## Pipeline (run notebooks in this order)

1. **`hipe_ocr_ner_extraction.ipynb`** — downloads the HIPE-2022 French dev TSV
   (`HIPE-2022-v2.1-hipe2020-dev-fr.tsv`) from the `hipe-eval/HIPE-2022-data` GitHub repo, parses
   its CoNLL-style format (`#`-comment lines mark `document_id`, blank lines are segment
   boundaries) into a flat token table, and scores each token against the impresso
   `OCR-quality-assessment-unigram` French bloom filter (`ocrqa-wp_v1.0.6-fr.bloom`, pulled via
   `huggingface_hub`) to flag whether it's a known French word form. Derives mention-span /
   ±5-±10-token-context / document-level OCR-known-rate features. Writes
   **`hipe2020_dev_fr_ocrqa.csv`** — the token-level table every downstream notebook reads.

2. **`nuextract3_ner.ipynb`** and **`gliner_ner.ipynb`** — both independently: reconstruct
   sentence-level text from the token table (`reconstruct_sentences`, duplicated in both
   notebooks — split on `MISC` containing `EndOfSentence`, respect `NoSpaceAfter` for
   detokenization, never cross document boundaries), then run one NER model over every sentence
   with labels `PERS/LOC/ORG/TIME/PROD` (HIPE's five gold coarse types):
   - **NuExtract3** (`numind/NuExtract3`, 4B structured-extraction LLM): one `model.generate()`
     call per sentence via a JSON extraction template; no native confidence, so
     `run_ner_with_confidence` approximates one as the mean per-token generation probability over
     the characters of the entity's verbatim text in the raw output. This is *not* a calibrated
     P(correct) — only a relative signal. Checkpoints to CSV every `CHECKPOINT_EVERY` (50)
     sentences and resumes from the checkpoint on rerun, since the full pass is expensive — run it
     via `sbatch`, not interactively. Output: `hipe2020_dev_fr_nuextract3_ner.csv` (no
     `start`/`end` columns).
   - **GLiNER multilingual** (`urchade/gliner_multi-v2.1`, 209M): batched `gliner_model.inference`
     over all sentences at once (`GLINER_THRESHOLD = 0.5`), using its native per-entity `score` —
     no approximation needed, cheap enough to run in one interactive pass. Output:
     `hipe2020_dev_fr_gliner_ner.csv` (includes `start`/`end` character offsets).

3. **`evaluate_ner_metrics.ipynb`** — evaluates one prediction CSV (`PRED_PATH`, toggle between the
   GLiNER and NuExtract3 outputs) against gold spans rebuilt from `hipe2020_dev_fr_ocrqa.csv`'s
   `NE-COARSE-LIT` BIO tags.
   - Gold and predicted spans are matched by **exact character boundary** within a sentence
     (`match_spans`): boundary+label match → TP; boundary match, different label → mislabel
     (counts as FN for the true type, feeds the confusion matrix, gold span not consumed);
     no boundary match → spurious FP (`true_label = "O"`); any never-matched gold span → FN.
   - NuExtract3 predictions have no character offsets, so spans are resolved by verbatim text
     search with a per-sentence cursor (to disambiguate repeated entity text within one sentence).
   - Metrics: confusion matrix, per-label + micro precision/recall/F1, Expected Calibration Error
     (ECE, 10 bins) and AUPRC — the latter two computed only over predictions that carry a
     `confidence` score, using `is_correct` (strict span+label match) as the binary outcome.
   - Section 8 re-runs the confusion matrix/PRF split into two groups — "OCR correct" vs "OCR
     error in span" (any token overlapping the entity's span flagged `ocr_word_known == False`) —
     to test the project's core question: does OCR quality correlate with NER correctness.

## Data conventions

- `MISC` column flags (space-separated, may combine): `EndOfSentence`, `EndOfLine`,
  `NoSpaceAfter`. Detokenization logic (build sentence text, decide spacing) is reimplemented in
  three places (`hipe_ocr_ner_extraction.ipynb` is token-only; `reconstruct_sentences` in both NER
  notebooks; `build_gold`/`build_token_offsets` in `evaluate_ner_metrics.ipynb`) — keep them in
  sync if the detokenization rule ever changes.
- Gold BIO tags (`NE-COARSE-LIT`) use lowercase types (`pers/loc/org/time/prod`); prediction CSVs
  and evaluation code use uppercase (`PERS/LOC/ORG/TIME/PROD`) — mapped via `LABEL_MAP` in
  `evaluate_ner_metrics.ipynb`.
- `ocr_word_known`: `True` = known French word form (bloom filter hit), `False` = not found
  (likely OCR error, or a rare/proper name), `None`/NaN = token normalized to nothing (pure
  punctuation) — treated as `True`/non-error for aggregate OCR-rate features and for the
  OCR-error-in-span grouping.
