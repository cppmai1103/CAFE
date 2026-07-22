"""Same job as src/ner/gliner/extract_ner_features.py (Phase 1 SS4.1 NER evidence extraction),
but swapping GLiNER2 for emanuelaboros/historical-ner-baseline
(https://huggingface.co/emanuelaboros/historical-ner-baseline) -- a standard HF
AutoModelForTokenClassification (BIO-tagging) fine-tuned for historical/OCR text, run via
transformers' token-classification pipeline with aggregation_strategy="simple" instead of
GLiNER2's per-type independent extraction.

Pipeline:
    train data CSV (see preprocessing_data.py) -> build sentence text -> historical-ner-
    baseline entity extraction -> per-candidate NER evidence features -> CSV.

Reuses src/ner/gliner/extract_ner_features.py's model-agnostic detokenization/chunking
helpers (build_sentence_texts, chunk_long_text, report_chunk_splits) directly rather than
duplicating them -- only the model-loading and entity-extraction steps differ.

Key architectural difference from GLiNER2: this model does single-label BIO tagging (one
predicted type per span, chosen by the model itself), not independent per-type sigmoid
scoring -- so unlike ner_features.csv, a given span here will (almost) never appear as more
than one candidate under different types.

Label mapping: the model's own entity_group values (pers/loc/org/prod/time, see the model
card) are upper-cased directly to the fixed HIPE scheme (PERS/LOC/ORG/PROD/TIME) used
throughout this project (src/ner/gliner/extract_ner_features.py's LABELS) -- no prompt
remapping needed since these labels already match 1:1.

Output: src/ner/historical_ner/data/ner_features.csv -- same column schema as
src/ner/gliner/extract_ner_features.py's ner_features.csv (document_id, sentence_id,
start_token_id, end_token_id, entity_text, predicted_entity_type, ner_score,
span_length_tokens, span_length_characters, sentence_chunked), so it's a drop-in
replacement for deduplicate_ner_features.py / everything downstream, if desired.

Usage:
    python src/ner/historical_ner/extract_ner_features.py
    python src/ner/historical_ner/extract_ner_features.py --limit 50  # quick smoke test
    python src/ner/historical_ner/extract_ner_features.py --load-data data/data_source/hipe2020/hipe2020_fr.csv --out data/hipe2020_fr/historical_ner/data_baseline/ner_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ner.gliner.extract_ner_features import TextChunk, build_sentence_texts, chunk_long_text, report_chunk_splits
from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA

MODEL_ID = "emanuelaboros/historical-ner-baseline"

# The model's own entity_group values (see the model card) upper-cased directly onto the
# fixed HIPE scheme every downstream script expects -- these already match 1:1, no prompt
# remapping needed (unlike GLiNER2's free-text prompt labels).
LABEL_MAP = {"pers": "PERS", "loc": "LOC", "org": "ORG", "prod": "PROD", "time": "TIME"}

DEFAULT_OUT = Path(__file__).parent / "data" / "ner_features.csv"


def extract_candidates_long(
    ner_pipeline,
    texts: list[str],
    batch_size: int,
    chunk_size: int,
    sentence_keys: list[tuple] | None = None,
) -> list[dict]:
    """Same word-chunking strategy as GLiNER2's extract_candidates_long (see that
    docstring) -- split any sentence over chunk_size words into non-overlapping chunks so
    the model's own subword-length limit is never silently truncating the tail of a long
    sentence, then remap each chunk's char offsets back to the whole sentence."""
    chunks_per_text = [chunk_long_text(text, chunk_size) for text in texts]
    report_chunk_splits(chunks_per_text, sentence_keys)

    all_chunk_texts = [chunk.text for chunks in chunks_per_text for chunk in chunks]
    chunk_entities: list[list[dict]] = []
    for i in tqdm(range(0, len(all_chunk_texts), batch_size), desc="NER inference", unit="batch"):
        batch = all_chunk_texts[i : i + batch_size]
        chunk_entities.extend(ner_pipeline(batch))

    results = []
    cursor = 0
    for chunks in tqdm(chunks_per_text, desc="Merging chunk results", unit="sentence"):
        text_entities = chunk_entities[cursor : cursor + len(chunks)]
        cursor += len(chunks)

        merged = []
        for chunk, entities in zip(chunks, text_entities):
            for ent in entities:
                merged.append(
                    {
                        # Recompute from char offsets rather than trusting the pipeline's
                        # own "word" field, which can carry tokenizer artifacts
                        # (leading/trailing whitespace, "##" pieces) depending on the
                        # underlying tokenizer.
                        "text": chunk.text[ent["start"] : ent["end"]],
                        "confidence": float(ent["score"]),
                        "start": ent["start"] + chunk.start_char,
                        "end": ent["end"] + chunk.start_char,
                        "label": ent["entity_group"],
                    }
                )
        results.append({"entities": merged, "chunked": len(chunks) > 1})

    return results


def build_candidates(
    sentences_df: pd.DataFrame,
    token_spans_df: pd.DataFrame,
    ner_pipeline,
    threshold: float,
    batch_size: int,
    chunk_size: int,
) -> pd.DataFrame:
    texts = sentences_df["sentence_text"].tolist()
    sentence_keys = list(zip(sentences_df["document_id"], sentences_df["sentence_id"]))
    results = extract_candidates_long(ner_pipeline, texts, batch_size, chunk_size, sentence_keys)

    token_spans_by_sentence: dict[tuple, list[tuple[int, int, int]]] = {
        key: list(zip(g["token_id"], g["start"], g["end"]))
        for key, g in token_spans_df.groupby(["document_id", "sentence_id"])
    }

    def token_id_range(document_id, sentence_id, start: int, end: int) -> tuple[int | None, int | None]:
        """token_id of the first/last train-data token overlapping this char span (same
        logic as GLiNER2's version)."""
        spans = token_spans_by_sentence.get((document_id, sentence_id), [])
        overlapping = [tid for tid, t_start, t_end in spans if t_end > start and t_start < end]
        if not overlapping:
            return None, None
        return min(overlapping), max(overlapping)

    n_unknown_label = 0
    n_below_threshold = 0
    records = []
    for srow, res in tqdm(
        zip(sentences_df.to_dict("records"), results), total=len(results), desc="Building candidates", unit="sentence"
    ):
        for ent in res["entities"]:
            hipe_label = LABEL_MAP.get(ent["label"].lower())
            if hipe_label is None:
                n_unknown_label += 1
                continue
            if ent["confidence"] < threshold:
                n_below_threshold += 1
                continue
            start_token_id, end_token_id = token_id_range(srow["document_id"], srow["sentence_id"], ent["start"], ent["end"])
            records.append(
                {
                    "document_id": srow["document_id"],
                    "sentence_id": srow["sentence_id"],
                    "start_token_id": start_token_id,
                    "end_token_id": end_token_id,
                    "entity_text": ent["text"],
                    "predicted_entity_type": hipe_label,
                    "ner_score": ent["confidence"],
                    "span_length_tokens": len(ent["text"].split()),
                    "span_length_characters": len(ent["text"]),
                    "sentence_chunked": res["chunked"],
                }
            )

    if n_unknown_label:
        print(f"Skipped {n_unknown_label} entity(ies) with a label outside {sorted(LABEL_MAP)}")
    if n_below_threshold:
        print(f"Skipped {n_below_threshold} entity(ies) below --threshold")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV produced by preprocessing_data.py")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="NER features output CSV path")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum confidence for a candidate to be kept")
    parser.add_argument("--batch-size", type=int, default=16, help="Pipeline batch size")
    parser.add_argument("--chunk-size", type=int, default=384, help="Word-chunk size for splitting long sentences")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sentences (smoke test)")
    args = parser.parse_args()

    print("=== Step 1: Load train data ===")
    print(f"Load data from {args.load_data}")
    tokens_df = pd.read_csv(args.load_data, dtype={"TOKEN": str, "MISC": str},
        # pandas' default NA-string sentinels ("NA", "null", "nan", ...) would otherwise
        # silently corrupt a genuine OCR token whose text happens to collide with one of
        # them (confirmed: one real token in hipe2020_fr is literally "NA") into a float
        # NaN despite the dtype=str hint above -- dtype coercion happens AFTER NA
        # detection, so it can't prevent this. keep_default_na=False turns that off
        # entirely, and na_values restores it only for the two genuinely-numeric columns
        # that still need a blank cell to become NaN.
        keep_default_na=False, na_values={"sentence_ocr_mean": [""], "document_ocr_mean": [""], "dictionary_score": [""]})
    tokens_df["MISC"] = tokens_df["MISC"].fillna("_")
    print(f"{tokens_df.shape[0]} tokens across {tokens_df['document_id'].nunique()} documents")

    if args.limit is not None:
        keys = tokens_df[["document_id", "sentence_id"]].drop_duplicates().head(args.limit)
        tokens_df = tokens_df.merge(keys, on=["document_id", "sentence_id"], how="inner")

    print("=== Step 2: Build sentence text ===")
    sentences_df, token_spans_df = build_sentence_texts(tokens_df)
    print(f"{sentences_df.shape[0]} sentences across {sentences_df['document_id'].nunique()} documents")

    print("=== Step 3: Load historical-ner-baseline model ===")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_ID} (device: {'cuda' if device == 0 else 'cpu'})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    # No explicit truncation kwarg needed/supported by this pipeline version -- it
    # truncates to tokenizer.model_max_length automatically (see
    # TokenClassificationPipeline.preprocess), which is why chunk_size above is a word
    # count, not a hard subword guarantee: it keeps chunks well under that limit in the
    # common case, and this is the safety net for the rare chunk that still overflows.
    ner_pipeline = pipeline(
        "token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device,
    )

    print("=== Step 4: Run entity extraction ===")
    candidates_df = build_candidates(sentences_df, token_spans_df, ner_pipeline, args.threshold, args.batch_size, args.chunk_size)
    print(f"{candidates_df.shape[0]} candidates extracted")

    print("=== Step 5: Save NER features ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(out_path, index=False)
    print(f"Saved NER features to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
