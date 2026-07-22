"""Extract Phase 1 SS4.3 context evidence features for NER candidates.

Input: the train data CSV produced by preprocessing_data.py -- which already carries
dictionary_score (True/False/None per token) and the precomputed sentence_ocr_mean --
plus deduplicate_ner_features.csv (see gliner/deduplicate_ner_features.py) for the
candidate spans, already resolved to pairwise non-overlapping per sentence.

Why a candidates CSV is a required input, not just train data: "context" is defined
relative to a candidate's span (start_token_id..end_token_id), which is not known ahead
of time from the train data alone -- it comes out of the NER extraction (and
deduplication) steps. Same reasoning as extract_ocr_features.py.

Pipeline: for every candidate, take up to CONTEXT_WINDOW train-data tokens immediately
before its span and up to CONTEXT_WINDOW immediately after, both clipped to the
candidate's own sentence (context never crosses a sentence boundary) -> aggregate their
dictionary_score, skipping punctuation tokens entirely, same convention as
extract_ocr_features.py's span aggregates.

Output: context_features.csv -- document_id, sentence_id, start_token_id, end_token_id,
left_context_ocr_mean_10, right_context_ocr_mean_10, context_ocr_min_10,
context_low_conf_word_fraction_10, sentence_ocr_mean, sentence_length,
context_window_length (one row per NER candidate, in the same order as --ner-features).
context_window_length is how many tokens were actually available on both sides combined
(<= 2 * CONTEXT_WINDOW; smaller near a sentence's start/end). All context_*/left_*/
right_* columns are None when the candidate's span didn't resolve to any train-data token
(shouldn't happen in practice; see gliner/extract_ner_features.py) or when no context
token on that side was scoreable.

Features (docs/phase1_manual_features.md SS4.3):
    left_context_ocr_mean_10            -- known-word rate over up to 10 tokens before the span
    right_context_ocr_mean_10           -- known-word rate over up to 10 tokens after the span
    context_ocr_min_10                  -- min known-word flag across both sides combined
    context_low_conf_word_fraction_10   -- fraction of unknown tokens across both sides combined
    sentence_ocr_mean                   -- known-word rate across the whole sentence
    sentence_length                     -- number of tokens in the sentence
    context_window_length               -- number of context tokens actually available

Usage:
    pip install -r requirements.txt
    python src/preprocessing/preprocessing_data.py
    python src/ner/gliner/extract_ner_features.py
    python src/ner/gliner/deduplicate_ner_features.py
    python src/phase1/feature_extraction/extract_context_features.py
    python src/phase1/feature_extraction/extract_context_features.py --ner-features /tmp/smoke_ner.csv --out /tmp/smoke_context.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA
from ner.gliner.deduplicate_ner_features import DEFAULT_OUT as DEFAULT_NER_FEATURES

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_baseline"
DEFAULT_OUT = DATA_DIR / "context_features.csv"

CONTEXT_WINDOW = 10  # tokens each side, per docs/phase1_manual_features.md SS4.3


def compute_context_features(ner_df: pd.DataFrame, tokens_df: pd.DataFrame) -> pd.DataFrame:
    """For every NER candidate, look at up to CONTEXT_WINDOW train-data tokens on each
    side of its span (start_token_id..end_token_id), clipped to the candidate's own
    sentence, and aggregate their dictionary_score into the SS4.3 context evidence
    features. Punctuation tokens (dictionary_score is None) are excluded entirely from
    the OCR aggregates -- sum(known) / count(scoreable) -- but still count toward
    context_window_length, which measures how much context was structurally available."""
    token_score_by_doc: dict[str, dict[int, float]] = {
        doc_id: dict(zip(g["token_id"], g["dictionary_score"].astype(float)))
        for doc_id, g in tokens_df.groupby("document_id")
    }
    sentence_bounds = (
        tokens_df.groupby(["document_id", "sentence_id"])["token_id"]
        .agg(min_token_id="min", max_token_id="max", sentence_length="size")
        .to_dict("index")
    )
    sentence_ocr_mean = tokens_df.groupby(["document_id", "sentence_id"])["sentence_ocr_mean"].first().to_dict()

    def scoreable_values(doc_id: str, token_ids: list[int]) -> list[float]:
        token_score = token_score_by_doc.get(doc_id, {})
        return [token_score[tid] for tid in token_ids if tid in token_score and pd.notna(token_score[tid])]

    records = []
    for row in tqdm(ner_df.to_dict("records"), total=len(ner_df), desc="Computing context features", unit="candidate"):
        doc_id = row["document_id"]
        sent_id = row["sentence_id"]
        start_tid, end_tid = row["start_token_id"], row["end_token_id"]
        bounds = sentence_bounds.get((doc_id, sent_id))

        if bounds is not None and pd.notna(start_tid) and pd.notna(end_tid):
            start_tid, end_tid = int(start_tid), int(end_tid)
            left_ids = list(range(max(bounds["min_token_id"], start_tid - CONTEXT_WINDOW), start_tid))
            right_ids = list(range(end_tid + 1, min(bounds["max_token_id"], end_tid + CONTEXT_WINDOW) + 1))

            left_values = scoreable_values(doc_id, left_ids)
            right_values = scoreable_values(doc_id, right_ids)
            combined_values = left_values + right_values

            left_context_ocr_mean_10 = sum(left_values) / len(left_values) if left_values else None
            right_context_ocr_mean_10 = sum(right_values) / len(right_values) if right_values else None
            context_ocr_min_10 = min(combined_values) if combined_values else None
            context_low_conf_word_fraction_10 = (
                sum(1 for v in combined_values if v == 0) / len(combined_values) if combined_values else None
            )
            context_window_length = len(left_ids) + len(right_ids)
        else:
            left_context_ocr_mean_10 = right_context_ocr_mean_10 = None
            context_ocr_min_10 = context_low_conf_word_fraction_10 = None
            context_window_length = None

        records.append(
            {
                "document_id": doc_id,
                "sentence_id": sent_id,
                "start_token_id": row["start_token_id"],
                "end_token_id": row["end_token_id"],
                "left_context_ocr_mean_10": left_context_ocr_mean_10,
                "right_context_ocr_mean_10": right_context_ocr_mean_10,
                "context_ocr_min_10": context_ocr_min_10,
                "context_low_conf_word_fraction_10": context_low_conf_word_fraction_10,
                "sentence_ocr_mean": sentence_ocr_mean.get((doc_id, sent_id)),
                "sentence_length": bounds["sentence_length"] if bounds is not None else None,
                "context_window_length": context_window_length,
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV produced by preprocessing_data.py (every split, filtered internally as needed)")
    parser.add_argument(
        "--ner-features",
        default=str(DEFAULT_NER_FEATURES),
        help="Deduplicated NER candidates CSV produced by gliner/deduplicate_ner_features.py",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Context features output CSV path")
    args = parser.parse_args()

    print("=== Step 1: Load train data and NER features ===")
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
    ner_df = pd.read_csv(args.ner_features)
    print(f"{tokens_df.shape[0]} tokens, {ner_df.shape[0]} candidates")

    print("=== Step 2: Compute context features ===")
    context_df = compute_context_features(ner_df, tokens_df)

    print("=== Step 3: Save context features ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    context_df.to_csv(out_path, index=False)
    print(f"Saved context features to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
