"""Extract Phase 1 SS4.2 OCR span evidence features for NER candidates.

Input: the train data CSV produced by preprocessing_data.py -- which already carries
dictionary_score (True/False/None per token, from the impresso OCR-quality-assessment
bloom filter) and the precomputed sentence_ocr_mean / document_ocr_mean -- plus
ner_features.csv (see extract_ner_features.py) for the candidate spans.

Why ner_features.csv is a required input, not just train data: a candidate's span is a
range of train-data tokens (start_token_id..end_token_id), not known ahead of time from
the train data alone -- it comes out of the NER extraction step.

Pipeline:
    train data CSV (dictionary_score/sentence_ocr_mean/document_ocr_mean already computed
    by preprocessing_data.py) + ner_features.csv candidates -> for each candidate,
    aggregate its covering tokens' dictionary_score, skipping punctuation tokens entirely,
    and attach its sentence/document OCR means -> OCR features CSV.

Output: ocr_features.csv -- document_id, sentence_id, start_token_id, end_token_id,
span_text, ocr_correct, span_ocr_mean, span_low_conf_word_fraction,
span_first_word_ocr, span_last_word_ocr, sentence_ocr_mean, document_ocr_mean (one row per
NER candidate, in the same order as ner_features.csv). ocr_correct is True iff every
scoreable token in the span is a known word (span_low_conf_word_fraction == 0). Every
span_* aggregate is computed over only the scoreable (non-punctuation) tokens in its
range -- sum(known) / count(scoreable) -- so a comma or period riding along in a span's
token range neither inflates the mean nor gets mistaken for the boundary word. All
span_* feature columns are None for a candidate whose start_token_id/end_token_id didn't
resolve to any scoreable train-data token (e.g. a span that is itself pure punctuation,
or -- shouldn't happen in practice -- one that resolved to no train-data token at all;
see extract_ner_features.py).

Features (docs/phase1_manual.md SS4.2):
    span_ocr_mean                 -- mean known-word rate across the span's tokens
    span_low_conf_word_fraction   -- fraction of the span's tokens that are unknown
    span_first_word_ocr           -- known-word flag of the span's first token
    span_last_word_ocr            -- known-word flag of the span's last token
    sentence_ocr_mean             -- known-word rate across the whole sentence
    document_ocr_mean             -- known-word rate across the whole document

Usage:
    pip install -r requirements.txt
    python preprocessing_data.py
    python extract_ner_features.py
    python extract_ocr_features.py
    python extract_ocr_features.py --ner-features /tmp/smoke_ner.csv --out /tmp/smoke_ocr.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from preprocessing_data import DEFAULT_OUT as DEFAULT_TRAIN_DATA
from extract_ner_features import DEFAULT_OUT as DEFAULT_NER_FEATURES

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "hipe2020_train_fr_ocr_features.csv"


def compute_span_ocr_features(ner_df: pd.DataFrame, tokens_df: pd.DataFrame) -> pd.DataFrame:
    """For every NER candidate, aggregate the dictionary_score of the train-data tokens
    its span covers (start_token_id..end_token_id, inclusive) into the SS4.2 OCR span
    evidence features, and attach its precomputed sentence_ocr_mean / document_ocr_mean
    (from preprocessing_data.py). Punctuation tokens (dictionary_score is None) are
    excluded entirely from span aggregates -- sum(known) / count(scoreable tokens) --
    rather than counted as "known", so a stray comma/period riding along in a span's
    token range can't dilute span_ocr_mean or masquerade as the boundary word in
    span_first_word_ocr/span_last_word_ocr."""
    scoreable = tokens_df[tokens_df["dictionary_score"].notna()].copy()
    scoreable["dictionary_score"] = scoreable["dictionary_score"].astype(float)

    token_ocr_by_doc: dict[str, dict[int, float]] = {
        doc_id: dict(zip(g["token_id"], g["dictionary_score"]))
        for doc_id, g in scoreable.groupby("document_id")
    }

    sentence_ocr_mean = tokens_df.groupby(["document_id", "sentence_id"])["sentence_ocr_mean"].first().to_dict()
    document_ocr_mean = tokens_df.groupby("document_id")["document_ocr_mean"].first().to_dict()

    records = []
    for row in tqdm(ner_df.to_dict("records"), total=len(ner_df), desc="Computing span OCR features", unit="candidate"):
        doc_id = row["document_id"]
        start_tid, end_tid = row["start_token_id"], row["end_token_id"]

        span_values: list[float] = []
        if pd.notna(start_tid) and pd.notna(end_tid):
            token_ocr = token_ocr_by_doc.get(doc_id, {})
            span_values = [
                token_ocr[tid] for tid in range(int(start_tid), int(end_tid) + 1) if tid in token_ocr
            ]

        if span_values:
            span_ocr_mean = sum(span_values) / len(span_values)
            span_low_conf_word_fraction = sum(1 for v in span_values if v == 0) / len(span_values)
            span_first_word_ocr = span_values[0]
            span_last_word_ocr = span_values[-1]
            ocr_correct = span_low_conf_word_fraction == 0
        else:
            span_ocr_mean = span_first_word_ocr = span_last_word_ocr = None
            span_low_conf_word_fraction = None
            ocr_correct = None

        records.append(
            {
                "document_id": doc_id,
                "sentence_id": row["sentence_id"],
                "start_token_id": start_tid,
                "end_token_id": end_tid,
                "span_text": row["entity_text"],
                "ocr_correct": ocr_correct,
                "span_ocr_mean": span_ocr_mean,
                "span_low_conf_word_fraction": span_low_conf_word_fraction,
                "span_first_word_ocr": span_first_word_ocr,
                "span_last_word_ocr": span_last_word_ocr,
                "sentence_ocr_mean": sentence_ocr_mean.get((doc_id, row["sentence_id"])),
                "document_ocr_mean": document_ocr_mean.get(doc_id),
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--train-data",
        default=str(DEFAULT_TRAIN_DATA),
        help="Train data CSV produced by preprocessing_data.py (with dictionary_score/sentence_ocr_mean/document_ocr_mean)",
    )
    parser.add_argument(
        "--ner-features",
        default=str(DEFAULT_NER_FEATURES),
        help="NER features CSV produced by extract_ner_features.py",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="OCR features output CSV path")
    args = parser.parse_args()

    print("=== Step 1: Load train data and NER features ===")
    tokens_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    tokens_df["MISC"] = tokens_df["MISC"].fillna("_")
    ner_df = pd.read_csv(args.ner_features)
    print(f"{tokens_df.shape[0]} tokens, {ner_df.shape[0]} candidates")

    print("=== Step 2: Compute span OCR features ===")
    ocr_df = compute_span_ocr_features(ner_df, tokens_df)

    print("=== Step 3: Save OCR features ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_df.to_csv(out_path, index=False)
    print(f"Saved OCR features to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
