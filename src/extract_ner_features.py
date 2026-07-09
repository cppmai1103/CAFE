"""Extract Phase 1 SS4.1 NER evidence features from HIPE-2022 (fr) train data.

Pipeline:
    train data CSV (see preprocessing_data.py) -> build sentence text -> GLiNER2 entity
    extraction -> per-candidate NER evidence features -> CSV.

Input: the token-level train data CSV produced by preprocessing_data.py -- the HIPE
source columns unchanged (TOKEN, NE-COARSE-LIT, ..., MISC), plus document_id,
sentence_id, and token_id.

Candidates are (span, predicted_type) pairs: GLiNER2 scores each entity type as an
independent sigmoid classifier over spans (no joint softmax across types), so the same
span can appear as more than one candidate if several types clear the threshold.

Output: ner_features.csv -- document_id, sentence_id, start_token_id, end_token_id,
entity_text, predicted_entity_type, ner_score, span_length_tokens,
span_length_characters, sentence_chunked (one row per candidate). No character offsets
are stored -- start_token_id/end_token_id are the token_id (from the train data CSV) of
the first/last train-data token overlapping the candidate span, so join on document_id +
start_token_id..end_token_id to recover the exact source rows and, if needed, their
character positions within the sentence; None if no token overlaps (shouldn't happen in
practice). entity_text is the candidate span's surface text, kept here for readability
even though it's also recoverable via that join. sentence_chunked is True if the

Features (docs/phase1_manual.md SS4.1):
    ner_score               -- the candidate's own confidence under its predicted type
    predicted_entity_type
    span_length_tokens
    span_length_characters

A few reconstructed "sentences" run far longer than the encoder's window --> `chunk_long_text` /
`extract_candidates_long` instead split any sentence longer than `chunk_size` words into
consecutive, non-overlapping `chunk_size`-word sub-sentences and run each one separately;
a sentence at or under `chunk_size` words is left untouched (a single "chunk" equal to
the whole sentence). Spans are remapped back to sentence-level character offsets.

Usage:
    pip install -r requirements.txt
    python preprocessing_data.py
    python extract_ner_features.py
    python extract_ner_features.py --limit 50 --out /tmp/smoke_test.csv  # quick smoke test
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from gliner2 import GLiNER2
from tqdm import tqdm

from preprocessing_data import DEFAULT_OUT as DEFAULT_TRAIN_DATA

GLINER_MODEL_ID = "fastino/gliner2-multi-v1"
LABELS = ["PERS", "LOC", "ORG", "TIME", "PROD"]

# Same word-boundary regex GLiNER2's own chunking utility uses, so chunk boundaries line
# up with how the processor itself tokenizes words.
_WORD_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_char: int


def chunk_long_text(text: str, chunk_size: int) -> list[TextChunk]:
    """Split text longer than chunk_size words into consecutive, non-overlapping
    chunk_size-word sub-sentences. Text at or under chunk_size words is left as-is,
    returned as a single chunk equal to the whole input."""
    tokens = [(m.start(), m.end()) for m in _WORD_PATTERN.finditer(text)]
    if len(tokens) <= chunk_size:
        return [TextChunk(text=text, start_char=0)]

    chunks = []
    start_word = 0
    while start_word < len(tokens):
        end_word = min(start_word + chunk_size, len(tokens))
        start_char = tokens[start_word][0]
        end_char = tokens[end_word - 1][1]
        chunks.append(TextChunk(text=text[start_char:end_char], start_char=start_char))
        start_word = end_word
    return chunks


DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "hipe2020_train_fr_gliner2_ner_features.csv"


def build_sentence_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group the token-level table (with sentence_id/token_id already assigned) into
    sentence-level text for model input, using MISC's NoSpaceAfter flag for
    detokenization. Mirrors gliner_ner.ipynb / evaluate_ner_metrics.ipynb.

    Also returns each token's character span within its own sentence text
    (document_id, sentence_id, token_id, start, end), so a candidate's start/end char
    offsets in ner_features.csv can be mapped back to the covering train-data rows via
    token_id, without redoing this detokenization."""
    sentences = []
    token_spans = []
    doc_id = None
    sent_id = None
    pieces: list[str] = []
    cur_len = 0
    no_space_before_next = False

    def flush():
        if pieces:
            sentences.append({"document_id": doc_id, "sentence_id": sent_id, "sentence_text": "".join(pieces)})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building sentence text", unit="token"):
        if row["document_id"] != doc_id or row["sentence_id"] != sent_id:
            flush()
            doc_id = row["document_id"]
            sent_id = row["sentence_id"]
            pieces = []
            cur_len = 0
            no_space_before_next = False

        if pieces and not no_space_before_next:
            pieces.append(" ")
            cur_len += 1
        token_text = str(row["TOKEN"])
        token_start = cur_len
        pieces.append(token_text)
        cur_len += len(token_text)
        token_spans.append(
            {
                "document_id": doc_id,
                "sentence_id": sent_id,
                "token_id": row["token_id"],
                "start": token_start,
                "end": cur_len,
            }
        )
        no_space_before_next = "NoSpaceAfter" in row["MISC"]

    flush()
    return pd.DataFrame(sentences), pd.DataFrame(token_spans)


def extract_candidates_long(
    extractor: GLiNER2,
    texts: list[str],
    labels: list[str],
    threshold: float,
    batch_size: int,
    chunk_size: int,
) -> list[dict]:
    """batch_extract_entities over arbitrarily long texts, via word chunking (see
    chunk_long_text). Returns one {"entities": {...}, "chunked": bool} dict per input
    text, with spans remapped to that text's own character offsets. "chunked" is True
    if the text was longer than chunk_size words and had to be split."""
    chunks_per_text = [chunk_long_text(text, chunk_size) for text in texts]

    all_chunk_texts = [chunk.text for chunks in chunks_per_text for chunk in chunks]
    chunk_results = []
    for i in tqdm(
        range(0, len(all_chunk_texts), batch_size), desc="GLiNER2 inference", unit="batch"
    ):
        batch = all_chunk_texts[i : i + batch_size]
        chunk_results.extend(
            extractor.batch_extract_entities(
                batch,
                labels,
                batch_size=len(batch),
                threshold=threshold,
                include_confidence=True,
                include_spans=True,
            )
        )

    results = []
    cursor = 0
    for chunks in tqdm(chunks_per_text, desc="Merging chunk results", unit="sentence"):
        text_results = chunk_results[cursor : cursor + len(chunks)]
        cursor += len(chunks)

        merged: dict[str, list[dict]] = {label: [] for label in labels}
        for chunk, chunk_result in zip(chunks, text_results):
            for label in labels:
                for span in chunk_result.get("entities", {}).get(label, []):
                    merged[label].append(
                        {
                            "text": span["text"],
                            "confidence": span["confidence"],
                            "start": span["start"] + chunk.start_char,
                            "end": span["end"] + chunk.start_char,
                        }
                    )

        results.append({"entities": merged, "chunked": len(chunks) > 1})

    return results


def build_candidates(
    sentences_df: pd.DataFrame,
    token_spans_df: pd.DataFrame,
    extractor: GLiNER2,
    threshold: float,
    batch_size: int,
    chunk_size: int,
) -> pd.DataFrame:
    texts = sentences_df["sentence_text"].tolist()
    official = extract_candidates_long(extractor, texts, LABELS, threshold, batch_size, chunk_size)

    token_spans_by_sentence: dict[tuple, list[tuple[int, int, int]]] = {
        key: list(zip(g["token_id"], g["start"], g["end"]))
        for key, g in token_spans_df.groupby(["document_id", "sentence_id"])
    }

    def token_id_range(document_id, sentence_id, start: int, end: int) -> tuple[int | None, int | None]:
        """token_id of the first/last train-data token overlapping this char span."""
        spans = token_spans_by_sentence.get((document_id, sentence_id), [])
        overlapping = [tid for tid, t_start, t_end in spans if t_end > start and t_start < end]
        if not overlapping:
            return None, None
        return min(overlapping), max(overlapping)

    records = []
    for srow, off in tqdm(
        zip(sentences_df.to_dict("records"), official), total=len(official), desc="Building candidates", unit="sentence"
    ):
        for label in LABELS:
            for span in off.get("entities", {}).get(label, []):
                span_text = span["text"]
                start_token_id, end_token_id = token_id_range(
                    srow["document_id"], srow["sentence_id"], span["start"], span["end"]
                )
                records.append(
                    {
                        "document_id": srow["document_id"],
                        "sentence_id": srow["sentence_id"],
                        "start_token_id": start_token_id,
                        "end_token_id": end_token_id,
                        "entity_text": span_text,
                        "predicted_entity_type": label,
                        "ner_score": span["confidence"],
                        "span_length_tokens": len(span_text.split()),
                        "span_length_characters": len(span_text),
                        "sentence_chunked": off["chunked"],
                    }
                )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--train-data",
        default=str(DEFAULT_TRAIN_DATA),
        help="Train data CSV produced by preprocessing_data.py",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="NER features output CSV path")
    parser.add_argument("--threshold", type=float, default=0.0, help="Official entity extraction threshold")
    parser.add_argument("--batch-size", type=int, default=16, help="GLiNER2 batch size")
    parser.add_argument(
        "--chunk-size", type=int, default=384, help="Word-chunk size for splitting long sentences"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sentences (smoke test)")
    args = parser.parse_args()

    print("=== Step 1: Load train data ===")
    print(f"Loading train data from {args.train_data}")
    tokens_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    tokens_df["MISC"] = tokens_df["MISC"].fillna("_")
    print(f"{tokens_df.shape[0]} tokens across {tokens_df['document_id'].nunique()} documents")

    if args.limit is not None:
        keys = tokens_df[["document_id", "sentence_id"]].drop_duplicates().head(args.limit)
        tokens_df = tokens_df.merge(keys, on=["document_id", "sentence_id"], how="inner")

    print("=== Step 2: Build sentence text ===")
    sentences_df, token_spans_df = build_sentence_texts(tokens_df)
    print(f"{sentences_df.shape[0]} sentences across {sentences_df['document_id'].nunique()} documents")

    print("=== Step 3: Load GLiNER2 model ===")
    print(f"Loading {GLINER_MODEL_ID}")
    extractor = GLiNER2.from_pretrained(GLINER_MODEL_ID)

    print("=== Step 4: Run entity extraction ===")
    candidates_df = build_candidates(
        sentences_df, token_spans_df, extractor, args.threshold, args.batch_size, args.chunk_size
    )
    print(f"{candidates_df.shape[0]} candidates extracted")

    print("=== Step 5: Save NER features ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(out_path, index=False)
    print(f"Saved NER features to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
