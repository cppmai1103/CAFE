"""Download and parse a HIPE-2022 TSV into token-level train data, with OCR
dictionary-membership features attached.

Output columns: 
- the source file's own columns (TOKEN, NE-COARSE-LIT, ..., MISC)
- document_id prepended and sentence_id, token_id, split, dictionary_score, sentence_ocr_mean, document_ocr_mean appended

Split (docs/phase1_manual.md SS6.1): 
- expert_train (50%)
- gate_train (20%)
- calibration (10%)
- test (20%)
Splitting is done:
- per document (every token/sentence of a document gets the same split), not per token/candidate
--> so no document's context leaks across splits

the assignment is a deterministic shuffle keyed on --split-seed. Downstream per-candidate files
(ner_features.csv, ocr_features.csv, context_features.csv, ...) don't carry split
themselves -- join back on document_id to recover it.

sentence_id is 0-indexed and resets at each new document; it's derived from MISC's
EndOfSentence flag, so the token immediately after a sentence-ending token starts the
next sentence_id.

token_id is 0-indexed and also resets at each new document -- it's just the row's
position among all tokens in its document, independent of sentence boundaries. Combined
with document_id, it uniquely identifies a row here, so extract_ner_features.py can
stamp each candidate in ner_features.csv with the start_token_id/end_token_id of the
train-data rows its span covers, without needing a separate join on character offsets.

dictionary_score / sentence_ocr_mean / document_ocr_mean are computed by
ocr_dictionary_check.py (see that module's docstring) -- this file just calls into it.

Usage:
    pip install -r requirements.txt
    python src/preprocessing/preprocessing_data.py
    python src/preprocessing/preprocessing_data.py --limit 50 --out /tmp/smoke_train_data.csv  # quick smoke test
    python src/preprocessing/preprocessing_data.py --split-seed 7  # different deterministic document split
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.ocr_dictionary_check import BLOOM_FILENAME, BLOOM_MODEL_ID, compute_dictionary_score, compute_ocr_means, get_bloomfilter

HIPE_URL = (
    "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/"
    "data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv"
)
DOC_ID_RE = re.compile(r"^#\s*hipe2022:document_id\s*=\s*(.+)$")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "hipe2020_train_fr_train_data.csv"

# Phase 1 data split proportions (docs/phase1_manual.md SS6.1).
SPLIT_PROPORTIONS = {"expert_train": 0.5, "gate_train": 0.2, "calibration": 0.1, "test": 0.2}
DEFAULT_SPLIT_SEED = 42


def load_hipe_tokens(url: str) -> pd.DataFrame:
    """Parse a HIPE-2022 TSV into a flat DataFrame with the source file's own columns
    (TOKEN, NE-COARSE-LIT, ..., MISC) unchanged and in their original order, plus a
    document_id column parsed from the '# hipe2022:document_id = ...' comment lines."""
    text = requests.get(url, timeout=60).text
    lines = text.splitlines()

    header = lines[0].split("\t")

    rows = []
    current_doc_id = None
    for line in lines[1:]:
        if not line.strip():
            continue  # blank line = segment boundary within the same document
        if line.startswith("#"):
            m = DOC_ID_RE.match(line.strip())
            if m:
                current_doc_id = m.group(1).strip()
            continue
        row = dict(zip(header, line.split("\t")))
        row["document_id"] = current_doc_id
        rows.append(row)

    return pd.DataFrame(rows, columns=["document_id", *header])


def assign_sentence_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Append a sentence_id column (0-indexed, resets at each new document) to the
    token-level table, using MISC's EndOfSentence flag as the sentence boundary."""
    sentence_ids = []
    doc_id = None
    sent_idx = 0
    for _, row in df.iterrows():
        if row["document_id"] != doc_id:
            doc_id = row["document_id"]
            sent_idx = 0
        sentence_ids.append(sent_idx)
        if "EndOfSentence" in row["MISC"]:
            sent_idx += 1

    out = df.copy()
    out["sentence_id"] = sentence_ids
    return out


def assign_token_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Append a token_id column (0-indexed, resets at each new document) to the
    token-level table -- the row's position among all tokens in its document,
    independent of sentence boundaries."""
    out = df.copy()
    out["token_id"] = df.groupby("document_id").cumcount()
    return out


def assign_splits(document_ids, seed: int = DEFAULT_SPLIT_SEED) -> dict:
    """Assign each document to one of Phase 1's four splits (docs/phase1_manual.md SS6.1:
    expert_train 50%, gate_train 20%, calibration 10%, test 20%). Splitting is done by
    document, not by individual token/candidate, so that no document's sentences leak
    across splits. Deterministic given `seed`."""
    unique_ids = sorted(set(document_ids))
    shuffled = np.random.RandomState(seed).permutation(unique_ids)

    n = len(shuffled)
    counts = {name: round(n * frac) for name, frac in SPLIT_PROPORTIONS.items()}
    counts["expert_train"] += n - sum(counts.values())  # absorb any rounding remainder

    split_of = {}
    cursor = 0
    for name in ["expert_train", "gate_train", "calibration", "test"]:
        for doc_id in shuffled[cursor : cursor + counts[name]]:
            split_of[doc_id] = name
        cursor += counts[name]
    return split_of


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hipe-url", default=HIPE_URL, help="HIPE-2022 TSV URL (default: hipe2020 train-fr)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Train data output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Only keep the first N sentences (smoke test)")
    parser.add_argument(
        "--split-seed", type=int, default=DEFAULT_SPLIT_SEED, help="Seed for the deterministic Phase 1 document split"
    )
    args = parser.parse_args()

    print("=== Step 1: Load HIPE tokens ===")
    print(f"Loading HIPE tokens from {args.hipe_url}")
    tokens_df = load_hipe_tokens(args.hipe_url)
    tokens_df["MISC"] = tokens_df["MISC"].fillna("_")
    tokens_df = assign_sentence_ids(tokens_df)
    tokens_df = assign_token_ids(tokens_df)
    print(f"{tokens_df.shape[0]} tokens across {tokens_df['document_id'].nunique()} documents")

    if args.limit is not None:
        keys = tokens_df[["document_id", "sentence_id"]].drop_duplicates().head(args.limit)
        tokens_df = tokens_df.merge(keys, on=["document_id", "sentence_id"], how="inner")

    print("=== Step 2: Assign Phase 1 data splits (per document) ===")
    split_of = assign_splits(tokens_df["document_id"].unique(), seed=args.split_seed)
    tokens_df["split"] = tokens_df["document_id"].map(split_of)
    doc_split_counts = tokens_df.drop_duplicates("document_id")["split"].value_counts()
    print(f"Documents per split:\n{doc_split_counts}")
    print(f"Tokens per split:\n{tokens_df['split'].value_counts()}")

    print("=== Step 3: Load French OCR-quality bloom filter ===")
    bloom_filter = get_bloomfilter(BLOOM_MODEL_ID, BLOOM_FILENAME)

    print("=== Step 4: Score token dictionary membership ===")
    tokens_df = compute_dictionary_score(tokens_df, bloom_filter)

    print("=== Step 5: Compute sentence/document OCR means ===")
    tokens_df = compute_ocr_means(tokens_df)

    print("=== Step 6: Save train data ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_df.to_csv(out_path, index=False)
    print(f"Saved train data to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
