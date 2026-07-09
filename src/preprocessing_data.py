"""Download and parse a HIPE-2022 TSV into token-level train data, with OCR
dictionary-membership features attached.

Output columns are the source file's own columns (TOKEN, NE-COARSE-LIT, ..., MISC)
unchanged and in their original order, with document_id prepended and sentence_id,
token_id, split, dictionary_score, sentence_ocr_mean, document_ocr_mean appended.

split assigns every document to one of Phase 1's four data splits (docs/phase1_manual.md
SS6.1): expert_train (50%), gate_train (20%), calibration (10%), test (20%). Splitting is
done per document (every token/sentence of a document gets the same split), not per
token/candidate, so no document's context leaks across splits; the assignment is a
deterministic shuffle keyed on --split-seed. Downstream per-candidate files
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

dictionary_score is True/False/None per token, from the impresso
OCR-quality-assessment-unigram French bloom filter (a list of known French word forms
built from Wikipedia + lexicons): True = known French word, False = unknown (likely OCR
error, or a rare/proper name -- the filter can't tell the two apart), None = punctuation
(not scoreable). "Known" is a proxy for correct OCR, not a verified fact -- there is no
continuous OCR confidence anywhere in the HIPE data, so this 0/1 dictionary-membership
signal stands in for it throughout.

sentence_ocr_mean / document_ocr_mean are the mean dictionary_score (as 0/1) across all
scoreable (non-punctuation) tokens in the row's sentence / document respectively -- the
same value repeated on every row of that sentence / document. Punctuation tokens
(dictionary_score is None) are excluded from both, so a comma or period can't dilute
them.

Usage:
    pip install -r requirements.txt
    python preprocessing_data.py
    python preprocessing_data.py --limit 50 --out /tmp/smoke_train_data.csv  # quick smoke test
    python preprocessing_data.py --split-seed 7  # different deterministic document split
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from pybloomfilter import BloomFilter
from tqdm import tqdm

HIPE_URL = (
    "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/"
    "data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv"
)
DOC_ID_RE = re.compile(r"^#\s*hipe2022:document_id\s*=\s*(.+)$")

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "hipe2020_train_fr_train_data.csv"

BLOOM_MODEL_ID = "impresso-project/OCR-quality-assessment-unigram"
BLOOM_FILENAME = "ocrqa-wp_v1.0.6-fr.bloom"

# Phase 1 data split proportions (docs/phase1_manual.md SS6.1).
SPLIT_PROPORTIONS = {"expert_train": 0.5, "gate_train": 0.2, "calibration": 0.1, "test": 0.2}
DEFAULT_SPLIT_SEED = 42

# Normalization table as documented on the bloom filter's model card
# (NFKC + lowercase + digits->'0' + strip punctuation). Ported from
# notebook/hipe_ocr_ner_extraction.ipynb.
QUOTES_PUNCT = "„•<>!\"#%&'’"
ASCII_PUNCT = "()*,./:;?"
BRACKETS_SPECIAL = "[]\\~_{}"
UNICODE_PUNCT = "\xa1\xab\xb7\xbb\xbf"
DASH_CARET = "—^`"
SPECIAL_SYMBOLS = "\xa6\xa7\xa3="
HYPHEN = "-"
DIGITS = "0123456789"

NORMALIZATION_TABLE = str.maketrans(
    {char: " " for char in (QUOTES_PUNCT + ASCII_PUNCT + BRACKETS_SPECIAL + UNICODE_PUNCT + DASH_CARET + SPECIAL_SYMBOLS + HYPHEN)}
    | {char: "0" for char in DIGITS}
)


def normalize_text(s: str, unicode_normalize: Optional[str] = "NFKC") -> str:
    if unicode_normalize:
        s = unicodedata.normalize(unicode_normalize, s).lower()
    return s.translate(NORMALIZATION_TABLE)


def get_bloomfilter(model_id: str, filename: str) -> BloomFilter:
    return BloomFilter.open(hf_hub_download(repo_id=model_id, filename=filename))


def word_is_known(token: str, bloom_filter: BloomFilter) -> Optional[bool]:
    """True: token is a known French word (bloom filter hit).
    False: token normalized to a non-empty string but was not found (likely OCR error,
    or a rare/proper name).
    None: token normalized to nothing (pure punctuation/whitespace token) -- not
    applicable."""
    normalized = normalize_text(token).strip()
    if not normalized:
        return None
    sub_tokens = normalized.split()
    return all(t in bloom_filter for t in sub_tokens)


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


def compute_dictionary_score(df: pd.DataFrame, bloom_filter: BloomFilter) -> pd.DataFrame:
    """Append dictionary_score (True/False/None) to the token-level table: True = known
    French word, False = unknown (likely OCR error, or a rare/proper name -- the bloom
    filter can't tell the two apart), None = punctuation/not scoreable."""
    out = df.copy()
    tqdm.pandas(desc="Scoring token dictionary membership", unit="token")
    out["dictionary_score"] = out["TOKEN"].progress_apply(lambda t: word_is_known(t, bloom_filter))
    return out


def compute_ocr_means(df: pd.DataFrame) -> pd.DataFrame:
    """Append sentence_ocr_mean / document_ocr_mean to the token-level table: the mean
    dictionary_score (as 0/1) across all scoreable (non-punctuation) tokens in the row's
    sentence / document. Punctuation tokens (dictionary_score is None) are excluded
    entirely from both means -- sum(known) / count(scoreable) -- rather than counted as
    "known", so they can't dilute the result."""
    scoreable = df[df["dictionary_score"].notna()].copy()
    scoreable["dictionary_score"] = scoreable["dictionary_score"].astype(float)

    sentence_mean = (
        scoreable.groupby(["document_id", "sentence_id"])["dictionary_score"]
        .mean()
        .rename("sentence_ocr_mean")
        .reset_index()
    )
    document_mean = (
        scoreable.groupby("document_id")["dictionary_score"].mean().rename("document_ocr_mean").reset_index()
    )

    out = df.merge(sentence_mean, on=["document_id", "sentence_id"], how="left")
    out = out.merge(document_mean, on="document_id", how="left")
    return out


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
