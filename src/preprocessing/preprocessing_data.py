"""Download and parse HIPE-2022's official hipe2020 train/dev/test TSVs into one
token-level CSV (data/data_source/hipe2020_<language>.csv, default language fr), with OCR
dictionary-membership features attached. --language selects which language subset to load
(default: fr; impresso's OCR-quality bloom filter covers French and German, see
ocr_dictionary_check.BLOOM_LANGUAGES). To point this at a different HIPE-2022 v2.1
dataset entirely (e.g. letemps instead of hipe2020 -- same
https://github.com/hipe-eval/HIPE-2022-data train/dev/test TSV layout, just a different
folder name), override --train-url/--dev-url/--test-url/--out directly rather than adding
a --dataset flag here -- see Usage below.

Lives in data/data_source/, not data/data_baseline/ -- data_baseline/ is reserved for the
OUTPUT of baseline runs (ner_features.csv, logistic_regression.csv, mlp_baseline.csv,
...), not this raw source data every one of them reads via --load-data. Every downstream
script's --load-data flag (see e.g. gliner/extract_ner_features.py) loads this one file
whole (every split, not just "train" -- the old --train-data name was misleading) and
filters by its own `split` column for whatever it actually needs.

Output columns:
- the source file's own columns (TOKEN, NE-COARSE-LIT, ..., MISC)
- document_id prepended and sentence_id, token_id, split, dictionary_score, sentence_ocr_mean, document_ocr_mean appended

Split: this is now a full-data run, so every document from all three official files is
kept -- no document is held back or resampled. `split` is tagged directly by which file a
document came from (train-fr.tsv -> "train", dev-fr.tsv -> "val", test-fr.tsv -> "test"),
not a random per-document split like the previous single-file version of this script
(that version only had a train-fr.tsv to work with and had to carve 70/10/20 out of it
itself; HIPE-2022 ships hipe2020/fr as three separate official files, so re-splitting one
of them and discarding the corpus's real held-out dev/test documents was never necessary
here -- see test/ajmc/preprocessing_data_ajmc.py for the same reasoning applied to ajmc
earlier). Downstream per-candidate files (ner_features.csv, ocr_features.csv,
context_features.csv, ...) don't carry split themselves -- join back on document_id to
recover it.

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
    python src/preprocessing/preprocessing_data.py --language de  # German hipe2020 subset -> hipe2020_de.csv
    python src/preprocessing/preprocessing_data.py \
      --train-url https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/letemps/fr/HIPE-2022-v2.1-letemps-train-fr.tsv \
      --dev-url https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/letemps/fr/HIPE-2022-v2.1-letemps-dev-fr.tsv \
      --test-url https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/letemps/fr/HIPE-2022-v2.1-letemps-test-fr.tsv \
      --out data/data_source/letemps_fr.csv  # a different HIPE-2022 dataset entirely
    python src/preprocessing/preprocessing_data.py --limit-per-split 50 --out data/data_source/.smoke/smoke_data.csv  # quick smoke test
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.ocr_dictionary_check import BLOOM_MODEL_ID, bloom_filename_for, compute_dictionary_score, compute_ocr_means, get_bloomfilter

LANGUAGES = ("fr", "de")
DEFAULT_LANGUAGE = "fr"
DOC_ID_RE = re.compile(r"^#\s*hipe2022:document_id\s*=\s*(.+)$")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_source"
DEFAULT_OUT = DATA_DIR / f"hipe2020_{DEFAULT_LANGUAGE}.csv"


def split_urls_for(language: str) -> dict[str, str]:
    """The three official hipe2020/<language> file URLs, tagged by which split they
    become (train-fr.tsv -> "train", dev-fr.tsv -> "val", test-fr.tsv -> "test") --
    same layout for every LANGUAGES entry, just the language segment changes."""
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
    base = f"https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/hipe2020/{language}"
    return {
        "train": f"{base}/HIPE-2022-v2.1-hipe2020-train-{language}.tsv",
        "val": f"{base}/HIPE-2022-v2.1-hipe2020-dev-{language}.tsv",
        "test": f"{base}/HIPE-2022-v2.1-hipe2020-test-{language}.tsv",
    }


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


def load_split(url: str, split_name: str, limit: int | None) -> pd.DataFrame:
    """Load one of the three official files and tag every row with split_name directly
    (no per-document random assignment -- see module docstring)."""
    df = load_hipe_tokens(url)
    df["MISC"] = df["MISC"].fillna("_")
    df = assign_sentence_ids(df)
    df = assign_token_ids(df)

    if limit is not None:
        keys = df[["document_id", "sentence_id"]].drop_duplicates().head(limit)
        df = df.merge(keys, on=["document_id", "sentence_id"], how="inner")

    df["split"] = split_name
    print(f"  {split_name}: {df.shape[0]} tokens across {df['document_id'].nunique()} documents (from {url})")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=LANGUAGES, help="HIPE-2022 hipe2020 language subset to load (default: fr)")
    parser.add_argument(
        "--train-url", default=None,
        help="HIPE-2022 train TSV URL (default: derived from --language; override to point at a different dataset "
        "entirely, or pass \"\" (empty string) to skip this split -- e.g. sonar/de has no train file at all)",
    )
    parser.add_argument(
        "--dev-url", default=None,
        help="HIPE-2022 dev TSV URL (default: derived from --language; becomes split='val'; \"\" to skip)",
    )
    parser.add_argument(
        "--test-url", default=None,
        help="HIPE-2022 test TSV URL (default: derived from --language; \"\" to skip)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    parser.add_argument("--limit-per-split", type=int, default=None, help="Only keep the first N sentences of each split (smoke test)")
    args = parser.parse_args()

    # None (flag not given) -> the --language default; "" (explicitly passed) -> skip this
    # split entirely, not fall back to the default -- "" is falsy same as None in Python,
    # so this has to check `is None` rather than truthiness, or --train-url "" would
    # silently resolve to the hipe2020 default instead of actually skipping.
    default_urls = split_urls_for(args.language)
    raw_urls = {"train": args.train_url, "val": args.dev_url, "test": args.test_url}
    split_urls = {
        split: (url if url is not None else default_urls[split])
        for split, url in raw_urls.items()
        if url != ""
    }
    if not split_urls:
        raise ValueError("all three splits were skipped (--train-url/--dev-url/--test-url all \"\") -- nothing to load")
    out_path = Path(args.out)

    print(f"=== Step 1: Load official {'/'.join(split_urls)} TSVs ===")
    split_dfs = []
    for split_name, url in tqdm(split_urls.items(), desc="Loading splits", unit="file"):
        split_dfs.append(load_split(url, split_name, args.limit_per_split))
    tokens_df = pd.concat(split_dfs, ignore_index=True)
    print(f"{tokens_df.shape[0]} tokens across {tokens_df['document_id'].nunique()} documents (all splits combined)")
    print(f"Tokens per split:\n{tokens_df['split'].value_counts()}")
    print(f"Documents per split:\n{tokens_df.drop_duplicates('document_id')['split'].value_counts()}")

    print(f"=== Step 2: Load {args.language} OCR-quality bloom filter ===")
    bloom_filter = get_bloomfilter(BLOOM_MODEL_ID, bloom_filename_for(args.language))

    print("=== Step 3: Score token dictionary membership ===")
    tokens_df = compute_dictionary_score(tokens_df, bloom_filter)

    print("=== Step 4: Compute sentence/document OCR means ===")
    tokens_df = compute_ocr_means(tokens_df)

    print("=== Step 5: Save train data ===")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_df.to_csv(out_path, index=False)
    print(f"Saved train data to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
