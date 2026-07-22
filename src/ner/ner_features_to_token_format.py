"""Convert deduplicate_ner_features.csv (span-level NER candidates, already resolved
to pairwise non-overlapping per sentence by deduplicate_ner_features.py) into the same
token-level format as the train data CSV, so gold and predicted labels sit side by side,
one row per token (including tokens that are not part of any entity).

Model-agnostic: --ner-features accepts any model's ner_features.csv (GLiNER2's own, or
historical-ner-baseline's, see src/ner/gliner/ and src/ner/historical_ner/) as long as it
has the shared document_id/sentence_id/start_token_id/end_token_id/predicted_entity_type/
ner_score schema -- the output "NER" column just holds whichever model's predictions were
passed in, so the same script/column serves every NER source rather than one per model.

Since the input is already deduplicated (see deduplicate_ner_features.py --
resolve_overlaps_for_sentence), at most one candidate ever covers any given token, so
each surviving candidate's span is simply exploded straight to its covering tokens.

Candidates below --threshold are
dropped first, so a token with no sufficiently-confident candidate over it gets "O" in the
NER column, instead of always being tagged with the model's best (possibly
near-zero-confidence) guess -- this is still the real filtering step even on deduplicated
input, since deduplication only resolves overlaps, it doesn't drop low-confidence lone
survivors (see deduplicate_ner_features.py's own docstring).

Output columns: doc_id, token_id, TOKEN, NE-COARSE-LIT, NE-COARSE-METO, NE-FINE-LIT,
NE-FINE-METO, NE-FINE-COMP, NE-NESTED, NEL-LIT, NEL-METO, MISC, NER, dictionary_score.

NER is an IOB2 tag ("B-PERS", "I-LOC", ..., "O") built from the winning candidate's
type and the token's position within that candidate's span (first covered token -> B-,
later ones -> I-), so it has the same shape as the gold NE-COARSE-LIT column.

Usage:
    python src/ner/ner_features_to_token_format.py
    python src/ner/ner_features_to_token_format.py --threshold 0.5
    python src/ner/ner_features_to_token_format.py --limit 5 --out data/data_baseline/.smoke/smoke_token_format.csv  # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# When run directly (python src/ner/ner_features_to_token_format.py), Python only
# auto-adds this file's own directory (src/ner/) to sys.path -- not src/ -- so the
# package-qualified "ner.gliner...." import below needs src/ added explicitly first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ner.gliner.deduplicate_ner_features import DEFAULT_OUT as DEFAULT_NER_FEATURES

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_LOAD_DATA = Path(__file__).parent.parent.parent / "data" / "data_source" / "hipe2020_fr.csv"
DEFAULT_THRESHOLD = 0.5


def default_out_path(threshold: float) -> Path:
    """Bakes --threshold into the default output filename (e.g. threshold=0.5 ->
    ..._threshold0.5.csv), so runs at different thresholds don't silently overwrite each
    other. Only used when --out isn't explicitly given (see main())."""
    return DATA_DIR / f"hipe2020_train_fr_ner_token_format_threshold{threshold}.csv"


OUTPUT_COLUMNS = [
    "doc_id",
    "token_id",
    "TOKEN",
    "NE-COARSE-LIT",
    "NE-COARSE-METO",
    "NE-FINE-LIT",
    "NE-FINE-METO",
    "NE-FINE-COMP",
    "NE-NESTED",
    "NEL-LIT",
    "NEL-METO",
    "MISC",
    "NER",
    "dictionary_score",
]


def explode_candidates_to_tokens(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (document_id, token_id, candidate) -- every token a surviving
    candidate's span covers, tagged with whether that token is the span's first
    (is_start), so the winning candidate per token can later be turned into a B-/I- tag."""
    rows = []
    for row in tqdm(
        candidates_df.itertuples(index=False),
        total=len(candidates_df),
        desc="Expanding candidate spans to tokens",
        unit="candidate",
    ):
        for token_id in range(int(row.start_token_id), int(row.end_token_id) + 1):
            rows.append(
                {
                    "document_id": row.document_id,
                    "token_id": token_id,
                    "predicted_entity_type": row.predicted_entity_type,
                    "ner_score": row.ner_score,
                    "is_start": token_id == int(row.start_token_id),
                }
            )
    return pd.DataFrame(rows, columns=["document_id", "token_id", "predicted_entity_type", "ner_score", "is_start"])


def build_ner_tags(train_df: pd.DataFrame, ner_features_df: pd.DataFrame, threshold: float) -> pd.Series:
    """Per-token IOB2 "NER" tag ("B-PERS", "I-LOC", ..., "O"), aligned to train_df's
    (document_id, token_id) rows. No best-candidate-per-token pick is needed here --
    deduplicate_ner_features.py already guarantees at most one candidate covers any given
    token, so exploding straight to tokens is enough."""
    print(f"{len(ner_features_df)} candidates before threshold filtering")
    candidates_df = ner_features_df.dropna(subset=["start_token_id", "end_token_id"]).copy()
    candidates_df = candidates_df[candidates_df["ner_score"] >= threshold]
    print(f"{len(candidates_df)} candidates remain with ner_score >= {threshold}")

    exploded_df = explode_candidates_to_tokens(candidates_df)
    print(f"{len(exploded_df)} tokens covered by at least one surviving candidate")

    prefix = exploded_df["is_start"].map({True: "B-", False: "I-"})
    exploded_df["tag"] = prefix + exploded_df["predicted_entity_type"]

    merged = train_df[["document_id", "token_id"]].merge(
        exploded_df[["document_id", "token_id", "tag"]], on=["document_id", "token_id"], how="left"
    )
    return merged["tag"].fillna("O")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV (gold labels)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="Deduplicated span-level NER candidates CSV (see deduplicate_ner_features.py) -- any model's, GLiNER2 or historical-ner-baseline")
    parser.add_argument(
        "--out",
        default=None,
        help="Token-level output CSV path (default: hipe2020_train_fr_ner_token_format_threshold<threshold>.csv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Minimum ner_score for a candidate to be eligible (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only keep the first N documents (smoke test)")
    args = parser.parse_args()
    out_path = Path(args.out) if args.out is not None else default_out_path(args.threshold)

    print("=== Step 1: Load train data (gold labels) ===")
    print(f"Load data from {args.load_data}")
    data_df = pd.read_csv(args.load_data, dtype={"TOKEN": str, "MISC": str},
        # pandas' default NA-string sentinels ("NA", "null", "nan", ...) would otherwise
        # silently corrupt a genuine OCR token whose text happens to collide with one of
        # them (confirmed: one real token in hipe2020_fr is literally "NA") into a float
        # NaN despite the dtype=str hint above -- dtype coercion happens AFTER NA
        # detection, so it can't prevent this. keep_default_na=False turns that off
        # entirely, and na_values restores it only for the two genuinely-numeric columns
        # that still need a blank cell to become NaN.
        keep_default_na=False, na_values={"sentence_ocr_mean": [""], "document_ocr_mean": [""], "dictionary_score": [""]})
    data_df["MISC"] = data_df["MISC"].fillna("_")
    print(f"{data_df.shape[0]} tokens across {data_df['document_id'].nunique()} documents")

    if args.limit is not None:
        doc_ids = data_df["document_id"].drop_duplicates().head(args.limit)
        data_df = data_df[data_df["document_id"].isin(doc_ids)].reset_index(drop=True)
        print(f"Limited to {data_df['document_id'].nunique()} documents ({data_df.shape[0]} tokens)")

    print("=== Step 2: Load NER features (predicted candidates) ===")
    print(f"Loading NER features from {args.ner_features}")
    ner_features_df = pd.read_csv(args.ner_features)
    if args.limit is not None:
        ner_features_df = ner_features_df[ner_features_df["document_id"].isin(doc_ids)].reset_index(drop=True)
    print(f"{ner_features_df.shape[0]} candidates across {ner_features_df['document_id'].nunique()} documents")

    print("=== Step 3: Filter by threshold and build NER tags ===")
    data_df["NER"] = build_ner_tags(data_df, ner_features_df, args.threshold)
    print(f"{(data_df['NER'] != 'O').sum()} tokens tagged with a NER entity type")

    print("=== Step 4: Assemble output columns ===")
    out_df = data_df.rename(columns={"document_id": "doc_id"})[OUTPUT_COLUMNS]

    print("=== Step 5: Save token-format output ===")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved token-format output to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
