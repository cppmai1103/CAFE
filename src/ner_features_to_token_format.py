"""Convert ner_features.csv (span-level GLiNER2 candidates) into the same token-level
format as the train data CSV, so gold and predicted labels sit side by side, one row per
token (including tokens that are not part of any entity).

For each token, the candidate with the highest ner_score among all candidates whose span
covers it (across every predicted_entity_type) is picked as that token's prediction.
Candidates below --threshold are dropped before picking, so a token with no
sufficiently-confident candidate over it gets "O" in the Gliner column, instead of always
being tagged with GLiNER's best (possibly near-zero-confidence) guess.

Output columns: doc_id, token_id, TOKEN, NE-COARSE-LIT, NE-COARSE-METO, NE-FINE-LIT,
NE-FINE-METO, NE-FINE-COMP, NE-NESTED, NEL-LIT, NEL-METO, MISC, Gliner, dictionary_score.

Gliner is an IOB2 tag ("B-PERS", "I-LOC", ..., "O") built from the winning candidate's
type and the token's position within that candidate's span (first covered token -> B-,
later ones -> I-), so it has the same shape as the gold NE-COARSE-LIT column.

Usage:
    python ner_features_to_token_format.py
    python ner_features_to_token_format.py --threshold 0.5
    python ner_features_to_token_format.py --limit 5 --out /tmp/smoke_token_format.csv  # quick smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TRAIN_DATA = DATA_DIR / "hipe2020_train_fr_train_data.csv"
DEFAULT_NER_FEATURES = DATA_DIR / "ner_features.csv"
DEFAULT_OUT = DATA_DIR / "hipe2020_train_fr_gliner_token_format.csv"

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
    "Gliner",
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


def pick_best_candidate_per_token(exploded_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest-ner_score candidate covering each (document_id, token_id)."""
    best_idx = exploded_df.groupby(["document_id", "token_id"])["ner_score"].idxmax()
    return exploded_df.loc[best_idx].reset_index(drop=True)


def build_gliner_tags(train_df: pd.DataFrame, ner_features_df: pd.DataFrame, threshold: float) -> pd.Series:
    """Per-token IOB2 "Gliner" tag ("B-PERS", "I-LOC", ..., "O"), aligned to train_df's
    (document_id, token_id) rows."""
    print(f"{len(ner_features_df)} candidates before threshold filtering")
    candidates_df = ner_features_df.dropna(subset=["start_token_id", "end_token_id"]).copy()
    candidates_df = candidates_df[candidates_df["ner_score"] >= threshold]
    print(f"{len(candidates_df)} candidates remain with ner_score >= {threshold}")

    exploded_df = explode_candidates_to_tokens(candidates_df)
    best_df = pick_best_candidate_per_token(exploded_df)
    print(f"{len(best_df)} tokens covered by at least one surviving candidate")

    prefix = best_df["is_start"].map({True: "B-", False: "I-"})
    best_df["tag"] = prefix + best_df["predicted_entity_type"]

    merged = train_df[["document_id", "token_id"]].merge(
        best_df[["document_id", "token_id", "tag"]], on=["document_id", "token_id"], how="left"
    )
    return merged["tag"].fillna("O")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (gold labels)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="Span-level GLiNER2 candidates CSV")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Token-level output CSV path")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum ner_score for a candidate to be eligible (default: 0.5)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only keep the first N documents (smoke test)")
    args = parser.parse_args()

    print("=== Step 1: Load train data (gold labels) ===")
    print(f"Loading train data from {args.train_data}")
    train_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    train_df["MISC"] = train_df["MISC"].fillna("_")
    print(f"{train_df.shape[0]} tokens across {train_df['document_id'].nunique()} documents")

    if args.limit is not None:
        doc_ids = train_df["document_id"].drop_duplicates().head(args.limit)
        train_df = train_df[train_df["document_id"].isin(doc_ids)].reset_index(drop=True)
        print(f"Limited to {train_df['document_id'].nunique()} documents ({train_df.shape[0]} tokens)")

    print("=== Step 2: Load NER features (predicted candidates) ===")
    print(f"Loading NER features from {args.ner_features}")
    ner_features_df = pd.read_csv(args.ner_features)
    if args.limit is not None:
        ner_features_df = ner_features_df[ner_features_df["document_id"].isin(doc_ids)].reset_index(drop=True)
    print(f"{ner_features_df.shape[0]} candidates across {ner_features_df['document_id'].nunique()} documents")

    print("=== Step 3: Pick winning candidate per token and build Gliner tags ===")
    train_df["Gliner"] = build_gliner_tags(train_df, ner_features_df, args.threshold)
    print(f"{(train_df['Gliner'] != 'O').sum()} tokens tagged with a Gliner entity type")

    print("=== Step 4: Assemble output columns ===")
    out_df = train_df.rename(columns={"document_id": "doc_id"})[OUTPUT_COLUMNS]

    print("=== Step 5: Save token-format output ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved token-format output to {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
