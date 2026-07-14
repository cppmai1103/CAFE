"""Compute label_reliable, the ground-truth target for Phase 1 baselines B0/B1/B3: per
docs/phase1_manual.md SS3, a candidate (a predicted span + type from ner_features.csv) is
"reliable" (label_reliable = 1) iff it matches a gold entity's type. Two modes for what
"matches" means (--mode):

    span_type (default): the candidate's exact start_token_id/end_token_id must equal a
        whole gold entity's boundaries, AND its type must match. A candidate that only
        partially overlaps a gold entity (e.g. missing a leading token) is NOT reliable
        here, even if every token it does cover has the right type -- see
        label_reliability()'s single dict lookup on build_gold_spans's closed spans.

    type_only: span boundaries are ignored entirely -- reliable iff EVERY token the
        candidate covers has the same gold type as its predicted_entity_type, regardless
        of whether NE-COARSE-LIT tagged that token B- or I- (see gold_type, which already
        strips the B-/I- prefix) and regardless of whether the candidate's own boundaries
        line up with a whole gold entity's. See label_reliability_type_only /
        build_gold_token_types.

span_type is what train_b3_logistic_regression.py, calibrate_ner_confidence.py (via
train_b3_logistic_regression.load_labeled_candidates), plot_b3_weights.py, and
analyze_data_splits.py all reuse (build_gold_spans/label_reliability, re-exported via
analyze_ocr_context_features.py) -- type_only is only wired up in this file's own
standalone CLI (--mode type_only) for now, not those other callers.

Run standalone (python -m / python src/gliner/label_reliability.py) to compute this over
deduplicate_ner_features.csv and save one row per candidate: document_id, sentence_id,
start_token_id, end_token_id, entity_text, predicted_entity_type, ner_score,
reliability_score (0/1, per whichever --mode was chosen).

Usage:
    python src/gliner/label_reliability.py
    python src/gliner/label_reliability.py --mode type_only
    python src/gliner/label_reliability.py --ner-features data/deduplicate_ner_features.csv --out data/label_reliability.csv

Output filename bakes in --mode (e.g. label_reliability_span_type.csv,
label_reliability_type_only.csv) unless --out is given explicitly, so the two modes'
outputs never overwrite each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gliner.deduplicate_ner_features import DEFAULT_OUT as DEFAULT_NER_FEATURES
from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_TRAIN_DATA

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def default_out_path(mode: str) -> Path:
    """Bakes --mode into the default output filename (e.g. mode=span_type ->
    label_reliability_span_type.csv), so the two modes' outputs don't overwrite each
    other. Only used when --out isn't explicitly given (see main())."""
    return DATA_DIR / f"label_reliability_{mode}.csv"

# HIPE's coarse bare types map 1:1 onto GLiNER2's predicted_entity_type values.
_TYPE_MAP = {"pers": "PERS", "loc": "LOC", "org": "ORG", "time": "TIME", "prod": "PROD"}


def gold_type(tag: str) -> str | None:
    """Normalize a gold NE-COARSE-LIT bare type to GLiNER's scheme, or None for "O"/an
    out-of-scope subtype (e.g. a component tag)."""
    if pd.isna(tag) or tag == "O":
        return None
    raw_type = tag.split("-", 1)[1]
    return _TYPE_MAP.get(raw_type.split(".", 1)[0])


def build_gold_spans(train_df: pd.DataFrame) -> dict[tuple, str]:
    """Close NE-COARSE-LIT's per-token IOB2 tags into gold spans, keyed by
    (document_id, start_token_id, end_token_id) -> entity_type, for exact-match lookup
    against candidate spans."""
    tags_df = train_df[["document_id", "token_id", "NE-COARSE-LIT"]].rename(columns={"NE-COARSE-LIT": "tag"})

    spans: dict[tuple, str] = {}
    current: dict | None = None

    def flush():
        nonlocal current
        if current is not None:
            spans[(current["document_id"], current["start"], current["end"])] = current["type"]
        current = None

    for row in tqdm(tags_df.itertuples(index=False), total=len(tags_df), desc="Closing gold spans", unit="token"):
        prefix = None if row.tag == "O" else row.tag.split("-", 1)[0]
        etype = gold_type(row.tag)
        same_doc = current is not None and current["document_id"] == row.document_id
        continues = same_doc and prefix == "I" and etype == current["type"]

        if not continues:
            flush()
        if etype is None:
            continue
        if continues:
            current["end"] = row.token_id
        else:
            current = {"document_id": row.document_id, "start": row.token_id, "end": row.token_id, "type": etype}

    flush()
    return spans


def label_reliability(candidates_df: pd.DataFrame, gold_spans: dict[tuple, str]) -> pd.Series:
    """label_reliable per candidate row: True iff (document_id, start_token_id,
    end_token_id) is an exact key in gold_spans AND that gold span's type matches this
    candidate's predicted_entity_type."""
    tqdm.pandas(desc="Matching candidates against gold spans", unit="candidate")

    def is_reliable(row) -> bool:
        if pd.isna(row["start_token_id"]) or pd.isna(row["end_token_id"]):
            return False
        key = (row["document_id"], int(row["start_token_id"]), int(row["end_token_id"]))
        return gold_spans.get(key) == row["predicted_entity_type"]

    return candidates_df.progress_apply(is_reliable, axis=1)


def build_gold_token_types(train_df: pd.DataFrame) -> dict[tuple, str | None]:
    """(document_id, token_id) -> gold type, B-/I- prefix already ignored (gold_type
    strips it), or None if the token isn't part of any gold entity ("O"). Used by
    label_reliability_type_only -- unlike build_gold_spans, this never closes tags into
    whole spans, so it has no notion of span boundaries at all."""
    tags_df = train_df[["document_id", "token_id", "NE-COARSE-LIT"]].rename(columns={"NE-COARSE-LIT": "tag"})
    return {(row.document_id, row.token_id): gold_type(row.tag) for row in tags_df.itertuples(index=False)}


def label_reliability_type_only(candidates_df: pd.DataFrame, gold_token_types: dict[tuple, str | None]) -> pd.Series:
    """label_reliable per candidate row, type_only mode: True iff EVERY token in
    [start_token_id, end_token_id] has the same gold type as predicted_entity_type.
    Ignores span boundaries entirely -- a candidate covering only part of a longer gold
    entity (e.g. missing a leading token) can still be reliable here, unlike
    label_reliability's span_type mode."""
    tqdm.pandas(desc="Matching candidates against gold token types (type_only)", unit="candidate")

    def is_reliable(row) -> bool:
        if pd.isna(row["start_token_id"]) or pd.isna(row["end_token_id"]):
            return False
        start, end = int(row["start_token_id"]), int(row["end_token_id"])
        document_id, predicted_type = row["document_id"], row["predicted_entity_type"]
        return all(gold_token_types.get((document_id, token_id)) == predicted_type for token_id in range(start, end + 1))

    return candidates_df.progress_apply(is_reliable, axis=1)


OUTPUT_COLUMNS = [
    "document_id",
    "sentence_id",
    "start_token_id",
    "end_token_id",
    "entity_text",
    "predicted_entity_type",
    "ner_score",
    "reliability_score",
]


MODES = ["span_type", "type_only"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="Token-level train data CSV (gold labels)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="Deduplicated span-level GLiNER2 candidates CSV (see deduplicate_ner_features.py)")
    parser.add_argument(
        "--out", default=None, help="Output CSV path (default: label_reliability_<mode>.csv)"
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="span_type",
        help="span_type: exact span boundaries + type must match (default). type_only: every token the "
        "candidate covers must have the matching gold type, span boundaries ignored.",
    )
    args = parser.parse_args()
    out_path = Path(args.out) if args.out is not None else default_out_path(args.mode)

    print(f"=== Step 1: Load train data and build gold lookup (mode={args.mode}) ===")
    print(f"Loading train data from {args.train_data}")
    train_df = pd.read_csv(args.train_data, dtype={"TOKEN": str, "MISC": str})
    train_df["token_id"] = train_df["token_id"].astype(int)
    if args.mode == "span_type":
        gold_lookup = build_gold_spans(train_df)
        print(f"{len(gold_lookup)} gold entity spans")
    else:
        gold_lookup = build_gold_token_types(train_df)
        print(f"{len(gold_lookup)} gold token types")

    print("=== Step 2: Load deduplicated NER candidates ===")
    print(f"Loading {args.ner_features}")
    candidates_df = pd.read_csv(args.ner_features)
    print(f"{len(candidates_df)} candidates loaded")

    print(f"=== Step 3: Compute reliability_score against gold (mode={args.mode}) ===")
    if args.mode == "span_type":
        candidates_df["reliability_score"] = label_reliability(candidates_df, gold_lookup).astype(int)
    else:
        candidates_df["reliability_score"] = label_reliability_type_only(candidates_df, gold_lookup).astype(int)
    n_reliable = int(candidates_df["reliability_score"].sum())
    print(f"{n_reliable} / {len(candidates_df)} candidates reliable ({n_reliable / len(candidates_df):.4%})")

    print("=== Step 4: Save output ===")
    out_df = candidates_df[OUTPUT_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
