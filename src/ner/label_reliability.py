"""Compute label_reliable, the ground-truth target for Phase 1 baselines B0/B1/B3: per
docs/phase1_manual_features.md SS3, a candidate (a predicted span + type from ner_features.csv) is
"reliable" (label_reliable = 1) iff it matches a gold entity's type. Two levels of output
granularity (--level):

    span (default): one row per candidate (document_id, sentence_id, start_token_id,
        end_token_id). Three modes for what "matches" means (--mode):

        span_type: the candidate's exact start_token_id/end_token_id must equal a whole
            gold entity's boundaries, AND its type must match. A candidate that only
            partially overlaps a gold entity (e.g. missing a leading token) is NOT
            reliable here, even if every token it does cover has the right type -- see
            label_reliability()'s single dict lookup on build_gold_spans's closed spans.

        type_only (default --mode): span boundaries are ignored entirely -- reliable iff
            EVERY token the candidate covers has the same gold type as its
            predicted_entity_type, regardless of whether NE-COARSE-LIT tagged that token
            B- or I- (see gold_type, which already strips the B-/I- prefix) and
            regardless of whether the candidate's own boundaries line up with a whole
            gold entity's. See label_reliability_type_only / build_gold_token_types.

        fuzzy: reliable iff AT LEAST ONE token the candidate covers has the same gold
            type as its predicted_entity_type -- unlike type_only (which requires every
            covered token to match), a candidate that over-extends into a leading/
            trailing non-entity token (e.g. "De Bruxelles" instead of gold "Bruxelles")
            is still reliable here, as long as it captures at least one gold token of the
            right type. See label_reliability_fuzzy / build_gold_token_types (same gold
            lookup as type_only).

    word: one row per (document_id, sentence_id, token_id) -- each candidate span is
        exploded into its individual covered tokens, which inherit its
        predicted_entity_type/ner_score. --mode is ignored at this granularity: the
        all()/any() distinction between type_only and fuzzy only matters when a candidate
        spans multiple tokens, so at word grain reliability is always a single-token type
        match (gold type of this one token == predicted_entity_type) -- the type_only
        definition applied per token. See build_word_level_rows.

Run standalone (python -m / python src/ner/label_reliability.py) to compute this over
deduplicate_ner_features.csv.

Usage:
    python src/ner/label_reliability.py
    python src/ner/label_reliability.py --mode type_only
    python src/ner/label_reliability.py --mode fuzzy
    python src/ner/label_reliability.py --level word
    python src/ner/label_reliability.py --ner-features data/deduplicate_ner_features.csv --out data/label_reliability.csv

Output filename bakes in --mode/--level (e.g. label_reliability_span_type.csv,
label_reliability_type_only.csv, label_reliability_span_level_fuzzy.csv,
label_reliability_word_level_type_only.csv) unless --out is given explicitly, so they
never overwrite each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ner.gliner.deduplicate_ner_features import DEFAULT_OUT as DEFAULT_NER_FEATURES
from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"


def default_out_path(mode: str, level: str = "span") -> Path:
    """Bakes --mode/--level into the default output filename (level=word ->
    label_reliability_word_level_type_only.csv, --mode ignored since word-level
    reliability is always the type_only check applied per token; level=span, mode=fuzzy
    -> label_reliability_span_level_fuzzy.csv; span_type/type_only at span level keep
    their original unqualified names -- other scripts (platt_scaling.py,
    prepare_data_logistic.py, plot_reliability_diagram.py, historical_ner/compare.py) have
    "label_reliability_type_only.csv" hardcoded as their own default input, so renaming it
    here would silently break those), so the different outputs don't overwrite each
    other. Only used when --out isn't explicitly given (see main())."""
    if level == "word":
        return DATA_DIR / "label_reliability_word_level_type_only.csv"
    if mode == "fuzzy":
        return DATA_DIR / "label_reliability_span_level_fuzzy.csv"
    return DATA_DIR / f"label_reliability_{mode}.csv"

# HIPE's coarse bare types map 1:1 onto GLiNER2's predicted_entity_type values. Matched
# case-insensitively (.lower() in gold_type below) since not every HIPE-2022 dataset
# spells its bare types the same way -- hipe2020/letemps use lowercase "pers"/"loc"/
# "org"/"time"/"prod", but newseye uses "PER"/"LOC"/"ORG"/"HumanProd" (no separate time
# type at all). "per"/"humanprod" are newseye-specific aliases for the same GLiNER2
# PERS/PROD output types; every other dataset's own spelling already matches a key here
# once lowercased, so this is purely additive, not a behavior change for hipe2020/letemps.
_TYPE_MAP = {
    "pers": "PERS", "per": "PERS",
    "loc": "LOC",
    "org": "ORG",
    "time": "TIME",
    "prod": "PROD", "humanprod": "PROD",
}


def gold_type(tag: str) -> str | None:
    """Normalize a gold NE-COARSE-LIT bare type to GLiNER's scheme, or None for "O"/an
    out-of-scope subtype (e.g. a component tag)."""
    if pd.isna(tag) or tag == "O":
        return None
    raw_type = tag.split("-", 1)[1]
    return _TYPE_MAP.get(raw_type.split(".", 1)[0].lower())


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


def label_reliability_fuzzy(candidates_df: pd.DataFrame, gold_token_types: dict[tuple, str | None]) -> pd.Series:
    """label_reliable per candidate row, fuzzy mode: True iff AT LEAST ONE token in
    [start_token_id, end_token_id] has the same gold type as predicted_entity_type.
    Same gold_token_types lookup as type_only, just any() instead of all() -- a candidate
    that over-extends its span into a leading/trailing non-entity or wrong-type token is
    still reliable here as long as it overlaps at least one gold token of the right type."""
    tqdm.pandas(desc="Matching candidates against gold token types (fuzzy)", unit="candidate")

    def is_reliable(row) -> bool:
        if pd.isna(row["start_token_id"]) or pd.isna(row["end_token_id"]):
            return False
        start, end = int(row["start_token_id"]), int(row["end_token_id"])
        document_id, predicted_type = row["document_id"], row["predicted_entity_type"]
        return any(gold_token_types.get((document_id, token_id)) == predicted_type for token_id in range(start, end + 1))

    return candidates_df.progress_apply(is_reliable, axis=1)


def build_token_text_lookup(train_df: pd.DataFrame) -> dict[tuple, str]:
    """(document_id, token_id) -> the token's own surface text (TOKEN column). Used by
    build_word_level_rows to fill in entity_text per exploded token, the same way
    build_gold_token_types looks up gold type per token."""
    return {(row.document_id, row.token_id): row.TOKEN for row in train_df[["document_id", "token_id", "TOKEN"]].itertuples(index=False)}


OUTPUT_COLUMNS_WORD = [
    "document_id",
    "sentence_id",
    "start_token_id",
    "end_token_id",
    "entity_text",
    "predicted_entity_type",
    "ner_score",
    "span_length_tokens",
    "span_length_characters",
    "reliability_score",
]


def build_word_level_rows(
    candidates_df: pd.DataFrame, gold_token_types: dict[tuple, str | None], token_text: dict[tuple, str],
) -> pd.DataFrame:
    """Explode every candidate span into one row per token it covers, with
    start_token_id/end_token_id both set to that token's own token_id (a word IS a
    length-1 span) and entity_text set to the token's own surface text. Also fills in
    span_length_tokens (always 1 at word grain -- degenerate/zero-variance, but B3/MLP's
    NUMERIC_FEATURES (phase1/modeling/logistic_regression.py) hard-require the column to
    exist) and span_length_characters (len(entity_text), still meaningful -- short vs long
    words). Together with reliability_score, this makes the output (OUTPUT_COLUMNS_WORD) a
    full drop-in replacement for deduplicate_ner_features.csv AND label_reliability.csv
    anywhere downstream that expects those columns (extract_ocr_features.py,
    extract_context_features.py, prepare_data_logistic.py, build_candidate_windows.py) --
    confirmed by actually running that chain, not just column-matching (sentence_chunked
    is the one deduplicate_ner_features.csv column deliberately NOT reproduced here, since
    it's span-extraction metadata prepare_data_logistic.py drops anyway -- see its
    errors="ignore" there).
    reliability_score compares that single token's own gold type against
    predicted_entity_type -- the type_only vs fuzzy all()/any() distinction only matters
    across multiple tokens, so at word grain it's just the type_only check applied to one
    token at a time."""
    rows = []
    for row in tqdm(candidates_df.itertuples(index=False), total=len(candidates_df), desc="Exploding candidates into per-token rows", unit="candidate"):
        if pd.isna(row.start_token_id) or pd.isna(row.end_token_id):
            continue
        start, end = int(row.start_token_id), int(row.end_token_id)
        for token_id in range(start, end + 1):
            reliable = gold_token_types.get((row.document_id, token_id)) == row.predicted_entity_type
            text = token_text.get((row.document_id, token_id), "")
            rows.append({
                "document_id": row.document_id,
                "sentence_id": row.sentence_id,
                "start_token_id": token_id,
                "end_token_id": token_id,
                "entity_text": text,
                "predicted_entity_type": row.predicted_entity_type,
                "ner_score": row.ner_score,
                "span_length_tokens": 1,
                "span_length_characters": len(text),
                "reliability_score": int(reliable),
            })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS_WORD)


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


MODES = ["span_type", "type_only", "fuzzy"]
LEVELS = ["span", "word"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV (gold labels)")
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="Deduplicated span-level GLiNER2 candidates CSV (see deduplicate_ner_features.py)")
    parser.add_argument(
        "--out", default=None, help="Output CSV path (default: label_reliability_<mode>.csv)"
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="type_only",
        help="span_type: exact span boundaries + type must match (default). type_only: every token the "
        "candidate covers must have the matching gold type, span boundaries ignored. fuzzy: at least one "
        "token the candidate covers must have the matching gold type. Ignored when --level word.",
    )
    parser.add_argument(
        "--level",
        choices=LEVELS,
        default="span",
        help="span (default): one row per candidate. word: one row per token the candidate covers "
        "(--mode is ignored, reliability is always a single-token type match).",
    )
    args = parser.parse_args()
    out_path = Path(args.out) if args.out is not None else default_out_path(args.mode, args.level)

    print(f"=== Step 1: Load train data and build gold lookup (level={args.level}, mode={args.mode}) ===")
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
    data_df["token_id"] = data_df["token_id"].astype(int)
    if args.level == "word":
        gold_lookup = build_gold_token_types(data_df)
        print(f"{len(gold_lookup)} gold token types")
    elif args.mode == "span_type":
        gold_lookup = build_gold_spans(data_df)
        print(f"{len(gold_lookup)} gold entity spans")
    else:
        gold_lookup = build_gold_token_types(data_df)
        print(f"{len(gold_lookup)} gold token types")

    print("=== Step 2: Load deduplicated NER candidates ===")
    print(f"Loading {args.ner_features}")
    candidates_df = pd.read_csv(args.ner_features)
    print(f"{len(candidates_df)} candidates loaded")

    print(f"=== Step 3: Compute reliability_score against gold (level={args.level}, mode={args.mode}) ===")
    if args.level == "word":
        token_text = build_token_text_lookup(data_df)
        out_df = build_word_level_rows(candidates_df, gold_lookup, token_text)
    else:
        if args.mode == "span_type":
            candidates_df["reliability_score"] = label_reliability(candidates_df, gold_lookup).astype(int)
        elif args.mode == "type_only":
            candidates_df["reliability_score"] = label_reliability_type_only(candidates_df, gold_lookup).astype(int)
        else:
            candidates_df["reliability_score"] = label_reliability_fuzzy(candidates_df, gold_lookup).astype(int)
        out_df = candidates_df[OUTPUT_COLUMNS]
    n_reliable = int(out_df["reliability_score"].sum())
    print(f"{n_reliable} / {len(out_df)} {args.level}-level rows reliable ({n_reliable / len(out_df):.4%})")

    print("=== Step 4: Save output ===")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
