"""Deduplicate historical-ner-baseline's ner_features.csv by MERGING span overlaps within
each sentence, rather than discarding the losers outright the way
src/ner/gliner/deduplicate_ner_features.py does for GLiNER2.

Why this needs its own version: GLiNER2 scores every (span, type) pair independently, so
two overlapping candidates genuinely are two competing hypotheses -- keeping only the
highest-scoring one and discarding the rest is correct there. historical-ner-baseline is
different: it's a single-label BIO tagger, and when overlapping candidates show up it's
usually not a competing hypothesis but the model's own tokenizer splitting ONE real word
into multiple adjacent entity fragments that each got tagged separately (e.g. OCR text
"Berlin" -> two fragments "BER"/"LIN", both landing on the same single train-data token
since "Berlin" is one token in the source data):

    EXP-1918-01-21-a-i0066,2,19,19,BER,LOC,0.9540882110595703,1,3,False
    EXP-1918-01-21-a-i0066,2,19,19,LIN,LOC,0.39546456933021545,1,3,False

Discarding "LIN" the way GLiNER2's dedup would leaves the recovered entity as just "BER"
-- silently truncating real text. Instead: transitively-overlapping candidates (found via
standard interval merging, sorted by start_token_id -- see merge_overlaps_for_sentence)
are merged into one final span. The final span's type and ner_score come from whichever
member had the highest ner_score (same "highest score wins" rule GLiNER2's dedup uses),
but its entity_text is the concatenation of every member's text, in the order they appear
in ner_features.csv (left-to-right through the sentence, same order the model itself
extracted them in) -- so "BER" + "LIN" -> "BERLIN", not just "BER".

Candidates with no start_token_id/end_token_id (no train-data token overlapped the span --
see extract_ner_features.py) are dropped before merging, since there's no token range to
compare them on.

Merge report (--conflicts-out, default data/ner_overlap_conflicts.json): one entry per
sentence that had at least one real merge, listing the final merged span alongside every
original candidate that fed into it -- so the merge decision (see
merge_overlaps_for_sentence) is auditable after the fact instead of only visible as a row
count. Sentences with no overlaps at all are omitted entirely.

Output row order: sorted by (document_id, start_token_id, end_token_id) -- i.e. the same
order the text itself reads in -- rather than encounter order (see deduplicate()).

Usage:
    python src/ner/historical_ner/deduplicate_ner_features.py
    python src/ner/historical_ner/deduplicate_ner_features.py --ner-features data/ner_features.csv --out data/deduplicate_ner_features.csv --conflicts-out data/ner_overlap_conflicts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_NER_FEATURES = DATA_DIR / "ner_features.csv"
DEFAULT_OUT = DATA_DIR / "deduplicate_ner_features.csv"
DEFAULT_CONFLICTS_OUT = DATA_DIR / "ner_overlap_conflicts.json"

OUTPUT_FIELDS = [
    "document_id", "sentence_id", "start_token_id", "end_token_id", "entity_text",
    "predicted_entity_type", "ner_score", "span_length_tokens", "span_length_characters", "sentence_chunked",
]


def _span_record(row) -> dict:
    """Plain-Python-typed dict for one candidate row, JSON-serializable (pandas/numpy
    scalars from itertuples aren't, natively)."""
    return {
        "start_token_id": int(row.start_token_id),
        "end_token_id": int(row.end_token_id),
        "entity_text": str(row.entity_text),
        "predicted_entity_type": str(row.predicted_entity_type),
        "ner_score": float(row.ner_score),
    }


def _merge_group_to_row(members: list) -> dict:
    """Turn one group of transitively-overlapping candidates (members, in the order they
    appear in ner_features.csv) into a single final output row: type/ner_score come from
    the highest-scoring member (same "highest score wins" rule as GLiNER2's dedup), but
    entity_text is every member's text concatenated in file order (not score order) --
    so a word split into fragments by the model's own tokenizer ("BER" + "LIN") comes back
    together as "BERLIN" instead of losing everything but the highest-scoring fragment."""
    winner = max(members, key=lambda row: row.ner_score)
    start_token_id = min(int(row.start_token_id) for row in members)
    end_token_id = max(int(row.end_token_id) for row in members)
    entity_text = "".join(str(row.entity_text) for row in members)
    return {
        "document_id": winner.document_id,
        "sentence_id": winner.sentence_id,
        "start_token_id": start_token_id,
        "end_token_id": end_token_id,
        "entity_text": entity_text,
        "predicted_entity_type": winner.predicted_entity_type,
        "ner_score": winner.ner_score,
        "span_length_tokens": len(entity_text.split()),
        "span_length_characters": len(entity_text),
        "sentence_chunked": bool(any(row.sentence_chunked for row in members)),
    }


def merge_overlaps_for_sentence(group: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """group: one sentence's candidates, already filtered to have non-null start/end
    token ids, in ner_features.csv's own row order (== extraction order, left to right
    through the sentence). Returns (merged_df, merges) -- merged_df has one row per
    connected group of transitively-overlapping candidates (see _merge_group_to_row);
    merges is one {"final": ..., "candidates": [...]} dict per group that actually merged
    more than one candidate (singletons aren't reported, same as GLiNER2's dedup only
    reporting spans that had real competition).

    Connected groups are found via standard 1D interval merging: sort by start_token_id
    (ties broken by original file order for determinism), sweep left to right, and fold a
    candidate into the current group whenever its start falls at or before the running
    group's max end_token_id so far. This correctly captures transitive overlaps (A
    overlaps B, B overlaps C, A doesn't directly overlap C -- all three still end up in
    one group), unlike checking pairwise overlap only against a single "kept" anchor."""
    ordered = group.reset_index(drop=True)
    ordered["file_order"] = ordered.index
    ordered = ordered.sort_values(["start_token_id", "end_token_id", "file_order"])

    merged_rows = []
    merges = []
    current_members: list = []
    current_end = None

    def flush():
        if not current_members:
            return
        merged_rows.append(_merge_group_to_row(current_members))
        if len(current_members) > 1:
            # Report candidates in the same file order used to build entity_text, not the
            # start_token_id-sorted order used to detect the merge -- so the JSON reads
            # the same left-to-right order a human would expect from the source text.
            file_ordered = sorted(current_members, key=lambda row: row.file_order)
            merges.append({
                "final": _merge_group_to_row(current_members),
                "candidates": [_span_record(row) for row in file_ordered],
            })

    for row in ordered.itertuples(index=False):
        start = int(row.start_token_id)
        if current_members and start <= current_end:
            current_members.append(row)
            current_end = max(current_end, int(row.end_token_id))
        else:
            flush()
            current_members = [row]
            current_end = int(row.end_token_id)
    flush()

    return pd.DataFrame(merged_rows, columns=OUTPUT_FIELDS), merges


def deduplicate(candidates_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drop candidates with no overlapping train-data token, then merge span overlaps
    sentence by sentence (see merge_overlaps_for_sentence). Returns (dedup_df,
    sentence_merges) -- sentence_merges is one {"document_id", "sentence_id", "merges":
    [...]} dict per sentence that had at least one real merge, in encounter order."""
    before = len(candidates_df)
    valid_df = candidates_df.dropna(subset=["start_token_id", "end_token_id"]).copy()
    dropped_no_token = before - len(valid_df)
    if dropped_no_token:
        print(f"Dropped {dropped_no_token} candidate(s) with no overlapping train-data token")

    groups = list(valid_df.groupby(["document_id", "sentence_id"], sort=False))
    merged_groups = []
    sentence_merges = []
    for (document_id, sentence_id), group in tqdm(groups, desc="Merging span overlaps per sentence", unit="sentence"):
        merged_df, merges = merge_overlaps_for_sentence(group)
        merged_groups.append(merged_df)
        if merges:
            sentence_merges.append({"document_id": document_id, "sentence_id": int(sentence_id), "merges": merges})

    dedup_df = pd.concat(merged_groups, ignore_index=True) if merged_groups else valid_df.iloc[0:0][OUTPUT_FIELDS]

    dedup_df = dedup_df.sort_values(["document_id", "start_token_id", "end_token_id"]).reset_index(drop=True)

    return dedup_df, sentence_merges


def verify_no_token_covered_twice(dedup_df: pd.DataFrame) -> pd.DataFrame:
    """Re-expand every final span to the individual train-data tokens it covers, and
    return the rows for any (document_id, token_id) covered by more than one surviving
    candidate -- should always be empty if merge_overlaps_for_sentence is correct;
    non-empty means a real bug, not just noise, since spans are supposed to be pairwise
    non-overlapping by construction."""
    rows = []
    for row in tqdm(dedup_df.itertuples(index=False), total=len(dedup_df), desc="Verifying no token is double-covered", unit="candidate"):
        for token_id in range(int(row.start_token_id), int(row.end_token_id) + 1):
            rows.append(
                {
                    "document_id": row.document_id,
                    "token_id": token_id,
                    "entity_text": row.entity_text,
                    "predicted_entity_type": row.predicted_entity_type,
                    "ner_score": row.ner_score,
                }
            )
    exploded = pd.DataFrame(rows, columns=["document_id", "token_id", "entity_text", "predicted_entity_type", "ner_score"])
    counts = exploded.groupby(["document_id", "token_id"]).size()
    offending_keys = counts[counts > 1].index
    if len(offending_keys) == 0:
        return exploded.iloc[0:0]
    return exploded.set_index(["document_id", "token_id"]).loc[offending_keys].reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="ner_features.csv (span-level historical-ner-baseline candidates)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Deduplicated candidates output CSV path")
    parser.add_argument(
        "--conflicts-out",
        default=str(DEFAULT_CONFLICTS_OUT),
        help="JSON report of merged spans per sentence and which candidates fed into each one",
    )
    args = parser.parse_args()

    print("=== Step 1: Load ner_features.csv ===")
    print(f"Loading {args.ner_features}")
    candidates_df = pd.read_csv(args.ner_features)
    print(f"{len(candidates_df)} candidates loaded")

    print("=== Step 2: Merge overlapping span fragments sentence by sentence (highest ner_score wins type/score, text is concatenated) ===")
    dedup_df, sentence_merges = deduplicate(candidates_df)
    n_merged_groups = sum(len(s["merges"]) for s in sentence_merges)
    n_fragments_absorbed = sum(len(m["candidates"]) - 1 for s in sentence_merges for m in s["merges"])
    print(f"{len(dedup_df)} candidates remain after merging ({len(candidates_df) - len(dedup_df)} fewer rows)")
    print(
        f"{len(sentence_merges)} sentence(s) had at least one merge: {n_merged_groups} final span(s) were built "
        f"from overlapping fragments, {n_fragments_absorbed} extra fragment(s) absorbed into a final span's text"
    )

    print("=== Step 3: Verify no token is covered by more than one surviving candidate ===")
    offending = verify_no_token_covered_twice(dedup_df)
    if len(offending) == 0:
        print("OK: every token is covered by at most one surviving candidate.")
    else:
        print(f"WARNING: {offending['token_id'].nunique()} token(s) still covered by multiple candidates:")
        print(offending.to_string(index=False))

    print("=== Step 4: Save deduplicated candidates ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("=== Step 5: Save merge report ===")
    conflicts_out_path = Path(args.conflicts_out)
    conflicts_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conflicts_out_path, "w") as f:
        json.dump(sentence_merges, f, indent=2, ensure_ascii=False)
    print(f"Saved {conflicts_out_path}")

    print(f"{len(dedup_df)} entities left after deduplication")
    print("=== Done ===")


if __name__ == "__main__":
    main()
