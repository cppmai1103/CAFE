"""Deduplicate ner_features.csv by resolving span overlaps within each sentence: sort a
sentence's candidates by ner_score descending, then greedily keep a candidate only if its
[start_token_id, end_token_id] range does not overlap any span already kept for that
sentence -- so the surviving set is pairwise non-overlapping by construction, regardless
of predicted_entity_type.

This differs from ner_features_to_token_format.py's pick_best_candidate_per_token: that
picks a winner independently per individual token, which can still leave two different
tokens of the same sentence "won" by two different overlapping spans (e.g. token 8 could
be won by span A and token 9 by span B, even though A and B overlap at token 9). Working
at the span level directly, as here, rules that out: once a span is kept, every candidate
overlapping ANY of its tokens is discarded, so a re-check afterwards (Step 3) should never
find a token covered twice.

Candidates with no start_token_id/end_token_id (no train-data token overlapped the span --
see extract_ner_features.py) are dropped before overlap resolution, since there's no token
range to compare them on.

Conflict report (--conflicts-out, default data/ner_overlap_conflicts.json): one entry per
sentence that had at least one real overlap, listing every discarded candidate alongside
the kept span(s) that beat it out -- so the greedy decision (see resolve_overlaps_for_sentence)
is auditable after the fact instead of only visible as a row count. Sentences with no
overlaps at all are omitted entirely.

Output row order: sorted by (document_id, start_token_id, end_token_id) -- i.e. the same
order the text itself reads in -- rather than the ner_score-descending order
resolve_overlaps_for_sentence naturally produces within each sentence (see deduplicate()).

Usage:
    python src/gliner/deduplicate_ner_features.py
    python src/gliner/deduplicate_ner_features.py --ner-features data/ner_features.csv --out data/deduplicate_ner_features.csv --conflicts-out data/ner_overlap_conflicts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "data_baseline"
DEFAULT_NER_FEATURES = DATA_DIR / "ner_features.csv"
DEFAULT_OUT = DATA_DIR / "deduplicate_ner_features.csv"
DEFAULT_CONFLICTS_OUT = DATA_DIR / "ner_overlap_conflicts.json"

SPAN_FIELDS = ["start_token_id", "end_token_id", "entity_text", "predicted_entity_type", "ner_score"]


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


def resolve_overlaps_for_sentence(group: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """group: one sentence's candidates, already filtered to have non-null start/end
    token ids. Returns (kept_df, conflicts) -- kept_df is the subset kept after greedy
    score-descending overlap resolution; conflicts is one {"kept": ...,
    "discarded_options": [...]} dict per kept span that had at least one overlapping
    competitor, so every rejected alternative for a given kept span is gathered in one
    place (e.g. "Société suisse du Grutli" ORG's entry lists every other type GLiNER2
    proposed for that same/overlapping token range -- LOC, PERS, TIME, ... -- as
    discarded_options), rather than one entry per discard."""
    ordered = group.sort_values("ner_score", ascending=False)
    kept_rows = []
    kept_ranges: list[tuple[int, int]] = []  # (start, end) inclusive, already-kept spans
    kept_records: list[dict] = []  # parallel to kept_ranges
    kept_discards: list[list[dict]] = []  # parallel to kept_ranges

    for row in ordered.itertuples(index=False):
        start, end = int(row.start_token_id), int(row.end_token_id)
        overlapping_idx = [i for i, (k_start, k_end) in enumerate(kept_ranges) if start <= k_end and k_start <= end]
        if not overlapping_idx:
            kept_rows.append(row)
            kept_ranges.append((start, end))
            kept_records.append(_span_record(row))
            kept_discards.append([])
        else:
            record = _span_record(row)
            for i in overlapping_idx:
                kept_discards[i].append(record)

    conflicts = [
        {"kept": kept_records[i], "discarded_options": kept_discards[i]}
        for i in range(len(kept_records))
        if kept_discards[i]
    ]
    return pd.DataFrame(kept_rows, columns=group.columns), conflicts


def deduplicate(candidates_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drop candidates with no overlapping train-data token, then resolve span overlaps
    sentence by sentence (see resolve_overlaps_for_sentence). Returns (dedup_df,
    sentence_conflicts) -- sentence_conflicts is one {"document_id", "sentence_id",
    "conflicts": [...]} dict per sentence that had at least one overlap, in encounter order."""
    before = len(candidates_df)
    valid_df = candidates_df.dropna(subset=["start_token_id", "end_token_id"]).copy()
    dropped_no_token = before - len(valid_df)
    if dropped_no_token:
        print(f"Dropped {dropped_no_token} candidate(s) with no overlapping train-data token")

    groups = list(valid_df.groupby(["document_id", "sentence_id"], sort=False))
    kept_groups = []
    sentence_conflicts = []
    for (document_id, sentence_id), group in tqdm(groups, desc="Resolving span overlaps per sentence", unit="sentence"):
        kept_df, conflicts = resolve_overlaps_for_sentence(group)
        kept_groups.append(kept_df)
        if conflicts:
            sentence_conflicts.append({"document_id": document_id, "sentence_id": int(sentence_id), "conflicts": conflicts})

    dedup_df = pd.concat(kept_groups, ignore_index=True) if kept_groups else valid_df.iloc[0:0]

    # resolve_overlaps_for_sentence appends survivors in ner_score-descending order, not
    # text order -- sort by (document_id, start_token_id) so rows land in the same order
    # the text itself reads, regardless of which score won each sentence's conflicts.
    dedup_df = dedup_df.sort_values(["document_id", "start_token_id", "end_token_id"]).reset_index(drop=True)

    return dedup_df, sentence_conflicts


def verify_no_token_covered_twice(dedup_df: pd.DataFrame) -> pd.DataFrame:
    """Re-expand every kept span to the individual train-data tokens it covers, and
    return the rows for any (document_id, token_id) covered by more than one surviving
    candidate -- should always be empty if resolve_overlaps_for_sentence is correct;
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
    parser.add_argument("--ner-features", default=str(DEFAULT_NER_FEATURES), help="ner_features.csv (span-level GLiNER2 candidates)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Deduplicated candidates output CSV path")
    parser.add_argument(
        "--conflicts-out",
        default=str(DEFAULT_CONFLICTS_OUT),
        help="JSON report of overlapping spans per sentence and which one was kept",
    )
    args = parser.parse_args()

    print("=== Step 1: Load ner_features.csv ===")
    print(f"Loading {args.ner_features}")
    candidates_df = pd.read_csv(args.ner_features)
    print(f"{len(candidates_df)} candidates loaded")

    print("=== Step 2: Resolve span overlaps sentence by sentence (keep highest ner_score) ===")
    dedup_df, sentence_conflicts = deduplicate(candidates_df)
    n_contested_spans = sum(len(s["conflicts"]) for s in sentence_conflicts)
    n_discarded = sum(len(c["discarded_options"]) for s in sentence_conflicts for c in s["conflicts"])
    print(f"{len(dedup_df)} candidates remain after deduplication ({len(candidates_df) - len(dedup_df)} removed)")
    print(
        f"{len(sentence_conflicts)} sentence(s) had at least one overlap: {n_contested_spans} kept span(s) had "
        f"competing candidates, {n_discarded} candidate(s) total discarded for overlapping a higher-scoring span"
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

    print("=== Step 5: Save overlap conflict report ===")
    conflicts_out_path = Path(args.conflicts_out)
    conflicts_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conflicts_out_path, "w") as f:
        json.dump(sentence_conflicts, f, indent=2, ensure_ascii=False)
    print(f"Saved {conflicts_out_path}")

    print(f"{len(dedup_df)} entities left after deduplication")
    print("=== Done ===")


if __name__ == "__main__":
    main()
