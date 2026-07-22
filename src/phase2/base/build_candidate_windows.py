"""Phase 2 Stage 0 (docs/phase2_learned_features.md SS33 checklist items 1-4): build one
candidate-specific context window per NER candidate, word-level (no subword tokenization
yet -- that's tokenize_windows.py, SS9-10).

For each candidate, this reuses exactly two of Phase 1's existing outputs (no new
candidate generation, no new labeling):
    train data CSV          -- document tokens + dictionary_score + split (context)
    label_reliability_*.csv -- document_id + start_token_id/end_token_id (inclusive-bounds
        token span, see gliner/extract_ner_features.py) + entity_text + predicted_type +
        ner_score + reliability_score (-> label_reliable), all already joined by
        gliner/label_reliability.py -- no separate ner_features.csv load needed here.

start_token_id/end_token_id are INCLUSIVE (first/last overlapping token), unlike
docs/phase2_learned_features.md SS2's target_end_doc, which is EXCLUSIVE
(doc_tokens[target_start_doc:target_end_doc]) -- converted once here
(target_end_doc = end_token_id + 1) so every downstream Phase 2 script can use plain
Python slicing.

dict_flags (SS6.1's simple vocabulary) come directly from dictionary_score
(ocr_dictionary_check.py): True -> GOOD, False -> BAD, None (punctuation) -> PUNCT.
There is no continuous per-token OCR confidence in this data (same caveat as
docs/pipeline_phase2.md SS0 raised for the old Phase 2 design) -- GOOD/BAD is a
dictionary-membership proxy, not a real OCR-quality signal.

Output is the SS5 "raw candidate-window object" shape, one JSON object per line, with
`split` and the raw `sentence_id`/`start_token_id`/`end_token_id` key columns added:
    candidate_id, document_id, sentence_id, start_token_id, end_token_id, split,
    span_text, window_tokens, target_start_window, target_end_window, dict_flags,
    predicted_type, ner_score, label_reliable

target_flags (SS6.2) and vocab-id integer encoding are NOT computed here -- they depend
on subword alignment (SS9), which needs a tokenizer and is tokenize_windows.py's job.
This script's window_tokens/dict_flags stay as plain word-level strings so this output is
tokenizer-agnostic and reusable across encoder choices (CamemBERT/XLM-R/...).

Usage:
    python src/phase2/base/build_candidate_windows.py
    python src/phase2/base/build_candidate_windows.py --window-left 32 --window-right 32
    python src/phase2/base/build_candidate_windows.py --limit 200 --print-examples 20  # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ner.label_reliability import default_out_path as default_label_reliability_path
from preprocessing.preprocessing_data import DEFAULT_OUT as DEFAULT_LOAD_DATA

PHASE2_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_phase2"
DEFAULT_LABEL_RELIABILITY = default_label_reliability_path("type_only")
DEFAULT_OUT = PHASE2_DATA_DIR / "phase2_candidate_windows.jsonl"

DEFAULT_WINDOW_LEFT = 16
DEFAULT_WINDOW_RIGHT = 16


def dict_flag_of(dictionary_score) -> str:
    """True -> GOOD, False -> BAD, punctuation (None/NaN) -> PUNCT -- see
    preprocessing/ocr_dictionary_check.py's word_is_known docstring."""
    if pd.isna(dictionary_score):
        return "PUNCT"
    return "GOOD" if bool(dictionary_score) else "BAD"


def build_document_tables(
    train_df: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[bool]], dict[str, str]]:
    """One row per document: ordered token text list, ordered dict_flag list, ordered
    no_space_after list (same order, by token_id), and document_id -> split. token_id is
    document-level and 0-indexed (preprocessing_data.py), so sorting by it recovers
    document reading order directly -- no need to go through sentence_id.

    no_space_after is only used by Step 4's sanity-check reconstruction below (see
    reconstruct_text) -- it's not part of the JSONL output schema, so it's fine that it's
    a Python bool list rather than the GOOD/BAD/PUNCT string vocabulary dict_flags uses."""
    train_df = train_df.sort_values(["document_id", "token_id"])
    tokens_by_doc = train_df.groupby("document_id")["TOKEN"].apply(list).to_dict()
    dict_flags_by_doc = (
        train_df.assign(dict_flag=train_df["dictionary_score"].map(dict_flag_of))
        .groupby("document_id")["dict_flag"]
        .apply(list)
        .to_dict()
    )
    no_space_after_by_doc = (
        train_df.assign(no_space_after=train_df["MISC"].apply(lambda m: "NoSpaceAfter" in str(m)))
        .groupby("document_id")["no_space_after"]
        .apply(list)
        .to_dict()
    )
    doc_to_split = train_df.drop_duplicates("document_id").set_index("document_id")["split"].to_dict()
    return tokens_by_doc, dict_flags_by_doc, no_space_after_by_doc, doc_to_split


def build_window(
    doc_tokens: list[str], doc_dict_flags: list[str], doc_no_space_after: list[bool],
    start_token_id: int, end_token_id: int,
    window_left: int, window_right: int,
) -> tuple[list[str], list[str], list[bool], int, int]:
    """docs/phase2_learned_features.md SS3: word-level candidate window centered on the target span.
    start_token_id/end_token_id are INCLUSIVE (Phase 1 convention); converted to an
    EXCLUSIVE end here so the rest of this function is plain Python slicing."""
    s = int(start_token_id)
    e = int(end_token_id) + 1  # inclusive -> exclusive

    window_start = max(0, s - window_left)
    window_end = min(len(doc_tokens), e + window_right)

    window_tokens = doc_tokens[window_start:window_end]
    dict_flags = doc_dict_flags[window_start:window_end]
    no_space_after = doc_no_space_after[window_start:window_end]
    target_start_window = s - window_start
    target_end_window = e - window_start
    return window_tokens, dict_flags, no_space_after, target_start_window, target_end_window


def reconstruct_text(tokens: list[str], no_space_after: list[bool]) -> str:
    """Join tokens the same way build_sentence_texts (src/ner/gliner/extract_ner_features.py)
    detokenizes them for real -- a space before token i unless token i-1's own
    NoSpaceAfter flag suppresses it (e.g. "d" + "'" + "auteurs" -> "d'auteurs", not
    "d ' auteurs") -- so Step 4's sanity check compares like for like against span_text,
    which was built by that same detokenization logic."""
    pieces: list[str] = []
    for i, token in enumerate(tokens):
        if pieces and not no_space_after[i - 1]:
            pieces.append(" ")
        pieces.append(token)
    return "".join(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-data", default=str(DEFAULT_LOAD_DATA), help="Token-level data CSV (document tokens, dictionary_score, split)")
    parser.add_argument(
        "--label-reliability", default=str(DEFAULT_LABEL_RELIABILITY),
        help="label_reliability_*.csv (see gliner/label_reliability.py) -- candidates, entity_text, predicted_type, ner_score, and reliability_score all come from this one file",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument("--window-left", type=int, default=DEFAULT_WINDOW_LEFT, help="Word tokens of left context (docs/phase2_learned_features.md SS3 default: 64)")
    parser.add_argument("--window-right", type=int, default=DEFAULT_WINDOW_RIGHT, help="Word tokens of right context (docs/phase2_learned_features.md SS3 default: 64)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N candidates (smoke test)")
    parser.add_argument("--print-examples", type=int, default=20, help="Print this many random examples for the SS33 sanity check (0 to skip)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --print-examples sampling")
    args = parser.parse_args()

    print("=== Step 1: Load train data and build per-document token/dict-flag tables ===")
    print(f"Loading {args.load_data}")
    data_df = pd.read_csv(args.load_data, dtype={"TOKEN": str, "MISC": str},
        # pandas' default NA-string sentinels ("NA", "null", "nan", ...) would otherwise
        # silently corrupt a genuine OCR token whose text happens to collide with one of
        # them (confirmed: one real token in hipe2020_fr is literally "NA") into a float
        # NaN despite the dtype=str hint above -- dtype coercion happens AFTER NA
        # detection, so it can't prevent this. keep_default_na=False turns that off
        # entirely, and na_values restores it only for the two genuinely-numeric columns
        # that still need a blank cell to become NaN.
        keep_default_na=False, na_values={"sentence_ocr_mean": [""], "document_ocr_mean": [""], "dictionary_score": [""]})
    tokens_by_doc, dict_flags_by_doc, no_space_after_by_doc, doc_to_split = build_document_tables(data_df)
    print(f"{len(tokens_by_doc)} documents")

    print("=== Step 2: Load candidates (entity_text, predicted_type, ner_score, reliability_score) ===")
    print(f"Loading {args.label_reliability}")
    candidates_df = pd.read_csv(args.label_reliability)
    print(f"{len(candidates_df)} candidates")

    before_dropna = len(candidates_df)
    candidates_df = candidates_df.dropna(subset=["start_token_id", "end_token_id"])
    if len(candidates_df) < before_dropna:
        print(f"Dropped {before_dropna - len(candidates_df)} candidate(s) with no overlapping token (start/end_token_id is null)")

    if args.limit is not None:
        candidates_df = candidates_df.head(args.limit)
        print(f"--limit {args.limit}: keeping first {len(candidates_df)} candidates")

    print("=== Step 3: Build candidate-specific windows ===")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_missing_doc = 0
    examples_for_sanity_check = []
    with open(out_path, "w") as f:
        for row in tqdm(candidates_df.to_dict("records"), total=len(candidates_df), desc="Building candidate windows", unit="candidate"):
            document_id = row["document_id"]
            if document_id not in tokens_by_doc:
                n_missing_doc += 1
                continue

            window_tokens, dict_flags, no_space_after, target_start_window, target_end_window = build_window(
                tokens_by_doc[document_id], dict_flags_by_doc[document_id], no_space_after_by_doc[document_id],
                row["start_token_id"], row["end_token_id"],
                args.window_left, args.window_right,
            )
            record = {
                "candidate_id": f"{document_id}__s{int(row['sentence_id'])}__{int(row['start_token_id'])}-{int(row['end_token_id'])}",
                "document_id": document_id,
                "sentence_id": int(row["sentence_id"]),
                "start_token_id": int(row["start_token_id"]),
                "end_token_id": int(row["end_token_id"]),
                "split": doc_to_split.get(document_id),
                "span_text": row["entity_text"],
                "window_tokens": window_tokens,
                "target_start_window": target_start_window,
                "target_end_window": target_end_window,
                "dict_flags": dict_flags,
                "predicted_type": row["predicted_entity_type"],
                "ner_score": float(row["ner_score"]),
                "label_reliable": int(row["reliability_score"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            # Not part of the JSONL schema -- stashed only for Step 4's in-memory sanity
            # check below, added after the write above so it never reaches the file.
            record["_no_space_after"] = no_space_after
            examples_for_sanity_check.append(record)

    if n_missing_doc:
        print(f"Skipped {n_missing_doc} candidate(s) whose document_id wasn't found in --load-data")
    print(f"Wrote {n_written} candidate windows to {out_path}")

    if args.print_examples:
        print(f"=== Step 4: Sanity check -- reconstruct target span from {min(args.print_examples, len(examples_for_sanity_check))} random windows ===")
        rng = random.Random(args.seed)
        sample = rng.sample(examples_for_sanity_check, min(args.print_examples, len(examples_for_sanity_check)))
        n_mismatch = 0
        for ex in sample:
            target_tokens = ex["window_tokens"][ex["target_start_window"]:ex["target_end_window"]]
            target_no_space_after = ex["_no_space_after"][ex["target_start_window"]:ex["target_end_window"]]
            reconstructed = reconstruct_text(target_tokens, target_no_space_after)
            match = reconstructed.strip() == str(ex["span_text"]).strip()
            n_mismatch += not match
            status = "OK" if match else "MISMATCH"
            print(f"[{status}] candidate={ex['candidate_id']} span_text={ex['span_text']!r} reconstructed={reconstructed!r}")
        if n_mismatch:
            print(f"WARNING: {n_mismatch}/{len(sample)} sampled windows' reconstructed target span didn't match span_text")
        else:
            print(f"All {len(sample)} sampled windows reconstructed their target span correctly")

    print("=== Done ===")


if __name__ == "__main__":
    main()
