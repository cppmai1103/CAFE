"""Phase 2 SS9 (Tokenization and subword alignment) + SS10 (Truncation rule): turn
build_candidate_windows.py's word-level windows into subword-level model inputs, using an
HF tokenizer, while guaranteeing the target span's subwords are never cut off.

Why this isn't a "write a giant file to disk" script like build_candidate_windows.py:
subword-tokenized candidates (input_ids/dict_flag_ids/target_flag_ids, one int per
subword, no padding yet since batches aren't formed here) would be several times larger
than the already-252MB word-level JSONL, and would need rebuilding every time the encoder
choice changes. tokenize_candidate_window() below is written to be imported directly by
the future training Dataset (checklist items 7+) and re-run on the fly per batch --
tokenizers are fast, so this isn't a real cost. This script itself is a verification/
statistics pass over build_candidate_windows.py's JSONL: it tokenizes every candidate,
confirms the target survives SS10's truncation rule, and reports how often
shrinking/truncation actually kicks in -- the SS33 checklist's "verify target span
survives tokenization/truncation" step.

SS9 alignment: a word may split into several subwords (Poincar6 -> Po/##in/##car/##6);
every subword inherits its whole word's dict_flag/target_flag (vocab.py). Special tokens
([CLS]/[SEP], i.e. wherever tokenizer.word_ids() is None) get SPECIAL for both.

SS10 truncation rule: the target span must NEVER be truncated. tokenize_candidate_window
tokenizes the untruncated window first; if it's over --max-length subwords, it shrinks
context word-by-word (alternating right/left, whichever currently has more context left)
and retokenizes, repeating until it fits -- the target span itself is never touched by
this shrinking. Only in the pathological case where the target ALONE already exceeds
--max-length subwords (essentially never on real newspaper NER spans) does the tokenizer's
own truncation=True get used as a fallback, and that candidate is flagged
target_truncated=True in the stats below so it's visible rather than silently wrong.

Usage:
    python src/phase2/tokenize_windows.py
    python src/phase2/tokenize_windows.py --encoder-name camembert-base --max-length 256
    python src/phase2/tokenize_windows.py --limit 500 --print-examples 20  # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.vocab import DICT_FLAG_VOCAB, TARGET_FLAG_VOCAB

DEFAULT_ENCODER_NAME = "camembert-base"
DEFAULT_MAX_LENGTH = 256


def tokenize_candidate_window(
    tokenizer: PreTrainedTokenizerBase,
    window_tokens: list[str], dict_flags: list[str],
    target_start_window: int, target_end_window: int,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    """SS9+SS10. Returns a dict of model-ready fields for one candidate:
    input_ids, dict_flag_ids, target_flag_ids, attention_mask (all 1s -- no padding at
    this single-example level, that's a batch-collation concern), n_subwords,
    n_left_dropped, n_right_dropped (how much context SS10 had to shrink), and
    target_truncated (True only in the pathological target-alone-too-long case)."""
    left = list(window_tokens[:target_start_window])
    left_flags = list(dict_flags[:target_start_window])
    target = list(window_tokens[target_start_window:target_end_window])
    target_flags = list(dict_flags[target_start_window:target_end_window])
    right = list(window_tokens[target_end_window:])
    right_flags = list(dict_flags[target_end_window:])

    n_left_start, n_right_start = len(left), len(right)

    while True:
        words = left + target + right
        word_dict_flags = left_flags + target_flags + right_flags
        target_start = len(left)
        target_end = len(left) + len(target)

        encoded = tokenizer(words, is_split_into_words=True, truncation=False)
        n_subwords = len(encoded["input_ids"])

        if n_subwords <= max_length:
            target_truncated = False
            break
        if not left and not right:
            # Pathological: the target span alone already exceeds max_length subwords.
            # SS10 has no answer for this case -- fall back to the tokenizer's own
            # truncation (default: cuts from the end) and flag it loudly rather than
            # silently producing an over-length sequence.
            encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
            target_truncated = True
            break

        # SS10: shrink context, never the target -- take from whichever side still has
        # more context left, so both sides get worn down roughly evenly.
        if len(right) >= len(left) and right:
            right, right_flags = right[:-1], right_flags[:-1]
        else:
            left, left_flags = left[1:], left_flags[1:]

    word_ids = encoded.word_ids()
    dict_flag_ids, target_flag_ids = [], []
    for word_id in word_ids:
        if word_id is None:
            dict_flag_ids.append(DICT_FLAG_VOCAB["SPECIAL"])
            target_flag_ids.append(TARGET_FLAG_VOCAB["SPECIAL"])
        else:
            dict_flag_ids.append(DICT_FLAG_VOCAB[word_dict_flags[word_id]])
            flag = "INSIDE_TARGET" if target_start <= word_id < target_end else "OUTSIDE"
            target_flag_ids.append(TARGET_FLAG_VOCAB[flag])

    return {
        "input_ids": encoded["input_ids"],
        "dict_flag_ids": dict_flag_ids,
        "target_flag_ids": target_flag_ids,
        "attention_mask": [1] * len(encoded["input_ids"]),
        "n_subwords": len(encoded["input_ids"]),
        "n_left_dropped": n_left_start - len(left),
        "n_right_dropped": n_right_start - len(right),
        "target_truncated": target_truncated,
    }


def target_survived(result: dict) -> bool:
    """SS33's sanity check: at least one subword must still be flagged INSIDE_TARGET,
    and (unless the pathological fallback fired) none of the target's subwords may have
    been dropped by shrinking -- shrinking only ever removes from left/right context."""
    if result["target_truncated"]:
        return False
    return TARGET_FLAG_VOCAB["INSIDE_TARGET"] in result["target_flag_ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see build_candidate_windows.py)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="HF tokenizer/encoder name (docs/phase2_learned_features.md SS11: CamemBERT or XLM-R)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max subword length (docs/phase2_learned_features.md SS10 default: 256)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N candidates (smoke test)")
    parser.add_argument("--print-examples", type=int, default=20, help="Print this many random examples (0 to skip)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --print-examples sampling")
    args = parser.parse_args()

    print("=== Step 1: Load tokenizer ===")
    print(f"Loading {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    print("=== Step 2: Load candidate windows ===")
    print(f"Loading {args.windows}")
    with open(args.windows) as f:
        lines = f.readlines()
    if args.limit is not None:
        lines = lines[: args.limit]
    print(f"{len(lines)} candidate windows")

    print("=== Step 3: Tokenize every candidate, verify the target survives SS10 truncation ===")
    n_shrunk = 0
    n_target_truncated = 0
    n_target_missing = 0
    subword_lengths = []
    examples = []
    for line in tqdm(lines, desc="Tokenizing candidate windows", unit="candidate"):
        record = json.loads(line)
        result = tokenize_candidate_window(
            tokenizer, record["window_tokens"], record["dict_flags"],
            record["target_start_window"], record["target_end_window"],
            max_length=args.max_length,
        )
        subword_lengths.append(result["n_subwords"])
        if result["n_left_dropped"] or result["n_right_dropped"]:
            n_shrunk += 1
        if result["target_truncated"]:
            n_target_truncated += 1
        if not target_survived(result):
            n_target_missing += 1
        examples.append((record, result))

    print(f"Subword length: min={min(subword_lengths)} max={max(subword_lengths)} mean={sum(subword_lengths) / len(subword_lengths):.1f}")
    print(f"{n_shrunk}/{len(lines)} candidates needed SS10 context shrinking to fit --max-length {args.max_length}")
    print(f"{n_target_truncated}/{len(lines)} candidates hit the pathological target-alone-too-long fallback")
    if n_target_missing:
        print(f"WARNING: {n_target_missing}/{len(lines)} candidates' target span did NOT survive tokenization/truncation")
    else:
        print(f"All {len(lines)} candidates' target span survived tokenization/truncation")

    if args.print_examples:
        print(f"=== Step 4: Print {min(args.print_examples, len(examples))} random examples ===")
        rng = random.Random(args.seed)
        for record, result in rng.sample(examples, min(args.print_examples, len(examples))):
            target_subword_ids = [i for i, t in enumerate(result["target_flag_ids"]) if t == TARGET_FLAG_VOCAB["INSIDE_TARGET"]]
            target_tokens = tokenizer.convert_ids_to_tokens([result["input_ids"][i] for i in target_subword_ids])
            status = "OK" if target_survived(result) else "MISSING"
            print(
                f"[{status}] candidate={record['candidate_id']} span_text={record['span_text']!r} "
                f"n_subwords={result['n_subwords']} dropped(L/R)={result['n_left_dropped']}/{result['n_right_dropped']} "
                f"target_subwords={target_tokens}"
            )

    print("=== Done ===")


if __name__ == "__main__":
    main()
