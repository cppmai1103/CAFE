"""phase2_simple's tokenization: a pair-encoded prompt instead of phase2's word-level
dict_flag/target_flag side-channel alignment (see model.py's module docstring for the
full input-format rationale).

Given one candidate window (build_candidate_windows.py's JSONL, reused unmodified -- see
DEFAULT_WINDOWS below, which points at phase2's shared output, not a phase2_simple-local
copy), this builds a tokenizer *pair*:
    text_a = window_tokens              (natural-flow context, target embedded, unmarked)
    text_b = [Entity] <span words> [\\Entity] [Type] <type word> [\\Type]
             [Confidence] <confidence as text> [\\Confidence]
tokenizer(text_a, text_b, is_split_into_words=True) then produces the encoder input
    <s> window context </s></s> [Entity] span [\\Entity] [Type] Person [\\Type]
        [Confidence] 0.98 [\\Confidence] </s>
(exact special tokens depend on the encoder family -- CamemBERT/RoBERTa-style shown above,
BERT-style is [CLS] ... [SEP] ... [SEP]) -- see model.py for how the 6 pooled positions
(h_cls/h_span/h_sep/h_entity/h_type/h_confidence) are read back out of this.

Why text_b's word list is built the way it is (see MARKER_WORD_OFFSETS below): each
bracketed tag ("[Entity]", "[\\Type]", ...) is ordinary text, not an added special token
-- it tokenizes into a handful of ordinary subwords via the frozen tokenizer's existing
vocabulary, no tokenizer.add_special_tokens/encoder.resize_token_embeddings needed. Every
tag's FIRST subword is what model.py pools from (word_ids()==the tag's word index, first
matching token) -- verified empirically (see git history) that a bidirectional encoder's
self-attention lets that first subword's hidden state summarize the whole tag+value it
introduces, which is the standard "marker token as feature" trick.

SS10-equivalent truncation rule, simplified vs phase2/tokenize_windows.py: only text_a
(context) is ever shrunk (same left/right-alternating heuristic, centered on the original
target position purely as a "keep the most relevant nearby words" heuristic -- the target
words themselves within text_a are never dropped by this loop, same as before). text_b
(the entity block) is *architecturally* never truncated: even in the pathological case
where text_a-shrunk-to-just-the-target-words plus text_b still exceeds max_length, the
tokenizer's own truncation="only_first" fallback is used, which is guaranteed to cut only
text_a. Unlike phase2's version, there is therefore no "did the target survive
truncation" failure mode to check for -- text_b holds an authoritative, always-intact copy
of the span/type/confidence, independent of whatever happens to text_a.

Usage:
    python src/phase2/simple/tokenize_windows.py
    python src/phase2/simple/tokenize_windows.py --encoder-name camembert-base --max-length 256
    python src/phase2/simple/tokenize_windows.py --limit 500 --print-examples 20  # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.simple.vocab import (
    CONFIDENCE_CLOSE, CONFIDENCE_OPEN, ENTITY_CLOSE, ENTITY_OPEN, TYPE_CLOSE, TYPE_DISPLAY_NAME, TYPE_OPEN,
)

DEFAULT_ENCODER_NAME = "bert-base-multilingual-cased"
DEFAULT_MAX_LENGTH = 256


def build_entity_block_words(
    span_words: list[str], predicted_type: str, ner_score: float, type_display_name: dict[str, str] | None = None,
) -> list[str]:
    """text_b's word list, plus (see MARKER_WORD_OFFSETS) the fixed word-index of every
    tag this candidate needs -- deterministic since span_words' length is the only
    variable part. type_display_name: None (default) uses the standard HIPE-2022
    TYPE_DISPLAY_NAME; pass vocab.py's type_display_name_from_file() output for a
    different NER source's own tagset (see dataset.py's same parameter)."""
    type_display_name = type_display_name if type_display_name is not None else TYPE_DISPLAY_NAME
    type_word = type_display_name[predicted_type]
    confidence_word = f"{ner_score:.2f}"
    return [
        ENTITY_OPEN, *span_words, ENTITY_CLOSE,
        TYPE_OPEN, type_word, TYPE_CLOSE,
        CONFIDENCE_OPEN, confidence_word, CONFIDENCE_CLOSE,
    ]


def marker_word_indices(n_span_words: int) -> dict[str, int]:
    """Word index (within text_b, i.e. what tokenizer.word_ids() reports for sequence 1)
    of each tag/value this candidate needs to pool. entity_open=0, span=[1, n_span_words],
    everything else follows at a fixed offset past the span. type_value/confidence_value
    are each a single word (see build_entity_block_words) -- model.py's "average" pool
    mode mean-pools over that one word's subwords, same idea as span_mask for h_span."""
    return {
        "entity_open": 0,
        "span_start": 1,
        "span_end": 1 + n_span_words,  # exclusive
        "type_open": 2 + n_span_words,
        "type_value": 3 + n_span_words,
        "confidence_open": 5 + n_span_words,
        "confidence_value": 6 + n_span_words,
    }


def tokenize_candidate_window(
    tokenizer: PreTrainedTokenizerBase,
    window_tokens: list[str], target_start_window: int, target_end_window: int,
    span_words: list[str], predicted_type: str, ner_score: float,
    max_length: int = DEFAULT_MAX_LENGTH,
    type_display_name: dict[str, str] | None = None,
) -> dict:
    """Returns input_ids/attention_mask/token_type_ids plus everything model.py's two
    type/confidence pooling modes need: the single-token positions
    sep_pos/entity_pos/type_pos/confidence_pos (mode "one" -- gather that one tag token),
    and span_mask/type_value_mask/confidence_value_mask (mode "average" -- mean-pool over
    the value's own subwords, 1 at every text_b token belonging to that word, 0 elsewhere).
    Both are always computed; which one a given model actually uses is a config choice on
    the model, not something this tokenizer needs to know about. n_subwords,
    n_left_dropped, n_right_dropped, context_truncated mirror phase2/tokenize_windows.py's
    stats fields for the SS33-style sanity pass below."""
    left = list(window_tokens[:target_start_window])
    target = list(window_tokens[target_start_window:target_end_window])
    right = list(window_tokens[target_end_window:])
    n_left_start, n_right_start = len(left), len(right)

    entity_block = build_entity_block_words(span_words, predicted_type, ner_score, type_display_name=type_display_name)
    idx = marker_word_indices(len(span_words))

    while True:
        text_a = left + target + right
        encoded = tokenizer(text_a, entity_block, is_split_into_words=True, truncation=False)
        n_subwords = len(encoded["input_ids"])

        if n_subwords <= max_length:
            context_truncated = False
            break
        if not left and not right:
            # Pathological: target words + entity block alone already exceed max_length.
            # truncation="only_first" is guaranteed to cut only text_a -- text_b (the
            # entity block, which is what pooling actually reads) is never touched.
            encoded = tokenizer(text_a, entity_block, is_split_into_words=True, truncation="only_first", max_length=max_length)
            context_truncated = True
            break

        if len(right) >= len(left) and right:
            right = right[:-1]
        else:
            left = left[1:]

    seq_ids = encoded.sequence_ids()
    word_ids = encoded.word_ids()

    sep_pos = next(i for i, s in enumerate(seq_ids) if s is None and i > 0)
    entity_pos = next(i for i, (s, w) in enumerate(zip(seq_ids, word_ids)) if s == 1 and w == idx["entity_open"])
    type_pos = next(i for i, (s, w) in enumerate(zip(seq_ids, word_ids)) if s == 1 and w == idx["type_open"])
    confidence_pos = next(i for i, (s, w) in enumerate(zip(seq_ids, word_ids)) if s == 1 and w == idx["confidence_open"])
    span_mask = [1 if (s == 1 and w is not None and idx["span_start"] <= w < idx["span_end"]) else 0 for s, w in zip(seq_ids, word_ids)]
    type_value_mask = [1 if (s == 1 and w == idx["type_value"]) else 0 for s, w in zip(seq_ids, word_ids)]
    confidence_value_mask = [1 if (s == 1 and w == idx["confidence_value"]) else 0 for s, w in zip(seq_ids, word_ids)]

    return {
        "input_ids": encoded["input_ids"],
        "token_type_ids": encoded.get("token_type_ids", [0] * len(encoded["input_ids"])),
        "attention_mask": encoded["attention_mask"],
        "sep_pos": sep_pos,
        "entity_pos": entity_pos,
        "type_pos": type_pos,
        "confidence_pos": confidence_pos,
        "span_mask": span_mask,
        "type_value_mask": type_value_mask,
        "confidence_value_mask": confidence_value_mask,
        "n_subwords": n_subwords if not context_truncated else len(encoded["input_ids"]),
        "n_left_dropped": n_left_start - len(left),
        "n_right_dropped": n_right_start - len(right),
        "context_truncated": context_truncated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see phase2/build_candidate_windows.py -- shared with phase2, not regenerated here)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="HF tokenizer/encoder name")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max combined (text_a + text_b) subword length")
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

    print("=== Step 3: Tokenize every candidate ===")
    n_shrunk = 0
    n_context_truncated = 0
    subword_lengths = []
    examples = []
    for line in tqdm(lines, desc="Tokenizing candidate windows", unit="candidate"):
        record = json.loads(line)
        span_words = record["window_tokens"][record["target_start_window"]:record["target_end_window"]]
        result = tokenize_candidate_window(
            tokenizer, record["window_tokens"], record["target_start_window"], record["target_end_window"],
            span_words, record["predicted_type"], record["ner_score"], max_length=args.max_length,
        )
        subword_lengths.append(result["n_subwords"])
        if result["n_left_dropped"] or result["n_right_dropped"]:
            n_shrunk += 1
        if result["context_truncated"]:
            n_context_truncated += 1
        examples.append((record, result))

    print(f"Subword length: min={min(subword_lengths)} max={max(subword_lengths)} mean={sum(subword_lengths) / len(subword_lengths):.1f}")
    print(f"{n_shrunk}/{len(lines)} candidates needed context shrinking to fit --max-length {args.max_length}")
    print(f"{n_context_truncated}/{len(lines)} candidates hit the pathological only_first fallback (entity block itself always intact)")

    if args.print_examples:
        print(f"=== Step 4: Print {min(args.print_examples, len(examples))} random examples ===")
        rng = random.Random(args.seed)
        for record, result in rng.sample(examples, min(args.print_examples, len(examples))):
            tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
            span_tokens = [t for t, m in zip(tokens, result["span_mask"]) if m]
            print(
                f"candidate={record['candidate_id']} span_text={record['span_text']!r} n_subwords={result['n_subwords']} "
                f"dropped(L/R)={result['n_left_dropped']}/{result['n_right_dropped']} "
                f"sep={tokens[result['sep_pos']]!r} entity={tokens[result['entity_pos']]!r} "
                f"type={tokens[result['type_pos']]!r} confidence={tokens[result['confidence_pos']]!r} span_subwords={span_tokens}"
            )

    print("=== Done ===")


if __name__ == "__main__":
    main()
