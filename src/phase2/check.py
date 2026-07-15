"""Quick inspection tool: prints the exact batch of tensors that gets passed into
Phase2Model.forward() (train.py/evaluate.py both build this same shape), for the first N
candidates in phase2_candidate_windows.jsonl. Useful for eyeballing the final input
format -- shapes, dtypes, padding, and per-candidate metadata -- without wiring up a full
training/evaluation run.

Usage:
    python src/phase2/check.py
    python src/phase2/check.py --n 5
    python src/phase2/check.py --n 3 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.dataset import DICT_FLAG_NAMES, SPLITS, TARGET_FLAG_NAMES, Phase2WindowDataset
from phase2.tokenize_windows import DEFAULT_ENCODER_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see build_candidate_windows.py)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="HF tokenizer name")
    parser.add_argument("--n", type=int, default=5, help="How many of the first candidates to include in the batch")
    parser.add_argument("--split", default=None, choices=list(SPLITS), help="Filter to train/val/test first (default: no filter, file order)")
    args = parser.parse_args()

    torch.set_printoptions(linewidth=200, edgeitems=6)

    print("=== Step 1: Load tokenizer and dataset ===")
    print(f"Loading {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    dataset = Phase2WindowDataset(args.windows, tokenizer, split=args.split)
    print(f"{len(dataset)} candidates available (split={args.split!r}); using the first {args.n}")

    print(f"=== Step 2: Build a batch from the first {args.n} candidates ===")
    batch = dataset.collate([dataset[i] for i in range(args.n)])

    print("=== Step 3: This is exactly what gets passed into Phase2Model.forward() ===")
    print('(model(batch["input_ids"], batch["dict_flag_ids"], batch["target_flag_ids"], '
          'batch["attention_mask"], batch["entity_type_id"], batch["ner_score"]))')
    print()
    for key in ["input_ids", "dict_flag_ids", "target_flag_ids", "attention_mask", "entity_type_id", "ner_score", "label_reliable"]:
        t = batch[key]
        print(f"{key:<16} dtype={str(t.dtype):<14} shape={tuple(t.shape)}")

    for key in ["input_ids", "dict_flag_ids", "target_flag_ids", "attention_mask"]:
        print(f"\n--- {key} ---")
        print(batch[key])

    print("\n--- entity_type_id, ner_score, label_reliable (per-candidate metadata) ---")
    for i in range(args.n):
        print(
            f"  [{i}] candidate_id={batch['candidate_id'][i]!r} span_text={batch['span_text'][i]!r} "
            f"entity_type_id={batch['entity_type_id'][i].item()} ner_score={batch['ner_score'][i].item():.4f} "
            f"label_reliable={batch['label_reliable'][i].item():.0f}"
        )

    print(f"\n=== Step 4: Decoded view -- every position INCLUDING padding, per candidate ===")
    padded_len = batch["input_ids"].shape[1]
    for i in range(args.n):
        real_len = int(batch["attention_mask"][i].sum().item())
        n_pad = padded_len - real_len
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i].tolist())
        print(f"\n--- [{i}] {batch['candidate_id'][i]} (span_text={batch['span_text'][i]!r}) "
              f"-- {real_len} real subwords + {n_pad} padding ---")
        print(f"{'pos':<5}{'token':<14}{'dict_flag':<12}{'target_flag':<16}{'attn_mask'}")
        for pos, tok in enumerate(tokens):
            dict_name = DICT_FLAG_NAMES[int(batch["dict_flag_ids"][i, pos])]
            target_name = TARGET_FLAG_NAMES[int(batch["target_flag_ids"][i, pos])]
            mask = int(batch["attention_mask"][i, pos])
            marker = "  <-- PAD" if mask == 0 else ("  <-- INSIDE_TARGET" if target_name == "INSIDE_TARGET" else "")
            print(f"{pos:<5}{tok:<14}{dict_name:<12}{target_name:<16}{mask}{marker}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
