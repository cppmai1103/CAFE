"""Quick inspection tool: prints the exact batch of tensors that gets passed into
Phase2SimpleModel.forward() (train.py/evaluate.py both build this same shape), for the
first N candidates in phase2_candidate_windows.jsonl. Useful for eyeballing the final
marker-block input format -- shapes, dtypes, padding, and which position each pooled
vector (h_cls/h_span/h_sep/h_entity/h_type/h_confidence) reads from -- without wiring up a
full training/evaluation run.

Usage:
    python src/phase2/simple/check.py
    python src/phase2/simple/check.py --n 5
    python src/phase2/simple/check.py --n 3 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.simple.dataset import SPLITS, Phase2SimpleWindowDataset
from phase2.simple.tokenize_windows import DEFAULT_ENCODER_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see phase2/build_candidate_windows.py)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="HF tokenizer name")
    parser.add_argument("--n", type=int, default=5, help="How many of the first candidates to include in the batch")
    parser.add_argument("--split", default=None, choices=list(SPLITS), help="Filter to train/val/test first (default: no filter, file order)")
    args = parser.parse_args()

    torch.set_printoptions(linewidth=200, edgeitems=6)

    print("=== Step 1: Load tokenizer and dataset ===")
    print(f"Loading {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    dataset = Phase2SimpleWindowDataset(args.windows, tokenizer, split=args.split)
    print(f"{len(dataset)} candidates available (split={args.split!r}); using the first {args.n}")

    print(f"=== Step 2: Build a batch from the first {args.n} candidates ===")
    batch = dataset.collate([dataset[i] for i in range(args.n)])

    print("=== Step 3: This is exactly what gets passed into Phase2SimpleModel.forward() ===")
    print('(model(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"], '
          'batch["sep_pos"], batch["entity_pos"], batch["type_pos"], batch["confidence_pos"], '
          'batch["span_mask"], batch["type_value_mask"], batch["confidence_value_mask"]))')
    print("(type_pos/confidence_pos are only used in type_confidence_pool='one' mode; "
          "type_value_mask/confidence_value_mask only in 'average' mode -- see model.py)")
    print()
    for key in ["input_ids", "token_type_ids", "attention_mask", "sep_pos", "entity_pos", "type_pos", "confidence_pos", "span_mask", "type_value_mask", "confidence_value_mask", "ner_score", "label_reliable"]:
        t = batch[key]
        print(f"{key:<16} dtype={str(t.dtype):<14} shape={tuple(t.shape)}")

    print("\n--- ner_score, label_reliable (per-candidate metadata) ---")
    for i in range(args.n):
        print(
            f"  [{i}] candidate_id={batch['candidate_id'][i]!r} span_text={batch['span_text'][i]!r} "
            f"ner_score={batch['ner_score'][i].item():.4f} label_reliable={batch['label_reliable'][i].item():.0f}"
        )

    print(f"\n=== Step 4: Decoded view -- every position INCLUDING padding, per candidate ===")
    padded_len = batch["input_ids"].shape[1]
    for i in range(args.n):
        real_len = int(batch["attention_mask"][i].sum().item())
        n_pad = padded_len - real_len
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i].tolist())
        sep_pos, entity_pos, type_pos, confidence_pos = (int(batch[k][i]) for k in ("sep_pos", "entity_pos", "type_pos", "confidence_pos"))
        print(f"\n--- [{i}] {batch['candidate_id'][i]} (span_text={batch['span_text'][i]!r}) "
              f"-- {real_len} real subwords + {n_pad} padding ---")
        print(f"{'pos':<5}{'token':<14}{'role':<14}{'attn_mask'}")
        for pos, tok in enumerate(tokens):
            mask = int(batch["attention_mask"][i, pos])
            role = ""
            if pos == 0:
                role = "CLS"
            elif pos == sep_pos:
                role = "SEP"
            elif pos == entity_pos:
                role = "ENTITY"
            elif pos == type_pos:
                role = "TYPE"
            elif pos == confidence_pos:
                role = "CONFIDENCE"
            elif batch["span_mask"][i, pos]:
                role = "SPAN"
            elif batch["type_value_mask"][i, pos]:
                role = "TYPE_VALUE"
            elif batch["confidence_value_mask"][i, pos]:
                role = "CONFIDENCE_VALUE"
            marker = "  <-- PAD" if mask == 0 else (f"  <-- {role}" if role else "")
            print(f"{pos:<5}{tok:<14}{role:<14}{mask}{marker}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
