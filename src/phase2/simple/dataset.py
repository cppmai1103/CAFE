"""PyTorch Dataset + batch collation for phase2_simple, on top of phase2's shared
build_candidate_windows.py JSONL and this folder's own tokenize_windows.py (the marker-
block pair-encoding scheme -- see model.py's module docstring for the full picture).

Split roles are identical to phase2/dataset.py's SPLITS (same document-level train/val/
test split, see preprocessing/preprocessing_data.py) -- reused verbatim so the two models
are trained and evaluated on exactly the same candidates.

Padding: input_ids pad with the tokenizer's own pad_token_id, token_type_ids/span_mask
pad with 0, attention_mask pads with 0. sep_pos/entity_pos/type_pos/confidence_pos are
single absolute positions per example, valid as-is under right-padding (no offset needed).

Usage (quality-check mode -- prints a decoded sample batch, doesn't train anything):
    python src/phase2/simple/dataset.py
    python src/phase2/simple/dataset.py --split train --batch-size 4 --print-batches 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.simple.tokenize_windows import DEFAULT_ENCODER_NAME, DEFAULT_MAX_LENGTH, tokenize_candidate_window

SPLITS = {
    "train": ("train",),
    "val": ("val",),
    "test": ("test",),
}


class Phase2SimpleWindowDataset(Dataset):
    def __init__(
        self, windows_path: str | Path, tokenizer, split: str | None = None, max_length: int = DEFAULT_MAX_LENGTH,
        type_display_name: dict[str, str] | None = None,
    ):
        """split: one of SPLITS's keys ("train"/"val"/"test"), or None to keep every
        candidate regardless of split. type_display_name: None (default) uses the
        standard HIPE-2022 TYPE_DISPLAY_NAME (see tokenize_windows.py); pass
        vocab.py's type_display_name_from_file() output if the windows file's
        predicted_type values come from a different NER source's own tagset (e.g.
        ajmc -- must match the display-name map the paired model's encoder text was
        actually trained on, see train.py's --labels-file)."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.type_display_name = type_display_name

        allowed_splits = set(SPLITS[split]) if split is not None else None
        records = []
        with open(windows_path) as f:
            for line in f:
                record = json.loads(line)
                if allowed_splits is None or record["split"] in allowed_splits:
                    records.append(record)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        span_words = record["window_tokens"][record["target_start_window"]:record["target_end_window"]]
        encoded = tokenize_candidate_window(
            self.tokenizer, record["window_tokens"], record["target_start_window"], record["target_end_window"],
            span_words, record["predicted_type"], record["ner_score"], max_length=self.max_length,
            type_display_name=self.type_display_name,
        )
        return {
            "candidate_id": record["candidate_id"],
            "document_id": record["document_id"],
            "sentence_id": record["sentence_id"],
            "start_token_id": record["start_token_id"],
            "end_token_id": record["end_token_id"],
            "split": record["split"],
            "span_text": record["span_text"],
            "input_ids": encoded["input_ids"],
            "token_type_ids": encoded["token_type_ids"],
            "attention_mask": encoded["attention_mask"],
            "sep_pos": encoded["sep_pos"],
            "entity_pos": encoded["entity_pos"],
            "type_pos": encoded["type_pos"],
            "confidence_pos": encoded["confidence_pos"],
            "span_mask": encoded["span_mask"],
            "type_value_mask": encoded["type_value_mask"],
            "confidence_value_mask": encoded["confidence_value_mask"],
            "ner_score": record["ner_score"],
            "label_reliable": record["label_reliable"],
        }

    def collate(self, batch: list[dict]) -> dict:
        """Pads every variable-length field to the batch's own max subword length --
        never to self.max_length, so short batches stay cheap."""
        lengths = [len(item["input_ids"]) for item in batch]
        max_len = max(lengths)
        pad_token_id = self.tokenizer.pad_token_id

        def pad(seq: list[int], pad_value: int) -> list[int]:
            return seq + [pad_value] * (max_len - len(seq))

        return {
            "candidate_id": [item["candidate_id"] for item in batch],
            "document_id": [item["document_id"] for item in batch],
            "sentence_id": torch.tensor([item["sentence_id"] for item in batch], dtype=torch.long),
            "start_token_id": torch.tensor([item["start_token_id"] for item in batch], dtype=torch.long),
            "end_token_id": torch.tensor([item["end_token_id"] for item in batch], dtype=torch.long),
            "split": [item["split"] for item in batch],
            "span_text": [item["span_text"] for item in batch],
            "input_ids": torch.tensor([pad(item["input_ids"], pad_token_id) for item in batch], dtype=torch.long),
            "token_type_ids": torch.tensor([pad(item["token_type_ids"], 0) for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in batch], dtype=torch.long),
            "sep_pos": torch.tensor([item["sep_pos"] for item in batch], dtype=torch.long),
            "entity_pos": torch.tensor([item["entity_pos"] for item in batch], dtype=torch.long),
            "type_pos": torch.tensor([item["type_pos"] for item in batch], dtype=torch.long),
            "confidence_pos": torch.tensor([item["confidence_pos"] for item in batch], dtype=torch.long),
            "span_mask": torch.tensor([pad(item["span_mask"], 0) for item in batch], dtype=torch.float32),
            "type_value_mask": torch.tensor([pad(item["type_value_mask"], 0) for item in batch], dtype=torch.float32),
            "confidence_value_mask": torch.tensor([pad(item["confidence_value_mask"], 0) for item in batch], dtype=torch.float32),
            "ner_score": torch.tensor([item["ner_score"] for item in batch], dtype=torch.float32),
            "label_reliable": torch.tensor([item["label_reliable"] for item in batch], dtype=torch.float32),
        }


def print_sample_batch(dataset: Phase2SimpleWindowDataset, batch: dict, n: int) -> None:
    """Quality check: decode each example's subwords and mark the 4 pooled positions +
    span mask, so misalignment would be visible at a glance."""
    for i in range(min(n, len(batch["candidate_id"]))):
        length = int(batch["attention_mask"][i].sum().item())
        tokens = dataset.tokenizer.convert_ids_to_tokens(batch["input_ids"][i, :length].tolist())

        print(f"--- {batch['candidate_id'][i]} (span_text={batch['span_text'][i]!r}, split={batch['split'][i]}) ---")
        print(f"ner_score={batch['ner_score'][i].item():.3f} label_reliable={int(batch['label_reliable'][i].item())}")
        sep_pos, entity_pos, type_pos, confidence_pos = (int(batch[k][i]) for k in ("sep_pos", "entity_pos", "type_pos", "confidence_pos"))
        for pos, tok in enumerate(tokens):
            marker = ""
            if pos == 0:
                marker = "  <-- CLS"
            elif pos == sep_pos:
                marker = "  <-- SEP"
            elif pos == entity_pos:
                marker = "  <-- ENTITY"
            elif pos == type_pos:
                marker = "  <-- TYPE"
            elif pos == confidence_pos:
                marker = "  <-- CONFIDENCE"
            elif batch["span_mask"][i, pos]:
                marker = "  <-- SPAN"
            elif batch["type_value_mask"][i, pos]:
                marker = "  <-- TYPE_VALUE"
            elif batch["confidence_value_mask"][i, pos]:
                marker = "  <-- CONFIDENCE_VALUE"
            print(f"{pos:<5}{tok:<14}{marker}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see phase2/build_candidate_windows.py)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="HF tokenizer name")
    parser.add_argument("--split", default=None, choices=list(SPLITS), help="Filter to train/val/test (default: no filter, every candidate)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max subword length per candidate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the printed sample")
    parser.add_argument("--print-batches", type=int, default=1, help="How many batches to print (0 to just report dataset size)")
    parser.add_argument("--seed", type=int, default=42, help="DataLoader shuffle seed")
    args = parser.parse_args()

    print("=== Step 1: Load tokenizer ===")
    print(f"Loading {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    print("=== Step 2: Build dataset ===")
    dataset = Phase2SimpleWindowDataset(args.windows, tokenizer, split=args.split, max_length=args.max_length)
    print(f"{len(dataset)} candidates (split={args.split!r})")

    if args.print_batches:
        print(f"=== Step 3: Print {args.print_batches} sample batch(es), batch_size={args.batch_size} ===")
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate, generator=generator)
        for i, batch in enumerate(loader):
            if i >= args.print_batches:
                break
            print(f"##### Batch {i} (padded length={batch['input_ids'].shape[1]}) #####")
            print_sample_batch(dataset, batch, n=args.batch_size)

    print("=== Done ===")


if __name__ == "__main__":
    main()
