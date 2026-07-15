"""PyTorch Dataset + batch collation for Phase 2, on top of build_candidate_windows.py's
JSONL and tokenize_windows.py's tokenize_candidate_window(). Tokenization happens per
item, on the fly (__getitem__), not precomputed to disk -- see tokenize_windows.py's
module docstring for why.

Split roles reuse the project's document-level train (70%) / val (10%) / test (20%)
split (see preprocessing/preprocessing_data.py) directly -- SPLITS exists mainly so every
Phase 2 script shares one place that names the three split values.

Padding: input_ids pad with the tokenizer's own pad_token_id, dict_flag_ids/
target_flag_ids pad with their vocab's PAD id (0 in both, see vocab.py), attention_mask
pads with 0 -- so padded positions are attended to by nothing and their flag embeddings
are never trained.

Usage (quality-check mode -- prints a decoded sample batch, doesn't train anything):
    python src/phase2/dataset.py
    python src/phase2/dataset.py --split train --batch-size 4 --print-batches 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.tokenize_windows import DEFAULT_ENCODER_NAME, DEFAULT_MAX_LENGTH, tokenize_candidate_window
from phase2.vocab import DICT_FLAG_VOCAB, ENTITY_TYPE_VOCAB, TARGET_FLAG_VOCAB

SPLITS = {
    "train": ("train",),
    "val": ("val",),
    "test": ("test",),
}

DICT_FLAG_NAMES = {v: k for k, v in DICT_FLAG_VOCAB.items()}
TARGET_FLAG_NAMES = {v: k for k, v in TARGET_FLAG_VOCAB.items()}


class Phase2WindowDataset(Dataset):
    def __init__(
        self, windows_path: str | Path, tokenizer, split: str | None = None, max_length: int = DEFAULT_MAX_LENGTH,
        splits: dict[str, tuple[str, ...]] = SPLITS,
    ):
        """split: one of splits's keys ("train"/"val"/"test"), or None to keep every
        candidate regardless of split. splits defaults to the module-level SPLITS
        (train/val/test, see preprocessing_data.py) -- only override it if the windows
        file on disk was built with different raw split labels than what SPLITS expects
        (e.g. mid-migration to a new split scheme)."""
        self.tokenizer = tokenizer
        self.max_length = max_length

        allowed_splits = set(splits[split]) if split is not None else None
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
        encoded = tokenize_candidate_window(
            self.tokenizer, record["window_tokens"], record["dict_flags"],
            record["target_start_window"], record["target_end_window"],
            max_length=self.max_length,
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
            "dict_flag_ids": encoded["dict_flag_ids"],
            "target_flag_ids": encoded["target_flag_ids"],
            "attention_mask": encoded["attention_mask"],
            "entity_type_id": ENTITY_TYPE_VOCAB[record["predicted_type"]],
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
            "dict_flag_ids": torch.tensor([pad(item["dict_flag_ids"], DICT_FLAG_VOCAB["PAD"]) for item in batch], dtype=torch.long),
            "target_flag_ids": torch.tensor([pad(item["target_flag_ids"], TARGET_FLAG_VOCAB["PAD"]) for item in batch], dtype=torch.long),
            "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in batch], dtype=torch.long),
            "entity_type_id": torch.tensor([item["entity_type_id"] for item in batch], dtype=torch.long),
            "ner_score": torch.tensor([item["ner_score"] for item in batch], dtype=torch.float32),
            "label_reliable": torch.tensor([item["label_reliable"] for item in batch], dtype=torch.float32),
        }


def print_sample_batch(dataset: Phase2WindowDataset, batch: dict, n: int) -> None:
    """Quality check: decode each example's subwords and print them aligned with their
    dict_flag/target_flag names, so misalignment (SS9) would be visible at a glance."""
    for i in range(min(n, len(batch["candidate_id"]))):
        length = int(batch["attention_mask"][i].sum().item())
        tokens = dataset.tokenizer.convert_ids_to_tokens(batch["input_ids"][i, :length].tolist())
        dict_flags = [DICT_FLAG_NAMES[x] for x in batch["dict_flag_ids"][i, :length].tolist()]
        target_flags = [TARGET_FLAG_NAMES[x] for x in batch["target_flag_ids"][i, :length].tolist()]

        print(f"--- {batch['candidate_id'][i]} (span_text={batch['span_text'][i]!r}, split={batch['split'][i]}) ---")
        print(f"entity_type_id={batch['entity_type_id'][i].item()} ner_score={batch['ner_score'][i].item():.3f} label_reliable={int(batch['label_reliable'][i].item())}")
        header = f"{'token':<14}{'dict_flag':<12}{'target_flag':<14}"
        print(header)
        for tok, df, tf in zip(tokens, dict_flags, target_flags):
            marker = "  <-- INSIDE_TARGET" if tf == "INSIDE_TARGET" else ""
            print(f"{tok:<14}{df:<12}{tf:<14}{marker}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see build_candidate_windows.py)")
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
    dataset = Phase2WindowDataset(args.windows, tokenizer, split=args.split, max_length=args.max_length)
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
