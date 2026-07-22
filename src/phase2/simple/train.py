"""phase2_simple's training loop -- structurally identical to phase2/train.py (same
split roles, same early-stopping-on-val-loss, same BCEWithLogitsLoss), adapted only for
Phase2SimpleModel's forward() signature (no dict_flag/target_flag/entity_type/ner_score
tensors -- the marker-block pair-encoding already put type/confidence into input_ids, see
model.py/tokenize_windows.py) and its much smaller trainable surface (classifier head
only, no side embeddings -- so no ablation flags either, there's nothing left to ablate).

Since the classifier is the only trainable component and the encoder forward pass never
needs a gradient to flow *through* it into anything trainable, validation AND training
forward passes could both technically run under torch.no_grad() up to the encoder output
-- but backward() still needs the encoder's *output* activations kept in the graph to
reach the classifier, so the encoder call itself stays outside no_grad() during training,
same structure as phase2/train.py, just simpler to reason about (no side-embedding inputs
feeding the frozen computation).

Usage:
    python src/phase2/simple/train.py
    python src/phase2/simple/train.py --batch-size 32 --lr 3e-4 --max-epochs 30
    python src/phase2/simple/train.py --limit 40 --max-epochs 2 --batch-size 4  # smoke test
    python src/phase2/simple/train.py --encoder-name xlm-roberta-base
"""

from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase1.modeling.training_curve import plot_training_curve
from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.simple.dataset import Phase2SimpleWindowDataset
from phase2.simple.model import (
    DEFAULT_ENCODER_NAME, DEFAULT_HEAD_DROPOUT, DEFAULT_HEAD_HIDDEN, DEFAULT_TYPE_CONFIDENCE_POOL,
    TYPE_CONFIDENCE_POOL_CHOICES, Phase2SimpleModel, print_parameter_breakdown, save_checkpoint, variant_name,
)
from phase2.simple.tokenize_windows import DEFAULT_MAX_LENGTH
from phase2.simple.vocab import type_display_name_from_file

CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent / "checkpoints" / "phase2_simple"
DEFAULT_CHECKPOINT_OUT = CHECKPOINTS_DIR / f"{variant_name()}.pt"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "figures" / "phase2_simple"


def run_epoch(model: Phase2SimpleModel, loader: DataLoader, loss_fn, optimizer, device: str, train: bool, desc: str) -> float:
    model.train(train)
    total_loss, total_n = 0.0, 0
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sep_pos = batch["sep_pos"].to(device)
        entity_pos = batch["entity_pos"].to(device)
        type_pos = batch["type_pos"].to(device)
        confidence_pos = batch["confidence_pos"].to(device)
        span_mask = batch["span_mask"].to(device)
        type_value_mask = batch["type_value_mask"].to(device)
        confidence_value_mask = batch["confidence_value_mask"].to(device)
        label = batch["label_reliable"].to(device)

        if train:
            optimizer.zero_grad()
            logits = model(input_ids, token_type_ids, attention_mask, sep_pos, entity_pos, type_pos, confidence_pos, span_mask, type_value_mask, confidence_value_mask)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(input_ids, token_type_ids, attention_mask, sep_pos, entity_pos, type_pos, confidence_pos, span_mask, type_value_mask, confidence_value_mask)
                loss = loss_fn(logits, label)

        total_loss += loss.item() * len(label)
        total_n += len(label)
    return total_loss / total_n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see phase2/build_candidate_windows.py)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="Frozen encoder")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max subword length per candidate")
    parser.add_argument("--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN, help="Classifier hidden width")
    parser.add_argument("--head-dropout", type=float, default=DEFAULT_HEAD_DROPOUT, help="Classifier dropout")
    parser.add_argument("--type-confidence-pool", default=DEFAULT_TYPE_CONFIDENCE_POOL, choices=TYPE_CONFIDENCE_POOL_CHOICES, help="How h_type/h_confidence are pooled: 'one' (marker token, default) or 'average' (mean over the value's own subwords, e.g. 'Location'/'0.87' -- see model.py)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 weight decay")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="Stop early after this many epochs with no val-loss improvement")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N train / N val candidates (smoke test)")
    parser.add_argument("--seed", type=int, default=42, help="torch/DataLoader shuffle seed")
    parser.add_argument("--out", default=None, help="Checkpoint output path (default: checkpoints/phase2_simple/<variant>.pt)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the training-curve plot into")
    parser.add_argument(
        "--labels-file", default=None,
        help="Same {TYPE: prompt wording} JSON file gliner/extract_ner_features.py's --labels-file reads "
        "(e.g. test/ajmc/labels.json) -- used as the marker-text TYPE tag's display-name map. Overrides the "
        "standard HIPE-2022 5-type TYPE_DISPLAY_NAME (see vocab.py). Must be passed again to evaluate.py "
        "(not stored in the checkpoint -- the frozen encoder is agnostic to which type words it saw).",
    )
    args = parser.parse_args()

    variant = variant_name(encoder_name=args.encoder_name, type_confidence_pool=args.type_confidence_pool)
    print(f"Variant: {variant}")

    type_display_name = type_display_name_from_file(args.labels_file) if args.labels_file else None
    if type_display_name:
        print(f"Custom type display-name map (from {args.labels_file}): {type_display_name}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load tokenizer and build train/val datasets ===")
    print(f"Loading {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    train_dataset = Phase2SimpleWindowDataset(args.windows, tokenizer, split="train", max_length=args.max_length, type_display_name=type_display_name)
    val_dataset = Phase2SimpleWindowDataset(args.windows, tokenizer, split="val", max_length=args.max_length, type_display_name=type_display_name)
    if args.limit is not None:
        train_dataset.records = train_dataset.records[: args.limit]
        val_dataset.records = val_dataset.records[: args.limit]
    print(f"{len(train_dataset)} train candidates, {len(val_dataset)} val candidates")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate)

    print("=== Step 2: Build model ===")
    model = Phase2SimpleModel(
        encoder_name=args.encoder_name, head_hidden=args.head_hidden, head_dropout=args.head_dropout,
        type_confidence_pool=args.type_confidence_pool,
    ).to(device)
    print_parameter_breakdown(model)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("=== Step 3: Train (train), early-stop on val ===")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True, desc=f"Epoch {epoch} train")
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False, desc=f"Epoch {epoch} val")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}: no val-loss improvement for {args.patience} epochs")
                break

    if best_state is None:
        warnings.warn("val loss never improved during training -- saving the last epoch's weights instead of a 'best' checkpoint")
        best_epoch = len(train_losses)
    else:
        model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} (train_loss={train_losses[best_epoch - 1]:.4f}, val_loss={val_losses[best_epoch - 1]:.4f})")

    print("=== Step 4: Save checkpoint ===")
    out_path = Path(args.out) if args.out is not None else CHECKPOINTS_DIR / f"{variant}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out_path)
    print(f"Saved {out_path}")

    print("=== Step 5: Plot train/val training curve ===")
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    curve_out_path = figures_dir / f"{variant}_track_training.png"
    plot_training_curve(train_losses, val_losses, best_epoch, f"{variant}: train vs val loss", curve_out_path)
    print(f"Saved {curve_out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
