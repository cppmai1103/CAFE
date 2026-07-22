"""phase2_expert's training loop -- structurally similar to phase2/train.py (same split
roles, same early-stopping-on-val-loss, same 6-argument forward() call), but the training
objective is BCEWithLogitsLoss PLUS a load-balancing auxiliary loss on the gate
(model.load_balance_loss, weight --lambda-balance, default 0.01, on by default) --
without it, the gate reliably collapses onto a single expert within a few epochs (an
unused expert gets ~no gradient once its alpha is ~0, so it never improves enough to
compete back -- see analyze_experts.py, which is what surfaced this happening in a real
run). The balance loss is only ever added into the TRAINING backward pass; val_loss
(used for early stopping/model selection, and everywhere else in the project a "loss"
means task performance) stays pure BCE. Pass --lambda-balance 0 to disable it and recover
the original unbalanced behavior. Other differences from phase2/train.py: which model
class gets built and which hyperparameters it takes
(num_experts/expert_hidden/gate_hidden/expert_dropout instead of
head_hidden/head_dropout -- see model.py's SS25/26 MoE head). Input pipeline (tokenizer,
Phase2WindowDataset, candidate windows JSONL) is reused directly from phase2 -- unchanged,
so both models train on exactly the same candidates/splits.

Per docs/phase2_learned_features.md SS26 "Recommended first MoE settings": K=4 experts, expert hidden
256, gate hidden 128, dropout 0.1 (all defaults below).

Usage:
    python src/phase2/expert/train.py
    python src/phase2/expert/train.py --batch-size 32 --lr 3e-4 --max-epochs 30
    python src/phase2/expert/train.py --limit 40 --max-epochs 2 --batch-size 4  # smoke test
    python src/phase2/expert/train.py --num-experts 8 --expert-hidden 128
    python src/phase2/expert/train.py --lambda-balance 0.05   # stronger balancing pressure
    python src/phase2/expert/train.py --lambda-balance 0      # disable load-balancing (old behavior)
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
from phase2.base.dataset import Phase2WindowDataset
from phase2.base.tokenize_windows import DEFAULT_MAX_LENGTH
from phase2.base.vocab import entity_type_vocab_from_file
from phase2.expert.model import (
    DEFAULT_D_SCORE, DEFAULT_D_TYPE, DEFAULT_ENCODER_NAME, DEFAULT_EXPERT_DROPOUT, DEFAULT_EXPERT_HIDDEN,
    DEFAULT_GATE_HIDDEN, DEFAULT_NUM_EXPERTS, Phase2ExpertModel, load_balance_loss, print_parameter_breakdown,
    save_checkpoint, variant_name,
)

CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent / "checkpoints" / "phase2_expert"
DEFAULT_CHECKPOINT_OUT = CHECKPOINTS_DIR / f"{variant_name()}.pt"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "figures" / "phase2_expert"
DEFAULT_LAMBDA_BALANCE = 0.01


def run_epoch(
    model: Phase2ExpertModel, loader: DataLoader, loss_fn, optimizer, device: str, train: bool, desc: str,
    lambda_balance: float = 0.0,
) -> tuple[float, float]:
    """Returns (avg BCE loss, avg load-balance loss) -- BCE is the pure task-performance
    number (used for early stopping/model selection); the balance loss is only ever added
    into the backward pass (train=True), never into what's reported as "val_loss", so
    early stopping stays a measure of task performance, not gate balance."""
    model.train(train)
    total_bce, total_balance, total_n = 0.0, 0.0, 0
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        input_ids = batch["input_ids"].to(device)
        dict_flag_ids = batch["dict_flag_ids"].to(device)
        target_flag_ids = batch["target_flag_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        entity_type_id = batch["entity_type_id"].to(device)
        ner_score = batch["ner_score"].to(device)
        label = batch["label_reliable"].to(device)

        if train:
            optimizer.zero_grad()
            logits, alpha = model(input_ids, dict_flag_ids, target_flag_ids, attention_mask, entity_type_id, ner_score, return_alpha=True)
            bce = loss_fn(logits, label)
            balance = load_balance_loss(alpha)
            (bce + lambda_balance * balance).backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits, alpha = model(input_ids, dict_flag_ids, target_flag_ids, attention_mask, entity_type_id, ner_score, return_alpha=True)
                bce = loss_fn(logits, label)
                balance = load_balance_loss(alpha)

        total_bce += bce.item() * len(label)
        total_balance += balance.item() * len(label)
        total_n += len(label)
    return total_bce / total_n, total_balance / total_n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see phase2/build_candidate_windows.py)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="Frozen encoder")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max subword length per candidate")
    parser.add_argument("--d-type", type=int, default=DEFAULT_D_TYPE, help="Entity-type embedding dim")
    parser.add_argument("--d-score", type=int, default=DEFAULT_D_SCORE, help="NER-score embedding dim")
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS, help="SS26 default: 4 latent experts")
    parser.add_argument("--expert-hidden", type=int, default=DEFAULT_EXPERT_HIDDEN, help="SS26 default: 256")
    parser.add_argument("--gate-hidden", type=int, default=DEFAULT_GATE_HIDDEN, help="SS26 default: 128")
    parser.add_argument("--expert-dropout", type=float, default=DEFAULT_EXPERT_DROPOUT, help="SS26 default: 0.1")
    parser.add_argument(
        "--labels-file", default=None,
        help="Same {TYPE: prompt wording} JSON file gliner/extract_ner_features.py's --labels-file reads "
        "(e.g. test/ajmc/labels.json) -- only the keys are used here. Overrides the standard HIPE-2022 "
        "5-type vocab (PERS/LOC/ORG/TIME/PROD) for a candidate-windows file whose predicted_type values "
        "come from a different NER source's own tagset (see phase2/train.py's --labels-file).",
    )
    parser.add_argument(
        "--lambda-balance", type=float, default=DEFAULT_LAMBDA_BALANCE,
        help="Weight on the load-balancing auxiliary loss (model.load_balance_loss) added to BCE during training only -- "
        "penalizes the gate for concentrating weight on few experts, so one expert doesn't permanently starve the others "
        "of gradient. 0 disables it (recovers the original no-balancing behavior).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 weight decay")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="Stop early after this many epochs with no val-loss improvement")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N train / N val candidates (smoke test)")
    parser.add_argument("--seed", type=int, default=42, help="torch/DataLoader shuffle seed")
    parser.add_argument("--out", default=None, help="Checkpoint output path (default: checkpoints/phase2_expert/<variant>.pt)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the training-curve plot into")
    args = parser.parse_args()

    variant = variant_name(encoder_name=args.encoder_name)
    print(f"Variant: {variant}")

    entity_type_vocab = entity_type_vocab_from_file(args.labels_file) if args.labels_file else None
    if entity_type_vocab:
        print(f"Custom entity-type vocab (from {args.labels_file}): {entity_type_vocab}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load tokenizer and build train/val datasets (reused from phase2) ===")
    print(f"Loading {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    train_dataset = Phase2WindowDataset(args.windows, tokenizer, split="train", max_length=args.max_length, entity_type_vocab=entity_type_vocab)
    val_dataset = Phase2WindowDataset(args.windows, tokenizer, split="val", max_length=args.max_length, entity_type_vocab=entity_type_vocab)
    if args.limit is not None:
        train_dataset.records = train_dataset.records[: args.limit]
        val_dataset.records = val_dataset.records[: args.limit]
    print(f"{len(train_dataset)} train candidates, {len(val_dataset)} val candidates")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate)

    print("=== Step 2: Build model ===")
    model = Phase2ExpertModel(
        encoder_name=args.encoder_name, d_type=args.d_type, d_score=args.d_score,
        num_experts=args.num_experts, expert_hidden=args.expert_hidden,
        gate_hidden=args.gate_hidden, expert_dropout=args.expert_dropout,
        entity_type_vocab=entity_type_vocab,
    ).to(device)
    print_parameter_breakdown(model)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"=== Step 3: Train (train), early-stop on val (lambda_balance={args.lambda_balance}) ===")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_balance = run_epoch(
            model, train_loader, loss_fn, optimizer, device, train=True, desc=f"Epoch {epoch} train",
            lambda_balance=args.lambda_balance,
        )
        val_loss, val_balance = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False, desc=f"Epoch {epoch} val")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} (balance={train_balance:.4f}) val_loss={val_loss:.4f} (balance={val_balance:.4f})")

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
