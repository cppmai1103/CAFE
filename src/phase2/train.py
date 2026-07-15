"""Phase 2 SS27 Stage 1: train model.py's Phase2Model (frozen encoder + side embeddings +
simple pooling + type/score embeddings + MLP head) -- the "minimal first model" from
SS34, no target-aware attention, no latent MoE (those are later additions, SS22/SS25).

Split roles match dataset.py's SPLITS: train = expert_train union gate_train (70%),
val = calibration (10%) -- early-stopped on, never backpropagated through. test (20%) is
untouched here; evaluate.py scores it separately once a checkpoint exists.

Unlike src/modeling/mlp_baseline.py's full-batch fit (small tabular model, ~80 features),
this is mini-batch SGD over a large frozen-encoder model -- one gradient step per batch,
not per epoch. Per docs/new_phase2.md SS29: batch size 16-32, lr 1e-3 (3e-4 fallback if
unstable), 10-30 epochs. Training-forward passes run WITHOUT torch.no_grad() (SS11:
gradients must reach the side embeddings through the frozen encoder); validation-forward
passes run WITH torch.no_grad() (pure inference, no backward pass planned, so there's no
reason to build the graph at all).

Ablations (docs/new_phase2.md SS31): --no-ner-score / --no-type / --no-dict-flag /
--no-target-flag each drop exactly one component from the full model -- see model.py's
module docstring for exactly what each one removes. --score-features {full,logit_only,
p_only} is a second, independent ablation dimension -- it doesn't remove NER-score
metadata, it simplifies ScoreMLP's input (only meaningful if --no-ner-score isn't also
given). The default (nothing passed) is the full model. model.variant_name() turns
whichever combination is active into a naming convention used for both the checkpoint and
(in evaluate.py) the scores CSV, so ablation runs never collide with each other or with
the full model's output:
    full model                    -> camembert_mlp
    --no-ner-score                -> camembert_mlp_without_ner_score
    --no-ner-score --no-type      -> camembert_mlp_without_ner_score_type
    --score-features logit_only   -> camembert_mlp_ner_logit_only
    --score-features p_only       -> camembert_mlp_ner_p_only

Output (paths default to <variant>.pt / <variant>_track_training.png unless --out /
--figures-dir override them):
    checkpoints_phase2/<variant>.pt -- best-val-loss epoch's weights + architecture
        config (model.save_checkpoint), loadable by evaluate.py via model.load_model.
    figures/modeling/<variant>_track_training.png -- train/val loss curve (reuses
        modeling/training_curve.py's plot_training_curve for visual consistency with
        B1/B3/the MLP baseline's own track_training.png plots).

Usage:
    python src/phase2/train.py
    python src/phase2/train.py --batch-size 32 --lr 3e-4 --max-epochs 30
    python src/phase2/train.py --limit 40 --max-epochs 2 --batch-size 4  # smoke test
    python src/phase2/train.py --no-ner-score   # ablation: saves checkpoints_phase2/camembert_mlp_without_ner_score.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.training_curve import plot_training_curve
from phase2.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.dataset import Phase2WindowDataset
from phase2.model import (
    DEFAULT_D_SCORE, DEFAULT_D_TYPE, DEFAULT_ENCODER_NAME, DEFAULT_HEAD_DROPOUT, DEFAULT_HEAD_HIDDEN,
    DEFAULT_SCORE_FEATURES, SCORE_FEATURES_CHOICES, Phase2Model, print_parameter_breakdown, save_checkpoint, variant_name,
)
from phase2.tokenize_windows import DEFAULT_MAX_LENGTH

CHECKPOINTS_DIR = Path(__file__).parent.parent.parent / "checkpoints_phase2"
DEFAULT_CHECKPOINT_OUT = CHECKPOINTS_DIR / "camembert_mlp.pt"
DEFAULT_FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "modeling"


def run_epoch(model: Phase2Model, loader: DataLoader, loss_fn, optimizer, device: str, train: bool, desc: str) -> float:
    model.train(train)
    total_loss, total_n = 0.0, 0
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
            logits = model(input_ids, dict_flag_ids, target_flag_ids, attention_mask, entity_type_id, ner_score)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(input_ids, dict_flag_ids, target_flag_ids, attention_mask, entity_type_id, ner_score)
                loss = loss_fn(logits, label)

        total_loss += loss.item() * len(label)
        total_n += len(label)
    return total_loss / total_n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see build_candidate_windows.py)")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER_NAME, help="Frozen encoder (docs/new_phase2.md SS11: CamemBERT or XLM-R)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max subword length per candidate")
    parser.add_argument("--d-type", type=int, default=DEFAULT_D_TYPE, help="Entity-type embedding dim (SS19: 32 or 64)")
    parser.add_argument("--d-score", type=int, default=DEFAULT_D_SCORE, help="NER-score embedding dim (SS19: 32 or 64)")
    parser.add_argument("--head-hidden", type=int, default=DEFAULT_HEAD_HIDDEN, help="Classifier hidden width (SS21)")
    parser.add_argument("--head-dropout", type=float, default=DEFAULT_HEAD_DROPOUT, help="Classifier dropout (SS21)")
    parser.add_argument("--no-ner-score", action="store_true", help="Ablation: drop the NER-score embedding")
    parser.add_argument("--no-type", action="store_true", help="Ablation: drop the entity-type embedding")
    parser.add_argument("--no-dict-flag", action="store_true", help="Ablation: drop the dictionary-flag side embedding")
    parser.add_argument("--no-target-flag", action="store_true", help="Ablation: drop the target-flag side embedding")
    parser.add_argument("--score-features", default=DEFAULT_SCORE_FEATURES, choices=SCORE_FEATURES_CHOICES, help="Ablation: simplify ScoreMLP's input (only matters if NER score isn't dropped)")
    parser.add_argument("--batch-size", type=int, default=16, help="SS29 default: 16 or 32")
    parser.add_argument("--lr", type=float, default=3e-4, help="SS29 default: 1e-3 (try 3e-4 if unstable)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 weight decay")
    parser.add_argument("--max-epochs", type=int, default=20, help="SS29 default: 10-30 epochs")
    parser.add_argument("--patience", type=int, default=3, help="Stop early after this many epochs with no val-loss improvement")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N train / N val candidates (smoke test)")
    parser.add_argument("--seed", type=int, default=42, help="torch/DataLoader shuffle seed")
    parser.add_argument("--out", default=None, help="Checkpoint output path (default: checkpoints_phase2/<variant>.pt, derived from the ablation flags)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to save the training-curve plot into")
    args = parser.parse_args()

    variant = variant_name(
        encoder_name=args.encoder_name,
        use_ner_score=not args.no_ner_score, use_type=not args.no_type,
        use_dict_flag=not args.no_dict_flag, use_target_flag=not args.no_target_flag,
        score_features=args.score_features,
    )
    print(f"Variant: {variant}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load tokenizer and build train/val datasets ===")
    print(f"Loading {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    train_dataset = Phase2WindowDataset(args.windows, tokenizer, split="train", max_length=args.max_length)
    val_dataset = Phase2WindowDataset(args.windows, tokenizer, split="val", max_length=args.max_length)
    if args.limit is not None:
        train_dataset.records = train_dataset.records[: args.limit]
        val_dataset.records = val_dataset.records[: args.limit]
    print(f"{len(train_dataset)} train (expert_train+gate_train) candidates, {len(val_dataset)} val (calibration) candidates")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate)

    print("=== Step 2: Build model ===")
    model = Phase2Model(
        encoder_name=args.encoder_name, d_type=args.d_type, d_score=args.d_score,
        head_hidden=args.head_hidden, head_dropout=args.head_dropout,
        use_ner_score=not args.no_ner_score, use_type=not args.no_type,
        use_dict_flag=not args.no_dict_flag, use_target_flag=not args.no_target_flag,
        score_features=args.score_features,
    ).to(device)
    print_parameter_breakdown(model)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("=== Step 3: Train (train=expert_train+gate_train), early-stop on val (calibration) ===")
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
    plot_training_curve(train_losses, val_losses, best_epoch, f"{variant}: train (expert_train+gate_train) vs val (calibration) loss", curve_out_path)
    print(f"Saved {curve_out_path}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
