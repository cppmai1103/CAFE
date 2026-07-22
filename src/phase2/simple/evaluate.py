"""Score a trained phase2_simple checkpoint (train.py's output) on one split, in the same
output shape as phase2/evaluate.py (and every Phase 1 baseline) --
document_id/sentence_id/start_token_id/end_token_id/split/ner_score/calibrated_score --
so it plugs directly into plot_reliability_diagram.py's --extra-score flag alongside
phase2's --camembert-mlp-score for a direct comparison (see compare.py in this folder).

Default split is test (docs/pipeline.md SS1: "test: final evaluation only") -- pass
--split val or --split train to inspect other splits, or --split "" for every candidate.

--out defaults to data_phase2_simple/test_results/<variant>_scores.csv, where <variant> is
read back from the CHECKPOINT's own saved config (model.variant_name()), same self-
describing-checkpoint convention as phase2/evaluate.py. Kept in its own test_results/
subfolder (name unchanged otherwise) since the default --split is test-only -- same
convention as modeling/platt_scaling.py/logistic_regression.py/mlp_baseline.py's
test_results/ output.

Usage:
    python src/phase2/simple/evaluate.py
    python src/phase2/simple/evaluate.py --checkpoint checkpoints/phase2_simple/camembert_simple_mlp.pt --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase1.modeling.metrics import (
    auroc, brier_score_loss, excess_aurc, expected_calibration_error, maximum_calibration_error_from_bins,
)
from phase2.base.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.simple.dataset import Phase2SimpleWindowDataset
from phase2.simple.model import load_model
from phase2.simple.train import DEFAULT_CHECKPOINT_OUT
from phase2.simple.vocab import type_display_name_from_file

DATA_PHASE2_SIMPLE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_phase2_simple"
TEST_RESULTS_DIR = DATA_PHASE2_SIMPLE_DIR / "test_results"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_OUT), help="Trained checkpoint (see train.py)")
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see phase2/build_candidate_windows.py)")
    parser.add_argument("--split", default="test", help="train/val/test (docs/pipeline.md SS1 default: test), or \"\" for every candidate")
    parser.add_argument("--batch-size", type=int, default=32, help="Eval batch size (no gradients, can be larger than training)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data_phase2_simple/test_results/<variant>_scores.csv)")
    parser.add_argument(
        "--labels-file", default=None,
        help="Same --labels-file passed to train.py for this checkpoint (e.g. test/ajmc/labels.json) -- must "
        "match, since it's not stored in the checkpoint (the frozen encoder is agnostic to which type words "
        "it saw). Omit if the checkpoint was trained on the standard HIPE-2022 5-type scheme.",
    )
    args = parser.parse_args()

    type_display_name = type_display_name_from_file(args.labels_file) if args.labels_file else None
    if type_display_name:
        print(f"Custom type display-name map (from {args.labels_file}): {type_display_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load checkpoint ===")
    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    print(f"encoder_name={model.encoder_name} variant={model.variant_name()}")

    print("=== Step 2: Load tokenizer and dataset ===")
    tokenizer = AutoTokenizer.from_pretrained(model.encoder_name)
    split = args.split or None
    dataset = Phase2SimpleWindowDataset(args.windows, tokenizer, split=split, type_display_name=type_display_name)
    print(f"{len(dataset)} candidates (split={split!r})")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)

    print("=== Step 3: Score every candidate ===")
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring candidates", unit="batch"):
            logits = model(
                batch["input_ids"].to(device), batch["token_type_ids"].to(device), batch["attention_mask"].to(device),
                batch["sep_pos"].to(device), batch["entity_pos"].to(device), batch["type_pos"].to(device),
                batch["confidence_pos"].to(device), batch["span_mask"].to(device),
                batch["type_value_mask"].to(device), batch["confidence_value_mask"].to(device),
            )
            calibrated_score = torch.sigmoid(logits).cpu().tolist()
            for i in range(len(batch["candidate_id"])):
                rows.append({
                    "document_id": batch["document_id"][i],
                    "sentence_id": int(batch["sentence_id"][i]),
                    "start_token_id": int(batch["start_token_id"][i]),
                    "end_token_id": int(batch["end_token_id"][i]),
                    "split": batch["split"][i],
                    "ner_score": float(batch["ner_score"][i]),
                    "calibrated_score": calibrated_score[i],
                    "label_reliable": int(batch["label_reliable"][i]),
                })
    scores_df = pd.DataFrame(rows)

    print("=== Step 4: Save scores CSV ===")
    out_path = Path(args.out) if args.out is not None else TEST_RESULTS_DIR / f"{model.variant_name()}_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df[KEY_COLS + ["split", "ner_score", "calibrated_score"]].to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print("=== Step 5: Compute metrics ===")
    labels = scores_df["label_reliable"].to_numpy()
    scores = scores_df["calibrated_score"].to_numpy()
    ece, bins_df = expected_calibration_error(scores, labels)
    mce = maximum_calibration_error_from_bins(bins_df)
    brier = brier_score_loss(labels, scores)
    auc = auroc(scores, labels)
    e_aurc = excess_aurc(scores, labels)
    print(f"{len(scores_df)} candidates -- Brier={brier:.4f} ECE={ece:.4f} MCE={mce:.4f} AUROC={auc:.4f} E-AURC={e_aurc:.4f}")

    print("=== Done ===")
    score_flag = "--simple-average-score" if model.type_confidence_pool == "average" else "--simple-one-score"
    label_flag = "--simple-average-label" if model.type_confidence_pool == "average" else "--simple-one-label"
    print(f"\nTo compare against phase2's without_dict_flag ablation (the fair baseline -- "
          f"see compare.py's module docstring for why), run (or just python src/phase2/simple/compare.py "
          f"if this is the default path/variant name):\n"
          f"  python src/phase2/simple/compare.py {score_flag} {out_path} {label_flag} {model.variant_name()}")


if __name__ == "__main__":
    main()
