"""Score a trained phase2_expert checkpoint (train.py's output) on one split, in the same
output shape as phase2/evaluate.py (and every Phase 1 baseline) --
document_id/sentence_id/start_token_id/end_token_id/split/ner_score/calibrated_score --
so it plugs directly into plot_reliability_diagram.py's --extra-score flag alongside
phase2's --camembert-mlp-score for a direct comparison (see compare.py in this folder).

Default split is test (docs/pipeline.md SS1: "test: final evaluation only") -- pass
--split val or --split train to inspect other splits, or --split "" for every candidate.

--out defaults to data_phase2_expert/test_results/<variant>_scores.csv, where <variant> is
read back from the CHECKPOINT's own saved config (model.variant_name()), same self-
describing-checkpoint convention as phase2/evaluate.py. Kept in its own test_results/
subfolder (name unchanged otherwise) since the default --split is test-only -- same
convention as modeling/platt_scaling.py/logistic_regression.py/mlp_baseline.py's
test_results/ output.

Usage:
    python src/phase2/expert/evaluate.py
    python src/phase2/expert/evaluate.py --checkpoint checkpoints/phase2_expert/camembert_experts.pt --split test
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
from phase2.base.dataset import Phase2WindowDataset
from phase2.expert.model import load_model
from phase2.expert.train import DEFAULT_CHECKPOINT_OUT

DATA_PHASE2_EXPERT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "data_phase2_expert"
TEST_RESULTS_DIR = DATA_PHASE2_EXPERT_DIR / "test_results"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_OUT), help="Trained checkpoint (see train.py)")
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see phase2/build_candidate_windows.py)")
    parser.add_argument("--split", default="test", help="train/val/test (docs/pipeline.md SS1 default: test), or \"\" for every candidate")
    parser.add_argument("--batch-size", type=int, default=32, help="Eval batch size (no gradients, can be larger than training)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data_phase2_expert/test_results/<variant>_scores.csv, variant read from the checkpoint itself)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load checkpoint ===")
    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    print(f"encoder_name={model.encoder_name} variant={model.variant_name()}")

    print("=== Step 2: Load tokenizer and dataset (reused from phase2) ===")
    tokenizer = AutoTokenizer.from_pretrained(model.encoder_name)
    split = args.split or None
    # entity_type_vocab comes from the checkpoint's own config (model.entity_type_vocab),
    # same reasoning as phase2/evaluate.py -- guarantees no train/eval vocab mismatch.
    dataset = Phase2WindowDataset(args.windows, tokenizer, split=split, entity_type_vocab=model.entity_type_vocab)
    print(f"{len(dataset)} candidates (split={split!r})")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)

    print("=== Step 3: Score every candidate (also recording the gate's per-expert alpha weights) ===")
    alpha_cols = [f"alpha_{k}" for k in range(model.num_experts)]
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring candidates", unit="batch"):
            logits, alpha = model(
                batch["input_ids"].to(device), batch["dict_flag_ids"].to(device), batch["target_flag_ids"].to(device),
                batch["attention_mask"].to(device), batch["entity_type_id"].to(device), batch["ner_score"].to(device),
                return_alpha=True,
            )
            calibrated_score = torch.sigmoid(logits).cpu().tolist()
            alpha_list = alpha.cpu().tolist()  # [B, K]
            for i in range(len(batch["candidate_id"])):
                row = {
                    "document_id": batch["document_id"][i],
                    "sentence_id": int(batch["sentence_id"][i]),
                    "start_token_id": int(batch["start_token_id"][i]),
                    "end_token_id": int(batch["end_token_id"][i]),
                    "split": batch["split"][i],
                    "ner_score": float(batch["ner_score"][i]),
                    "calibrated_score": calibrated_score[i],
                    "label_reliable": int(batch["label_reliable"][i]),
                }
                row.update(zip(alpha_cols, alpha_list[i]))
                rows.append(row)
    scores_df = pd.DataFrame(rows)

    print("=== Step 4: Save scores CSV ===")
    out_path = Path(args.out) if args.out is not None else TEST_RESULTS_DIR / f"{model.variant_name()}_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # alpha_* columns are extra -- plot_reliability_diagram.py only reads calibrated_score
    # and ignores unknown columns, so this same CSV also feeds compare.py unmodified.
    scores_df[KEY_COLS + ["split", "ner_score", "calibrated_score"] + alpha_cols].to_csv(out_path, index=False)
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
    print(f"\nTo compare against phase2's full model, run:\n"
          f"  python src/phase2/expert/compare.py --experts-score {out_path}")
    print(f"\nTo see which expert(s) actually drove these scores (usage distribution + "
          f"pairwise similarity), run:\n"
          f"  python src/phase2/expert/analyze_experts.py --scores {out_path}")


if __name__ == "__main__":
    main()
