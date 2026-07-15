"""Score a trained Phase 2 checkpoint (train.py's output) on one split, in the same
output shape as src/modeling/platt_scaling.py/logistic_regression.py/mlp_baseline.py --
document_id/sentence_id/start_token_id/end_token_id/split/ner_score/calibrated_score --
so it plugs directly into plot_reliability_diagram.py's --camembert-mlp-score flag for the
actual reliability-diagram/metrics-bar/ROC/risk-coverage plots.

Default split is test (docs/pipeline.md SS1: "test: final evaluation only") -- pass
--split val or --split train to inspect other splits, or --split "" for every candidate.

This script also prints the same Brier/ECE/MCE/AUROC/E-AURC summary
plot_reliability_diagram.py computes (reusing modeling/metrics.py directly) as a quick
console check, but the actual plots are produced by running plot_reliability_diagram.py
separately (this script prints the exact command to run at the end) -- keeping "compute
a score" and "compare/plot scores against other baselines" as separate concerns, same as
every other baseline in this project.

--out defaults to data_phase2/<variant>_scores.csv, where <variant> is read back from the
CHECKPOINT's own saved config (model.variant_name()) -- not re-derived from CLI flags you'd
have to remember to repeat -- so scoring an ablation checkpoint (see train.py's --no-*
flags) never collides with the full model's (or another ablation's) scores CSV.

Usage:
    python src/phase2/evaluate.py
    python src/phase2/evaluate.py --checkpoint checkpoints/phase2/camembert_mlp.pt --split test
    python src/phase2/evaluate.py --checkpoint checkpoints/phase2/camembert_mlp_without_ner_score.pt  # writes data_phase2/camembert_mlp_without_ner_score_scores.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.metrics import (
    auroc, brier_score_loss, excess_aurc, expected_calibration_error, maximum_calibration_error_from_bins,
)
from phase2.build_candidate_windows import DEFAULT_OUT as DEFAULT_WINDOWS
from phase2.dataset import Phase2WindowDataset
from phase2.model import encoder_short_name, load_model
from phase2.train import DEFAULT_CHECKPOINT_OUT

DATA_PHASE2_DIR = Path(__file__).parent.parent.parent / "data" / "data_phase2"

KEY_COLS = ["document_id", "sentence_id", "start_token_id", "end_token_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_OUT), help="Trained checkpoint (see train.py)")
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="phase2_candidate_windows.jsonl (see build_candidate_windows.py)")
    parser.add_argument("--split", default="test", help="train/val/test (docs/pipeline.md SS1 default: test), or \"\" for every candidate")
    parser.add_argument("--batch-size", type=int, default=32, help="Eval batch size (no gradients, can be larger than training)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data_phase2/<variant>_scores.csv, variant read from the checkpoint itself)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=== Step 1: Load checkpoint ===")
    print(f"Loading {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    print(f"encoder_name={model.encoder_name} variant={model.variant_name()}")

    print("=== Step 2: Load tokenizer and dataset ===")
    tokenizer = AutoTokenizer.from_pretrained(model.encoder_name)
    split = args.split or None
    dataset = Phase2WindowDataset(args.windows, tokenizer, split=split)
    print(f"{len(dataset)} candidates (split={split!r})")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate)

    print("=== Step 3: Score every candidate ===")
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring candidates", unit="batch"):
            logits = model(
                batch["input_ids"].to(device), batch["dict_flag_ids"].to(device), batch["target_flag_ids"].to(device),
                batch["attention_mask"].to(device), batch["entity_type_id"].to(device), batch["ner_score"].to(device),
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
    out_path = Path(args.out) if args.out is not None else DATA_PHASE2_DIR / f"{model.variant_name()}_scores.csv"
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
    is_full_model = model.variant_name() == f"{encoder_short_name(model.encoder_name)}_mlp"
    if is_full_model:
        print(f"\nTo compare against B0/B1/B3/MLP and produce the standard plots, run:\n"
              f"  python src/modeling/plot_reliability_diagram.py --platt-scaling-score data/platt_scaling.csv "
              f"--logistic-score data/logistic_regression.csv --mlp-score data/mlp_baseline.csv --camembert-mlp-score {out_path}")
    else:
        print(f"\nThis is an ablation variant -- to compare it against the full model, add it as an --extra-score, e.g.:\n"
              f"  python src/modeling/plot_reliability_diagram.py --camembert-mlp-score data_phase2/camembert_mlp_scores.csv "
              f"--extra-score {model.variant_name()}={out_path} --figures-dir figures/ablation")


if __name__ == "__main__":
    main()
