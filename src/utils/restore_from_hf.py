"""Restore data/, checkpoints/, and figures/ from the private Hugging Face Hub repos
written by backup_to_hf.py -- for setting up a fresh rented server (see CLAUDE.md)
without redoing every pipeline stage from scratch.

Mirrors backup_to_hf.py's repo layout exactly: the dataset repo's root maps straight
onto data/, and the model repo's checkpoints/ and figures/ subfolders map straight onto
this repo's own checkpoints/ and figures/ -- so restoring is just downloading each repo
into REPO_ROOT, no path remapping needed.

Auth: uses whatever token `hf auth login` already cached (~/.cache/huggingface/token),
or HF_TOKEN if set -- a read-only token is enough here, unlike backup_to_hf.py.

Usage:
    python src/utils/restore_from_hf.py                                    # both
    python src/utils/restore_from_hf.py --skip-data                        # checkpoints/+figures/ only
    python src/utils/restore_from_hf.py --skip-checkpoints
    python src/utils/restore_from_hf.py --dataset-repo myuser/cafe-tei-data --model-repo myuser/cafe-tei-checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Where to restore the dataset repo into")
    parser.add_argument("--dataset-repo", default=None, help="Dataset repo id, e.g. myuser/cafe-tei-data (default: <your-username>/cafe-tei-data)")
    parser.add_argument("--model-repo", default=None, help="Model repo id, e.g. myuser/cafe-tei-checkpoints (default: <your-username>/cafe-tei-checkpoints)")
    parser.add_argument("--skip-data", action="store_true", help="Don't restore data/")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Don't restore checkpoints/ or figures/")
    args = parser.parse_args()

    print("=== Step 1: Check Hugging Face auth ===")
    api = HfApi()
    who = api.whoami()
    username = who["name"]
    print(f"Logged in as: {username}")

    dataset_repo = args.dataset_repo or f"{username}/cafe-tei-data"
    model_repo = args.model_repo or f"{username}/cafe-tei-checkpoints"

    if not args.skip_data:
        data_dir = Path(args.data_dir)
        print(f"\n=== Step 2: Download {dataset_repo} -> {data_dir} ===")
        data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=dataset_repo, repo_type="dataset", local_dir=str(data_dir))
        print(f"Restored {data_dir}")
    else:
        print("\n=== Step 2: Skipped data/ restore (--skip-data) ===")

    if not args.skip_checkpoints:
        print(f"\n=== Step 3: Download {model_repo} -> {REPO_ROOT} (checkpoints/, figures/) ===")
        snapshot_download(repo_id=model_repo, repo_type="model", local_dir=str(REPO_ROOT))
        print(f"Restored {REPO_ROOT / 'checkpoints'} and {REPO_ROOT / 'figures'}")
    else:
        print("\n=== Step 3: Skipped checkpoints/+figures/ restore (--skip-checkpoints) ===")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
