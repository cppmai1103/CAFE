"""Back up data/, checkpoints/, and figures/ to private Hugging Face Hub repos --
this server is rented (see CLAUDE.md), so nothing under those directories (all
git-ignored, see .gitignore) survives if the instance is torn down.

Splits the backup across two repos, matching Hub conventions: a dataset repo for
data/, a model repo for checkpoints/ (figures/ rides along in the model repo since
it's small and tied to those checkpoints' training runs).

Auth: uses whatever token `hf auth login` already cached (~/.cache/huggingface/token),
or HF_TOKEN if set. Run `hf auth login` first (needs a token with Write access) --
this script does not prompt for a token itself, so a missing/read-only login fails
fast with HfHubHTTPError rather than hanging on an interactive prompt.

Re-running after a new training run only uploads changed files (upload_folder diffs
against the repo), so incremental syncs are cheap.

Usage:
    python src/utils/backup_to_hf.py                       # both data/ and checkpoints/+figures/
    python src/utils/backup_to_hf.py --skip-data            # checkpoints/+figures/ only
    python src/utils/backup_to_hf.py --skip-checkpoints
    python src/utils/backup_to_hf.py --dataset-repo myuser/cafe-tei-data --model-repo myuser/cafe-tei-checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"


def dir_size_human(path: Path) -> str:
    total_bytes = float(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()))
    for unit in ("B", "KB", "MB", "GB"):
        if total_bytes < 1024:
            return f"{total_bytes:.1f}{unit}"
        total_bytes /= 1024
    return f"{total_bytes:.1f}TB"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory to upload as the dataset repo")
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_CHECKPOINTS_DIR), help="Directory to upload into the model repo, under checkpoints/")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR), help="Directory to upload into the model repo, under figures/")
    parser.add_argument("--dataset-repo", default=None, help="Dataset repo id, e.g. myuser/cafe-tei-data (default: <your-username>/cafe-tei-data)")
    parser.add_argument("--model-repo", default=None, help="Model repo id, e.g. myuser/cafe-tei-checkpoints (default: <your-username>/cafe-tei-checkpoints)")
    parser.add_argument("--public", action="store_true", help="Create the repos as public (default: private)")
    parser.add_argument("--skip-data", action="store_true", help="Don't upload data/")
    parser.add_argument("--skip-checkpoints", action="store_true", help="Don't upload checkpoints/ or figures/")
    args = parser.parse_args()

    print("=== Step 1: Check Hugging Face auth ===")
    api = HfApi()
    who = api.whoami()
    username = who["name"]
    print(f"Logged in as: {username} (token role: {who.get('auth', {}).get('accessToken', {}).get('role', 'unknown')})")

    dataset_repo = args.dataset_repo or f"{username}/cafe-tei-data"
    model_repo = args.model_repo or f"{username}/cafe-tei-checkpoints"
    private = not args.public

    if not args.skip_data:
        data_dir = Path(args.data_dir)
        print(f"\n=== Step 2: Create dataset repo {dataset_repo} (private={private}) ===")
        api.create_repo(dataset_repo, repo_type="dataset", private=private, exist_ok=True)

        print(f"=== Step 3: Upload {data_dir} ({dir_size_human(data_dir)}) -> {dataset_repo} ===")
        api.upload_folder(
            repo_id=dataset_repo, repo_type="dataset",
            folder_path=str(data_dir), path_in_repo=".",
            commit_message="Backup data/",
        )
        print(f"Done: https://huggingface.co/datasets/{dataset_repo}")
    else:
        print("\n=== Step 2-3: Skipped data/ upload (--skip-data) ===")

    if not args.skip_checkpoints:
        checkpoints_dir = Path(args.checkpoints_dir)
        figures_dir = Path(args.figures_dir)
        print(f"\n=== Step 4: Create model repo {model_repo} (private={private}) ===")
        api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)

        print(f"=== Step 5: Upload {checkpoints_dir} ({dir_size_human(checkpoints_dir)}) -> {model_repo}/checkpoints ===")
        api.upload_folder(
            repo_id=model_repo, repo_type="model",
            folder_path=str(checkpoints_dir), path_in_repo="checkpoints",
            commit_message="Backup checkpoints/",
        )

        print(f"=== Step 6: Upload {figures_dir} ({dir_size_human(figures_dir)}) -> {model_repo}/figures ===")
        api.upload_folder(
            repo_id=model_repo, repo_type="model",
            folder_path=str(figures_dir), path_in_repo="figures",
            commit_message="Backup figures/",
        )
        print(f"Done: https://huggingface.co/{model_repo}")
    else:
        print("\n=== Step 4-6: Skipped checkpoints/+figures/ upload (--skip-checkpoints) ===")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
