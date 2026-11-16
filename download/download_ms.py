#!/usr/bin/env python
"""Download a dataset from ModelScope.

Note: this cluster can reach ``www.modelscope.cn`` but NOT the bare
``modelscope.cn`` host (connection times out). The SDK defaults to the bare
host, so ``MODELSCOPE_ENDPOINT`` must be set before importing modelscope.
This script sets it itself, so no manual export is required.

Usage:
    python download_data.py                                      # default: AI-ModelScope/MATH-500
    python download_data.py --dataset AI-ModelScope/gsm8k
    python download_data.py --dataset AI-ModelScope/gsm8k --local-dir ./data/gsm8k
"""

import argparse
import os
import time

# MUST happen before modelscope is imported: the SDK reads this at config time.
os.environ.setdefault("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")

DEFAULT_DATASET = "AI-ModelScope/MATH-500"
DEFAULT_LOCAL_DIR = "./data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a ModelScope dataset.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset id, e.g. AI-ModelScope/gsm8k (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Base output directory; the dataset is saved under <local-dir>/<name> "
        f"(default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch/tag/commit to download (default: the dataset's default revision)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from modelscope import snapshot_download

    name = args.dataset.split("/")[-1]
    local_dir = os.path.join(args.local_dir, name)

    print(f"endpoint : {os.environ['MODELSCOPE_ENDPOINT']}")
    print(f"dataset  : {args.dataset}")
    print(f"local_dir: {local_dir}")

    start = time.time()
    path = snapshot_download(
        args.dataset,
        repo_type="dataset",
        revision=args.revision,
        local_dir=local_dir,
    )
    print(f"done in {time.time() - start:.1f}s -> {path}")

    for root, _, files in os.walk(path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            print(f"  {os.path.getsize(fp):>12}  {os.path.relpath(fp, path)}")


if __name__ == "__main__":
    main()
