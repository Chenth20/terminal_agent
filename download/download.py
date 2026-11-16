#!/usr/bin/env python
r"""Download models / datasets from ModelScope or HuggingFace.

Network notes for this cluster (verified 2026-10):
  * ModelScope: ``www.modelscope.cn`` works, the bare ``modelscope.cn`` host
    times out (SDK default) -> endpoint forced to ``https://www.modelscope.cn``.
  * HuggingFace: ``huggingface.co`` is unreachable (poisoned DNS). Use the
    ``hf-mirror.com`` mirror. The xet transfer protocol gets connection resets,
    so it is disabled; default timeouts (10s) are too short and are raised.
  All defaults can be overridden by exporting the env vars before running.

Output path: ``--local-dir`` is REQUIRED and there is no implicit fallback --
downloads always land exactly where you point them. The conventional root on
this cluster is ``/storage/openpsi/users/chen/code_data`` (``--data-root`` only
matters when you use ``--subdir`` instead of ``--local-dir``).

Examples:
    # ModelScope (default source)
    python download.py --repo Qwen/Qwen2.5-0.5B-Instruct \
        --local-dir /storage/openpsi/users/chen/code_data/models/Qwen2.5-0.5B-Instruct
    python download.py --repo AI-ModelScope/gsm8k --type dataset \
        --local-dir /storage/openpsi/users/chen/code_data/datasets/gsm8k

    # HuggingFace (via hf-mirror.com)
    python download.py --source hf --repo Qwen/Qwen2.5-0.5B-Instruct \
        --local-dir /storage/openpsi/users/chen/code_data/models/Qwen2.5-0.5B-Instruct
    python download.py --source hf --repo openai/gsm8k --type dataset \
        --include "main/*" --local-dir /storage/openpsi/users/chen/code_data/datasets/gsm8k

    # Shorthand: put it under the data root as <type>s/<repo name>
    python download.py --repo Qwen/Qwen2.5-7B-Instruct --subdir
    python download.py --repo Qwen/Qwen2.5-7B-Instruct --subdir --data-root /storage/openpsi/users/chen/code_data

    # Only some files
    python download.py --repo Qwen/Qwen2.5-7B-Instruct --include "*.json" "*.safetensors" \
        --local-dir /storage/openpsi/users/chen/code_data/models/Qwen2.5-7B-Instruct

Private / gated repos: export MODELSCOPE_API_TOKEN or HF_TOKEN (never pass
tokens on the command line). Do not send HF tokens through hf-mirror.com
unless you trust the mirror.
"""

import argparse
import os
import sys
import time

# Must be set BEFORE importing modelscope / huggingface_hub (read at import time).
_ENV_DEFAULTS = {
    "MODELSCOPE_ENDPOINT": "https://www.modelscope.cn",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_ETAG_TIMEOUT": "60",
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
}
for _k, _v in _ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

DEFAULT_DATA_ROOT = "/storage/openpsi/users/chen/code_data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a model or dataset from ModelScope / HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1] if "Examples:" in __doc__ else None,
    )
    p.add_argument("--repo", required=True, help="Repo id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--source",
        choices=["modelscope", "ms", "huggingface", "hf"],
        default="modelscope",
        help="Download source (default: modelscope)",
    )
    p.add_argument(
        "--type",
        dest="repo_type",
        choices=["model", "dataset"],
        default="model",
        help="Repo type (default: model)",
    )
    p.add_argument(
        "--local-dir",
        default=None,
        help="Output directory (required unless --subdir is used)",
    )
    p.add_argument(
        "--subdir",
        action="store_true",
        help=f"Auto-construct path as <data-root>/<type>s/<repo-name> "
        f"(data-root defaults to {DEFAULT_DATA_ROOT})",
    )
    p.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory when --subdir is used (default: {DEFAULT_DATA_ROOT})",
    )
    p.add_argument("--revision", default=None, help="Branch / tag / commit")
    p.add_argument("--include", nargs="+", default=None, help="Glob patterns to download only")
    p.add_argument("--exclude", nargs="+", default=None, help="Glob patterns to skip")
    p.add_argument("--max-workers", type=int, default=8, help="Parallel file downloads")
    p.add_argument("--retries", type=int, default=3, help="Retries on network errors (resumes)")
    return p.parse_args()


def download_modelscope(args: argparse.Namespace, local_dir: str) -> str:
    from modelscope import snapshot_download

    return snapshot_download(
        args.repo,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=local_dir,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
        max_workers=args.max_workers,
    )


def download_huggingface(args: argparse.Namespace, local_dir: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        args.repo,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=local_dir,
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
        max_workers=args.max_workers,
    )


def main() -> int:
    args = parse_args()
    is_hf = args.source in ("huggingface", "hf")
    endpoint = os.environ["HF_ENDPOINT"] if is_hf else os.environ["MODELSCOPE_ENDPOINT"]

    if args.local_dir:
        local_dir = args.local_dir
    elif args.subdir:
        local_dir = os.path.join(
            args.data_root, f"{args.repo_type}s", args.repo.split("/")[-1]
        )
    else:
        print(
            "error: --local-dir is required. Specify an explicit path, e.g.:\n"
            f"  --local-dir /storage/openpsi/users/chen/code_data/models/{args.repo.split('/')[-1]}\n"
            "Or use --subdir to auto-place under the default data root.",
            file=sys.stderr,
        )
        return 2

    fn = download_huggingface if is_hf else download_modelscope

    print(f"source   : {'huggingface' if is_hf else 'modelscope'} ({endpoint})")
    print(f"repo     : {args.repo} [{args.repo_type}]")
    print(f"local_dir: {local_dir}")

    start = time.time()
    for attempt in range(1, args.retries + 1):
        try:
            path = fn(args, local_dir)
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:  # network errors: both SDKs resume partial files on retry
            print(f"[attempt {attempt}/{args.retries}] {type(e).__name__}: {e}", file=sys.stderr)
            if attempt == args.retries:
                return 1
            time.sleep(5 * attempt)

    total = 0
    print(f"done in {time.time() - start:.1f}s -> {path}")
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # skip .cache / .git metadata
        for f in sorted(files):
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            total += size
            print(f"  {size:>14,}  {os.path.relpath(fp, path)}")
    print(f"total: {total / 1024**2:,.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
