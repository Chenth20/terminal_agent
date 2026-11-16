#!/usr/bin/env python3
"""Verify that an environment can run this checkout's Qwen3.5 agent SFT.

Run inside the image / conda env, on a node with a GPU:

    python deploy/verify_env.py            # full check (needs CUDA)
    python deploy/verify_env.py --no-gpu   # login node: skip GPU/kernel checks

Exit code 0 = every required check passed. Anything printed as FAIL must be
fixed before launching a multi-node job; WARN is informational.
"""

import argparse
import importlib
import os
import sys

REQUIRED = {
    "torch": "2.6.0",
    "transformers": "5.12.1",
    "datasets": "4.0.0",
    "accelerate": "1.11.0",
    "peft": "0.18.1",
    "trl": "0.24.0",
    "deepspeed": "0.18.4",
    "liger_kernel": "0.8.4",
    "fla": "0.5.2",
    "wandb": "0.28.2",
}

OPTIONAL = ("causal_conv1d", "flash_attn", "vllm", "sglang")

results = []


def record(status, name, detail):
    results.append((status, name, detail))
    print(f"{status:4} {name:28} {detail}")


def module_version(name):
    mod = importlib.import_module(name)
    return getattr(mod, "__version__", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="skip CUDA / kernel / fast-path checks")
    args = ap.parse_args()

    py = "%d.%d.%d" % sys.version_info[:3]
    record("PASS" if sys.version_info[:2] == (3, 11) else "WARN", "python", f"{py} (image ships 3.11)")

    for name, want in REQUIRED.items():
        try:
            have = module_version(name)
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            record("FAIL", name, f"missing ({type(exc).__name__}: {exc})")
            continue
        record("PASS" if str(have) == want else "WARN", name, f"{have} (validated: {want})")

    for name in OPTIONAL:
        try:
            record("PASS", name, module_version(name))
        except Exception as exc:  # noqa: BLE001
            record("WARN", name, f"absent ({type(exc).__name__}) - fine unless you need it")

    try:
        import llamafactory
        from llamafactory.extras.env import VERSION
        from llamafactory.extras.misc import check_dependencies

        origin = os.path.dirname(os.path.abspath(llamafactory.__file__))
        # The launchers export PYTHONPATH=<repo>/src; if that is missing the image's baked
        # snapshot runs instead of this checkout (a silent, fixed-bug-regressing failure mode).
        in_repo = "llama-factory/src/llamafactory" in origin.replace(os.sep, "/")
        record("PASS" if in_repo else "FAIL", "llamafactory", f"{VERSION} @ {origin}")
        check_dependencies()
        record("PASS", "check_dependencies", "ok")
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "llamafactory", f"{type(exc).__name__}: {exc}")

    if not args.no_gpu:
        try:
            import torch

            record("PASS" if torch.cuda.is_available() else "FAIL", "cuda", f"torch cuda {torch.version.cuda}")
            if torch.cuda.is_available():
                gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                mem = min(torch.cuda.get_device_properties(i).total_memory for i in range(len(gpus))) / 2**30
                record("PASS" if mem >= 120 else "WARN", "gpu", f"{len(gpus)}x {gpus[0]}, min {mem:.0f} GiB")
            import transformers.models.qwen3_5.modeling_qwen3_5 as q

            fast = getattr(q, "is_fast_path_available", False)
            record("PASS" if fast else "WARN", "qwen3.5 fast path", "fla + causal_conv1d active" if fast else "torch fallback (slow)")
        except Exception as exc:  # noqa: BLE001
            record("FAIL", "cuda", f"{type(exc).__name__}: {exc}")

    fails = [r for r in results if r[0] == "FAIL"]
    print(f"\n{len(results) - len(fails)}/{len(results)} checks not failing; {len(fails)} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())