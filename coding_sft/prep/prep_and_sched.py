#!/usr/bin/env python3
"""One entry point that produces everything a run needs from a raw dataset, with the DEFAULT settings:

  1. (optional) prep:  <name>.jsonl + dataset_info.json   from a source jsonl, one row per compaction segment
                       (prepare_terminal_horizon.py --no-chunk --bad-pair mask --overlong drop, --workers = cores)
  2. tokenized cache:  <dir>/tokenized_qwen3_5_text_262144               (build_tokenized.py, CPU)
  3. trajectory sched: <dir>/tokenized_qwen3_5_text_262144_sched16       (build_traj_sched.py: step = 64 complete
                       trajectories, dummy-row padding, max_steps/steps_per_epoch/G in manifest.json)

The batch definition ("effective batch 64" = 64 complete trajectories) and the checkpoint policy
(save_steps=1 + keep_weights_every=steps_per_epoch) are fixed defaults everywhere - see
coding_sft/launch/train_th3k_c312.sh. Running this is idempotent: every step is skipped when its output exists.

    python coding_sft/prep/prep_and_sched.py --src <source>.jsonl --dir /storage/.../data/<name> --name <name> \
        [--manifest <source>.manifest.jsonl] [--world 16] [--epochs 8] [--skip-prep]
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="", help="raw sharegpt jsonl (one row per conversation / compaction segment)")
    ap.add_argument("--dir", required=True, help="dataset dir, e.g. /storage/.../data/terminal_horizon_qwen38_flash")
    ap.add_argument("--name", required=True, help="dataset name inside that dir (dataset_info.json key)")
    ap.add_argument("--manifest", default="", help="manifest jsonl, one line per source row (job/env_key/sample); "
                                                  "default <dir>/<name>.manifest.jsonl")
    ap.add_argument("--world", type=int, default=16, help="total GPU ranks of the training job")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--order", choices=("group_by_length", "global_sort", "random", "random_fixed"), default="global_sort",
                    help="step composition (build_traj_sched.py --order); sched dir suffix: _gsort / _rand / _randfix")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    ap.add_argument("--skip-prep", action="store_true", help="<name>.jsonl already exists")
    ap.add_argument("--think-cap", type=int, default=0, help="prepare_terminal_horizon.py --think-cap (chars), 0 = off")
    ap.add_argument("--overlong", choices=("drop", "truncate", "tail"), default="drop",
                    help="prepare_terminal_horizon.py --overlong policy for rows above 262144 tokens")
    ap.add_argument("--config", default="coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml")
    ap.add_argument("--tokenizer", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B",
                    help="tokenizer dir for prepare_terminal_horizon.py token budgeting; MUST match the model of "
                         "--config (the 9B tokenizer's pre-tokenizer regex differs from the 4B one)")
    args = ap.parse_args()

    d = args.dir.rstrip("/")
    os.makedirs(d, exist_ok=True)
    jsonl = os.path.join(d, f"{args.name}.jsonl")
    tok = os.path.join(d, "tokenized_qwen3_5_text_262144")
    suffix = {"global_sort": "_gsort", "random": "_rand", "random_fixed": "_randfix", "group_by_length": ""}[args.order]
    sched = f"{tok}_sched{args.world}{suffix}"
    manifest = args.manifest or os.path.join(d, f"{args.name}.manifest.jsonl")

    if not args.skip_prep and not os.path.exists(jsonl):
        assert args.src, "--src is required when <dir>/<name>.jsonl does not exist"
        run([sys.executable, "coding_sft/prep/prepare_terminal_horizon.py", "--no-chunk", "--max-tokens", "262144",
             "--workers", str(args.workers), "--bad-pair", "mask", "--overlong", args.overlong, "--think-cap", str(args.think_cap),
             "--tokenizer", args.tokenizer, "--input", args.src, "--out-dir", d, "--name", args.name])

    if not os.path.exists(os.path.join(tok, "dataset_dict.json")):
        run([sys.executable, "coding_sft/prep/build_tokenized.py", args.config,
             "--set", f"dataset_dir={d}", f"dataset={args.name}", f"tokenized_path={tok}", "overwrite_cache=true",
             "preprocessing_num_workers=32"])

    # Keep the source-row mapping alongside the prepared data.
    local_manifest = os.path.join(d, f"{args.name}.manifest.jsonl")
    if os.path.exists(manifest) and not os.path.exists(local_manifest):
        import shutil
        shutil.copy(manifest, local_manifest)
        print(f"copied manifest -> {local_manifest}")

    if not os.path.exists(os.path.join(sched, "manifest.json")):
        assert os.path.exists(manifest), (
            f"trajectory schedule needs a manifest (one line per cache row, job/env_key/sample): {manifest}.\n"
            "Write it next to the dataset (the arena builder emits TerminalHorizon-*.manifest.jsonl in the same "
            "order as the jsonl) or pass --manifest.")
        run([sys.executable, "coding_sft/prep/build_traj_sched.py", "--cache", tok, "--manifest-src", manifest, "--rows-jsonl", jsonl,
             "--world", str(args.world), "--epochs", str(args.epochs), "--order", args.order, "--out", sched])

    m = json.load(open(os.path.join(sched, "manifest.json")))
    print("\nready:")
    print(f"  dataset dir      : {d}")
    print(f"  tokenized cache  : {tok}")
    print(f"  trajectory sched : {sched}")
    print(f"  trajectories     : {m['n_trajs']}  (rows {m['n_source_rows']}, dummy {m['dummy_rows']})")
    print(f"  steps            : {m['steps']} ({m['steps_per_epoch']}/epoch x {m['epochs']})  ga = {m['G']}")
    print(f"  training overrides: tokenized_path={sched} disable_shuffling=true "
          f"max_steps={m['steps']} gradient_accumulation_steps={m['G']} "
          f"keep_weights_every={m['steps_per_epoch']}")


if __name__ == "__main__":
    main()