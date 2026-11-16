#!/usr/bin/env python3
"""Freeze the 64-trajectory grouping that all three arms must follow.

Baseline (whole-trajectory) uses HF's ``group_by_length``, which sorts by token length and shuffles
in mega-batches with a per-process RNG. That order is not reproducible across arms unless it is
computed once. This script computes it in the *training* environment (transformers 5.x inside the SIF)
and writes ``group_schedule.json``: one list of 64 trajectory indices per optimizer step, all epochs.

Trajectories per optimizer step = 64 (the TerminalHorizon recipe's effective batch), on 16 GPUs
(2 nodes x 8) that is ga = 4 micro-batches of 16 rows.

    python coding_sft/prep/build_group_schedule.py --tokenized <cache> --out <path>
"""

import argparse
import glob
import json
import os

import pyarrow as pa
import torch
from transformers.trainer_pt_utils import get_length_grouped_indices


def arrow_lengths(tokenized_path: str) -> list[int]:
    """Read per-row sequence lengths straight from the arrow shards (version-independent)."""
    split_dirs = glob.glob(os.path.join(tokenized_path, "*", "dataset_info.json"))
    if not split_dirs:  # dataset saved without split dirs
        candidates = [tokenized_path]
    else:
        candidates = [os.path.dirname(p) for p in split_dirs if os.path.basename(os.path.dirname(p)) == "train"]
    lengths: list[int] = []
    for d in candidates:
        for shard in sorted(glob.glob(os.path.join(d, "*.arrow"))):
            with pa.memory_map(shard, "r") as src:
                table = pa.ipc.open_stream(src).read_all()
            if "input_ids" not in table.column_names:
                continue
            for row in table.column("input_ids").to_pylist():
                lengths.append(len(row))
    return lengths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized", required=True, help="whole-trajectory tokenized cache (defines the order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajs-per-step", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset-shortest", type=int, default=0, help="debug: only schedule the N shortest trajectories")
    args = ap.parse_args()

    all_lengths = arrow_lengths(args.tokenized)
    subset = list(range(len(all_lengths)))
    if args.subset_shortest:
        subset = sorted(subset, key=lambda i: all_lengths[i])[: args.subset_shortest]
    lengths = [all_lengths[i] for i in subset]
    n = len(lengths)
    print(f"{n} trajectories (of {len(all_lengths)}), token length p50 {sorted(lengths)[n // 2]}, max {max(lengths)}")

    groups: list[list[int]] = []
    pads: list[int] = []
    for ep in range(args.epochs):
        gen = torch.Generator()
        gen.manual_seed(args.seed + ep)
        order = [int(i) for i in get_length_grouped_indices(lengths, args.trajs_per_step, generator=gen)]
        for s in range(0, n, args.trajs_per_step):
            group = [subset[i] for i in order[s : s + args.trajs_per_step]]  # indices into the FULL cache
            pad = args.trajs_per_step - len(group)
            if pad:
                group = group + group[:pad]  # repeat own members; every arm pads identically
            groups.append(group)
            pads.append(pad)

    payload = {
        "groups": groups,
        "pad_counts": pads,
        "epochs": args.epochs,
        "trajs_per_step": args.trajs_per_step,
        "n_trajs": n,
        "subset": subset,
        "seed": args.seed,
        "lengths": all_lengths,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    per_ep = (n + args.trajs_per_step - 1) // args.trajs_per_step
    print(f"{len(groups)} optimizer steps ({per_ep}/epoch x {args.epochs}), padded rows {sum(pads)}")
    print(f"out {args.out}")


if __name__ == "__main__":
    main()