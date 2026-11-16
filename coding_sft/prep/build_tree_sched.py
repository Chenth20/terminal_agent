#!/usr/bin/env python3
"""Physically re-order an arm-B tree cache so that every optimizer step consumes exactly 64 TRAJECTORIES.

Why: the recipe's "effective batch 64" means 64 trajectories (abl_mask_const_* : one row = one trajectory).
In the tree cache one trajectory is 1..14 rows (one per compaction segment, times --split-tokens sub-rows), so
`ga` can no longer be a constant and the sampler must keep a trajectory's rows inside one step. We therefore
(1) group trajectories: `--order global_sort` (default) = one global sort of all trajectories by whole-trajectory
token count (descending), 64 consecutive per step, identical every epoch, no RNG; `--order group_by_length` =
`get_length_grouped_indices(batch_size=64, Generator(42+epoch))` - NOTE this is NOT the HF trainer's grouping
(the trainer passes batch_size = per_device_bs * ga = 4 -> mega-batches of 200, drawn from the global RNG), it
is kept only to describe the already-built *_sched16 caches of 2026-10-29. Tail step padded by repeating its
own members.
(2) expand every step into its rows, (3) lay the rows out rank-major for `disable_shuffling: true` consumption:
accelerate's BatchSamplerShard hands global micro-batch i to rank i % world, so step s occupies rows
[world*G*s, world*G*(s+1)) and a step with fewer rows than world*G is filled with DUMMY rows (8 pad tokens,
all labels IGNORE, trunk_len 8, no branches) that add 0 to the loss sum and 0 to num_items_in_batch
(verified in the A-nopack debug arm: identical 8-step losses).

G (micro-batches per rank per step) = ceil(max rows in any step / world); it is written to the manifest and
the launcher reads it as gradient_accumulation_steps. The trainer's token-mean loss (num_items_in_batch summed
over all ranks and micro-batches) is then exactly "sum CE over the 64 trajectories' supervised tokens / their count",
the same quantity the baseline computes for the same 64 trajectories.

    python coding_sft/prep/build_tree_sched.py --cache DIR --manifest-src SRC.manifest.jsonl --world 16 --epochs 8 --out DIR_sched
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from datasets import DatasetDict
from transformers.trainer_pt_utils import get_length_grouped_indices

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_io import read_cache  # noqa: E402
from prepare_tree_generic import FEATURES, build_dataset  # noqa: E402

IGNORE_INDEX = -100
COLS = ("input_ids", "labels", "position_ids", "tree_forks", "tree_branch_lens")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="tree cache dir (prepare_tree_generic.py output, with manifest.json)")
    ap.add_argument("--manifest-src", default=None,
                    help="source manifest jsonl (job/env_key/sample per source row) to identify trajectories; "
                         "default: every source row is its own trajectory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select-score", type=float, default=None,
                    help="keep only source rows whose manifest-src `score` equals this (e.g. 1 = solved subset)")
    ap.add_argument("--world", type=int, required=True, help="ranks (GPUs) the trainer runs on")
    ap.add_argument("--trajs-per-step", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tail", choices=("drop", "repeat"), default="drop",
                    help="what to do with the n %% trajs_per_step leftover trajectories of an epoch: 'drop' (default) "
                         "skips them (guarantees every step is a real 64-trajectory batch); 'repeat' cyclically repeats "
                         "the leftover members to fill 64 (legacy - a 1-2 sample 'batch' spikes the grad norm)")
    ap.add_argument("--order", choices=("group_by_length", "global_sort"), default="global_sort",
                    help="global_sort (default): one global length sort, 64 consecutive trajectories per step, same every "
                         "epoch; group_by_length: HF get_length_grouped_indices(batch_size=64, seed 42+epoch) (legacy)")
    ap.add_argument("--pad-token-id", type=int, default=248044)
    args = ap.parse_args()
    t0 = time.time()

    man = json.load(open(os.path.join(args.cache, "manifest.json"), encoding="utf-8"))
    per = man["per_row"]
    cols = read_cache(args.cache, list(COLS))
    trunk_len = [int(r["trunk_tokens"]) for r in per]
    assert len(cols["input_ids"]) == len(per), (len(cols["input_ids"]), len(per))
    for i, r in enumerate(per):
        assert len(cols["input_ids"][i]) == r["tokens"], (i, len(cols["input_ids"][i]), r["tokens"])

    # ---- trajectories: source rows of one (job, env_key, sample) ----
    if args.manifest_src:
        src = [json.loads(l) for l in open(args.manifest_src, encoding="utf-8")]
        key = lambda sr: (src[sr]["job"], src[sr]["env_key"], src[sr]["sample"])  # noqa: E731
    else:
        key = lambda sr: sr  # noqa: E731
    keep = set(range(len(per)))
    if args.select_score is not None:
        assert args.manifest_src, "--select-score needs --manifest-src"
        keep = {i for i, r in enumerate(per) if float(src[r["src_row"]].get("score") or 0) == args.select_score}
        print(f"[{time.time() - t0:.0f}s] --select-score {args.select_score}: {len(keep)}/{len(per)} rows", flush=True)
    traj_rows: dict = {}
    for i, r in enumerate(per):
        if i in keep:
            traj_rows.setdefault(key(r["src_row"]), []).append(i)
    trajs = list(traj_rows.keys())
    rows_of = [traj_rows[k] for k in trajs]
    # per-trajectory length for grouping = whole-trajectory (all segments, think inline) token count, i.e. what the
    # baseline's group_by_length would see; approx_conv_tokens is that estimate per source row
    seg_len = {}
    for r in per:
        if r.get("part", 0) == 0:
            seg_len[r["src_row"]] = r["approx_conv_tokens"]
    lengths = [sum(seg_len[per[i]["src_row"]] for i in rs if per[i].get("part", 0) == 0) for rs in rows_of]
    n = len(trajs)
    print(f"[{time.time() - t0:.0f}s] {len(per)} rows -> {n} trajectories; rows/traj max {max(map(len, rows_of))}; "
          f"traj tokens p50 {int(np.median(lengths))} max {max(lengths)}", flush=True)

    # ---- schedule ----
    groups: list[list[int]] = []
    pads: list[int] = []
    for ep in range(args.epochs):
        if args.order == "global_sort":
            order = sorted(range(n), key=lambda i: (-lengths[i], i))   # longest first, ties by index: pure function of the data
        else:
            gen = torch.Generator()
            gen.manual_seed(args.seed + ep)
            order = [int(i) for i in get_length_grouped_indices(lengths, args.trajs_per_step, generator=gen)]
        for s in range(0, n, args.trajs_per_step):
            g = order[s: s + args.trajs_per_step]
            pad = args.trajs_per_step - len(g)
            if pad and args.tail == "drop":
                # DROP the leftover trajectories: a step made of 1-2 unique trajectories (global_sort: the N shortest)
                # is not a "64-trajectory batch". Its token-mean gradient is dominated by those 1-2 samples -> a
                # grad-norm spike (measured 2026-10-31: 0.5 -> 11.7 at the epoch tail of dsv41_history) that the
                # clip has to absorb, and with a deterministic schedule it recurs at the same step every epoch.
                # Cost: n % 64 trajectories (<= 63, the shortest ones) are not trained on.
                if s == 0:
                    raise SystemExit(f"--tail drop: only {len(g)} trajectories (< {args.trajs_per_step}); refusing to drop the whole epoch")
                print(f"[tail] epoch {ep}: dropping the last {len(g)} trajectories (indices {sorted(g)})", flush=True)
                break
            if pad:
                # legacy repeat-padding: cycle the group's own members until the group is full
                g = (g * (args.trajs_per_step // max(1, len(g)) + 1))[: args.trajs_per_step]
            groups.append(g)
            pads.append(pad)
    step_rows = [[i for t in g for i in rows_of[t]] for g in groups]
    max_rows = max(map(len, step_rows))
    G = -(-max_rows // args.world)
    slots = args.world * G
    print(f"[{time.time() - t0:.0f}s] {len(groups)} steps ({len(groups) // args.epochs}/epoch x {args.epochs}); rows/step "
          f"p50 {int(np.median(list(map(len, step_rows))))} max {max_rows} -> G = ga = {G}, {slots} slots/step, "
          f"fill {sum(map(len, step_rows)) / (slots * len(groups)):.1%}", flush=True)

    # ---- lay out: step s -> slots [slots*s, slots*(s+1)); slot j -> rank j % world, micro-batch j // world ----
    out = {k: [] for k in (*COLS, "trunk_len")}
    dummy_ids = np.full(8, args.pad_token_id, dtype=np.int32)
    n_dummy = 0
    steps_meta = []
    for s, rows in enumerate(step_rows):
        # longest rows first so every rank's micro-batch sequence is similar (balances the ZeRO-3 step time)
        rows = sorted(rows, key=lambda i: -per[i]["tokens"])
        sup = 0
        for j in range(slots):
            if j < len(rows):
                i = rows[j]
                for k in COLS:
                    out[k].append(cols[k][i])
                out["trunk_len"].append(trunk_len[i])
                sup += int((cols["labels"][i] != IGNORE_INDEX).sum())
            else:
                out["input_ids"].append(dummy_ids)
                out["labels"].append(np.full(8, IGNORE_INDEX, dtype=np.int64))
                out["position_ids"].append(np.arange(8, dtype=np.int32))
                out["tree_forks"].append(np.zeros(0, dtype=np.int64))
                out["tree_branch_lens"].append(np.zeros(0, dtype=np.int64))
                out["trunk_len"].append(8)
                n_dummy += 1
        exp_sup = sum(per[i]["branch_tokens"] for i in rows)
        assert sup == exp_sup, (s, sup, exp_sup)
        steps_meta.append({"step": s, "trajs": groups[s], "pad_trajs": pads[s], "rows": len(rows), "supervised": sup,
                           "tokens": sum(per[i]["tokens"] for i in rows), "max_row": per[rows[0]]["tokens"]})
    assert len(out["input_ids"]) == slots * len(groups)

    ds = build_dataset(out)
    assert ds.features == FEATURES
    os.makedirs(args.out, exist_ok=True)
    DatasetDict({"train": ds}).save_to_disk(args.out)
    manifest = {
        "source_cache": args.cache, "manifest_src": args.manifest_src, "select_score": args.select_score, "world": args.world, "G": G,
        "gradient_accumulation_steps": G, "trajs_per_step": args.trajs_per_step, "epochs": args.epochs, "seed": args.seed,
        "order": args.order, "tail": args.tail,
        "tail_dropped_per_epoch": (n % args.trajs_per_step if args.tail == "drop" and n % args.trajs_per_step else 0),
        "n_trajs": n, "n_source_rows": len(per), "steps": len(groups), "steps_per_epoch": len(groups) // args.epochs,
        "rows": slots * len(groups), "dummy_rows": n_dummy, "slots_per_step": slots,
        "supervised_tokens_per_epoch": sum(m["supervised"] for m in steps_meta[: len(groups) // args.epochs]),
        "traj_lengths": lengths, "per_step": steps_meta,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"[{time.time() - t0:.0f}s] sched cache: {len(ds)} rows ({n_dummy} dummy) -> {args.out}; "
          f"train with max_steps={len(groups)} (or num_train_epochs=1) gradient_accumulation_steps={G} disable_shuffling=true "
          f"per_device_train_batch_size=1 on {args.world} ranks", flush=True)


if __name__ == "__main__":
    main()
