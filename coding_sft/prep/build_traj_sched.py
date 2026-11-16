#!/usr/bin/env python3
"""Physically re-order a FLAT (non-tree) tokenized cache so that every optimizer step consumes exactly 64
COMPLETE TRAJECTORIES.

Why: the recipe's "effective batch 64" means 64 trajectories, but one trajectory written by
prepare_terminal_horizon.py in compaction mode is 1..14 rows (one per compaction segment), so `ga` can no
longer be a constant and the sampler must keep all rows of a trajectory inside the same step. Same design
as build_tree_sched.py, minus the tree columns:

(1) group trajectories exactly like the baseline would (`get_length_grouped_indices` on the per-trajectory
    token count, torch Generator seed 42+epoch, 64 per step, tail step padded by repeating its own members),
(2) expand every step into its rows, (3) lay them out rank-major for `disable_shuffling: true`: accelerate's
BatchSamplerShard hands global micro-batch i to rank i % world, so step s occupies rows
[world*G*s, world*G*(s+1)) and a step with fewer rows than world*G is filled with DUMMY rows (8 pad tokens,
all labels IGNORE_INDEX) that add 0 to the loss sum and 0 to num_items_in_batch.

G = ceil(max rows in any step / world) is written to the manifest and must be used as
gradient_accumulation_steps; train with max_steps = `steps` (or num_train_epochs=1).

    python coding_sft/prep/build_traj_sched.py --cache DIR --manifest-src SRC.manifest.jsonl --world 16 \
        --epochs 8 --out DIR_sched
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset, DatasetDict, DatasetInfo, Features
from datasets.table import InMemoryTable
from transformers.trainer_pt_utils import get_length_grouped_indices

IGNORE_INDEX = -100
COLS = ("input_ids", "attention_mask", "labels")
PAD_TOKEN_ID = 248044


def read_cache(path: str, columns) -> tuple[dict[str, list], pa.Schema]:
    """Read the list columns of a saved-to-disk LF cache straight from the arrow shards (row order kept)."""
    train_dir = os.path.join(path, "train")
    with open(os.path.join(train_dir, "state.json"), encoding="utf-8") as f:
        files = [d["filename"] for d in json.load(f)["_data_files"]]
    out = {c: [] for c in columns}
    schema = None
    for fn in files:
        with pa.memory_map(os.path.join(train_dir, fn), "r") as src:
            table = pa.ipc.open_stream(src).read_all()
        schema = table.schema
        for c in columns:
            for chunk in table.column(c).chunks:
                offsets = chunk.offsets.to_numpy()
                values = chunk.values.to_numpy(zero_copy_only=False)
                out[c].extend(values[offsets[i]: offsets[i + 1]].copy() for i in range(len(offsets) - 1))
    return out, schema


def list_column(arrays: list[np.ndarray], dtype: pa.DataType, rows_per_chunk: int = 128) -> pa.ChunkedArray:
    chunks = []
    for s in range(0, len(arrays), rows_per_chunk):
        part = arrays[s: s + rows_per_chunk]
        offsets = np.zeros(len(part) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([len(a) for a in part])
        values = pa.array(np.concatenate(part) if part else np.zeros(0, dtype=dtype.to_pandas_dtype()), type=dtype)
        chunks.append(pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values))
    return pa.chunked_array(chunks, type=pa.list_(dtype))


def build_dataset(cols: dict[str, list], schema: pa.Schema) -> Dataset:
    n = len(cols["input_ids"])
    arrays = {
        "input_ids": list_column(cols["input_ids"], pa.int32()),
        "attention_mask": list_column(cols["attention_mask"], pa.int32()),
        "labels": list_column(cols["labels"], pa.int64()),
        "images": pa.nulls(n),
        "videos": pa.nulls(n),
        "audios": pa.nulls(n),
    }
    table = pa.Table.from_arrays([arrays[f.name] for f in schema], schema=schema)
    return Dataset(InMemoryTable(table), info=DatasetInfo(features=Features.from_arrow_schema(schema)))


def replicate_schedule(args, rows_of, lengths, traj_rows, src, src_row, t0):
    """Loss-Length Replication: per-trajectory pass counts r_i, then a random order of the replicated pool sliced into
    64-trajectory steps with NO trajectory twice in the same step. Returns (groups, pads, meta)."""
    L = np.asarray(lengths, dtype=np.float64)
    n = len(L)
    E = args.epochs
    # ---- difficulty: the probe's gsort step loss, mapped through the probe sched's trajectory membership ----
    pm = json.load(open(args.probe_sched, encoding="utf-8"))
    assert pm["order"] == "global_sort", "probe must be a global_sort schedule (step = length bucket)"
    spe = pm["steps_per_epoch"]
    step_loss = {}
    for line in open(args.probe_log, encoding="utf-8"):
        if line.strip():
            x = json.loads(line)
            if "loss" in x:
                step_loss[x["current_steps"]] = x["loss"]
    ep0 = (args.probe_epoch - 1) * spe
    # the probe sched indexes trajectories by ITS OWN traj order; match by trajectory key via its manifest_src rows
    psrc = [json.loads(l) for l in open(pm["manifest_src"], encoding="utf-8")]
    p_src_row = [int(json.loads(l)["src_row"]) for l in open(pm["rows_jsonl"], encoding="utf-8")]
    p_traj_rows: dict = {}
    for i, r in enumerate(p_src_row):
        sr = psrc[r]
        p_traj_rows.setdefault((sr["job"], sr["env_key"], sr["sample"]), []).append(i)
    p_keys = list(p_traj_rows.keys())
    key_loss = {}
    for s in range(spe):
        step_meta = pm["per_step"][ep0 + s]
        assert step_meta["step"] == ep0 + s
        for t in set(step_meta["trajs"]):
            key_loss[p_keys[t]] = step_loss[ep0 + s + 1]
    keys = list(traj_rows.keys())
    # trajectories the probe dropped at its epoch tail (tail=drop: the n % 64 shortest) have no bucket loss ->
    # give them the loss of the last (shortest) bucket
    tail_loss = step_loss[ep0 + spe]
    missing = [k for k in keys if k not in key_loss]
    assert len(missing) <= pm.get("tail_dropped_per_epoch", 0), ("probe data != this data", missing[:3])
    ell = np.array([key_loss.get(k, tail_loss) for k in keys], dtype=np.float64)
    # ---- pass counts under the uniform-E token budget ----
    w = (ell / ell.mean()) ** args.alpha * (L / np.median(L)) ** args.beta
    budget = E * L.sum()
    lo, hi = 1e-4, 1e4
    for _ in range(80):
        Z = (lo * hi) ** 0.5
        r = np.clip(np.round(E * w / Z), args.rmin, args.rmax)
        if (r * L).sum() > budget:
            lo = Z
        else:
            hi = Z
    r = np.clip(np.round(E * w / hi), args.rmin, args.rmax).astype(int)
    q = np.argsort(L)
    dec = lambda a, i: float(a[q[i * n // 10:(i + 1) * n // 10]].mean())  # noqa: E731
    print(f"[{time.time() - t0:.0f}s] replicate: alpha={args.alpha} beta={args.beta} rmin={args.rmin} rmax={args.rmax} "
          f"-> token budget {(r * L).sum() / L.sum():.2f} x data (target {E}); passes by length decile "
          f"p10 {dec(r, 0):.1f} / p50 {dec(r, 4):.1f} / p90 {dec(r, 9):.1f}; loss decile p10 {dec(ell, 0):.3f} p90 {dec(ell, 9):.3f}",
          flush=True)
    # ---- order: random over the replicated pool, no trajectory twice per step ----
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    remaining = r.copy()
    groups, pads = [], []
    while remaining.sum() >= args.trajs_per_step:
        # candidates with the most remaining copies first (ties random) -> the pool drains evenly, no traj starves
        noise = torch.rand(n, generator=gen).numpy()
        cand = [i for i in np.lexsort((noise, -remaining)) if remaining[i] > 0]
        if len(cand) < args.trajs_per_step:
            break
        g = [int(i) for i in cand[: args.trajs_per_step]]
        # shuffle inside the step so slot order is not "most-remaining first"
        perm = torch.randperm(len(g), generator=gen).tolist()
        g = [g[j] for j in perm]
        for i in g:
            remaining[i] -= 1
        groups.append(g)
        pads.append(0)
    leftover = int(remaining.sum())
    print(f"[{time.time() - t0:.0f}s] replicate: {len(groups)} steps of {args.trajs_per_step}; {leftover} replica(s) "
          f"dropped at the tail; unique trajs {n}, planned passes sum {int(r.sum())}", flush=True)
    meta = {"alpha": args.alpha, "beta": args.beta, "rmin": args.rmin, "rmax": args.rmax, "probe_sched": args.probe_sched,
            "probe_log": args.probe_log, "probe_epoch": args.probe_epoch, "passes": r.tolist(), "probe_loss": ell.tolist(),
            "token_budget_epochs": float((r * L).sum() / L.sum()), "tail_replicas_dropped": leftover}
    return groups, pads, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="flat tokenized cache (build_tokenized.py output)")
    ap.add_argument("--manifest-src", required=True, help="SOURCE manifest jsonl (one line per SOURCE row, job/env_key/sample)")
    ap.add_argument("--rows-jsonl", required=True,
                    help="the prepped jsonl the cache was tokenized from (prepare_terminal_horizon.py output, one line per "
                         "cache row, carrying `src_row` = index into --manifest-src). REQUIRED because prepare_terminal_horizon "
                         "shuffles trajectories and drops over-long ones: cache row i is NOT manifest line i (the 2026-10-29 "
                         "qwen38_flash _sched16 assumed it was and mis-grouped 10,997/11,001 rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", type=int, required=True, help="ranks (GPUs) the trainer runs on")
    ap.add_argument("--trajs-per-step", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tail", choices=("drop", "repeat"), default="drop",
                    help="what to do with the n %% trajs_per_step leftover trajectories of an epoch: 'drop' (default) "
                         "skips them (guarantees every step is a real 64-trajectory batch); 'repeat' cyclically repeats "
                         "the leftover members to fill 64 (legacy - a 1-2 sample 'batch' spikes the grad norm)")
    ap.add_argument("--order", choices=("group_by_length", "global_sort", "random", "random_fixed", "replicate"), default="global_sort",
                    help="global_sort (default): sort ALL trajectories by whole-trajectory token count (descending) once "
                         "and slice 64 consecutive trajectories per step, identical in every epoch - fully deterministic, "
                         "no RNG. group_by_length: HF get_length_grouped_indices with batch_size=64 (NOT what the HF "
                         "trainer does with per_device_bs=1, ga=4: it uses mega-batches of 50*4=200, re-drawn from the "
                         "global RNG; kept only for the already-built _sched16 caches). random: a fresh trajectory-level "
                         "permutation per epoch (seed+epoch). random_fixed: one permutation from `seed`, same in every epoch.")
    # ---- --order replicate (Loss-Length Replication): train hard/long trajectories more often under the SAME
    # token budget as --epochs uniform passes. r_i = clip(round(E/Z * (loss_i/mean)^alpha * (len_i/median)^beta),
    # rmin, rmax), Z bisected so sum(r_i * len_i) == E * sum(len_i). Difficulty loss_i comes from a FINISHED
    # global_sort probe run: in a gsort schedule step s of an epoch is a narrow length bucket of 64 trajectories,
    # so the probe's per-step loss (trainer_log.jsonl) maps every trajectory to its bucket loss.
    # Refs: Rho-1 (loss-based selection), DoReMi (excess-loss domain reweighting), Muennighoff'23 (repeating
    # data <=4 epochs is nearly free -> rmin), length-aware mixing for long-context SFT.
    ap.add_argument("--probe-sched", default=None, help="replicate: manifest.json of the gsort sched the probe run used")
    ap.add_argument("--probe-log", default=None, help="replicate: trainer_log.jsonl of the finished gsort probe run")
    ap.add_argument("--probe-epoch", type=int, default=2, help="replicate: 1-based probe epoch to read losses from (2 = after warmup)")
    ap.add_argument("--alpha", type=float, default=4.0, help="replicate: loss exponent")
    ap.add_argument("--beta", type=float, default=0.5, help="replicate: length exponent")
    ap.add_argument("--rmin", type=int, default=2)
    ap.add_argument("--rmax", type=int, default=16)
    ap.add_argument("--pad-token-id", type=int, default=PAD_TOKEN_ID)
    args = ap.parse_args()
    t0 = time.time()

    cols, schema = read_cache(args.cache, COLS)
    n_rows = len(cols["input_ids"])
    lens = np.array([len(a) for a in cols["input_ids"]])
    print(f"[{time.time() - t0:.0f}s] {n_rows} rows, tokens p50 {int(np.median(lens))} max {lens.max()}", flush=True)

    # ---- trajectories: cache rows of one (job, env_key, sample). cache row order == prepped jsonl row order; the
    # prepped row's `src_row` points into the SOURCE manifest, which carries the trajectory key ----
    src = [json.loads(line) for line in open(args.manifest_src, encoding="utf-8")]
    src_row = []
    with open(args.rows_jsonl, encoding="utf-8") as f:
        for line in f:
            src_row.append(int(json.loads(line)["src_row"]))
    assert len(src_row) == n_rows, (len(src_row), n_rows)
    assert max(src_row) < len(src), (max(src_row), len(src))
    traj_rows: dict = {}
    for i, r in enumerate(src_row):
        sr = src[r]
        traj_rows.setdefault((sr["job"], sr["env_key"], sr["sample"]), []).append(i)
    # sanity: every trajectory's rows must carry distinct source rows (no duplicate segment)
    for k, rs in traj_rows.items():
        assert len({src_row[i] for i in rs}) == len(rs), ("duplicate source row inside trajectory", k)
    rows_of = list(traj_rows.values())
    lengths = [int(lens[rs].sum()) for rs in rows_of]
    n = len(rows_of)
    print(f"[{time.time() - t0:.0f}s] {n_rows} rows -> {n} trajectories; rows/traj max {max(map(len, rows_of))}; "
          f"traj tokens p50 {int(np.median(lengths))} max {max(lengths)}", flush=True)

    # ---- schedule ----
    groups: list[list[int]] = []
    pads: list[int] = []
    replicate_meta = None
    if args.order == "replicate":
        assert args.probe_sched and args.probe_log, "--order replicate needs --probe-sched and --probe-log"
        groups, pads, replicate_meta = replicate_schedule(args, rows_of, lengths, traj_rows, src, src_row, t0)
    for ep in (range(args.epochs) if args.order != "replicate" else ()):
        if args.order == "global_sort":
            # one global length sort, longest first (step 0 = the 64 longest trajectories, OOM shows up at once);
            # ties broken by trajectory index so the order is a pure function of the data
            order = sorted(range(n), key=lambda i: (-lengths[i], i))
        elif args.order == "random":
            # no length grouping at all: a seeded random permutation per epoch, sliced 64/step. Within a step the
            # layout below still sorts rows longest-first into slots (rank j%world, micro-batch j//world), so each
            # rank walks the step's whole length descent -> per-rank total load stays balanced.
            gen = torch.Generator()
            gen.manual_seed(args.seed + ep)
            order = torch.randperm(n, generator=gen).tolist()
        elif args.order == "random_fixed":
            # random's mixed-length steps (fill ~75%, ga 22 vs 60 for global_sort on dsv41_history, ~16% faster per
            # epoch) but ONE permutation drawn from `seed` and reused in every epoch, so the step composition is
            # identical across epochs like global_sort (per-step loss stays comparable epoch to epoch).
            gen = torch.Generator()
            gen.manual_seed(args.seed)
            order = torch.randperm(n, generator=gen).tolist()
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
    spe_out = len(groups) // args.epochs   # replicate: "epoch" = 1/E of the token budget (checkpoint cadence only)
    print(f"[{time.time() - t0:.0f}s] {len(groups)} steps ({spe_out}/epoch x {args.epochs}); "
          f"rows/step p50 {int(np.median(list(map(len, step_rows))))} max {max_rows} -> G = ga = {G}, "
          f"{slots} slots/step, fill {sum(map(len, step_rows)) / (slots * len(groups)):.1%}", flush=True)

    # ---- lay out: step s -> slots [slots*s, slots*(s+1)); slot j -> rank j % world, micro-batch j // world ----
    out = {k: [] for k in COLS}
    dummy_ids = np.full(8, args.pad_token_id, dtype=np.int32)
    n_dummy = 0
    steps_meta = []
    for s, rows in enumerate(step_rows):
        rows = sorted(rows, key=lambda i: -lens[i])  # longest first: balances the ZeRO-3 step time per rank
        for j in range(slots):
            if j < len(rows):
                i = rows[j]
                for k in COLS:
                    out[k].append(cols[k][i])
            else:
                out["input_ids"].append(dummy_ids)
                out["attention_mask"].append(np.ones(8, dtype=np.int32))
                out["labels"].append(np.full(8, IGNORE_INDEX, dtype=np.int64))
                n_dummy += 1
        steps_meta.append({"step": s, "trajs": groups[s], "pad_trajs": pads[s], "rows": len(rows),
                           "supervised": int(sum((cols["labels"][i] != IGNORE_INDEX).sum() for i in rows)),
                           "tokens": int(lens[rows].sum())})
    assert len(out["input_ids"]) == slots * len(groups)

    ds = build_dataset(out, schema)
    os.makedirs(args.out, exist_ok=True)
    DatasetDict({"train": ds}).save_to_disk(args.out)
    manifest = {
        "source_cache": args.cache, "manifest_src": args.manifest_src, "rows_jsonl": args.rows_jsonl, "world": args.world, "G": G,
        "gradient_accumulation_steps": G, "trajs_per_step": args.trajs_per_step, "epochs": args.epochs,
        "seed": args.seed, "order": args.order, "tail": args.tail,
        "tail_dropped_per_epoch": (n % args.trajs_per_step if args.tail == "drop" and n % args.trajs_per_step else 0),
        "n_trajs": n, "n_source_rows": n_rows, "steps": len(groups),
        "steps_per_epoch": spe_out, "rows": slots * len(groups), "dummy_rows": n_dummy,
        "slots_per_step": slots, "traj_lengths": lengths, "per_step": steps_meta,
        "replicate": replicate_meta,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"[{time.time() - t0:.0f}s] sched cache: {len(ds)} rows ({n_dummy} dummy) -> {args.out}; train with "
          f"max_steps={len(groups)} gradient_accumulation_steps={G} disable_shuffling=true on {args.world} ranks",
          flush=True)


if __name__ == "__main__":
    main()