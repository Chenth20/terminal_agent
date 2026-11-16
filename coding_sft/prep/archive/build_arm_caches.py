#!/usr/bin/env python3
"""Materialise the per-arm training caches that follow the frozen group schedule.

Both arms consume their cache with ``disable_shuffling: true`` (SequentialSampler) on 16 ranks with
``per_device_train_batch_size: 1``. accelerate's ``BatchSamplerShard`` hands global batch ``i`` to rank
``i % 16``, so optimizer step ``s`` with ``ga`` micro-batches per rank consumes rows
``[16*ga*s, 16*ga*(s+1))``. This script lays rows out exactly like that.

baseline  -> ``<base cache>_sched``: the 64 whole trajectories of group ``s`` at rows [64s, 64s+64)  (ga = 4).
A (perdh) -> ``<perdh cache>_sched_pack``: every per-step row of the same 64 trajectories, packed into
             exactly ``16*G`` bins of <= cutoff tokens per step (LPT-balanced, neat-packing layout:
             sub-sequence ids in ``attention_mask``, restarting ``position_ids``, ``packing_params``),
             at rows [16*G*s, 16*G*(s+1))  (ga = G, printed at the end and written to the manifest).

Before packing, the supervised token sequence of every trajectory in the perdh cache (concatenation of
its per-step rows, last turn only) is asserted to be identical to the baseline cache's supervised tokens
for that trajectory, so both arms optimise exactly the same targets with the same ``num_items_in_batch``.

Run inside the training container (datasets 4.x) - CPU only, ~60 GB RAM:

    python coding_sft/prep/build_arm_caches.py \
        --base-cache  /storage/openpsi/users/chen/data/th3k_align/tokenized_qwen3_5_text_262144 \
        --perdh-cache /storage/openpsi/users/chen/data/th3k_align_perdh/tokenized_qwen3_5_text_262144 \
        --perdh-jsonl /storage/openpsi/users/chen/data/th3k_align_perdh/th3k_perdh.jsonl \
        --schedule    /storage/openpsi/users/chen/data/th3k_align/group_schedule.json
"""

import argparse
import glob
import heapq
import json
import os
import time

import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, Features, Value

try:
    from datasets import List as SeqFeature  # datasets >= 4
except ImportError:  # datasets 3.x
    from datasets import Sequence as SeqFeature

IGNORE_INDEX = -100
MAX_SU_SEQ_IDX = 2**32  # same sentinel LlamaFactory writes for "no images in this pack"


def read_cache(path: str, columns: list[str]) -> dict[str, list[np.ndarray]]:
    """Read list columns of a saved-to-disk LF cache straight from its arrow shards (row order preserved)."""
    train_dir = os.path.join(path, "train")
    with open(os.path.join(train_dir, "state.json"), encoding="utf-8") as f:
        files = [d["filename"] for d in json.load(f)["_data_files"]]
    out: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for fn in files:
        with pa.memory_map(os.path.join(train_dir, fn), "r") as src:
            table = pa.ipc.open_stream(src).read_all()
        for c in columns:
            for chunk in table.column(c).chunks:
                offsets = chunk.offsets.to_numpy()
                values = chunk.values.to_numpy(zero_copy_only=False)
                out[c].extend(values[offsets[i] : offsets[i + 1]].copy() for i in range(len(offsets) - 1))
    return out


def perdh_index(jsonl: str) -> list[tuple[int, int]]:
    """(traj_idx, step) per row of the perdh jsonl, read from the trailing fields without parsing contents."""
    idx = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            tail = line[line.rfind('"traj_idx"') :].rstrip().rstrip("}")
            d = json.loads("{" + tail + "}")
            idx.append((int(d["traj_idx"]), int(d["step"])))
    return idx


def lpt_bins(lengths: list[int], n_bins: int) -> list[list[int]]:
    """Longest-processing-time-first: spread rows over exactly n_bins bins with balanced token loads."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    heap = [(0, b) for b in range(n_bins)]
    heapq.heapify(heap)
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i in order:
        load, b = heapq.heappop(heap)
        bins[b].append(i)
        heapq.heappush(heap, (load + lengths[i], b))
    return bins


def ffd_count(lengths: list[int], capacity: int) -> int:
    """Lower-ish bound on bins: first-fit-decreasing."""
    loads: list[int] = []
    for L in sorted(lengths, reverse=True):
        for j, cur in enumerate(loads):
            if cur + L <= capacity:
                loads[j] = cur + L
                break
        else:
            loads.append(L)
    return len(loads)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--perdh-cache", required=True)
    ap.add_argument("--perdh-jsonl", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--cutoff", type=int, default=262144)
    ap.add_argument("--base-out", default=None, help="default <base-cache>_sched")
    ap.add_argument("--perdh-out", default=None, help="default <perdh-cache>_sched_pack")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--world", type=int, default=16, help="number of ranks (GPUs) the trainer runs on")
    ap.add_argument("--perdh-unpacked-out", default=None,
                    help="also write an UNPACKED A cache: one perdh row per micro-batch, each step padded to "
                         "world*G_u rows with zero-label dummy rows (checks that packing itself does not change the loss)")
    ap.add_argument("--pad-token-id", type=int, default=248044, help="pad token used for dummy rows (Qwen3.5 <|endoftext|>)")
    args = ap.parse_args()
    WORLD = args.world
    base_out = args.base_out or args.base_cache.rstrip("/") + "_sched"
    perdh_out = args.perdh_out or args.perdh_cache.rstrip("/") + "_sched_pack"

    t0 = time.time()
    with open(args.schedule, encoding="utf-8") as f:
        sched = json.load(f)
    groups: list[list[int]] = sched["groups"]
    TPS = sched["trajs_per_step"]
    assert all(len(g) == TPS for g in groups), f"every group must hold {TPS} trajectories (padded)"
    assert TPS % WORLD == 0, (TPS, WORLD)

    base = read_cache(args.base_cache, ["input_ids", "attention_mask", "labels"])
    n_traj = len(base["input_ids"])
    print(f"[{time.time() - t0:.0f}s] base cache: {n_traj} trajectories")
    assert n_traj == len(sched["lengths"]), (n_traj, len(sched["lengths"]))

    # ---------------- baseline: whole trajectories in schedule order -----------------
    if not args.skip_base:
        order = [t for g in groups for t in g]
        feats = Features(
            {
                "input_ids": SeqFeature(Value("int32")),
                "attention_mask": SeqFeature(Value("int32")),
                "labels": SeqFeature(Value("int64")),
                "images": Value("null"),
                "videos": Value("null"),
                "audios": Value("null"),
            }
        )
        ds = Dataset.from_dict(
            {
                "input_ids": [base["input_ids"][t] for t in order],
                "attention_mask": [base["attention_mask"][t] for t in order],
                "labels": [base["labels"][t] for t in order],
                "images": [None] * len(order),
                "videos": [None] * len(order),
                "audios": [None] * len(order),
            },
            features=feats,
        )
        DatasetDict({"train": ds}).save_to_disk(base_out)
        print(f"[{time.time() - t0:.0f}s] baseline cache: {len(order)} rows -> {base_out}  (ga = {TPS // WORLD})")

    # ---------------- A: per-step rows, verified, packed per group -----------------
    rows = perdh_index(args.perdh_jsonl)
    perdh = read_cache(args.perdh_cache, ["input_ids", "labels"])
    assert len(perdh["input_ids"]) == len(rows), (len(perdh["input_ids"]), len(rows))
    print(f"[{time.time() - t0:.0f}s] perdh cache: {len(rows)} rows")
    by_traj: dict[int, list[int]] = {}
    for r, (t, k) in enumerate(rows):
        by_traj.setdefault(t, []).append(r)
    for t, rs in by_traj.items():
        steps = [rows[r][1] for r in rs]
        assert steps == list(range(len(rs))), f"traj {t}: steps not contiguous {steps[:5]}..."

    # supervised-token identity: baseline(traj) == concat over steps of perdh(traj, step)
    n_sup_total = 0
    used = sorted({t for g in groups for t in g})
    for t in used:
        b = base["labels"][t]
        b_sup = b[b != IGNORE_INDEX]
        p_sup = np.concatenate([perdh["labels"][r][perdh["labels"][r] != IGNORE_INDEX] for r in by_traj[t]])
        assert b_sup.shape == p_sup.shape and np.array_equal(b_sup, p_sup), (
            f"traj {t}: supervised tokens differ (base {b_sup.shape[0]} vs perdh {p_sup.shape[0]})"
        )
        n_sup_total += int(b_sup.shape[0])
    print(f"[{time.time() - t0:.0f}s] supervised tokens identical for all {len(used)} scheduled trajectories ({n_sup_total:,} tokens)")

    plen = [len(x) for x in perdh["input_ids"]]
    assert max(plen) <= args.cutoff
    step_rows: list[list[int]] = [[r for t in g for r in by_traj[t]] for g in groups]  # duplicates for padded members
    n_min = [ffd_count([plen[r] for r in rs], args.cutoff) for rs in step_rows]
    G = max(1, (max(n_min) + WORLD - 1) // WORLD)
    while True:
        B = WORLD * G
        plans = [lpt_bins([plen[r] for r in rs], B) for rs in step_rows]
        worst = max(max(sum(plen[rs[i]] for i in b) for b in bins) for rs, bins in zip(step_rows, plans))
        if worst <= args.cutoff:
            break
        G += 1
    print(f"[{time.time() - t0:.0f}s] bins/step min {min(n_min)} max {max(n_min)} -> G = {G} micro-batches/rank, {B} bins/step, fullest bin {worst:,} tokens")

    feats = Features(
        {
            "input_ids": SeqFeature(Value("int32")),
            "attention_mask": SeqFeature(Value("int32")),
            "position_ids": SeqFeature(Value("int32")),
            "labels": SeqFeature(Value("int64")),
            "packing_params": {
                "sequence_boundaries": SeqFeature(Value("int64")),
                "image_subseq_ids": SeqFeature(Value("int64")),
                "video_subseq_ids": SeqFeature(Value("int64")),
                "audio_subseq_ids": SeqFeature(Value("int64")),
                "right_padding_length": Value("int64"),
            },
            "images": Value("null"),
            "videos": Value("null"),
            "audios": Value("null"),
        }
    )
    manifest = {"G": G, "bins_per_step": B, "cutoff": args.cutoff, "world": WORLD, "trajs_per_step": TPS, "base_ga": TPS // WORLD, "steps": []}
    shards = []
    for s, (rs, bins) in enumerate(zip(step_rows, plans)):
        cols = {k: [] for k in ("input_ids", "attention_mask", "position_ids", "labels", "packing_params")}
        step_tok = step_sup = 0
        bin_tok = []
        for b in bins:  # bin index j -> micro-batch j // WORLD on rank j % WORLD
            members = [rs[i] for i in b]
            members.sort(key=lambda r: (rows[r][0], rows[r][1]))  # deterministic order inside the bin
            ids, att, pos, lab, bounds = [], [], [], [], [0]
            for j, r in enumerate(members):
                x = perdh["input_ids"][r]
                ids.append(x)
                att.append(np.full(len(x), j + 1, dtype=np.int32))
                pos.append(np.arange(len(x), dtype=np.int32))
                lab.append(perdh["labels"][r])
                bounds.append(bounds[-1] + len(x))
            ids = np.concatenate(ids); lab = np.concatenate(lab)
            assert len(ids) <= args.cutoff and len(members) > 0
            cols["input_ids"].append(ids.astype(np.int32))
            cols["attention_mask"].append(np.concatenate(att))
            cols["position_ids"].append(np.concatenate(pos))
            cols["labels"].append(lab)
            cols["packing_params"].append(
                {
                    "sequence_boundaries": bounds,
                    "image_subseq_ids": [MAX_SU_SEQ_IDX],
                    "video_subseq_ids": [MAX_SU_SEQ_IDX],
                    "audio_subseq_ids": [MAX_SU_SEQ_IDX],
                    "right_padding_length": 0,
                }
            )
            step_tok += len(ids); step_sup += int((lab != IGNORE_INDEX).sum()); bin_tok.append(len(ids))
        n = len(cols["input_ids"])
        cols["images"] = [None] * n; cols["videos"] = [None] * n; cols["audios"] = [None] * n
        shards.append(Dataset.from_dict(cols, features=feats))
        manifest["steps"].append(
            {"step": s, "rows": len(rs), "bins": n, "tokens": step_tok, "supervised": step_sup,
             "bin_tokens_min": min(bin_tok), "bin_tokens_max": max(bin_tok)}
        )
        base_sup = sum(int((base["labels"][t] != IGNORE_INDEX).sum()) for t in groups[s])
        assert step_sup == base_sup, (s, step_sup, base_sup)
        print(f"[{time.time() - t0:.0f}s] step {s:2d}: {len(rs):5d} rows -> {n} bins, {step_tok / 1e6:6.1f}M tokens "
              f"(bin {min(bin_tok) / 1e3:.0f}K..{max(bin_tok) / 1e3:.0f}K), {step_sup / 1e6:.2f}M supervised == baseline")

    from datasets import concatenate_datasets

    if args.perdh_unpacked_out:
        # one row per micro-batch, rows of step s at [world*Gu*s, world*Gu*(s+1)); shorter steps are filled with
        # dummy rows (8 pad tokens, all labels IGNORE) that contribute 0 to the loss sum and 0 to num_items_in_batch
        Gu = (max(len(rs) for rs in step_rows) + WORLD - 1) // WORLD
        ufeats = Features(
            {
                "input_ids": SeqFeature(Value("int32")),
                "attention_mask": SeqFeature(Value("int32")),
                "labels": SeqFeature(Value("int64")),
                "images": Value("null"),
                "videos": Value("null"),
                "audios": Value("null"),
            }
        )
        ucols = {k: [] for k in ("input_ids", "attention_mask", "labels")}
        dummy = np.full(8, args.pad_token_id, dtype=np.int32)
        n_dummy = 0
        for s, rs in enumerate(step_rows):
            # rank-major interleave is irrelevant for correctness (every rank's rows are summed); keep traj/step order
            members = sorted(rs, key=lambda r: (rows[r][0], rows[r][1]))
            for r in members:
                x = perdh["input_ids"][r]
                ucols["input_ids"].append(x.astype(np.int32))
                ucols["attention_mask"].append(np.ones(len(x), dtype=np.int32))
                ucols["labels"].append(perdh["labels"][r])
            for _ in range(WORLD * Gu - len(members)):
                ucols["input_ids"].append(dummy)
                ucols["attention_mask"].append(np.ones(8, dtype=np.int32))
                ucols["labels"].append(np.full(8, IGNORE_INDEX, dtype=np.int64))
                n_dummy += 1
        n = len(ucols["input_ids"])
        ucols["images"] = [None] * n; ucols["videos"] = [None] * n; ucols["audios"] = [None] * n
        uds = Dataset.from_dict(ucols, features=ufeats)
        DatasetDict({"train": uds}).save_to_disk(args.perdh_unpacked_out)
        manifest["unpacked"] = {"G": Gu, "rows_per_step": WORLD * Gu, "dummy_rows": n_dummy, "path": args.perdh_unpacked_out}
        print(f"[{time.time() - t0:.0f}s] A-unpacked cache: {n} rows ({n_dummy} dummy) -> {args.perdh_unpacked_out}")
        print(f"TRAIN A-UNPACKED WITH: tokenized_path={args.perdh_unpacked_out} gradient_accumulation_steps={Gu} packing=false")

    ds = concatenate_datasets(shards)
    DatasetDict({"train": ds}).save_to_disk(perdh_out, num_shards={"train": len(shards)})
    with open(os.path.join(perdh_out, "pack_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    tot = sum(x["tokens"] for x in manifest["steps"])
    print(f"[{time.time() - t0:.0f}s] A cache: {len(ds)} rows ({tot / 1e9:.2f}B tokens) -> {perdh_out}")
    print(f"TRAIN WITH: tokenized_path={perdh_out} gradient_accumulation_steps={G} packing=true neat_packing=true")


if __name__ == "__main__":
    main()
