#!/usr/bin/env python3
"""Arm-B (tree) stress / throughput benchmark on ONE GPU with the real 4B model and real cache rows.

For every selected row: TreeDataCollator -> fwd + bwd (grad checkpointing, liger, no optimizer / ZeRO),
report peak memory, wall-clock, supervised tokens/s and -- with --profile -- where the time goes
(GDN trunk scan / GDN branch loop / flex attention / rest).

    python coding_sft/checks/bench_tree.py --cache .../flash_full_tree/tokenized_qwen3_5_text_262144 \\
        --tokens 100000 200000 300000 400000 500000 600000        # nearest row to each size
    python coding_sft/checks/bench_tree.py --rows 12 77 --profile

Memory note: the optimizer / ZeRO-3 partition is NOT included (that is ~2 GB/GPU on 16 ranks for 4B,
but the fp32 master + Adam states are sharded, so activations dominate at these lengths).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
import tree_attn  # noqa: E402
from tree_attn import STATS, TreeDataCollator, apply_tree_patch  # noqa: E402

MODEL = "/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text"
PAD = 248044


def load_rows(cache: str, rows: list[int] | None, tokens: list[int] | None):
    from datasets import load_from_disk

    ds = load_from_disk(cache)["train"]
    man_path = os.path.join(cache, "manifest.json")
    if rows is None:
        assert tokens, "--rows or --tokens"
        if os.path.exists(man_path):
            per = json.load(open(man_path))["per_row"]
            lens = np.array([r["tokens"] for r in per])
        else:
            lens = np.array([len(ds[i]["input_ids"]) for i in range(len(ds))])
        rows = [int(np.argmin(np.abs(lens - t))) for t in tokens]
    return ds, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/storage/openpsi/users/chen/data/flash_full_tree/tokenized_qwen3_5_text_262144")
    ap.add_argument("--rows", type=int, nargs="+", default=None)
    ap.add_argument("--tokens", type=int, nargs="+", default=None)
    ap.add_argument("--profile", action="store_true", help="torch.profiler breakdown of the last row")
    ap.add_argument("--no-bwd", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="time the (repeat)th run so compile/autotune is excluded")
    args = ap.parse_args()

    ds, rows = load_rows(args.cache, args.rows, args.tokens)
    print(f"rows {rows}  T_MULT={tree_attn.T_MULT} MAX_SEG={tree_attn.MAX_SEG} flex_compile={os.environ.get('TREE_FLEX_COMPILE', '1')} "
          f"ATTN_SPLIT={tree_attn.ATTN_SPLIT} SPLIT_MULT={tree_attn.SPLIT_MULT}", flush=True)

    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    apply_liger_kernel_to_qwen3_5()
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    apply_tree_patch()
    coll = TreeDataCollator(pad_token_id=PAD)
    print(f"weights {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    results = []
    for row in rows:
        r = ds[row]
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in coll([r]).items()}
        T = batch["input_ids"].shape[1]
        n_sup = int((batch["labels"] != -100).sum())
        n_br = len(r["tree_branch_lens"])
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok = True
        dt = float("nan")
        try:
            for it in range(args.repeat):
                for k in STATS:
                    STATS[k] = 0
                torch.cuda.synchronize()
                t0 = time.time()
                out = model(**batch, use_cache=False)
                if not args.no_bwd:
                    out.loss.backward()
                torch.cuda.synchronize()
                dt = time.time() - t0
                loss = float(out.loss)
                del out
                model.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError as e:
            ok = False
            print(f"row {row:5d} T_pad {T:7d} -> OOM peak_alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
                  f"resv {torch.cuda.max_memory_reserved() / 2**30:.1f} GB ({str(e)[:300]})", flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if ok:
            peak = torch.cuda.max_memory_allocated() / 2**30
            resv = torch.cuda.max_memory_reserved() / 2**30
            print(f"row {row:5d} T_pad {T:7d} trunk {int(r['trunk_len']):6d} q_split {tree_attn._CTX['ctx'].q_split if tree_attn._CTX['ctx'] else 0:7d} branches {n_br:3d} sup {n_sup:7d} | "
                  f"loss {loss:.4f} | {dt:6.1f}s  {T / dt:8.0f} tok/s  {n_sup / dt:8.0f} sup-tok/s | "
                  f"peak {peak:6.1f} GB (resv {resv:6.1f}) | calls {dict(STATS)}", flush=True)
            results.append({"row": row, "T": T, "sup": n_sup, "s": dt, "peak": peak})
        del batch

    if args.profile and rows:
        from torch.profiler import ProfilerActivity, profile

        r = ds[rows[-1]]
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in coll([r]).items()}
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            out = model(**batch, use_cache=False)
            if not args.no_bwd:
                out.loss.backward()
            torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        ev = prof.key_averages()
        total = sum(e.device_time_total for e in ev) or 1.0

        def bucket(name: str) -> str:
            n = name.lower()
            if n.startswith("chunk_") or "delta_rule" in n or "wy_fast" in n or "fused_recurrent" in n:
                return "gdn (fla)"
            if "flex" in n or "triton_tem_fused" in n or "flex_attention" in n:
                return "flex attention"
            if "causal_conv1d" in n:
                return "conv1d"
            if "gemm" in n or "cutlass" in n or "matmul" in n or "sm90" in n or "sm80" in n:
                return "gemm"
            if "cross_entropy" in n or "liger" in n or "rms" in n or "silu" in n:
                return "liger (norm/ce/mlp)"
            return "other"

        agg: dict[str, float] = {}
        for e in ev:
            agg[bucket(e.key)] = agg.get(bucket(e.key), 0.0) + e.device_time_total
        print("\nCUDA time breakdown (last row):")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v / 1e6:8.2f} s  {100 * v / total:5.1f}%")
        print("\ntop 15 kernels:")
        for e in sorted(ev, key=lambda e: -e.device_time_total)[:15]:
            print(f"  {e.device_time_total / 1e6:7.2f}s {e.count:6d}x  {e.key[:110]}")

    if results:
        print("\nsummary:")
        for x in results:
            print(f"  T={x['T']:7d}  {x['s']:6.1f}s  peak {x['peak']:5.1f} GB  {x['T'] / x['s']:7.0f} tok/s")


if __name__ == "__main__":
    main()
