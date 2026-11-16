#!/usr/bin/env python3
"""M2: arm-B tree forward vs arm-A per-step rows on the REAL model and the REAL cache.

For a tree row (one trajectory) every branch_k must give the same per-token NLL as arm-A's row (traj, k)
run alone: arm A encodes the same supervised tokens with the same visible context. Also compares the
gradient of the summed branch loss (tree) with the sum of the per-row gradients (arm A).

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_tree_vs_perdh.py            # shortest + longest row
    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_tree_vs_perdh.py --rows 0 5 --max-tokens 60000
"""

import argparse
import json
import os
import sys

for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prep"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))

import numpy as np
import torch
import torch.nn.functional as F
from cache_io import perdh_index, read_cache
from datasets import load_from_disk

from tree_attn import TreeDataCollator, apply_tree_patch

DATA = "/storage/openpsi/users/chen/data"
MODEL = "/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text"
PAD = 248044
IGNORE = -100


def ce_chunked(logits: torch.Tensor, labels: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Per-position CE(logits[i-1], labels[i]) for every position (ignore_index filtering left to the caller)."""
    L = logits.shape[0]
    out = []
    for i in range(1, L, chunk):
        j = min(i + chunk, L)
        out.append(F.cross_entropy(logits[i - 1: j - 1].float(), labels[i:j], reduction="none", ignore_index=IGNORE))
    return torch.cat(out)


def ce_sum(logits: torch.Tensor, labels: torch.Tensor, n: int, chunk: int = 2048) -> torch.Tensor:
    """sum_CE / n over the supervised positions, computed in chunks so the [T, vocab] CE never materialises."""
    L = logits.shape[0]
    total = logits.new_zeros((), dtype=torch.float32)
    for i in range(1, L, chunk):
        j = min(i + chunk, L)
        total = total + F.cross_entropy(logits[i - 1: j - 1].float(), labels[i:j], reduction="sum", ignore_index=IGNORE)
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-cache", default=f"{DATA}/th3k_align_tree/tokenized_qwen3_5_text_262144_dbg")
    ap.add_argument("--schedule", default=f"{DATA}/th3k_align/group_schedule_dbg.json")
    ap.add_argument("--perdh-cache", default=f"{DATA}/th3k_align_perdh/tokenized_qwen3_5_text_262144")
    ap.add_argument("--perdh-jsonl", default=f"{DATA}/th3k_align_perdh/th3k_perdh.jsonl")
    ap.add_argument("--rows", type=int, nargs="+", default=None, help="tree cache rows (default: shortest + longest)")
    ap.add_argument("--max-tokens", type=int, default=70000)
    ap.add_argument("--grad", action="store_true", help="also compare gradients (needs ~2x memory)")
    args = ap.parse_args()

    tree = load_from_disk(args.tree_cache)["train"]
    sched = json.load(open(args.schedule))
    groups, TPS = sched["groups"], sched["trajs_per_step"]
    rows_idx = perdh_index(args.perdh_jsonl)
    by_traj: dict[int, list[int]] = {}
    for r, (t, k) in enumerate(rows_idx):
        by_traj.setdefault(t, []).append(r)
    perdh = read_cache(args.perdh_cache, ["input_ids", "labels"])

    rows = args.rows
    if rows is None:
        lens = [len(tree[i]["input_ids"]) for i in range(len(tree))]
        rows = [int(np.argmin(lens)), int(np.argmax(lens))]

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    model.eval()
    apply_tree_patch()
    coll = TreeDataCollator(pad_token_id=PAD)

    worst = 0.0
    worst_mean = 0.0
    agg_all = [0.0, 0.0, 0.0]
    for row in rows:
        s, j = divmod(row, TPS)
        traj = groups[s][j]
        r = tree[row]
        trunk_len = int(r["trunk_len"])
        forks = [int(x) for x in r["tree_forks"]]
        lens = [int(x) for x in r["tree_branch_lens"]]
        n_sub = max(1, int(np.searchsorted(np.cumsum([trunk_len] + lens), args.max_tokens)))
        n_sub = min(n_sub, len(lens))
        cut = trunk_len + sum(lens[:n_sub])
        row_d = {
            "input_ids": np.asarray(r["input_ids"][:cut], dtype=np.int32),
            "labels": np.asarray(r["labels"][:cut], dtype=np.int64),
            "position_ids": np.asarray(r["position_ids"][:cut], dtype=np.int32),
            "tree_forks": forks[:n_sub], "tree_branch_lens": lens[:n_sub], "trunk_len": trunk_len,
        }
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in coll([row_d]).items()}
        print(f"\nrow {row} (step {s}, member {j}, traj {traj}): {len(r['input_ids'])} tok total, trunk {trunk_len}, "
              f"{len(lens)} branches -> checking {n_sub} branches / {cut} tok / T_pad {batch['input_ids'].shape[1]}")
        lab = batch["labels"][0]
        fwd = {k: v for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            logits = model(**fwd, use_cache=False).logits[0]
        nll_tree_all = ce_chunked(logits, lab, chunk=4096)  # nll_tree_all[p-1] = CE(logit[p-1], lab[p])

        agg = [0.0, 0.0, 0.0]  # sum nll_tree, sum nll_A, n tokens
        for b in range(n_sub):
            rk = by_traj[traj][b]
            ids = torch.tensor(perdh["input_ids"][rk], dtype=torch.long)
            labk = torch.tensor(perdh["labels"][rk], dtype=torch.long)
            with torch.no_grad():
                lg = model(input_ids=ids[None].cuda(), attention_mask=torch.ones_like(ids)[None].cuda(),
                           use_cache=False).logits[0]
            nll_ref_all = ce_chunked(lg, labk.cuda(), chunk=4096)
            sup_ref = (labk != IGNORE).nonzero().flatten()
            ref = nll_ref_all[sup_ref - 1]
            # the same tokens inside the tree: branch token 0 lives at label position fork_b, the rest at s_b+j
            st = int(batch["tree_branch_starts"][0, b])
            pos = torch.tensor([forks[b]] + list(range(st + 1, st + lens[b])), device=lab.device)
            got = nll_tree_all[pos - 1]
            assert got.numel() == ref.numel() == lens[b], (got.numel(), ref.numel(), lens[b])
            d = (got - ref).abs()
            worst = max(worst, d.max().item())
            worst_mean = max(worst_mean, d.mean().item())
            agg[0] += float(got.sum()); agg[1] += float(ref.sum()); agg[2] += lens[b]
            agg_all[0] += float(got.sum()); agg_all[1] += float(ref.sum()); agg_all[2] += lens[b]
            print(f"  branch {b:2d}: {lens[b]:4d} tok | NLL tree {got.mean():.5f} A {ref.mean():.5f} "
                  f"| max|d| {d.max():.4f} mean|d| {d.mean():.5f}")
            del lg, nll_ref_all
        n_tok = max(agg[2], 1.0)
        print(f"  ROW token-weighted: tree {agg[0] / n_tok:.6f} A {agg[1] / n_tok:.6f} "
              f"|d| {abs(agg[0] - agg[1]) / n_tok:.6f} over {int(agg[2])} supervised tokens")
        del logits, nll_tree_all
        torch.cuda.empty_cache()

    # Per-token |d| is dominated by the bf16 kernel difference (flex vs FA2: mean ~0.02, tail up to 25 on
    # identical inputs), so it is reported but NOT used as the criterion. What must match is the supervised-token
    # weighted mean NLL: tree_agg - A_agg.
    print(f"\nworst per-token |d| {worst:.4f}, worst branch mean|d| {worst_mean:.5f} (kernel noise, informational)")
    print(f"token-weighted |d| {abs(agg_all[0] - agg_all[1]) / max(agg_all[2], 1.0):.6f} over {int(agg_all[2])} "
          f"supervised tokens -> {'PASS' if abs(agg_all[0] - agg_all[1]) / max(agg_all[2], 1.0) < 1e-2 else 'FAIL'} "
          f"(limit 0.01)")

    if args.grad:
        row = rows[0]
        s, j = divmod(row, TPS)
        traj = groups[s][j]
        r = tree[row]
        trunk_len, forks, lens = int(r["trunk_len"]), [int(x) for x in r["tree_forks"]], [int(x) for x in r["tree_branch_lens"]]
        n_sub = min(len(lens), max(1, int(np.searchsorted(np.cumsum([trunk_len] + lens), args.max_tokens))))
        cut = trunk_len + sum(lens[:n_sub])
        row_d = {
            "input_ids": np.asarray(r["input_ids"][:cut], dtype=np.int32),
            "labels": np.asarray(r["labels"][:cut], dtype=np.int64),
            "position_ids": np.asarray(r["position_ids"][:cut], dtype=np.int32),
            "tree_forks": forks[:n_sub], "tree_branch_lens": lens[:n_sub], "trunk_len": trunk_len,
        }
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in coll([row_d]).items()}
        N = int((batch["labels"] != IGNORE).sum())
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        names = [n for n, p in model.named_parameters() if ("layers.3.self_attn.q_proj" in n or "layers.0.linear_attn.out_proj" in n)]
        if not names:
            names = [n for n, p in model.named_parameters() if p.dim() == 2][:2]
        model.zero_grad()
        fwd = {k: v for k, v in batch.items() if k != "labels"}
        lg = model(**fwd, use_cache=False).logits[0]
        ce_sum(lg, batch["labels"][0], N).backward()
        del lg
        g_tree = {n: p.grad.detach().clone() for n, p in model.named_parameters() if n in names}
        model.zero_grad()
        for b in range(n_sub):
            rk = by_traj[traj][b]
            ids = torch.tensor(perdh["input_ids"][rk], dtype=torch.long)
            labk = torch.tensor(perdh["labels"][rk], dtype=torch.long)
            lg = model(input_ids=ids[None].cuda(), attention_mask=torch.ones_like(ids)[None].cuda(),
                       use_cache=False).logits[0]
            ce_sum(lg, labk.cuda(), N).backward()
            del lg
        for n in names:
            g_ref = dict(model.named_parameters())[n].grad.detach()
            cos = float(F.cosine_similarity(g_tree[n].flatten(), g_ref.flatten(), dim=0))
            rel = float((g_tree[n] - g_ref).norm() / (g_ref.norm() + 1e-12))
            print(f"  grad {n}: cosine {cos:.6f} rel {rel:.4f} -> {'PASS' if cos > 0.99 else 'FAIL'}")


if __name__ == "__main__":
    main()