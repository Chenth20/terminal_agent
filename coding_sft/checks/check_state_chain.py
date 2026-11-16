#!/usr/bin/env python3
"""M3: which recurrent state does a branch have to start from? (arm B fork-state convention)

Part A pins down fla's `chunk_gated_delta_rule` convention with the raw op (no model): chaining
[0,cut) -> [cut,T) from the returned final_state must reproduce the single-pass output, and does so only if
final_state is the state *at* position `cut` (i.e. after consuming tokens [0,cut)). Part B then runs the real
tree GDN with a fork far away from the other forks, under both conventions:

    TREE_FORK_STATE=own    initial_state = state at fork          <- must match arm A
    TREE_FORK_STATE=next   initial_state = state at the NEXT fork  <- off by one segment

Only the convention that reproduces arm A's per-token NLL is admissible.

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_state_chain.py
"""

import os
import sys

for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
from tree_attn import TreeLayout, apply_tree_patch  # noqa: E402

TRUNK, FORKS, LENS = 384, [96, 192, 288], [64, 48, 80]
VOCAB = 256


def part_a() -> None:
    print("=== part A: fla chunk_gated_delta_rule state convention (raw op) ===", flush=True)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    B, T, H, K, V = 1, 320, 32, 128, 128
    for dtype in (torch.float32, torch.bfloat16):
        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
        beta = torch.rand(B, T, H, dtype=dtype, device="cuda") * 0.5
        g = -torch.rand(B, T, H, dtype=dtype, device="cuda") * 0.5

        def run(a, b, state):
            return chunk_gated_delta_rule(
                q[:, a:b].contiguous(), k[:, a:b].contiguous(), v[:, a:b].contiguous(),
                g=g[:, a:b].contiguous(), beta=beta[:, a:b].contiguous(),
                initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )

        o_full, _ = run(0, T, None)
        for cut in (192, 250):  # 192 = chunk-aligned (64*3), 250 = misaligned
            o1, s1 = run(0, cut, None)
            o2, _ = run(cut, T, s1)
            d1 = (o1.float() - o_full.float()[:, :cut]).abs()
            d2 = (o2.float() - o_full.float()[:, cut:]).abs()
            print(f"  {str(dtype):15s} cut={cut:4d}: prefix max|d| {d1.max():.6f} mean {d1.mean():.2e} | "
                  f"suffix max|d| {d2.max():.6f} mean {d2.mean():.2e}", flush=True)


def make_sample():
    g = torch.Generator().manual_seed(7)
    trunk = torch.randint(2, VOCAB, (TRUNK,), generator=g)
    branches = [torch.randint(2, VOCAB, (n,), generator=g) for n in LENS]
    ids = torch.cat([trunk] + branches)
    labels = torch.cat([torch.full((TRUNK,), -100), *[b.clone() for b in branches]])
    pos = torch.cat([torch.arange(TRUNK), *[torch.arange(f, f + n) for f, n in zip(FORKS, LENS)]])
    return ids, labels, pos


def branch_refs(model, ids):
    refs = []
    for b, (f, n) in enumerate(zip(FORKS, LENS)):
        br = ids[TRUNK + sum(LENS[:b]): TRUNK + sum(LENS[:b]) + n]
        seq = torch.cat([ids[:f], br])
        p = torch.cat([torch.arange(f), torch.arange(f, f + n)])
        with torch.no_grad():
            lg = model(input_ids=seq[None].cuda(), position_ids=p[None].cuda(), use_cache=False).logits[0].float()
        at = torch.arange(f - 1, f - 1 + n, device=lg.device)
        refs.append(-torch.log_softmax(lg[at], -1).gather(1, br.cuda()[:, None]).squeeze(1))
    return refs


def part_b(dtype) -> None:
    from check_tree_synth import tiny_config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    print(f"\n=== part B: tree GDN branch state, tiny model, {dtype} ===", flush=True)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config(["linear_attention"] * 4)).cuda().eval().to(dtype)
    ids, labels, pos = make_sample()
    refs = branch_refs(model, ids)

    layout = TreeLayout(TRUNK, FORKS, LENS, pad_id=0)
    batch = layout.build(ids.tolist(), labels.tolist(), pos.tolist())
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    for mode in ("next", "own"):
        os.environ["TREE_FORK_STATE"] = mode
        with torch.no_grad():
            lg = model(**{k: v for k, v in batch.items()}, use_cache=False).logits[0].float()
        worst = 0.0
        line = []
        for b, (f, n) in enumerate(zip(FORKS, LENS)):
            st = layout.starts[b]
            at = torch.tensor([f - 1] + list(range(st, st + n - 1)), device=lg.device)
            br = ids[TRUNK + sum(LENS[:b]): TRUNK + sum(LENS[:b]) + n]
            got = -torch.log_softmax(lg[at], -1).gather(1, br.cuda()[:, None]).squeeze(1)
            d = (got - refs[b]).abs()
            worst = max(worst, d.max().item())
            line.append(f"b{b}: tree {got.mean():.5f} A {refs[b].mean():.5f} max|d| {d.max():.5f}")
        print(f"  TREE_FORK_STATE={mode:4s} worst max|d| {worst:.6f} -> {'PASS' if worst < 0.02 else 'FAIL'}",
              flush=True)
        for x in line:
            print("      " + x, flush=True)


def main() -> None:
    apply_tree_patch()
    part_a()
    part_b(torch.float32)
    part_b(torch.bfloat16)


if __name__ == "__main__":
    main()