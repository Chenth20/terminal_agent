#!/usr/bin/env python3
"""M1: synthetic-tree parity for the arm-B patch, split so each failure is diagnosable.

Phase 1 (GDN only, all-linear tiny model, bf16): the tree GDN (trunk segment chaining + branch
  initial_state + packed conv) must reproduce, per branch token, the NLL of running that branch as a plain
  causal continuation of its prefix (arm A's target), and the same gradient. Same fla kernels on both sides,
  so a mismatch here is a logic bug, not noise.
Phase 2 (full attention only, all-full tiny model, fp32): the flex BlockMask must reproduce an *explicit*
  dense tree mask fed to the eager/sdpa path. fp32 + same projections -> tight tolerances.
Phase 3 (real layer mix, bf16): end-to-end sanity, report only.

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_tree_synth.py
"""

import os
import sys

for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
from tree_attn import STATS, TreeLayout, apply_tree_patch  # noqa: E402

MODEL = "/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text"
# multi-block: TRUNK 300 + branches pushes T_pad to 640 => 5 flex blocks, so the block-selection logic
# (which trunk blocks a branch may read, and the pads between segments) is actually exercised. The earlier
# 37-token version fit in a single block and could not catch a block-index bug.
TRUNK, FORKS, LENS = 300, [100, 200], [60, 90]


def tiny_config(layer_types):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    base = Qwen3_5TextConfig.from_pretrained(MODEL).to_dict()
    rope = dict(base.get("rope_parameters") or {})
    rope.update({"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                 "mrope_section": [2, 1, 1]})
    cfg = dict(base)
    cfg.update(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=len(layer_types),
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, layer_types=list(layer_types),
        max_position_embeddings=8192, rope_parameters=rope, pad_token_id=0, eos_token_id=1, attention_dropout=0.0,
    )
    cfg.pop("architectures", None)
    return Qwen3_5TextConfig(**cfg)


def make_sample(vocab: int, g: torch.Generator):
    trunk = torch.randint(2, vocab, (TRUNK,), generator=g)
    branches = [torch.randint(2, vocab, (n,), generator=g) for n in LENS]
    ids = torch.cat([trunk] + branches)
    cache_labels = torch.cat([torch.full((TRUNK,), -100), *[b.clone() for b in branches]])
    pos = torch.cat([torch.arange(TRUNK), *[torch.arange(f, f + n) for f, n in zip(FORKS, LENS)]])
    return ids, cache_labels, pos


def branch_targets(ids, b):
    off = TRUNK + sum(LENS[:b])
    return ids[off: off + LENS[b]]


def run_phase1(model_ctor, patch):
    print("\n=== phase 1: GDN tree logic (all linear-attention layers, bf16) ===")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    g = torch.Generator().manual_seed(7)
    ids, cache_labels, pos = make_sample(256, g)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config(["linear_attention"] * 4)).cuda().eval().to(torch.bfloat16)

    refs = []
    for b in range(len(LENS)):
        f = FORKS[b]
        br = branch_targets(ids, b)
        seq = torch.cat([ids[:f], br])
        p = torch.cat([torch.arange(f), torch.arange(f, f + len(br))])
        with torch.no_grad():
            lg = model(input_ids=seq[None].cuda(), position_ids=p[None].cuda(), use_cache=False).logits[0].float()
        positions = torch.arange(f - 1, f - 1 + len(br), device=lg.device)
        refs.append(-torch.log_softmax(lg[positions], -1).gather(1, br.cuda()[:, None]).squeeze(1))

    patch()
    layout = TreeLayout(TRUNK, FORKS, LENS, pad_id=0)
    batch = layout.build(ids.tolist(), cache_labels.tolist(), pos.tolist())
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    STATS.update(gdn_tree=0, gdn_orig=0, attn_tree=0, attn_orig=0)
    with torch.no_grad():
        out = model(**batch, use_cache=False)
    print("  path counters:", STATS)
    logits = out.logits[0].float()
    worst = 0.0
    for b in range(len(LENS)):
        f, s, n = FORKS[b], layout.starts[b], LENS[b]
        positions = torch.tensor([f - 1] + list(range(s, s + n - 1)), device=logits.device)
        got = -torch.log_softmax(logits[positions], -1).gather(1, branch_targets(ids, b).cuda()[:, None]).squeeze(1)
        d = (got - refs[b]).abs()
        worst = max(worst, d.max().item())
        print(f"  branch {b}: NLL tree {got.mean():.6f} ref {refs[b].mean():.6f} | max|d| {d.max():.6f}")
    print(f"  worst per-token |d| {worst:.6f} ->", "PASS" if worst < 0.05 else "FAIL")

    model.train()
    name, param = next((n, p) for n, p in model.named_parameters() if "layers.0.linear_attn" in n and p.dim() == 2)
    model.zero_grad()
    model(**batch, use_cache=False).loss.backward()
    g_tree = param.grad.detach().clone()
    model.zero_grad()
    for b in range(len(LENS)):
        f = FORKS[b]
        br = branch_targets(ids, b)
        seq = torch.cat([ids[:f], br])
        p = torch.cat([torch.arange(f), torch.arange(f, f + len(br))])
        lab = torch.cat([torch.full((f,), -100, dtype=torch.long), br])
        model(input_ids=seq[None].cuda(), position_ids=p[None].cuda(), labels=lab[None].cuda(), use_cache=False,
              num_items_in_batch=torch.tensor(sum(LENS)).cuda()).loss.backward()
    g_ref = param.grad.detach().clone()
    cos = float(F.cosine_similarity(g_tree.flatten(), g_ref.flatten(), dim=0))
    rel = float((g_tree - g_ref).norm() / (g_ref.norm() + 1e-12))
    print(f"  grad {name}: cosine {cos:.6f} rel {rel:.4f} ->", "PASS" if cos > 0.99 else "FAIL")


def dense_tree_mask(seg: torch.Tensor, vis_end: torch.Tensor, dtype=torch.float32):
    T = seg.numel()
    q = torch.arange(T, device=seg.device)[:, None]
    kv = torch.arange(T, device=seg.device)[None, :]
    sq, sk = seg[:, None], seg[None, :]
    allow = ((sk == 0) & (kv < vis_end[:, None])) | ((sq > 0) & (sq == sk) & (kv <= q))
    allow = allow | ((sq < 0) & (q == kv))
    m = torch.zeros((1, 1, T, T), dtype=dtype, device=seg.device)
    m.masked_fill_(~allow[None, None], float("-inf"))
    return m


def run_phase2(patch):
    print("\n=== phase 2: full-attention tree mask (all full-attention layers, fp32) ===")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    g = torch.Generator().manual_seed(7)
    ids, cache_labels, pos = make_sample(256, g)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config(["full_attention"] * 2)).cuda().eval().to(torch.float32)

    patch()
    layout = TreeLayout(TRUNK, FORKS, LENS, pad_id=0)
    batch = layout.build(ids.tolist(), cache_labels.tolist(), pos.tolist())
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    STATS.update(gdn_tree=0, gdn_orig=0, attn_tree=0, attn_orig=0)
    with torch.no_grad():
        lg_tree = model(**batch, use_cache=False).logits[0].float()
    print("  path counters:", STATS)
    seg = batch["tree_seg"][0]
    vis = batch["tree_vis_end"][0]
    m = dense_tree_mask(seg, vis)
    with torch.no_grad():
        # reference: the same layout and positions through the original path with an explicit dense tree mask
        lg_ref = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], attention_mask=m,
                       use_cache=False).logits[0].float()
    sup = batch["labels"][0] != -100
    d = (lg_tree[sup] - lg_ref[sup]).abs()
    print(f"  {int(sup.sum())} supervised positions, max|d| {d.max():.6f}, mean|d| {d.mean():.6f} ->",
          "PASS" if d.max() < 1e-3 else "FAIL")
    # also the loss path (labels already carry the fork-position trick)
    with torch.no_grad():
        l_tree = float(model(**batch, use_cache=False).loss)
        l_ref = float(model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                            attention_mask=m, labels=batch["labels"], use_cache=False).loss)
    print(f"  loss tree {l_tree:.6f} ref {l_ref:.6f} |d| {abs(l_tree - l_ref):.6f} ->",
          "PASS" if abs(l_tree - l_ref) < 1e-3 else "FAIL")


def main() -> None:
    patch_applied = {"done": False}

    def patch():
        if not patch_applied["done"]:
            apply_tree_patch()
            patch_applied["done"] = True

    run_phase1(None, patch)
    run_phase2(patch)


if __name__ == "__main__":
    main()