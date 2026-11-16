#!/usr/bin/env python3
"""fla's varlen causal_conv1d computes offsets in int32: with C = 8192 channels (Qwen3.5-4B GDN) a packed
length > 2^31 / C = 262144 tokens reads garbage (NaN grads, flash step 1) or faults (IMA at 275K).
tree_attn._conv_varlen now convolves the packed branches in chunks of whole segments below that limit.

This check runs the raw fla kernel (expected: wrong/IMA past 262144 -> only up to 265K here) and the
chunked _conv_varlen (expected: exact) against a per-segment Dao-AILab causal_conv1d_fn reference on
EVERY segment, fwd + bwd, at the lengths that appear in the flash cache (max branch-token sum 274,858).

    sbatch --reservation=rsi --export=ALL,SCRIPT=coding_sft/checks/check_conv_len.py coding_sft/launch/slurm/run_check_state.slurm
"""

import os
import sys

import torch
from causal_conv1d import causal_conv1d_fn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))
import tree_attn  # noqa: E402
from tree_attn import _conv_varlen  # noqa: E402

C, W = 8192, 4


def reference(x_cl: torch.Tensor, weight, bias, cu: list[int], dy_cl: torch.Tensor):
    """Per-segment Dao-AILab conv (channel-first [1, C, L]); returns y, dx, dw, db."""
    xl = x_cl.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    bb = bias.detach().clone().requires_grad_(True)
    ys = [causal_conv1d_fn(x=xl[:, :, a:b].contiguous(), weight=w, bias=bb, activation="silu") for a, b in zip(cu, cu[1:])]
    y = torch.cat(ys, dim=-1)
    y.backward(dy_cl)
    return y.detach(), xl.grad, w.grad, bb.grad


def main() -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = (torch.randn(C, W, device="cuda", generator=gen) * 0.3).bfloat16().requires_grad_(True)
    bias = (torch.randn(C, device="cuda", generator=gen) * 0.1).bfloat16().requires_grad_(True)
    print(f"default CONV_MAX_ELEMS={tree_attn.CONV_MAX_ELEMS} -> max packed len {tree_attn.CONV_MAX_ELEMS // C}", flush=True)
    # (L, chunk limit in elements); limit 2**40 = no chunking (raw fla kernel) to locate the bwd threshold
    D = 2**31 - 2**24  # tree_attn default
    cases = [(255_000, D), (262_200, D), (274_858, D), (294_912, D), (294_912, 2**30)]
    for L, lim in cases:
        tree_attn.CONV_MAX_ELEMS = lim
        n_seg = 120
        cuts = torch.linspace(0, L, n_seg + 1).long().tolist()
        cuts[-1] = L
        cu_t = torch.tensor(cuts, dtype=torch.int32, device="cuda")
        x = torch.randn(1, C, L, device="cuda", generator=gen).bfloat16()  # channel-first like tree_gdn_forward
        dy = torch.randn(1, C, L, device="cuda", generator=gen).bfloat16()
        y_ref, dx_ref, dw_ref, db_ref = reference(x, weight, bias, cuts, dy)
        try:
            xl = x.clone().requires_grad_(True)
            y = _conv_varlen(xl, weight, bias, "silu", cu_t)
            y.backward(dy)
            torch.cuda.synchronize()
        except Exception as e:  # IMA poisons the context: report and stop
            print(f"L={L:7d} lim={lim / 2**31:.3f}*2^31 -> {type(e).__name__}: {str(e)[:120]}", flush=True)
            return
        dx, dw, db = xl.grad, weight.grad.clone(), bias.grad.clone()
        weight.grad = None
        bias.grad = None
        fin = all(bool(torch.isfinite(t).all()) for t in (y, dx, dw, db))
        e_y = float((y.float() - y_ref.float()).abs().max())
        ex = (dx.float() - dx_ref.float()).abs().amax(dim=1)[0]  # [L]
        bad = (ex > 0.25).nonzero().flatten()
        where = f"bad tokens {bad.numel()} first {int(bad[0]) if bad.numel() else -1}" 
        rel_w = float((dw.float() - dw_ref.float()).norm() / dw_ref.float().norm())
        rel_b = float((db.float() - db_ref.float()).norm() / db_ref.float().norm())
        print(f"L={L:7d} L*C={L * C / 2**31:.3f}*2^31 lim={lim / 2**31:.3f}*2^31 chunk_len={min(L, lim // C)} | finite {fin} "
              f"| max|y-ref| {e_y:.3g} max|dx-ref| {float(ex.max()):.3g} ({where}) | rel dw {rel_w:.2e} db {rel_b:.2e}", flush=True)
        del x, dy, xl, y, dx, y_ref, dx_ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
