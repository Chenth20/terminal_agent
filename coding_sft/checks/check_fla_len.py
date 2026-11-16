#!/usr/bin/env python3
"""Does fla's chunk_gated_delta_rule (varlen) stay finite/correct when the packed length crosses 2^18?

With H=32, K=V=128 the per-chunk state tensor [NT, H, K, V] has NT*2^19 elements, i.e. it reaches 2^31 at
L = 262144 (NT = 4096, chunk 64) -> an int32 offset overflow would show up exactly there.

For every L: random packed batch of segments (Qwen3.5-4B GDN shapes), forward + backward, then
  * finiteness of output and all grads,
  * the FIRST segment's output/grads vs the same segment run alone (L small, known good).

    sbatch --reservation=rsi --export=ALL,SCRIPT=coding_sft/checks/check_fla_len.py coding_sft/launch/slurm/run_check_state.slurm
"""

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

H, K, V = 32, 128, 128
SEG0 = 4096  # first segment, compared against a stand-alone run


def make(L: int, n_seg: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)  # noqa: E731
    q = r(1, L, H, K).bfloat16()
    k = r(1, L, H, K).bfloat16()
    v = r(1, L, H, V).bfloat16()
    g = -F.softplus(r(1, L, H)).float() * 0.1
    beta = torch.rand(1, L, H, device="cuda", generator=gen).bfloat16()
    S0 = (r(n_seg, H, K, V) * 0.1).float()
    # n_seg segments: [0, SEG0) then n_seg-1 roughly equal pieces of the rest
    cuts = torch.linspace(SEG0, L, n_seg, device="cpu").long().tolist()
    cu = torch.tensor([0] + cuts, dtype=torch.int32, device="cuda")
    assert cu.numel() == n_seg + 1 and int(cu[-1]) == L
    do = r(1, L, H, V).bfloat16()
    return q, k, v, g, beta, S0, cu, do


def run(q, k, v, g, beta, S0, cu, do):
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta, S0)]
    o, _ = chunk_gated_delta_rule(*leaves[:3], g=leaves[3], beta=leaves[4], initial_state=leaves[5],
                                  output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    o.backward(do)
    return o.detach(), [t.grad for t in leaves]


def main() -> None:
    names = ["dq", "dk", "dv", "dg", "dbeta", "dS0"]
    for L in [200_000, 249_856, 261_888, 262_144, 262_400, 270_000, 275_000]:
        q, k, v, g, beta, S0, cu, do = make(L, n_seg=100, seed=L)
        o, gr = run(q, k, v, g, beta, S0, cu, do)
        fin = {"o": bool(torch.isfinite(o).all())} | {n: bool(torch.isfinite(x).all()) for n, x in zip(names, gr)}
        # reference: first segment alone
        s = slice(0, SEG0)
        cu0 = torch.tensor([0, SEG0], dtype=torch.int32, device="cuda")
        o0, gr0 = run(q[:, s], k[:, s], v[:, s], g[:, s], beta[:, s], S0[:1], cu0, do[:, s])
        err = {"o": float((o[:, s].float() - o0.float()).abs().max())}
        for n, a, b in zip(names, gr, gr0):
            a_ = a[:1] if n == "dS0" else a[:, s]
            err[n] = float((a_.float() - b.float()).abs().max())
        print(f"L={L:7d} NT={-(-L // 64):5d} NT*H*K*V={-(-L // 64) * H * K * V / 2**31:.3f}*2^31 | finite {fin} | "
              f"seg0 max|diff| " + " ".join(f"{n}={e:.3g}" for n, e in err.items()), flush=True)
        del q, k, v, g, beta, S0, do, o, gr, o0, gr0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
