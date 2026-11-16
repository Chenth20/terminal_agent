#!/usr/bin/env python3
"""Why does long-context flex_attention fall back to the dense math path (77 GiB alloc) in the tree run?

Prints the dispatch code of torch's flex_attention HOP and probes which backend each call actually uses.
Cheap, CPU-only, no model.

    SCRIPT=coding_sft/checks/probe_flex.py sbatch coding_sft/launch/slurm/run_check_state.slurm
"""

import inspect
import re
import sys

import torch

print("torch", torch.__version__, torch.__file__, flush=True)

import os as _os

mod_file = _os.path.join(_os.path.dirname(torch.__file__), "_higher_order_ops", "flex_attention.py")
src = open(mod_file).read().splitlines()
print(f"\n=== {mod_file} : lines mentioning the fallback ===", flush=True)
for i, line in enumerate(src):
    if re.search(r"sdpa_dense|math_attention|_fused_flex_attention|require_fused|fallback|is_compiling", line):
        print(f"{i + 1:5d}: {line}", flush=True)

print("\n=== torch.nn.attention.flex_attention.flex_attention signature ===", flush=True)
from torch.nn.attention.flex_attention import flex_attention  # noqa: E402

print(inspect.signature(flex_attention), flush=True)
print("\n=== BlockMask.from_kv_blocks signature ===", flush=True)
from torch.nn.attention.flex_attention import BlockMask  # noqa: E402

print(inspect.signature(BlockMask.from_kv_blocks), flush=True)


def probe(backend_hint: str) -> None:
    """Run one tiny fused-vs-math comparison and report which kernel was used."""
    T = 512
    q = torch.randn(1, 4, T, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 4, T, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 4, T, 64, dtype=torch.bfloat16)

    def mask_mod(b, h, qi, ki):
        return (qi >= ki) & (ki % 2 == 0)

    from torch.nn.attention.flex_attention import create_block_mask

    bm = create_block_mask(mask_mod, 1, 1, T, T, device="cpu")
    with torch.no_grad():
        out = flex_attention(q, k, v, block_mask=bm)
    print(f"  {backend_hint}: out finite={bool(torch.isfinite(out).all())} "
          f"block_size={bm.BLOCK_SIZE} kv_num_blocks={tuple(bm.kv_num_blocks.shape)}", flush=True)


print("\n=== tiny forward (CPU, eager) ===", flush=True)
probe("eager cpu")

print("\n=== sys.argv note ===", sys.argv, flush=True)