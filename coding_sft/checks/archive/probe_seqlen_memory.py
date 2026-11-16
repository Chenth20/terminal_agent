"""Memory probe: full-size Qwen3.5-4B fwd+bwd on one GPU at increasing sequence lengths.

Mirrors the training path's kernels (FA2, GatedDeltaNet fast path, liger fused-linear-CE, grad
checkpointing) but WITHOUT the optimizer / ZeRO; reports peak memory per length so we can see
whether activations (not optimizer state) are what breaks 256K.
    python coding_sft/checks/probe_seqlen_memory.py 65536 131072 196608 262144
"""
import sys
import time

import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
from transformers import AutoModelForImageTextToText

import transformers.models.qwen3_5.modeling_qwen3_5 as q

assert q.is_fast_path_available, "GatedDeltaNet fast path missing"
apply_liger_kernel_to_qwen3_5()
path = "/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B"
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.train()
print("weights GB", torch.cuda.memory_allocated() / 2**30)
for L in map(int, sys.argv[1:]):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    x = torch.randint(0, 100000, (1, L), device="cuda")
    try:
        t0 = time.time()
        out = model(input_ids=x, labels=x)
        out.loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"L={L:7d}  OK  peak_alloc={torch.cuda.max_memory_allocated()/2**30:6.1f} GB  peak_resv={torch.cuda.max_memory_reserved()/2**30:6.1f} GB  fwd+bwd={dt:6.1f}s  loss={out.loss.item():.3f}", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"L={L:7d}  OOM  peak_resv={torch.cuda.max_memory_reserved()/2**30:6.1f} GB  ({str(e)[:90]})", flush=True)
    model.zero_grad(set_to_none=True)
    del x
    torch.cuda.empty_cache()
