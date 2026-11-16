# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GatedDeltaNet precision audit on the REAL training path (coding_analysis/training.md).

Loads the model exactly like LlamaFactory does for training (bf16, fa2, liger, LF's Qwen3.5 GPU
patch), wraps the FLA helpers that allocate the chunk state `h` (fwd) and `dh` (bwd) and the kernel
entry `chunk_gated_delta_rule`, runs one forward/backward on a real trajectory and reports:
  * which GDN forward ran (LF patch vs transformers), use_cache / cache_params
  * dtypes of q / k / v / g / beta entering the kernel, initial_state, output_final_state
  * dtype of the chunk state h returned by chunk_gated_delta_rule_fwd_h and dh from ..._bwd_dhu
  * whether final_state / dh0 exist
  * package versions + sha256 of the FLA kernel sources
Expected for the bf16 baseline (coding_analysis/training.md): q/k/v bf16, g fp32, h/dh bf16, final_state None,
accumulators fp32 (verified from the Triton source: b_h*/b_dh* = tl.zeros(..., tl.float32)).

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/precision_audit.py --len 4096 --out /tmp/audit.json
"""

import argparse
import hashlib
import json
import os


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

import torch
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments
from llamafactory.model import load_model


def dt(x):
    return str(x.dtype).replace("torch.", "") if torch.is_tensor(x) else ("None" if x is None else type(x).__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="storage/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--data", default="storage/data/tb_agent/th3k/th3k_mask.jsonl")
    ap.add_argument("--len", type=int, default=4096)
    ap.add_argument("--out", default="storage/runs/precision_audit.json")
    args = ap.parse_args()

    import fla
    import fla.ops.common.chunk_delta_h as cdh
    import fla.ops.gated_delta_rule.chunk as gdr
    import transformers
    from llamafactory.model import patcher

    rec = {"versions": {"transformers": transformers.__version__, "fla": fla.__version__, "torch": torch.__version__}, "events": []}
    for name, mod in (("chunk_delta_h.py", cdh), ("gated_delta_rule/chunk.py", gdr), ("llamafactory/model/patcher.py", patcher)):
        rec["versions"][name] = {"path": mod.__file__, "sha256": hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()[:16]}

    # ---- wrap FLA helpers
    orig_fwd_h, orig_bwd_dhu, orig_entry = cdh.chunk_gated_delta_rule_fwd_h, cdh.chunk_gated_delta_rule_bwd_dhu, gdr.chunk_gated_delta_rule

    def fwd_h(*a, **kw):
        h, v_new, final_state = orig_fwd_h(*a, **kw)
        rec["events"].append({"fn": "chunk_gated_delta_rule_fwd_h", "h": dt(h), "h_shape": list(h.shape), "v_new": dt(v_new), "final_state": dt(final_state),
                              "initial_state": dt(kw.get("initial_state")), "output_final_state": kw.get("output_final_state")})
        return h, v_new, final_state

    def bwd_dhu(*a, **kw):
        dh, dh0, dv2 = orig_bwd_dhu(*a, **kw)
        rec["events"].append({"fn": "chunk_gated_delta_rule_bwd_dhu", "dh": dt(dh), "dh0": dt(dh0), "dv2": dt(dv2)})
        return dh, dh0, dv2

    def entry(q, k, v, g=None, beta=None, **kw):
        rec["events"].append({"fn": "chunk_gated_delta_rule", "q": dt(q), "k": dt(k), "v": dt(v), "g": dt(g), "beta": dt(beta),
                              "initial_state": dt(kw.get("initial_state")), "output_final_state": kw.get("output_final_state"),
                              "cu_seqlens": dt(kw.get("cu_seqlens")), "use_qk_l2norm_in_kernel": kw.get("use_qk_l2norm_in_kernel")})
        return orig_entry(q, k, v, g=g, beta=beta, **kw)

    # the kernel-entry: LF's patch captured `chunk_gated_delta_rule` as a closure at import, so hook the
    # autograd Function it calls (the tensors here are exactly what the Triton kernels receive)
    orig_fn_forward = gdr.ChunkGatedDeltaRuleFunction.forward

    def fn_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, *rest, **kw):
        cu = rest[1] if len(rest) > 1 else kw.get("cu_seqlens")
        rec["events"].append({"fn": "chunk_gated_delta_rule", "q": dt(q), "k": dt(k), "v": dt(v), "g": dt(g), "beta": dt(beta),
                              "initial_state": dt(initial_state), "output_final_state": output_final_state, "cu_seqlens": dt(cu)})
        return orig_fn_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, *rest, **kw)

    gdr.ChunkGatedDeltaRuleFunction.forward = staticmethod(fn_forward)
    cdh.chunk_gated_delta_rule_fwd_h, cdh.chunk_gated_delta_rule_bwd_dhu = fwd_h, bwd_dhu
    gdr.chunk_gated_delta_rule_fwd_h = fwd_h  # the autograd Function imports the name into its module
    gdr.chunk_gated_delta_rule_bwd_dhu = bwd_dhu
    gdr.chunk_gated_delta_rule = entry
    patcher.chunk_gated_delta_rule = entry  # LF patch binds the symbol at patch time -> set before load_model
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    mq.chunk_gated_delta_rule = entry if hasattr(mq, "chunk_gated_delta_rule") else None

    # ---- model exactly as LF training loads it
    model_args = ModelArguments(model_name_or_path=args.model, flash_attn="fa2", enable_liger_kernel=True)
    model_args.compute_dtype = torch.bfloat16  # set by the parser from bf16: true in training
    tok = AutoTokenizer.from_pretrained(args.model)
    finetuning_args = FinetuningArguments(stage="sft", finetuning_type="full")
    model = load_model(tok, model_args, finetuning_args, is_trainable=True)
    # the HF Trainer runs under bf16 autocast (bf16: true); without a trainer we mirror it with
    # model weights in bf16 (ZeRO-3 keeps fp32 master weights in the optimizer, not in the module)
    model = model.to(torch.bfloat16).cuda().train()
    gdn = model.model.layers[0].linear_attn
    rec["gdn_forward"] = f"{gdn.forward.__func__.__module__}.{gdn.forward.__func__.__qualname__}"
    rec["gdn_chunk_fn"] = f"{gdn.chunk_gated_delta_rule.__module__}.{getattr(gdn.chunk_gated_delta_rule, '__name__', '?')}"
    rec["config"] = {"use_cache": getattr(model.config, "use_cache", None), "mamba_ssm_dtype": getattr(model.config, "mamba_ssm_dtype", None),
                     "dtype": str(model.dtype), "A_log.dtype": dt(gdn.A_log), "dt_bias.dtype": dt(gdn.dt_bias), "in_proj_qkv.weight": dt(gdn.in_proj_qkv.weight)}

    template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen3_5_text", cutoff_len=10**9, enable_thinking=True))
    conv = json.loads(open(args.data).readline())["conversations"]
    ids, labels = [], []
    for s, t in template.encode_multiturn(tok, conv, None, None):
        if len(ids) + len(s) + len(t) > args.len:
            break
        ids += s + t
        labels += [IGNORE_INDEX] * len(s) + t
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=torch.tensor([ids]).cuda(), labels=torch.tensor([labels]).cuda(), use_cache=False)
    out.loss.backward()
    rec["loss"] = out.loss.item()
    rec["grad_dtype_sample"] = {n: dt(p.grad) for n, p in list(model.named_parameters())[:3]}

    ev = rec["events"]
    kinds = {e["fn"] for e in ev}
    rec["summary"] = {
        "gdn_forward_is_lf_patch": "patcher" in rec["gdn_forward"],
        "kernel_entry_captured": "chunk_gated_delta_rule" in kinds,
        "fwd_h_captured": "chunk_gated_delta_rule_fwd_h" in kinds,
        "bwd_dhu_captured": "chunk_gated_delta_rule_bwd_dhu" in kinds,
        "qkv": sorted({(e["q"], e["k"], e["v"]) for e in ev if e["fn"] == "chunk_gated_delta_rule"}),
        "g": sorted({e["g"] for e in ev if e["fn"] == "chunk_gated_delta_rule"}),
        "h": sorted({e["h"] for e in ev if e["fn"] == "chunk_gated_delta_rule_fwd_h"}),
        "dh": sorted({e["dh"] for e in ev if e["fn"] == "chunk_gated_delta_rule_bwd_dhu"}),
        "final_state": sorted({e["final_state"] for e in ev if e["fn"] == "chunk_gated_delta_rule_fwd_h"}),
        "dh0": sorted({e["dh0"] for e in ev if e["fn"] == "chunk_gated_delta_rule_bwd_dhu"}),
        "n_fwd_h_calls": sum(e["fn"] == "chunk_gated_delta_rule_fwd_h" for e in ev),
        "n_bwd_dhu_calls": sum(e["fn"] == "chunk_gated_delta_rule_bwd_dhu" for e in ev),
    }
    src = open(cdh.__file__).read()
    rec["summary"]["accumulators_fp32_in_source"] = "b_h1 = tl.zeros([BV, 64], dtype=tl.float32)" in src and "b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)" in src
    rec["summary"]["h_stored_as_pointer_dtype"] = "tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty)" in src
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
