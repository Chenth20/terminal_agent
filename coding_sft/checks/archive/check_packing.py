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

"""Is LlamaFactory packing safe for Qwen3.5 (GatedDeltaNet + full attention)?

Builds exactly what LF's `neat_packing` + FA2 collator feeds the model: input_ids = [A ; B],
position_ids restart at 0 for B, attention_mask = None. Then compares the per-token loss of B
inside the pack with B alone. Three variants:
  1. LF neat_packing as-is (position_ids only)                       -> what `packing: true, neat_packing: true` does
  2. + cu_seq_lens_q / seq_idx kwargs (what the linear-attention layers need)
  3. plain `packing: true` (no neat): position_ids 0..N-1 continuous   -> cross-sample attention by design
B inside the pack must equal B alone (to bf16 noise) for packing to be correct.

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_packing.py
"""

import argparse
import json
import os


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


def encode(template, tok, conv, max_len):
    ids, labels = [], []
    for s, t in template.encode_multiturn(tok, conv, None, None):
        if len(ids) + len(s) + len(t) > max_len:
            break
        ids += s + t
        labels += [IGNORE_INDEX] * len(s) + t
    return ids, labels


@torch.no_grad()
def token_losses(model, ids, labels, position_ids, extra=None):
    x = torch.tensor([ids]).cuda()
    out = model(input_ids=x, position_ids=torch.tensor([position_ids]).cuda(), attention_mask=None, **(extra or {}))
    logits = out.logits[0, :-1].float()
    y = torch.tensor(labels[1:]).cuda()
    return F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX, reduction="none")  # per position


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--data", default="/storage/openpsi/users/chen/data/terminal_horizon_dbg128/dbg128.jsonl")
    ap.add_argument("--len", type=int, default=4096)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="fp32 separates real contamination from bf16 GEMM-shape noise")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen3_5_text", cutoff_len=10**9, enable_thinking=True))
    rows = [json.loads(l) for l in open(args.data)]
    a_ids, a_lab = encode(template, tok, rows[0]["conversations"], args.len)
    b_ids, b_lab = encode(template, tok, rows[1]["conversations"], args.len)
    la, lb = len(a_ids), len(b_ids)
    print(f"A={la} tokens, B={lb} tokens")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32, attn_implementation="flash_attention_2" if args.dtype == "bf16" else "sdpa").cuda().eval()

    ref = token_losses(model, b_ids, b_lab, list(range(lb)))
    ref_mean = ref[torch.tensor(b_lab[1:]).cuda() != IGNORE_INDEX].mean().item()
    print(f"B alone                       : loss {ref_mean:.5f}")

    packed_ids, packed_lab = a_ids + b_ids, a_lab + b_lab
    # packed loss index p predicts packed_lab[p+1]; for p = la+q that is b_lab[q+1] == B-alone loss index q.
    # Index la-1 (A's last token predicting B's first token) has no counterpart in B-alone -> excluded.
    mask_b = torch.tensor(packed_lab[1:]).cuda() != IGNORE_INDEX
    mask_b[:la] = False
    rb = ref[torch.tensor(b_lab[1:]).cuda() != IGNORE_INDEX]

    def report(name, pos, extra=None):
        l = token_losses(model, packed_ids, packed_lab, pos, extra)
        lb_ = l[mask_b]
        diff = (lb_ - rb).abs()
        print(f"{name:30s}: loss {lb_.mean().item():.5f}  max|Δtoken| {diff.max().item():.4f}  mean|Δ| {diff.mean().item():.5f}  -> {'OK (<= noise floor of [B;A] below)' if diff.max().item() < 0.4 else 'CONTAMINATED'}")

    # 1. LF neat_packing + FA2: position ids restart, attention_mask None
    pos_neat = list(range(la)) + list(range(lb))
    report("neat_packing (position_ids)", pos_neat)
    # 2. + explicit varlen kwargs for the linear-attention layers
    cu = torch.tensor([0, la, la + lb], dtype=torch.int32).cuda()
    seq_idx = torch.tensor([[0] * la + [1] * lb], dtype=torch.int32).cuda()
    report("neat + cu_seq_lens_q/seq_idx", pos_neat, {"cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": max(la, lb), "max_length_k": max(la, lb), "seq_idx": seq_idx})
    report("neat + cu_seq_lens only", pos_neat, {"cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": max(la, lb), "max_length_k": max(la, lb)})
    report("neat + seq_idx only", pos_neat, {"seq_idx": seq_idx})
    # 3. plain packing: continuous positions (cross-sample attention expected)
    report("packing w/o neat (continuous)", list(range(la + lb)))
    # noise floor: B alone but through the varlen kernels (cu_seqlens=[0,lb]) and in bf16 -> expected |Δ| from kernel choice only
    cu1 = torch.tensor([0, lb], dtype=torch.int32).cuda()
    l1 = token_losses(model, b_ids, b_lab, list(range(lb)), {"cu_seq_lens_q": cu1, "cu_seq_lens_k": cu1, "max_length_q": lb, "max_length_k": lb, "seq_idx": torch.zeros((1, lb), dtype=torch.int32).cuda()})
    d = (l1[torch.tensor(b_lab[1:]).cuda() != IGNORE_INDEX] - rb).abs()
    print(f"{'noise floor (B alone, varlen)':30s}: loss {l1[torch.tensor(b_lab[1:]).cuda() != IGNORE_INDEX].mean().item():.5f}  max|Δtoken| {d.max().item():.4f}  mean|Δ| {d.mean().item():.5f}")
    # B first in the pack: nothing precedes it, so any Δ beyond the noise floor = A leaking backwards (impossible if causal) / kernel bug
    packed2_ids, packed2_lab = b_ids + a_ids, b_lab + a_lab
    cu2 = torch.tensor([0, lb, la + lb], dtype=torch.int32).cuda()
    l2 = token_losses(model, packed2_ids, packed2_lab, list(range(lb)) + list(range(la)), {"cu_seq_lens_q": cu2, "cu_seq_lens_k": cu2, "max_length_q": max(la, lb), "max_length_k": max(la, lb), "seq_idx": torch.tensor([[0] * lb + [1] * la], dtype=torch.int32).cuda()})
    m2 = torch.tensor(packed2_lab[1:]).cuda() != IGNORE_INDEX; m2[lb - 1:] = False
    d2 = (l2[m2] - rb).abs()
    for name, extra in (("[B;A] cu_seq_lens only (no seq_idx)", {"cu_seq_lens_q": cu2, "cu_seq_lens_k": cu2, "max_length_q": max(la, lb), "max_length_k": max(la, lb)}),
                        ("[B;A] seq_idx only (no cu_seq_lens)", {"seq_idx": torch.tensor([[0] * lb + [1] * la], dtype=torch.int32).cuda()})):
        l3 = token_losses(model, packed2_ids, packed2_lab, list(range(lb)) + list(range(la)), extra)
        d3 = (l3[m2] - rb).abs()
        print(f"{name:36s}: loss {l3[m2].mean().item():.5f}  max|Δtoken| {d3.max().item():.4f}  mean|Δ| {d3.mean().item():.5f}")
    # 4. end-to-end through LF's real collator (neat_packing + FA2) -> whatever kwargs it emits go to the model
    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    coll = SFTDataCollatorWith4DAttentionMask(template=template, model=model, tokenizer=tok, processor=None, pad_to_multiple_of=8,
                                              label_pad_token_id=IGNORE_INDEX, neat_packing=True, attn_implementation="flash_attention_2", compute_dtype=torch.bfloat16)
    pad = 8 - (la + lb) % 8 if (la + lb) % 8 else 0
    feat = coll([{"input_ids": packed_ids + [tok.pad_token_id] * pad, "attention_mask": [1] * la + [2] * lb + [0] * pad,
                  "position_ids": pos_neat + [0] * pad, "labels": packed_lab + [IGNORE_INDEX] * pad}])
    feat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in feat.items() if k not in ("labels",)}
    feat.pop("attention_mask", None)
    with torch.no_grad():
        logits = model(**feat).logits[0, :-1].float()
    l4 = F.cross_entropy(logits, torch.tensor(packed_lab[1:]).cuda(), ignore_index=IGNORE_INDEX, reduction="none")
    d4 = (l4[mask_b] - rb).abs()
    print(f"{'LF collator neat_packing (patched?)':36s}: loss {l4[mask_b].mean().item():.5f}  max|Δtoken| {d4.max().item():.4f}  mean|Δ| {d4.mean().item():.5f}  kwargs={sorted(k for k in feat if k not in ('input_ids','position_ids'))}")
    print(f"{'[B;A] + varlen, B at front':30s}: loss {l2[m2].mean().item():.5f}  max|Δtoken| {d2.max().item():.4f}  mean|Δ| {d2.mean().item():.5f}")


if __name__ == "__main__":
    main()
