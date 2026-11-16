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

"""Multi-trajectory packing check for Qwen3.5 through LlamaFactory's REAL packed pipeline.

Runs `PackedSupervisedDatasetProcessor` (greedy knapsack into cutoff_len, neat_packing attention ids,
restarting position ids, right padding) + `SFTDataCollatorWith4DAttentionMask` (unpad, cu_seqlens /
seq_idx for the linear-attention layers) on K trajectories, then compares the per-token loss of EVERY
sub-sequence inside the pack with the same trajectory run alone. Also a forward/backward gradient
check: grad of the packed loss (sum over sub-sequences) vs sum of per-sample grads.

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_packing_multi.py --k 6 --cutoff 24576
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
from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--data", default="/storage/openpsi/users/chen/data/terminal_horizon_dbg128/dbg128.jsonl")
    ap.add_argument("--k", type=int, default=6, help="trajectories to pack")
    ap.add_argument("--sample-len", type=int, default=4096, help="truncate each trajectory to this many tokens")
    ap.add_argument("--cutoff", type=int, default=16384, help="pack length (LF cutoff_len)")
    ap.add_argument("--grad", action="store_true", help="also compare gradients (slow, needs ~40 GB)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    data_args = DataArguments(template="qwen3_5_text", cutoff_len=args.cutoff, enable_thinking=True, packing=True, neat_packing=True)
    template = get_template_and_fix_tokenizer(tok, data_args)
    rows = [json.loads(l) for l in open(args.data)][: args.k]
    # vary lengths so sub-sequences are not all equal
    samples = [encode(template, tok, r["conversations"], args.sample_len - 512 * (i % 3)) for i, r in enumerate(rows)]
    print("sample lengths:", [len(s[0]) for s in samples])

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    model.train()  # dropout is 0 in Qwen3.5; keeps the same code path as training
    for p in model.parameters():
        p.requires_grad_(args.grad)

    # ---- references: each sample alone
    refs = []
    for ids, lab in samples:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]).cuda(), attention_mask=torch.ones(1, len(ids), dtype=torch.long).cuda())
            l = F.cross_entropy(out.logits[0, :-1].float(), torch.tensor(lab[1:]).cuda(), ignore_index=IGNORE_INDEX, reduction="none")
        refs.append(l[torch.tensor(lab[1:]).cuda() != IGNORE_INDEX].cpu())
        del out

    # ---- LF packed pipeline: processor -> collator
    proc = PackedSupervisedDatasetProcessor(template=template, tokenizer=tok, processor=None, data_args=data_args)
    # bypass _encode_data_example by feeding pre-encoded ids through the same packing code
    batch_input_ids = [s[0] for s in samples]
    batch_labels = [s[1] for s in samples]
    lengths = [len(x) for x in batch_input_ids]
    from llamafactory.data.processor.processor_utils import greedy_knapsack

    length2indexes = {}
    for i, L in enumerate(lengths):
        length2indexes.setdefault(L, []).append(i)
    knapsacks = greedy_knapsack(list(lengths), args.cutoff)  # LF sorts in place; keep `lengths` indexable
    print(f"knapsacks: {[len(k) for k in knapsacks]} samples each (cutoff {args.cutoff})")

    coll = SFTDataCollatorWith4DAttentionMask(template=template, model=model, tokenizer=tok, processor=None, pad_to_multiple_of=8,
                                              label_pad_token_id=IGNORE_INDEX, neat_packing=True, attn_implementation="flash_attention_2", compute_dtype=torch.bfloat16)
    worst = 0.0
    for knapsack in knapsacks:
        members, p_ids, p_pos, p_att, p_lab = [], [], [], [], []
        for i, L in enumerate(knapsack):
            idx = length2indexes[L].pop()
            members.append(idx)
            p_ids += batch_input_ids[idx]
            p_pos += list(range(L))
            p_att += [i + 1] * L
            p_lab += batch_labels[idx]
        pad = args.cutoff - len(p_ids) + 1  # LF pads to cutoff_len + 1
        p_ids += [tok.pad_token_id] * pad; p_pos += [0] * pad; p_att += [0] * pad; p_lab += [IGNORE_INDEX] * pad
        feat = coll([{"input_ids": p_ids, "attention_mask": p_att, "position_ids": p_pos, "labels": p_lab}])
        labels = feat.pop("labels").cuda()
        feat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in feat.items()}
        feat.pop("attention_mask", None)
        print("  collator kwargs:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in feat.items() if k not in ("input_ids", "position_ids")})
        with torch.no_grad():
            logits = model(**feat).logits[0]
        L = logits.shape[0]
        per_tok = torch.cat([F.cross_entropy(logits[i : min(i + 2048, L - 1)].float(), labels[0, i + 1 : min(i + 2048, L - 1) + 1], ignore_index=IGNORE_INDEX, reduction="none") for i in range(0, L - 1, 2048)])
        del logits
        off = 0
        for i, idx in enumerate(members):
            n = lengths[idx]
            seg = per_tok[off : off + n - 1]  # positions off..off+n-2 predict tokens off+1..off+n-1 (within the sub-sequence)
            seg_lab = labels[0, off + 1 : off + n]
            got = seg[seg_lab != IGNORE_INDEX].cpu()
            d = (got - refs[idx]).abs()
            worst = max(worst, d.max().item())
            print(f"  pack member {i} (sample {idx}, pos {off}..{off + n - 1}): loss {got.mean():.5f} vs alone {refs[idx].mean():.5f}  max|Δ| {d.max():.4f} mean|Δ| {d.mean():.5f}")
            off += n
    print(f"worst per-token |Δ| across all packed members: {worst:.4f}  ->", "OK (bf16 kernel-path noise, < 0.5)" if worst < 0.5 else "CONTAMINATED")

    if args.grad:
        # gradient parity on the first knapsack: d(sum of member losses)/dW packed vs sum of alone grads
        for p in model.parameters():
            p.requires_grad_(True)
        name, p_ref = next((n, p) for n, p in model.named_parameters() if "layers.0.linear_attn" in n and p.dim() == 2)
        knapsack = knapsacks[0]
        members = []
        for L in knapsack:
            members.append(lengths.index(L))
        g_alone = torch.zeros_like(p_ref)
        for idx in members:
            ids, lab = samples[idx]
            model.zero_grad()
            out = model(input_ids=torch.tensor([ids]).cuda(), labels=torch.tensor([lab]).cuda(), num_items_in_batch=torch.tensor(1).cuda())
            out.loss.backward()
            g_alone += p_ref.grad
        p_ids, p_pos, p_att, p_lab = [], [], [], []
        for i, idx in enumerate(members):
            p_ids += samples[idx][0]; p_pos += list(range(lengths[idx])); p_att += [i + 1] * lengths[idx]; p_lab += samples[idx][1]
        pad = (-len(p_ids)) % 8
        p_ids += [tok.pad_token_id] * pad; p_pos += [0] * pad; p_att += [0] * pad; p_lab += [IGNORE_INDEX] * pad
        feat = coll([{"input_ids": p_ids, "attention_mask": p_att, "position_ids": p_pos, "labels": p_lab}])
        feat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in feat.items()}
        feat.pop("attention_mask", None)
        model.zero_grad()
        out = model(**feat, num_items_in_batch=torch.tensor(1).cuda())
        out.loss.backward()
        rel = (p_ref.grad - g_alone).norm() / g_alone.norm()
        print(f"grad parity on {name}: ||g_packed - sum g_alone|| / ||sum g_alone|| = {rel:.4f}  ->", "OK" if rel < 0.05 else "MISMATCH")


if __name__ == "__main__":
    main()
