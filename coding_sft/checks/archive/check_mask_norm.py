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

"""How does a per-turn loss mask interact with the loss normalisation? (1 GPU, exact arithmetic check)

Takes one trajectory, builds labels (a) unmasked and (b) with assistant turn T masked (labels=-100 on
that span, tokens kept in input_ids). Then checks, with the model + liger exactly as in training:

  1. num_items_in_batch (what HF Trainer computes as `labels[...,1:].ne(-100).sum()` over the whole
     effective batch) drops by exactly the number of masked targets  -> denominator shrinks.
  2. The loss returned with that num_items_in_batch equals sum of per-token CE over the REMAINING
     targets / remaining count, i.e. masked tokens contribute 0 to the numerator and 0 to the denominator.
  3. Gradient: d loss/d W with the mask == gradient of the mean over remaining tokens (masked turn
     contributes nothing even though it is in the context).

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_mask_norm.py --len 6144 --mask-turn 3
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--data", default="/storage/openpsi/users/chen/data/terminal_horizon_dbg128/dbg128.jsonl")
    ap.add_argument("--len", type=int, default=6144)
    ap.add_argument("--mask-turn", type=int, default=1)
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen3_5_text", cutoff_len=10**9, enable_thinking=True))
    conv = [json.loads(l) for l in open(args.data)][args.row]["conversations"]
    pairs = template.encode_multiturn(tok, conv, None, None)
    ids, lab, lab_m, spans = [], [], [], []
    for t, (s, r) in enumerate(pairs):
        if len(ids) + len(s) + len(r) > args.len:
            break
        spans.append((len(ids) + len(s), len(ids) + len(s) + len(r)))
        ids += s + r
        lab += [IGNORE_INDEX] * len(s) + r
        lab_m += [IGNORE_INDEX] * len(s) + ([IGNORE_INDEX] * len(r) if t == args.mask_turn else r)
    n_all = sum(l != IGNORE_INDEX for l in lab[1:])
    n_rem = sum(l != IGNORE_INDEX for l in lab_m[1:])
    n_masked_targets = n_all - n_rem
    print(f"{len(spans)} turns, {len(ids)} tokens; targets: all={n_all}, after masking turn {args.mask_turn}: {n_rem} (masked {n_masked_targets} = {n_masked_targets / n_all:.1%})")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    from llamafactory.hparams import ModelArguments
    from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel
    apply_liger_kernel(model.config, ModelArguments(model_name_or_path=args.model, enable_liger_kernel=True), is_trainable=True, require_logits=False)
    model.train()
    x = torch.tensor([ids]).cuda()

    # reference per-token CE (plain HF logits path, no liger), to compare numerators
    def per_token_ce():
        out = model(input_ids=x, skip_logits=False)
        lg = out.logits[0, :-1]
        y = torch.tensor(lab[1:]).cuda()
        return torch.cat([F.cross_entropy(lg[i : i + 1024].float(), y[i : i + 1024], ignore_index=IGNORE_INDEX, reduction="none") for i in range(0, lg.shape[0], 1024)])

    with torch.no_grad():
        ce = per_token_ce()
    keep = torch.tensor(lab_m[1:]).cuda() != IGNORE_INDEX
    allm = torch.tensor(lab[1:]).cuda() != IGNORE_INDEX
    ref_all = ce[allm].sum() / n_all
    ref_rem = ce[keep].sum() / n_rem
    ref_wrong = ce[keep].sum() / n_all  # what you would get if the denominator did NOT shrink
    print(f"\n(1)+(2) token-mean over all targets      : {ref_all.item():.5f}")
    print(f"        token-mean over REMAINING targets: {ref_rem.item():.5f}   <- expected with mask")
    print(f"        remaining sum / ORIGINAL count   : {ref_wrong.item():.5f}   <- would be the value if the denominator did not shrink")
    # HF Trainer's denominator: labels[..., 1:].ne(-100).sum()  (computed from the collated labels, so masked turn excluded)
    hf_denom = torch.tensor([lab_m]).cuda()[..., 1:].ne(IGNORE_INDEX).sum().item()
    print(f"        HF Trainer num_items_in_batch from the masked labels = {hf_denom} (== remaining {n_rem}: {hf_denom == n_rem})")

    with torch.no_grad():
        loss_m = model(input_ids=x, labels=torch.tensor([lab_m]).cuda(), num_items_in_batch=torch.tensor(hf_denom).cuda()).loss.float().item()
        loss_u = model(input_ids=x, labels=torch.tensor([lab]).cuda(), num_items_in_batch=torch.tensor(n_all).cuda()).loss.float().item()
    print(f"        model+liger loss, masked labels + HF denominator: {loss_m:.5f}  (matches remaining-mean: {abs(loss_m - ref_rem.item()) < 2e-3})")
    print(f"        model+liger loss, unmasked               : {loss_u:.5f}  (matches all-mean: {abs(loss_u - ref_all.item()) < 2e-3})")

    # (3) gradient: masked loss grad == grad of (sum of CE over remaining)/n_rem; compare with the same thing built by hand
    name, p = next((n, p) for n, p in model.named_parameters() if "layers.1.linear_attn.in_proj_qkv" in n)
    model.gradient_checkpointing_enable()
    model.zero_grad()
    model(input_ids=x, labels=torch.tensor([lab_m]).cuda(), num_items_in_batch=torch.tensor(hf_denom).cuda()).loss.backward()
    g_mask = p.grad.detach().clone()
    model.zero_grad()
    # by hand: logits path, CE only over remaining positions, divided by n_rem
    (per_token_ce()[keep].sum() / n_rem).backward()
    g_hand = p.grad.detach().clone()
    model.zero_grad()
    (per_token_ce()[allm].sum() / n_all).backward()
    g_all = p.grad.detach().clone()
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    print(f"\n(3) grad of {name}:")
    print(f"    ||g_masked(liger) - g_hand(remaining-mean)|| / ||g_hand|| = {rel(g_mask, g_hand):.4f}  (bf16 noise expected < 0.05)")
    print(f"    ||g_masked - g_unmasked|| / ||g_unmasked||              = {rel(g_mask, g_all):.4f}  (the masked turn's contribution)")


if __name__ == "__main__":
    main()
