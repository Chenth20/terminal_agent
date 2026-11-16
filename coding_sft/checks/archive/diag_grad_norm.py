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

"""Where does the step-0 gradient norm come from? (1 GPU, base model, token-mean loss)

Prints (a) per-token loss split by token class (think text / JSON action / structural tokens such as
<think>, </think>, <|im_end|>, '{', '"analysis"'), (b) the grad-norm contribution of each parameter
group (embeddings+lm_head, linear-attention, full-attention, MLP, norms), and (c) the grad norm with
the K highest-loss tokens masked out, to see whether a few tokens dominate.

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/diag_grad_norm.py --len 3072
"""

import argparse
import json
import os
from collections import defaultdict


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


def classify(tok, ids, labels):
    """token class for every supervised position: struct / think / action."""
    cls = ["-"] * len(ids)
    in_think = False
    for i, (t, l) in enumerate(zip(ids, labels)):
        if l == IGNORE_INDEX:
            continue
        s = tok.decode([t])
        if "<think>" in s:
            in_think, cls[i] = True, "struct"
        elif "</think>" in s:
            in_think, cls[i] = False, "struct"
        elif "<|im_end|>" in s or s.strip() in ("{", "}", "[", "]", "],", "},", '"', ",", ":"):
            cls[i] = "struct"
        else:
            cls[i] = "think" if in_think else "action"
    return cls


def group(name):
    if "embed" in name or "lm_head" in name:
        return "embed/lm_head"
    if "linear_attn" in name:
        return "linear_attn (GatedDeltaNet)"
    if "self_attn" in name:
        return "self_attn (full)"
    if "mlp" in name:
        return "mlp"
    return "norms/other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--data", default="/storage/openpsi/users/chen/data/terminal_horizon_dbg128/dbg128.jsonl")
    ap.add_argument("--rows", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--len", type=int, default=3072)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen3_5_text", cutoff_len=10**9, enable_thinking=True))
    rows = [json.loads(l) for l in open(args.data)]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    model.gradient_checkpointing_enable()
    model.train()

    samples = [encode(template, tok, rows[r]["conversations"], args.len) for r in args.rows]
    n_items = sum(sum(1 for l in lab[1:] if l != IGNORE_INDEX) for _, lab in samples)
    print(f"{len(samples)} samples, {sum(len(s[0]) for s in samples)} tokens, {n_items} targets (token-mean loss)")

    # (a) per-token loss by class
    cls_loss, cls_n = defaultdict(float), defaultdict(int)
    top = []
    for ids, lab in samples:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]).cuda()).logits[0, :-1].float()
        l = F.cross_entropy(logits, torch.tensor(lab[1:]).cuda(), ignore_index=IGNORE_INDEX, reduction="none").cpu()
        cls = classify(tok, ids, lab)
        for i in range(1, len(ids)):
            if lab[i] != IGNORE_INDEX:
                cls_loss[cls[i]] += l[i - 1].item(); cls_n[cls[i]] += 1
                top.append((l[i - 1].item(), tok.decode(ids[max(0, i - 6) : i]), tok.decode([ids[i]])))
        del logits
    print("\n(a) loss by token class (share of total loss mass):")
    tot = sum(cls_loss.values())
    for c in ("think", "action", "struct"):
        print(f"   {c:7s}: n={cls_n[c]:6d} ({cls_n[c] / n_items:5.1%} of targets)  mean loss {cls_loss[c] / max(cls_n[c], 1):.3f}  loss mass {cls_loss[c] / tot:5.1%}")
    top.sort(reverse=True)
    print("   highest-loss targets:")
    for l, ctx, t in top[:8]:
        print(f"     {l:6.2f}  ...{ctx[-40:]!r} -> {t!r}")
    print(f"   top-1% of targets carry {sum(x[0] for x in top[: n_items // 100]) / tot:.1%} of the loss mass")

    # (b) grad norm by parameter group, token-mean over both samples
    def run(mask_top_k=0):
        model.zero_grad(set_to_none=True)
        thr = top[mask_top_k - 1][0] if mask_top_k else float("inf")
        for ids, lab in samples:
            lab2 = list(lab)
            if mask_top_k:
                with torch.no_grad():
                    lg = model(input_ids=torch.tensor([ids]).cuda()).logits[0, :-1].float()
                l = F.cross_entropy(lg, torch.tensor(lab[1:]).cuda(), ignore_index=IGNORE_INDEX, reduction="none").cpu()
                for i in range(1, len(ids)):
                    if lab[i] != IGNORE_INDEX and l[i - 1] >= thr:
                        lab2[i] = IGNORE_INDEX
                del lg
            out = model(input_ids=torch.tensor([ids]).cuda(), labels=torch.tensor([lab2]).cuda(), num_items_in_batch=torch.tensor(n_items).cuda())
            out.loss.backward()
        sq = defaultdict(float)
        for n, p in model.named_parameters():
            if p.grad is not None:
                sq[group(n)] += p.grad.float().pow(2).sum().item()
        total = sum(sq.values()) ** 0.5
        return total, sq

    total, sq = run()
    print(f"\n(b) global grad norm (pre-clip) = {total:.3f}")
    for g, v in sorted(sq.items(), key=lambda x: -x[1]):
        print(f"   {g:28s} norm {v ** 0.5:7.3f}  share {v / total ** 2:5.1%}")
    for k in (n_items // 100, n_items // 20):
        t2, _ = run(mask_top_k=k)
        print(f"(c) grad norm with top {k} ({k / n_items:.0%}) highest-loss targets masked: {t2:.3f}")


if __name__ == "__main__":
    main()
