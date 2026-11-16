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

"""Single-GPU checks of the loss path used by LlamaFactory's SFT trainer for Qwen3.5.

1. GatedDeltaNet fast path is active (no "fast path is not available" warning; fla + causal_conv1d importable).
2. Liger fused linear-CE forward == plain HF forward on the same input (loss parity).
3. Right padding (collator pad_to_multiple_of=8) does not change the loss of a sample
   (GatedDeltaNet conv/recurrent state must ignore padded positions via attention_mask).
4. Loss normalisation: with num_items_in_batch the per-sample loss is sum_CE/num_items (token-mean over
   the effective batch); without it HF uses per-microbatch mean. Reports which one the model class takes.

    python coding_sft/checks/check_loss_norm.py --model .../Qwen__Qwen3.5-4B-text --data dbg128.jsonl
"""

import argparse
import inspect
import json
import os
import warnings


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


def encode(template, tok, conv, max_len):
    pairs = template.encode_multiturn(tok, conv, None, None)
    ids, labels = [], []
    for s, t in pairs:
        if len(ids) + len(s) + len(t) > max_len:
            break
        ids += s + t
        labels += [IGNORE_INDEX] * len(s) + t
    return ids, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--template", default="qwen3_5_text")
    ap.add_argument("--data", default="/storage/openpsi/users/chen/data/terminal_horizon_dbg128/dbg128.jsonl")
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    template = get_template_and_fix_tokenizer(tok, DataArguments(template=args.template, cutoff_len=10**9, enable_thinking=True))
    rows = [json.loads(l) for l in open(args.data)]
    a_ids, a_lab = encode(template, tok, rows[0]["conversations"], args.max_len)
    b_ids, b_lab = encode(template, tok, rows[1]["conversations"], args.max_len // 2)
    print(f"sample A: {len(a_ids)} tokens, {sum(l != IGNORE_INDEX for l in a_lab)} targets; sample B: {len(b_ids)} tokens, {sum(l != IGNORE_INDEX for l in b_lab)} targets")

    import fla, causal_conv1d  # noqa: F401  (import errors here = fast path unavailable)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
        fast_warn = [str(x.message) for x in w if "fast path" in str(x.message).lower()]
    print("model class:", type(model).__name__, "| accepts_loss_kwargs attr:", getattr(type(model), "accepts_loss_kwargs", "<none>"),
          "| forward **kwargs:", any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(model.forward).parameters.values()))
    print("fast-path warnings at load:", fast_warn or "none")

    def fwd(m, ids, labels, pad_to=None, **kw):
        n = len(ids)
        if pad_to:
            pad = (-n) % pad_to
            ids = ids + [tok.pad_token_id] * pad
            labels = labels + [IGNORE_INDEX] * pad
        x = torch.tensor([ids]).cuda()
        y = torch.tensor([labels]).cuda()
        am = torch.zeros_like(x)
        am[:, :n] = 1
        with torch.no_grad():
            out = m(input_ids=x, attention_mask=am, labels=y, **kw)
        return out.loss.float().item()

    # 1/2: plain HF forward, then liger
    import logging as _l
    _l.getLogger().setLevel(_l.INFO)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        base_a = fwd(model, a_ids, a_lab)
        fast_warn = [str(x.message) for x in w if "fast path" in str(x.message).lower()]
    print("fast-path warnings at forward:", fast_warn or "none")
    base_a_pad = fwd(model, a_ids, a_lab, pad_to=8)
    base_a_pad64 = fwd(model, a_ids, a_lab, pad_to=64)
    print(f"HF loss A: {base_a:.6f} | right-padded to x8: {base_a_pad:.6f} | to x64: {base_a_pad64:.6f}")

    from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel
    from llamafactory.hparams import ModelArguments
    apply_liger_kernel(model.config, ModelArguments(model_name_or_path=args.model, enable_liger_kernel=True), is_trainable=True, require_logits=False)
    model.train()  # skip_logits defaults to self.training
    for m_ in model.modules():
        m_.training = True
    lig_a = fwd(model, a_ids, a_lab)
    lig_a_pad = fwd(model, a_ids, a_lab, pad_to=8)
    print(f"liger loss A: {lig_a:.6f} | right-padded x8: {lig_a_pad:.6f}   (HF {base_a:.6f})")

    # 4: normalisation
    n_a = sum(l != IGNORE_INDEX for l in a_lab[1:])
    n_b = sum(l != IGNORE_INDEX for l in b_lab[1:])
    la, lb = fwd(model, a_ids, a_lab), fwd(model, b_ids, b_lab)
    la_n = fwd(model, a_ids, a_lab, num_items_in_batch=torch.tensor(n_a + n_b).cuda())
    lb_n = fwd(model, b_ids, b_lab, num_items_in_batch=torch.tensor(n_a + n_b).cuda())
    print(f"per-microbatch means: A={la:.5f} B={lb:.5f} -> naive GA avg {(la + lb) / 2:.5f}")
    print(f"token-mean over effective batch (A+B targets={n_a + n_b}): {la_n + lb_n:.5f}  (expected {(la * n_a + lb * n_b) / (n_a + n_b):.5f})")
    ok = abs((la_n + lb_n) - (la * n_a + lb * n_b) / (n_a + n_b)) < 2e-3
    print("num_items_in_batch honoured by model+liger:", ok)
    print("ALL OK" if ok and abs(base_a - lig_a) < 2e-3 and abs(base_a - base_a_pad64) < 2e-3 and not fast_warn else "PROBLEMS")


if __name__ == "__main__":
    main()
