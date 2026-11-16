#!/usr/bin/env python3
"""1-GPU smoke test of the A-arm (perdh) group-packed cache through LlamaFactory's REAL training path.

Loads bins straight from ``tokenized_qwen3_5_text_262144_sched_pack`` (the rows the trainer will feed),
runs them through ``SFTDataCollatorWith4DAttentionMask(neat_packing=True)`` + the liger-patched
``Qwen3_5ForCausalLM`` and checks:

  1. ``num_items_in_batch`` accounting: sum over the 304 bins of step 0 of (labels != -100) equals the
     baseline cache's supervised count for group 0 (what both arms will divide by);
  2. packed loss == alone loss: for a bin, the per-sub-sequence NLL (computed from logits) matches running
     each member row alone (no cross-row leakage through FA2 / GatedDeltaNet / conv state);
  3. the collator's kwargs (cu_seq_lens, seq_idx) are consistent with packing_params.

    CUDA_VISIBLE_DEVICES=0 python coding_sft/checks/check_perdh_pack.py --bins 0 1 --max-bin-tokens 65536
"""

import argparse
import json
import os


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel


def per_token_nll(model, feat: dict, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(**feat).logits[0]
    L = logits.shape[0]
    out = []
    for i in range(0, L - 1, 4096):
        j = min(i + 4096, L - 1)
        out.append(F.cross_entropy(logits[i:j].float(), labels[0, i + 1 : j + 1], ignore_index=IGNORE_INDEX, reduction="none"))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--pack", default="/storage/openpsi/users/chen/data/th3k_align_perdh/tokenized_qwen3_5_text_262144_sched_pack")
    ap.add_argument("--base", default="/storage/openpsi/users/chen/data/th3k_align/tokenized_qwen3_5_text_262144_sched")
    ap.add_argument("--bins", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-bin-tokens", type=int, default=98304, help="truncate the bin to its first sub-sequences fitting this")
    args = ap.parse_args()

    pack = load_from_disk(args.pack)["train"]
    base = load_from_disk(args.base)["train"]
    with open(os.path.join(args.pack, "pack_manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    B = man["bins_per_step"]
    print(f"pack cache {len(pack)} rows, {B} bins/step, G={man['G']}; base cache {len(base)} rows")

    # 1. step-0 supervised count parity (what num_items_in_batch sums to on 16 ranks)
    sup_pack = sum(int((torch.tensor(pack[i]["labels"]) != IGNORE_INDEX).sum()) for i in range(B))
    sup_base = sum(int((torch.tensor(base[i]["labels"]) != IGNORE_INDEX).sum()) for i in range(64))
    print(f"step 0 supervised tokens: perdh {sup_pack:,} vs baseline {sup_base:,} ->", "OK" if sup_pack == sup_base else "MISMATCH")
    assert sup_pack == sup_base

    tok = AutoTokenizer.from_pretrained(args.model)
    data_args = DataArguments(template="qwen3_5_text", cutoff_len=262144, enable_thinking=True, packing=True, neat_packing=True)
    template = get_template_and_fix_tokenizer(tok, data_args)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    apply_liger_kernel(model.config, ModelArguments(model_name_or_path=args.model, enable_liger_kernel=True), is_trainable=True, require_logits=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    coll = SFTDataCollatorWith4DAttentionMask(
        template=template, model=model, tokenizer=tok, processor=None, pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX, neat_packing=True, attn_implementation="flash_attention_2", compute_dtype=torch.bfloat16,
    )

    worst = 0.0
    for b in args.bins:
        row = pack[b]
        bounds = row["packing_params"]["sequence_boundaries"]
        # keep the leading sub-sequences that fit the token budget (bins are ~220K; a 65K slice is plenty for the check)
        n_sub = max([1] + [i for i in range(1, len(bounds)) if bounds[i] <= args.max_bin_tokens])
        end = bounds[n_sub]
        feat_in = {
            "input_ids": row["input_ids"][:end],
            "attention_mask": row["attention_mask"][:end],
            "position_ids": row["position_ids"][:end],
            "labels": row["labels"][:end],
            "packing_params": {**row["packing_params"], "sequence_boundaries": bounds[: n_sub + 1]},
        }
        feat = coll([feat_in])
        labels = feat.pop("labels").cuda()
        feat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in feat.items()}
        feat.pop("attention_mask", None)
        cu = feat["cu_seq_lens_q"].tolist()
        # 3. collator boundaries == packing_params (modulo the pad_to_multiple_of tail, which gets unpadded)
        print(f"bin {b}: {n_sub} sub-sequences, {end} tokens; collator cu_seq_lens {cu[:4]}... vs bounds {bounds[:4]}...")
        assert cu[: n_sub + 1] == bounds[: n_sub + 1], (cu[: n_sub + 1], bounds[: n_sub + 1])
        nll_packed = per_token_nll(model, feat, labels)
        # 2. each member alone
        for i in range(n_sub):
            s, e = bounds[i], bounds[i + 1]
            ids = torch.tensor([row["input_ids"][s:e]]).cuda()
            lab = torch.tensor([row["labels"][s:e]]).cuda()
            alone = per_token_nll(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, lab)
            m = lab[0, 1:] != IGNORE_INDEX
            a = alone[m]
            p = nll_packed[s : e - 1][m]
            d = (a - p).abs()
            worst = max(worst, d.max().item())
            print(f"   member {i} (pos {s}..{e - 1}, {int(m.sum())} targets): packed {p.mean():.5f} alone {a.mean():.5f} max|d| {d.max():.4f}")
    print(f"worst per-token |d|: {worst:.4f} ->", "OK" if worst < 0.5 else "CONTAMINATED")

    # liger training-path loss on one bin honours num_items_in_batch (sum / N)
    model.train()
    row = pack[args.bins[0]]
    bounds = row["packing_params"]["sequence_boundaries"]
    n_sub = max([1] + [i for i in range(1, len(bounds)) if bounds[i] <= args.max_bin_tokens])
    end = bounds[n_sub]
    feat = coll([{k: row[k][:end] for k in ("input_ids", "attention_mask", "position_ids", "labels")} | {"packing_params": {**row["packing_params"], "sequence_boundaries": bounds[: n_sub + 1]}}])
    feat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in feat.items()}
    feat.pop("attention_mask", None)
    n = int((feat["labels"][0, 1:] != IGNORE_INDEX).sum())
    with torch.no_grad():
        l1 = model(**feat, num_items_in_batch=torch.tensor(n).cuda()).loss.item()
        l2 = model(**feat, num_items_in_batch=torch.tensor(2 * n).cuda()).loss.item()
    print(f"liger train loss: N={n} -> {l1:.5f}; 2N -> {l2:.5f} (ratio {l1 / l2:.3f}, expect 2.000) ->", "OK" if abs(l1 / l2 - 2) < 0.01 else "BAD")
    print("sanity: packed token-mean from logits", nll_packed[labels[0, 1:] != IGNORE_INDEX].mean().item() if args.bins[0] == b else "n/a")


if __name__ == "__main__":
    main()
