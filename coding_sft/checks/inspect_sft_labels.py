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

"""Run a training YAML through LlamaFactory's real data pipeline (tokenizer + template + processor)
on CPU and check the supervision mask before spending GPU time.

    python coding_sft/checks/inspect_sft_labels.py coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml --n 3

Checks per example:
  * len(input_ids) == len(labels) and at least one supervised token
  * user / terminal-observation text is never supervised
  * every assistant turn in the row is supervised (mask_history=false), and supervised text
    starts with the template's <think> opener exactly once per turn (no double-think)
  * no example hits cutoff_len (which would silently cut an assistant turn mid-JSON)
"""

import argparse
import os
import sys

import yaml


# Terminal output legitimately contains literal "<image>"/"<video>"/"<audio>" strings (e.g.
# `ctr images inspect <image>`); the Qwen3.5 template's multimodal plugin would treat them as
# media placeholders and raise. Point the placeholders at strings that never occur in text.
# Must run before llamafactory is imported (constants are read at import time).
for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=2, help="examples to print")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--set", nargs="*", default=[], help="override yaml keys, e.g. --set dataset=dbg128 dataset_dir=/x")
    ap.add_argument("--raw", default=None, help="raw jsonl of the same dataset: verify `loss: 0` turns are unsupervised")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for k in ("deepspeed", "tokenized_path", "enable_liger_kernel", "flash_attn", "use_cce"):
        cfg.pop(k, None)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg.update(
        {
            "bf16": False,
            "use_cpu": True,
            "report_to": "none",
            "max_samples": args.max_samples,
            "overwrite_cache": True,
            "preprocessing_num_workers": 4,
            "output_dir": "/tmp/inspect_sft_labels",
        }
    )
    model_args, data_args, training_args, _, _ = get_train_args(cfg)
    tok_mod = load_tokenizer(model_args)
    tok = tok_mod["tokenizer"]
    template = get_template_and_fix_tokenizer(tok, data_args)
    ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok_mod)["train_dataset"]
    raw_masked = None
    if args.raw:
        import json

        raw_masked = []
        for line in open(args.raw, encoding="utf-8"):
            conv = json.loads(line)["conversations"]
            raw_masked.append([mi // 2 for mi in range(1, len(conv), 2) if conv[mi].get("loss", 1) in (0, False)])
        print(f"raw file: {len(raw_masked)} rows, {sum(len(x) for x in raw_masked)} turns flagged loss=0")

    open_ids = tok.encode(template.thought_words[0], add_special_tokens=False)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    n_at_cutoff = 0
    problems = 0
    lens, sup_counts = [], []
    for i, row in enumerate(ds):
        ids, labels = row["input_ids"], row["labels"]
        assert len(ids) == len(labels)
        assert len(row["attention_mask"]) == len(ids) and all(row["attention_mask"]), "attention_mask must be all-ones (no attention masking of observations)"
        # labels are UNshifted (model/trainer shifts once): wherever labels[p] != IGNORE it must equal ids[p]
        if any(l != IGNORE_INDEX and l != t for l, t in zip(labels, ids)):
            print(f"[{i}] labels are not aligned with input_ids (pre-shifted?)"); problems += 1
        if ids[-1] != im_end and labels[-1] != im_end:
            pass  # trailing '\n' after <|im_end|> is part of the assistant slot; checked below via span end
        lens.append(len(ids)); sup_counts.append(sum(1 for l in labels if l != IGNORE_INDEX))
        sup = [t for t in labels if t != IGNORE_INDEX]
        if not sup:
            print(f"[{i}] NO supervised tokens"); problems += 1; continue
        if len(ids) >= data_args.cutoff_len:
            n_at_cutoff += 1
        # count assistant turns by '<|im_start|>assistant' positions
        n_assist = sum(1 for p in range(len(ids) - 1) if ids[p] == im_start and tok.decode([ids[p + 1]]).strip() == "assistant")
        # supervised spans
        spans = []
        p = 0
        while p < len(labels):
            if labels[p] != IGNORE_INDEX:
                q = p
                while q < len(labels) and labels[q] != IGNORE_INDEX:
                    q += 1
                spans.append((p, q)); p = q
            else:
                p += 1
        masked_turns = raw_masked[i] if raw_masked is not None else []
        if len(spans) != n_assist - len(masked_turns):
            print(f"[{i}] assistant turns={n_assist}, masked={len(masked_turns)}, supervised spans={len(spans)}"); problems += 1
        if masked_turns:
            # the masked turn's text must still be in the input (context) but carry no label
            heads = [p for p in range(len(ids) - 1) if ids[p] == im_start and tok.decode([ids[p + 1]]).strip() == "assistant"]
            for t in masked_turns:
                a, b = heads[t], heads[t + 1] if t + 1 < len(heads) else len(ids)
                if any(l != IGNORE_INDEX for l in labels[a:b]):
                    print(f"[{i}] masked turn {t} still has supervised tokens"); problems += 1
                if "<think>" not in tok.decode(ids[a:b]):
                    print(f"[{i}] masked turn {t} missing from context"); problems += 1
        for s, e in spans:
            txt = tok.decode(ids[s:e])
            # a span must be exactly one assistant message: starts with the thought opener,
            # ends with the turn terminator, and is never preceded by an un-masked user header
            if not txt.startswith(template.thought_words[0]):
                print(f"[{i}] supervised span does not start with thought opener: {txt[:80]!r}"); problems += 1
            if not txt.rstrip("\n").endswith("<|im_end|>"):
                print(f"[{i}] supervised span does not end with <|im_end|> (truncated turn?): {txt[-80:]!r}"); problems += 1
            if "<|im_start|>" in txt:
                print(f"[{i}] role header inside supervised span: {txt[:80]!r}"); problems += 1
            if txt.count("<think>") != 1:
                print(f"[{i}] span has {txt.count('<think>')} <think> openers (expect 1): {txt[:120]!r}"); problems += 1
        if i < args.n:
            print(f"===== example {i}: len={len(ids)} supervised={len(sup)} assistant_turns={n_assist} =====")
            print("--- first 600 chars of input ---")
            print(tok.decode(ids)[:600])
            print("--- first supervised span ---")
            print(tok.decode(ids[spans[0][0]:spans[0][1]])[:800])
            print("--- last supervised span (tail) ---")
            print(tok.decode(ids[spans[-1][0]:spans[-1][1]])[-300:])
    # the eos (<|im_end|>) of every assistant turn must be supervised
    print(f"\nchecked {len(ds)} examples; at cutoff_len={data_args.cutoff_len}: {n_at_cutoff}; problems={problems}")
    if lens:
        print(f"seq len min/median/max: {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}  supervised tokens total {sum(sup_counts)} "
              f"({sum(sup_counts)/max(sum(lens),1):.1%} of all tokens)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
