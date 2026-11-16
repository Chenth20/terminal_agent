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

"""Export the text-only (language model) part of a Qwen3.5 VLM checkpoint.

Qwen3.5-4B ships as Qwen3_5ForConditionalGeneration: 4.21B language params + 334M vision tower
+ 121M MTP head. For text-only agent SFT the vision tower and MTP head are never run and (vision)
are frozen by LlamaFactory, but they are still loaded, sharded by ZeRO-3, carried through every
checkpoint save, and force the VLM processor / mm-plugin code path (which is why terminal output
containing the literal string "<image>" needed an env-var workaround).

This writes a Qwen3_5ForCausalLM checkpoint (model_type qwen3_5_text) with identical language
weights, tokenizer and chat template; the logits are bit-identical (verified by greedy decode).

    python coding_sft/prep/export_qwen3_5_text_only.py \
        --src /storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B \
        --dst /storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true", help="compare logits with the full VLM on a prompt (needs a GPU)")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.src)
    text_cfg = cfg.get_text_config()
    # Qwen3_5ForCausalLM ignores `mtp.*` / `model.visual.*` on load, so loading the VLM checkpoint
    # through the text class keeps exactly the language weights.
    model = Qwen3_5ForCausalLM.from_pretrained(args.src, config=text_cfg, dtype=torch.bfloat16)
    print("language params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    model.save_pretrained(args.dst, safe_serialization=True, max_shard_size="5GB")

    tok = AutoTokenizer.from_pretrained(args.src)
    tok.save_pretrained(args.dst)
    for name in ("chat_template.jinja", "generation_config.json", "LICENSE", "README.md"):
        p = Path(args.src) / name
        if p.exists() and not (Path(args.dst) / name).exists():
            shutil.copy(p, Path(args.dst) / name)
    print("saved to", args.dst)

    if args.verify:
        from transformers import Qwen3_5ForConditionalGeneration

        x = tok("<|im_start|>user\nList files.<|im_end|>\n<|im_start|>assistant\n<think>\n", return_tensors="pt").input_ids.cuda()
        text = Qwen3_5ForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16, device_map="cuda").eval()
        full = Qwen3_5ForConditionalGeneration.from_pretrained(args.src, dtype=torch.bfloat16, device_map="cuda").eval()
        with torch.no_grad():
            lt, lf = text(x).logits, full(input_ids=x).logits
            gt, gf = text.generate(x, max_new_tokens=32, do_sample=False), full.generate(input_ids=x, max_new_tokens=32, do_sample=False)
        print("max |logit diff|:", (lt - lf).abs().max().item(), "| greedy identical:", torch.equal(gt, gf))


if __name__ == "__main__":
    main()
