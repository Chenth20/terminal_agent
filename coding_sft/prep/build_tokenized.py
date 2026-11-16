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

"""Build (and sanity-check) LlamaFactory's `tokenized_path` cache for a training yaml on CPU.

Same code path as training (`get_dataset` with the yaml's template / cutoff / packing settings),
no GPU, no model weights. Prints row count, token totals and the supervised fraction so a bad
dataset is caught before a 16-GPU job is launched.

    python coding_sft/prep/build_tokenized.py <train.yaml> --set dataset=... tokenized_path=... overwrite_cache=true
"""

import argparse
import os


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

import yaml

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for k in ("deepspeed", "enable_liger_kernel", "flash_attn"):
        cfg.pop(k, None)
    cfg.update({"bf16": False, "use_cpu": True, "report_to": "none", "output_dir": "/tmp/build_tokenized"})
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    assert cfg.get("tokenized_path"), "tokenized_path must be set"
    if os.path.isdir(cfg["tokenized_path"]) and not cfg.get("overwrite_cache"):
        print(f"{cfg['tokenized_path']} exists; pass overwrite_cache=true to rebuild")
        return

    model_args, data_args, training_args, _, _ = get_train_args(cfg)
    tok_mod = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tok_mod["tokenizer"], data_args)
    ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok_mod)["train_dataset"]
    n_tok = n_sup = 0
    for row in ds:
        n_tok += len(row["input_ids"])
        n_sup += sum(1 for l in row["labels"] if l != IGNORE_INDEX)
    print(f"tokenized {len(ds)} rows -> {data_args.tokenized_path}: {n_tok / 1e6:.1f}M tokens, {n_sup / 1e6:.1f}M supervised ({n_sup / max(n_tok, 1):.1%})")


if __name__ == "__main__":
    main()
