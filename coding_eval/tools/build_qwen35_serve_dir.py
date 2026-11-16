#!/usr/bin/env python3
"""Assemble a Qwen3.5-4B serving directory for an SFT text-only checkpoint.

The SFT checkpoints under /storage/openpsi/users/chen/checkpoints/abl_* were trained from the
text-only export (Qwen3_5ForCausalLM, 426 language_model tensors, no vision tower, no `mtp.*`
head). The AIS serving image (sglang 0.5.18, commit 85b53914) only registers
Qwen3_5ForConditionalGeneration, and the production profile (profiles/qwen35-4b.sh) runs
EAGLE MTP with the checkpoint's own `mtp.*` head. To serve the SFT weights with *identical*
parameters to the Qwen3.5-4B run, this script builds a VLM-format directory:

  * config.json / preprocessor_config.json / video_preprocessor_config.json / vocab.json /
    merges.txt                                            <- original Qwen__Qwen3.5-4B
  * tokenizer.json / tokenizer_config.json / chat_template.jinja <- SFT checkpoint (what training
                                                             tokenized with; history <think> kept)
  * model-00001-of-00002.safetensors = SFT model.safetensors (426 language_model.* tensors, bf16)
  * model-00002-of-00002.safetensors = base visual.* + mtp.* tensors (312 tensors, bf16)
  * model.safetensors.index.json covering exactly the 738 base keys

Shapes/dtypes of the 426 language_model tensors are checked against the base checkpoint.

Usage:
  build_qwen35_serve_dir.py <sft_ckpt_dir> <out_dir> [--base <base_dir>] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DEFAULT = "/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B"
LM_PREFIX = "model.language_model."
COPY_FROM_BASE = [
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
]
# Tokenizer comes from the SFT checkpoint: the text-only export re-serialized it under transformers 5
# (Qwen2 converter regex, no audio tokens) and that is what training tokenized with.
COPY_FROM_SFT = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def st_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sft")
    ap.add_argument("out")
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sft, out, base = Path(a.sft), Path(a.out), Path(a.base)

    sft_st = sft / "model.safetensors"
    if not sft_st.is_file():
        sys.exit(f"missing {sft_st}")
    if out.exists():
        if not a.force:
            sys.exit(f"{out} exists (use --force)")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    base_index = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    base_keys = set(base_index)
    lm_keys = {k for k in base_keys if k.startswith(LM_PREFIX)}
    extra_keys = base_keys - lm_keys
    sft_hdr = st_header(sft_st)
    if set(sft_hdr) != lm_keys:
        only_sft = sorted(set(sft_hdr) - lm_keys)[:5]
        only_base = sorted(lm_keys - set(sft_hdr))[:5]
        sys.exit(f"language_model key mismatch: only_sft={only_sft} only_base={only_base}")

    # shape check of the language model against base. dtype may differ: the base stores
    # linear_attn.A_log / linear_attn.norm.weight in fp32, LlamaFactory saved everything in bf16;
    # sglang's weight loaders copy_() into fp32 params, so bf16 -> fp32 is exact, only report it.
    base_hdrs = {fn: st_header(base / fn) for fn in sorted(set(base_index.values()))}
    bad, dtype_diff = [], []
    for k, meta in sft_hdr.items():
        bm = base_hdrs[base_index[k]][k]
        if bm["shape"] != meta["shape"]:
            bad.append((k, bm["shape"], meta["shape"]))
        elif bm["dtype"] != meta["dtype"]:
            dtype_diff.append((k, bm["dtype"], meta["dtype"]))
    if bad:
        sys.exit(f"shape mismatch on {len(bad)} tensors, e.g. {bad[:3]}")
    if dtype_diff:
        print(f"[{out.name}] note: {len(dtype_diff)} tensors differ in storage dtype vs base "
              f"(e.g. {dtype_diff[0]}); harmless (upcast on load)")

    # shard 1: SFT language model (hard link if same FS, else copy)
    shard1 = "model-00001-of-00002.safetensors"
    try:
        os.link(sft_st, out / shard1)
        how = "hardlink"
    except OSError:
        shutil.copy2(sft_st, out / shard1)
        how = "copy"
    print(f"[{out.name}] {shard1} <- {sft_st} ({how}, {len(sft_hdr)} tensors)")

    # shard 2: base visual + mtp
    shard2 = "model-00002-of-00002.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for k in extra_keys:
        by_file.setdefault(base_index[k], []).append(k)
    for fn, keys in by_file.items():
        with safe_open(base / fn, framework="pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k).contiguous()
    save_file(tensors, str(out / shard2), metadata={"format": "pt"})
    print(f"[{out.name}] {shard2} <- base visual+mtp ({len(tensors)} tensors)")

    weight_map = {k: shard1 for k in lm_keys}
    weight_map.update({k: shard2 for k in extra_keys})
    total = sum((out / s).stat().st_size for s in (shard1, shard2))
    json.dump(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
        open(out / "model.safetensors.index.json", "w"),
        indent=2,
    )

    for fn in COPY_FROM_BASE:
        if (base / fn).exists():
            shutil.copy2(base / fn, out / fn)
    for fn in COPY_FROM_SFT:
        shutil.copy2(sft / fn, out / fn)
    tc = json.load(open(out / "tokenizer_config.json"))
    assert "chat_template" not in tc, "expected chat_template.jinja to be the single source of truth"

    # sanity: index keys == base keys, every referenced file exists
    idx = json.load(open(out / "model.safetensors.index.json"))["weight_map"]
    assert set(idx) == base_keys, "index key set != base key set"
    for s in set(idx.values()):
        assert (out / s).is_file(), s
    # provenance
    json.dump(
        {"sft_checkpoint": str(sft.resolve()), "base": str(base.resolve()),
         "language_model_tensors": len(lm_keys), "visual_mtp_tensors": len(extra_keys),
         "tokenizer_and_chat_template": "from sft checkpoint"},
        open(out / "SERVE_DIR_PROVENANCE.json", "w"), indent=2,
    )
    print(f"[{out.name}] OK: {len(idx)} tensors indexed, config from base, tokenizer+chat_template from SFT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
