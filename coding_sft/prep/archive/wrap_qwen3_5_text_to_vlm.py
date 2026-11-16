"""Inverse of export_qwen3_5_text_only.py: wrap a text-only Qwen3_5ForCausalLM SFT checkpoint back into the
Qwen3_5ForConditionalGeneration layout so it can be served by sglang (which only registers the VLM class
and the MTP draft class for Qwen3.5).

    language weights  <- SFT checkpoint (keys already `model.language_model.*`, tie_word_embeddings)
    model.visual.*    <- base checkpoint (never trained, never used for text)
    mtp.*             <- base checkpoint (draft head; speculative decoding is lossless, only the accept rate
                         depends on how far the SFT moved the target)
    config.json       <- base config with text_config replaced by the SFT config
    tokenizer / chat_template.jinja / generation_config.json <- SFT checkpoint
    preprocessor_config.json / video_preprocessor_config.json <- base (VLM processor needs them)

    python wrap_qwen3_5_text_to_vlm.py --base /storage/.../Qwen__Qwen3.5-4B \
        --sft /storage/.../checkpoints/abl_base_ais2n --dst /storage/.../models/sft/abl_base_ais2n-final
"""
import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

TEXT_CFG_DROP = {"architectures", "transformers_version", "bos_token_id", "pad_token_id", "use_cache"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    base, sft, dst = Path(a.base), Path(a.sft), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    base_cfg = json.load(open(base / "config.json"))
    sft_cfg = json.load(open(sft / "config.json"))
    assert sft_cfg["architectures"] == ["Qwen3_5ForCausalLM"], sft_cfg["architectures"]
    text_cfg = {k: v for k, v in sft_cfg.items() if k not in TEXT_CFG_DROP}
    text_cfg["use_cache"] = True
    cfg = dict(base_cfg)
    cfg["text_config"] = text_cfg
    cfg["tie_word_embeddings"] = text_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings"))

    tensors = {}
    sft_files = sorted(sft.glob("*.safetensors"))
    for f in sft_files:
        with safe_open(f, "pt") as s:
            for k in s.keys():
                assert k.startswith("model.language_model.") or k == "lm_head.weight", k
                tensors[k] = s.get_tensor(k)
    n_lang = len(tensors)

    idx = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    extra = {k: v for k, v in idx.items() if k.startswith("model.visual.") or k.startswith("mtp.")}
    by_file = {}
    for k, f in extra.items():
        by_file.setdefault(f, []).append(k)
    for f, keys in by_file.items():
        with safe_open(base / f, "pt") as s:
            for k in keys:
                tensors[k] = s.get_tensor(k)
    base_lang = [k for k in idx if k.startswith("model.language_model.")]
    missing = sorted(set(base_lang) - set(tensors))
    assert not missing, f"SFT checkpoint lacks {len(missing)} language tensors, e.g. {missing[:3]}"
    print(f"language {n_lang} (base has {len(base_lang)}), visual+mtp {len(extra)}, total {len(tensors)}")
    dtypes = {}
    for k, t in tensors.items():
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
    print("dtypes:", dtypes)

    out = dst / "model.safetensors"
    save_file(tensors, str(out), metadata={"format": "pt"})
    json.dump(
        {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
         "weight_map": {k: "model.safetensors" for k in tensors}},
        open(dst / "model.safetensors.index.json", "w"), indent=2)
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        shutil.copy(sft / name, dst / name)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "merges.txt", "vocab.json", "LICENSE"):
        p = base / name
        if p.exists():
            shutil.copy(p, dst / name)
    (dst / "README.md").write_text(
        f"Serving wrapper: language weights from {sft}, vision tower + MTP head from {base}. "
        "Built by coding_sft/prep/wrap_qwen3_5_text_to_vlm.py; do not train from this.\n")
    print("saved", dst, f"{out.stat().st_size / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
