#!/usr/bin/env python3
"""Build a serve dir for a BASE Qwen3.5 checkpoint that uses the SFT chat template.

Why: the SFT checkpoints are trained with `chat_template.jinja` patched so that every
historical assistant turn renders its `<think>...</think>` block (the shipped Qwen3.5
template renders it only after the last user turn). Serving a base model with the
shipped template therefore gives the model a different prompt distribution than an SFT
checkpoint: the base model never sees its own earlier reasoning, the SFT model always
does. This builds a directory whose weights are the untouched base model but whose
chat template is byte-identical to the SFT serving template, so the two can be compared.

Weights and tokenizer are hard-linked (no copy). Only `chat_template.jinja` and the
inline `chat_template` field of `tokenizer_config.json` are rewritten.

    python coding_sft/prep/make_patchtpl_serve_dir.py \
        --src /storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B \
        --dst /storage/openpsi/users/chen/checkpoints/serve/base-4b-patchtpl \
        --template /storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text/chat_template.jinja
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

# Files never hard-linked (they are rewritten or irrelevant to serving).
SKIP = {"chat_template.jinja", "tokenizer_config.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--template", required=True, help="chat_template.jinja to install")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    template = Path(args.template).read_text()
    if "{%- if true %}" not in template:
        raise SystemExit(f"{args.template} is not the history-thinking template")
    dst.mkdir(parents=True, exist_ok=True)

    linked = 0
    for path in sorted(src.iterdir()):
        if not path.is_file() or path.name in SKIP or path.name.startswith("."):
            continue
        target = dst / path.name
        if target.exists():
            continue
        os.link(path, target)
        linked += 1

    (dst / "chat_template.jinja").write_text(template)

    # transformers>=5 reads the .jinja file, older versions read the inline field:
    # write both so the served template can never fall back to the shipped one.
    tok_cfg = json.loads((src / "tokenizer_config.json").read_text())
    had_inline = "chat_template" in tok_cfg
    tok_cfg["chat_template"] = template
    (dst / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2) + "\n")

    prov = {
        "base_weights": str(src),
        "patch": "chat_template.jinja + tokenizer_config.json chat_template replaced with the "
        "SFT serving template (history <think> rendered on every assistant turn); weights untouched",
        "template_source": str(args.template),
        "template_sha256": sha256(dst / "chat_template.jinja"),
        "src_template_sha256": sha256(src / "chat_template.jinja"),
        "tokenizer_config_had_inline_template": had_inline,
        "hardlinked_files": linked,
    }
    (dst / "SERVE_DIR_PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")

    print(json.dumps(prov, indent=2))
    print("total files:", sum(1 for _ in dst.iterdir()))


if __name__ == "__main__":
    main()
