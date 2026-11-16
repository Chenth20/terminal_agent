#!/usr/bin/env python3
"""Select the trajectory set for the base/A/B alignment suite.

The B arm (trunk + branches) runs one tree row per trajectory:
    keep-mode tree row = whole trajectory (trunk, thinks inline) + one branch (think_k + output_k)
    per assistant turn, i.e. tree_len = traj_len + sum(think_k + output_k + branch overhead).
VRAM caps a row at ~262k tokens (same budget as the baseline recipe), so the suite uses only
trajectories whose TREE row fits. From those, sample --n stratified by trajectory length.

Outputs: <out-dir>/th3k_align.jsonl + dataset_info.json (whole-trajectory rows, `mask` policy
loss flags preserved from the source).
"""

import argparse
import json
import os
import re

from tokenizers import Tokenizer

THINK_RE = re.compile(r"<think>\n(.*?)\n</think>\n\n", re.DOTALL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/storage/openpsi/users/chen/data/terminal_horizon_th3k_mask/th3k_mask.jsonl")
    ap.add_argument("--tokenizer", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="th3k_align")
    ap.add_argument("--n", type=int, default=306)
    ap.add_argument("--tree-budget", type=int, default=250000, help="max tree-row tokens (keep mode)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]

    kept: list[tuple[int, int, int]] = []  # (row_idx, traj_tokens, tree_tokens)
    for i, r in enumerate(rows):
        c = r["conversations"]
        texts = [m["content"] for m in c]
        lens = [len(e.ids) for e in tok.encode_batch(texts)]
        slack = 8 * len(c)
        traj = sum(lens) + slack
        # branch k duplicates think_k + output_k (assistant content); branch overhead ~8 tokens each
        asst = sum(lens[j] for j in range(1, len(c), 2))
        tree = traj + asst + 8 * (len(c) // 2)
        if tree <= args.tree_budget:
            kept.append((i, traj, tree))

    print(f"{len(rows)} source trajs -> {len(kept)} fit tree budget {args.tree_budget}")
    kept.sort(key=lambda x: x[1])
    stride = max(1, len(kept) // args.n)
    chosen = kept[::stride][: args.n]
    chosen_idx = sorted(i for i, _, _ in chosen)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in chosen_idx:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

    info = {
        args.name: {
            "file_name": f"{args.name}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "loss_tag": "loss",
            },
        }
    }
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    trajs = [t for _, t, _ in chosen]
    trees = [t for _, _, t in chosen]
    print(f"selected {len(chosen_idx)}: traj tokens p50 {sorted(trajs)[len(trajs)//2]} max {max(trajs)}")
    print(f"               tree tokens p50 {sorted(trees)[len(trees)//2]} max {max(trees)}")
    print(f"out {out_path}")


if __name__ == "__main__":
    main()