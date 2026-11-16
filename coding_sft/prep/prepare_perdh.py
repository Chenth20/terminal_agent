#!/usr/bin/env python3
"""Expand whole trajectories into per-decision-history (per-step) rows.

Row k of a trajectory = all prior turns WITH THINK STRIPPED + current user turn + current assistant
turn (think kept, supervised). This reproduces the generation-time context of Terminus2 with
``interleaved_thinking=false`` (what DeepSeek saw when TH-3K was collected).

Each output row records ``traj_idx`` (0-based index into the input jsonl) and ``step`` so the packer
can map rows back to the group schedule.

    python coding_sft/prep/prepare_perdh.py \
        --input /storage/.../th3k_align.jsonl --out-dir /storage/.../perdh
"""

import argparse
import json
import os
import re

THINK_RE = re.compile(r"<think>\n.*?\n</think>\n\n", re.DOTALL)


def strip_think(content: str) -> str:
    return THINK_RE.sub("", content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="th3k_perdh")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.jsonl")

    n_rows = 0
    steps_per_traj: list[int] = []
    with open(args.input, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for traj_idx, line in enumerate(fin):
            c = json.loads(line)["conversations"]
            n_steps = len(c) // 2
            steps_per_traj.append(n_steps)
            for k in range(n_steps):
                msgs = []
                for j in range(k):  # history turns: user verbatim, assistant think-stripped
                    msgs.append({"role": "user", "content": c[2 * j]["content"]})
                    hist_asst = {"role": "assistant", "content": strip_think(c[2 * j + 1]["content"])}
                    msgs.append(hist_asst)
                msgs.append({"role": "user", "content": c[2 * k]["content"]})
                cur = dict(c[2 * k + 1])  # keeps the `loss` flag if the source row had one
                msgs.append(cur)
                fout.write(
                    json.dumps({"conversations": msgs, "traj_idx": traj_idx, "step": k}, ensure_ascii=False) + "\n"
                )
                n_rows += 1

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

    s = sorted(steps_per_traj)
    print(f"{len(steps_per_traj)} trajs -> {n_rows} rows; steps/traj p50 {s[len(s)//2]} mean {sum(s)/len(s):.1f} max {s[-1]}")
    print(f"out {out_path}")


if __name__ == "__main__":
    main()