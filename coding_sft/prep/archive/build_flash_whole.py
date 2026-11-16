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

"""Build the "flash full, reasoning visible, whole trajectory" SFT set (experiment 2 of the
condition-mismatch study) from TerminalHorizon-dsv41-96k-SFT.

Input: one conversation per compaction segment, assistant content ``<think>{reasoning}</think>{json}``
(no newlines around the tags). This script is a thin driver over
``prepare_terminal_horizon.py`` (same normalisation, JSON action validation and no-chunk policy as
the th3k_mask reference data): every assistant turn is rewritten to the ``qwen3_5_text`` template's
``("<think>\\n", "\\n</think>\\n\\n")`` layout, malformed actions are kept in context but flagged
``loss: 0`` (``--bad-pair mask``), and a conversation is written only if the WHOLE conversation fits
``--max-tokens`` (``--overlong drop``: never truncated mid-trajectory; isolated to
``<name>_overlong.jsonl``). The manifest's job / env_key / segment / status / score are copied into
each row as meta (ignored by LlamaFactory, useful for later slicing).

Must run inside the training container (the login node's `tokenizers` cannot load the Qwen3.5
tokenizer):

    python coding_sft/prep/build_flash_whole.py --out-dir /storage/openpsi/users/chen/data/flash_full_whole
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

from tokenizers import Tokenizer


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_terminal_horizon import (  # noqa: E402
    Stats,
    accumulate_token_stats,
    check_task_complete,
    normalize_assistant,
    validate_schema,
    whole_trajectory,
)


MANIFEST_KEYS = ("job", "env_key", "sample", "segment", "n_segments", "status", "score", "n_compactions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/storage/openpsi/users/chen/code_sft_data/TerminalHorizon-dsv41-96k-SFT.jsonl")
    ap.add_argument("--manifest", default="/storage/openpsi/users/chen/code_sft_data/TerminalHorizon-dsv41-96k-SFT.manifest.jsonl")
    ap.add_argument("--tokenizer", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--out-dir", default="/storage/openpsi/users/chen/data/flash_full_whole")
    ap.add_argument("--name", default="flash_whole")
    ap.add_argument("--max-tokens", type=int, default=262144, help="= training cutoff_len; longer conversations are dropped")
    ap.add_argument("--bad-pair", choices=["drop", "cut", "mask", "keep"], default="mask")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: only read the first N rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    manifest = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    stats = Stats()
    reasons = Counter()
    kept: list[tuple[int, list[dict]]] = []  # (source row, conversation)
    overlong: list[tuple[int, int]] = []  # (source row, approx tokens) - the text is NOT re-written (3.5 GB input)
    drop_by_status = Counter()

    with open(args.input, encoding="utf-8") as f:
        for li, line in enumerate(f):
            if args.max_rows and li >= args.max_rows:
                break
            stats.rows += 1
            meta = manifest[li] if li < len(manifest) else {}
            conv = json.loads(line).get("conversations")
            if not isinstance(conv, list) or not validate_schema(conv):
                stats.dropped_schema += 1
                reasons["schema"] += 1
                continue

            fixed: list[dict] = []
            bad_pairs: set[int] = set()
            for i, m in enumerate(conv):
                if m["role"] == "user":
                    fixed.append({"role": "user", "content": m["content"].strip()})
                    continue
                res = normalize_assistant(m["content"])
                if res is None:
                    bad_pairs.add(i // 2)
                    stats.bad_turns += 1
                    fixed.append({"role": "assistant", "content": m["content"]})
                    continue
                content, has_think = res
                stats.assistant_turns += 1
                stats.assistant_with_think += has_think
                stats.empty_think += not has_think
                fixed.append({"role": "assistant", "content": content})

            tok_lens = [len(e.ids) for e in tok.encode_batch([m["content"] for m in fixed])]
            for k in range(0, len(fixed), 2):
                if k // 2 not in bad_pairs and tok_lens[0] + tok_lens[k] + tok_lens[k + 1] + 24 > args.max_tokens:
                    bad_pairs.add(k // 2)
                    stats.oversize_pairs += 1
            if len(bad_pairs) * 2 >= len(fixed):
                stats.dropped_bad_turn += 1
                reasons["all_pairs_bad"] += 1
                continue
            stats.trajs_with_bad += bool(bad_pairs)
            windows = whole_trajectory(fixed, tok_lens, args.max_tokens, bad_pairs, stats, "drop", args.bad_pair)
            if windows is None or windows == "bad":
                stats.dropped_bad_turn += 1
                reasons["bad_action" if windows == "bad" else "no_first_pair"] += 1
                continue
            if windows == "overlong":
                stats.dropped_overlong += 1
                reasons["overlong"] += 1
                drop_by_status[(meta.get("status"), meta.get("score"))] += 1
                overlong.append((li, sum(tok_lens) + 8 * len(fixed)))
                continue
            check_task_complete(windows, stats)
            kept.append((li, windows[0]))
            if stats.rows % 500 == 0:
                print(f"[progress] {stats.rows} rows read, {len(kept)} kept, {stats.dropped_overlong} overlong", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.name}.jsonl")
    with open(path, "w", encoding="utf-8") as fo:
        for ti, (src_row, conv) in enumerate(kept):
            meta = accumulate_token_stats([conv], tok, stats)[0]
            m = manifest[src_row] if src_row < len(manifest) else {}
            row = {"conversations": conv, "traj_id": ti, "src_row": src_row, **meta, **{k: m.get(k) for k in MANIFEST_KEYS}}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats.windows = len(kept)
    print(f"[write] {path}: {len(kept)} conversations")
    if overlong:
        opath = os.path.join(args.out_dir, f"{args.name}_overlong.jsonl")
        with open(opath, "w", encoding="utf-8") as fo:
            for src_row, ntok in overlong:
                m = manifest[src_row] if src_row < len(manifest) else {}
                fo.write(json.dumps({"src_row": src_row, "approx_tokens": ntok, **{k: m.get(k) for k in MANIFEST_KEYS}}) + "\n")
        print(f"[write] {opath}: {len(overlong)} over-long conversations (> {args.max_tokens} tokens; src_row index only, NOT registered)")

    info = {
        args.name: {
            "file_name": os.path.basename(path),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                     "loss_tag": "loss"},
        }
    }
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as fo:
        json.dump(info, fo, indent=2)

    wt = sorted(stats.window_tokens)
    tt = sorted(stats.target_tokens)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    kept_status = Counter((m.get("status"), m.get("score")) for src, _ in kept for m in [manifest[src] if src < len(manifest) else {}])
    print("---- stats ----")
    print(f"rows read                 : {stats.rows}")
    print(f"dropped (schema)          : {stats.dropped_schema}")
    print(f"dropped (bad action)      : {stats.dropped_bad_turn}")
    print(f"dropped (overlong)        : {stats.dropped_overlong}  by (status, score): {dict(drop_by_status)}")
    print(f"kept                      : {len(kept)}  by (status, score): {dict(kept_status)}")
    print(f"bad assistant turns       : {stats.bad_turns} (masked loss=0: {stats.masked_turns})  oversize pairs: {stats.oversize_pairs}")
    print(f"assistant turns kept      : {stats.assistant_turns}  with-think rate: {stats.assistant_with_think / max(stats.assistant_turns, 1):.3f}  empty-think: {stats.empty_think}")
    print(f"task_complete             : declared-then-continued {stats.task_complete_mid} turns, missing at end {stats.task_complete_missing_end} convs")
    print(f"conv tokens p50/p90/p99/max : {q(wt, 0.5)} / {q(wt, 0.9)} / {q(wt, 0.99)} / {wt[-1] if wt else 0}  (budget {args.max_tokens}, sum {sum(wt)})")
    print(f"target tokens p50/p90/p99/max : {q(tt, 0.5)} / {q(tt, 0.9)} / {q(tt, 0.99)} / {tt[-1] if tt else 0}  (sum {sum(tt)})")
    tot = max(stats.think_tokens + stats.action_tokens + stats.obs_tokens, 1)
    print(f"token split think/action/obs  : {stats.think_tokens / tot:.1%} / {stats.action_tokens / tot:.1%} / {stats.obs_tokens / tot:.1%}")
    print(f"reasons                   : {dict(reasons)}")
    print(f"steps per epoch at bs 64  : {-(-len(kept) // 64)}")


if __name__ == "__main__":
    main()
