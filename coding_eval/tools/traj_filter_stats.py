#!/usr/bin/env python3
"""Per-trajectory quality filter statistics for a TerminalHorizon-format jsonl.

Computes per-row: steps, think p50/p90 (chars), cmds/step, ends_with_task_complete, and reports how many
rows survive candidate 'th3k-like' filters. Optionally writes the surviving row indices.

Usage: traj_filter_stats.py <jsonl> [--write-keep <path>] [--cmd-min 2.2] [--think-lo 1500] [--think-hi 12000]
       [--p90-max 20000] [--steps-max 120] [--require-tail]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def row_stats(conv: list[dict]) -> dict:
    a = [m for m in conv if m["role"] == "assistant"]
    th, ncmd = [], 0
    for m in a:
        c = m["content"]
        t = THINK.search(c)
        th.append(len(t.group(1)) if t else 0)
        body = c[t.end():] if t else c
        try:
            j = json.loads(body)
        except json.JSONDecodeError:
            continue
        ncmd += len(j.get("commands", []) or [])
    last = a[-1]["content"] if a else ""
    s = sorted(th)
    return dict(
        steps=len(a),
        think_p50=statistics.median(th) if th else 0,
        think_p90=s[int(0.9 * len(s))] if s else 0,
        cmd=ncmd / max(1, len(a)),
        tail=('"task_complete": true' in last) or ('"task_complete":true' in last),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--write-keep", default="")
    ap.add_argument("--cmd-min", type=float, default=2.2)
    ap.add_argument("--think-lo", type=float, default=1500)
    ap.add_argument("--think-hi", type=float, default=12000)
    ap.add_argument("--p90-max", type=float, default=20000)
    ap.add_argument("--steps-max", type=int, default=120)
    ap.add_argument("--require-tail", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            d = json.loads(line)
            rows.append(row_stats(d.get("conversations") or d.get("messages") or []))
    n = len(rows)

    def cnt(pred) -> int:
        return sum(1 for r in rows if pred(r))

    def keep(r) -> bool:
        return (r["cmd"] >= args.cmd_min and args.think_lo <= r["think_p50"] <= args.think_hi
                and r["think_p90"] < args.p90_max and r["steps"] <= args.steps_max
                and (r["tail"] or not args.require_tail))

    q = lambda key, qs: [round(sorted(r[key] for r in rows)[min(n - 1, int(n * x))], 1) for x in qs]
    print(f"{args.jsonl}: {n} rows")
    print(f"  cmd/step p10/50/90       : {q('cmd', (.1, .5, .9))}")
    print(f"  think_p50 p10/50/90      : {q('think_p50', (.1, .5, .9))}")
    print(f"  think_p90 p10/50/90      : {q('think_p90', (.1, .5, .9))}")
    print(f"  steps p10/50/90          : {q('steps', (.1, .5, .9))}")
    print(f"  ends task_complete       : {cnt(lambda r: r['tail'])}")
    print(f"  cmd>={args.cmd_min}: {cnt(lambda r: r['cmd'] >= args.cmd_min)}   "
          f"think_p50 in [{args.think_lo},{args.think_hi}]: {cnt(lambda r: args.think_lo <= r['think_p50'] <= args.think_hi)}   "
          f"think_p90<{args.p90_max}: {cnt(lambda r: r['think_p90'] < args.p90_max)}   steps<={args.steps_max}: {cnt(lambda r: r['steps'] <= args.steps_max)}")
    kept = [i for i, r in enumerate(rows) if keep(r)]
    print(f"  KEEP (all filters{' + tail' if args.require_tail else ''}): {len(kept)}")
    if args.write_keep:
        with open(args.write_keep, "w") as f:
            json.dump(kept, f)
        print(f"  wrote {args.write_keep}")


if __name__ == "__main__":
    main()