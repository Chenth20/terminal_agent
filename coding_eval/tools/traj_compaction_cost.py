#!/usr/bin/env python3
"""Measure post-compaction re-exploration and command diversity in tb21 eval trajectories.

For each DONE trajectory:
  - find steps whose user/system message is a compaction handoff ("Performed context summarization" /
    "Here are the answers the other agent provided")
  - count the fraction of the next K agent steps that are pure inspection (ls/cat/head/tail/sed -n/grep/wc/find/file)
  - command diversity = unique keystrokes / total keystrokes
  - duplicate-command rate = 1 - diversity

Usage: traj_compaction_cost.py <job_dir> [...]
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

INSPECT = re.compile(
    r"^\s*(ls|cat|head|tail|sed -n|grep|wc|find|file|stat|tree|du|df|pwd|echo)\b"
)
HANDOFF = ("Performed context summarization", "Here are the answers the other agent provided")


def analyse(traj: Path):
    rows = []
    with traj.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    cmds, inspect_after, after_total, compactions = [], 0, 0, 0
    pending = None  # steps left to attribute to a compaction
    for r in rows:
        msg = str(r.get("message") or "")
        if any(h in msg for h in HANDOFF):
            compactions += 1
            pending = 5
            continue
        if r.get("source") not in ("agent", "assistant"):
            continue
        bash = [c for c in (r.get("tool_calls") or []) if c.get("function_name") == "bash_command"]
        ks = [(c.get("arguments") or {}).get("keystrokes") or "" for c in bash]
        for k in ks:
            cmds.append(k.strip())
            if pending is not None:
                after_total += 1
                if INSPECT.match(k):
                    inspect_after += 1
        if pending is not None and ks:
            pending -= 1
            if pending <= 0:
                pending = None
    div = len(set(cmds)) / len(cmds) if cmds else 0
    post = inspect_after / after_total if after_total else 0
    return compactions, post, div, len(cmds)


def main(dirs):
    for d in dirs:
        recs = []
        for tj in sorted(Path(d).glob("tasks/*/s*/task.json")):
            meta = json.loads(tj.read_text())
            if meta.get("status") != "DONE":
                continue
            tr = tj.parent / "trajectory.jsonl"
            if not tr.exists():
                continue
            c, post, div, n = analyse(tr)
            if n == 0:
                continue
            recs.append((c, post, div, n))
        if not recs:
            print(f"{Path(d).name[:44]}: no DONE trajectories")
            continue
        print(
            f"{Path(d).name[:44]:46s} n={len(recs):3d} "
            f"compactions/traj={statistics.mean(r[0] for r in recs):4.2f} "
            f"inspect-of-next-5={statistics.mean(r[1] for r in recs) * 100:5.1f}% "
            f"cmd-diversity={statistics.mean(r[2] for r in recs):.3f} "
            f"cmds/traj={statistics.median(r[3] for r in recs):5.0f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])