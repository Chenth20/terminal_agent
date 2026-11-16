#!/usr/bin/env python3
"""Classify DONE-but-score-0 tasks of a tb21 eval job into failure categories.

Signals per task (from trajectory.jsonl + task.json + tbench_test.log):
  TTL          agent_ms >= 235 min (killed at 4 h)
  CAP512       >= 500 agent steps (max_turns)
  NO_DECLARE   never called mark_task_complete and neither TTL nor CAP (ended otherwise, e.g. session died)
  DECLARED_*   declared complete, then tests failed:
     DECLARED_NOTEST   tbench log shows 0 tests collected / import error / no pytest run (deliverable missing)
     DECLARED_PARTIAL  some tests passed, some failed
     DECLARED_ALLFAIL  all tests failed
Also reports: steps, think p50, cmds, declare step, compactions, test counts.

Usage: classify_tb21_errors.py <job_dir> [--examples N]
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PASSED = re.compile(r"(\d+) passed")
FAILED = re.compile(r"(\d+) failed")
ERRORS = re.compile(r"(\d+) error")


def test_counts(log: str) -> tuple[int, int, int]:
    tail = log[-4000:]
    p = int(PASSED.search(tail).group(1)) if PASSED.search(tail) else 0
    f = int(FAILED.search(tail).group(1)) if FAILED.search(tail) else 0
    e = int(ERRORS.search(tail).group(1)) if ERRORS.search(tail) else 0
    return p, f, e


def analyse(d: Path) -> dict | None:
    meta = json.loads((d / "task.json").read_text())
    if meta.get("status") != "DONE" or meta.get("score") == 1:
        return None
    tr = d / "trajectory.jsonl"
    if not tr.exists():
        return None
    rows = []
    for line in tr.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    ag = [r for r in rows if r.get("source") == "agent"]
    if not ag:
        return None
    declare_idx = None
    think = []
    ncmd = 0
    for i, r in enumerate(ag):
        think.append(len(r.get("reasoning_content") or ""))
        calls = r.get("tool_calls") or []
        for c in calls:
            if c.get("function_name") == "mark_task_complete" and declare_idx is None:
                declare_idx = i
            if c.get("function_name") == "bash_command":
                ncmd += 1
    compactions = sum(1 for r in rows if r.get("source") == "system")
    agent_min = (meta.get("timings") or {}).get("agent_ms", 0) / 60000
    ttl = agent_min >= 235
    cap = len(ag) >= 500
    log = (d / "tbench_test.log").read_text(errors="replace") if (d / "tbench_test.log").exists() else ""
    p, f, e = test_counts(log)
    if ttl:
        cat = "TTL"
    elif cap:
        cat = "CAP512"
    elif declare_idx is None:
        cat = "NO_DECLARE"
    else:
        if p + f + e == 0 or "no tests ran" in log or "ERROR collecting" in log or "ModuleNotFoundError" in log[-3000:]:
            cat = "DECLARED_NOTEST"
        elif p > 0:
            cat = "DECLARED_PARTIAL"
        else:
            cat = "DECLARED_ALLFAIL"
    return dict(env=meta["env_key"], s=meta.get("sample_index"), cat=cat, steps=len(ag), think_p50=statistics.median(think),
                cmds=ncmd, declare_step=declare_idx, compactions=compactions, agent_min=agent_min, tests=(p, f, e),
                last_fail=re.findall(r"FAILED [^\n]{0,120}", log[-3000:])[:2])


def main():
    job = Path(sys.argv[1])
    recs = [r for r in (analyse(d) for d in sorted(job.glob("tasks/*/s*")) if d.is_dir()) if r]
    cats = Counter(r["cat"] for r in recs)
    print(f"{job.name}: failed DONE = {len(recs)}")
    for k, v in cats.most_common():
        grp = [r for r in recs if r["cat"] == k]
        print(f"  {k:18s} {v:4d}  steps p50={statistics.median(r['steps'] for r in grp):5.0f} think p50={statistics.median(r['think_p50'] for r in grp):6.0f} "
              f"compactions={statistics.mean(r['compactions'] for r in grp):4.1f} agent_min p50={statistics.median(r['agent_min'] for r in grp):5.0f}")
    out = job / "error_classes.json"
    out.write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()