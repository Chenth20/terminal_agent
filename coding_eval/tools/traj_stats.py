#!/usr/bin/env python3
"""Per-job trajectory statistics for tb21 eval dirs under /storage/openpsi/users/chen/output/<job>.

Handles both trajectory schemas:
  - 1.0.0/1.0.1: {"source": "agent", "reasoning_content", "tool_calls": [{function_name: bash_command, arguments:{keystrokes,duration}}], "message": "analysis/plan text"}
  - 1.2.96 / 1.3.96 (this repo's stream harness): same, plus `observation`, `metrics`.

Usage: traj_stats.py <job_dir> [<job_dir> ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def analyse(traj: Path, meta: dict) -> dict:
    rows = []
    with traj.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    agent = [r for r in rows if r.get("source") in ("agent", "assistant")]
    think, ncmd, durs, declares = [], [], [], 0
    for r in agent:
        think.append(len(r.get("reasoning_content") or ""))
        calls = r.get("tool_calls") or []
        names = [c.get("function_name") for c in calls]
        if "mark_task_complete" in names or (r.get("message") or "").find("task_complete") >= 0:
            declares += 1
        bash = [c for c in calls if c.get("function_name") == "bash_command"]
        ncmd.append(len(bash))
        for c in bash:
            try:
                durs.append(float((c.get("arguments") or {}).get("duration") or 0))
            except (TypeError, ValueError):
                pass
    compactions = sum(1 for r in rows if "summarization" in str(r.get("message") or "")[:120])
    t = meta.get("timings") or {}
    agent_ms = t.get("agent_ms") or 0
    return dict(
        status=meta.get("status"),
        score=meta.get("score"),
        steps=len(agent),
        think_p50=pct(think, 0.5),
        think_mean=statistics.mean(think) if think else 0,
        cmd_per_step=statistics.mean(ncmd) if ncmd else 0,
        dur_p50=pct(durs, 0.5),
        dur_ge30=sum(1 for d in durs if d >= 30) / len(durs) if durs else 0,
        declares=declares,
        compactions=compactions,
        agent_min=agent_ms / 60000,
        ttl=agent_ms >= 235 * 60000,
    )


def main(dirs: list[str]) -> None:
    for d in dirs:
        base = Path(d)
        per = []
        for tj in sorted(base.glob("tasks/*/s*/task.json")):
            meta = json.loads(tj.read_text())
            tr = tj.parent / "trajectory.jsonl"
            if not tr.exists():
                continue
            row = analyse(tr, meta)
            if row["steps"] == 0:
                continue
            per.append(row)
        name = base.name[:44]
        if not per:
            print(f"{name}: no trajectories")
            continue
        done = [r for r in per if r["status"] == "DONE"]
        passed = [r for r in done if r["score"] == 1]
        ttl = [r for r in done if r["ttl"]]
        ge512 = [r for r in done if r["steps"] >= 512]
        print(f"\n== {name}  traj={len(per)} DONE={len(done)} pass={len(passed)} "
              f"pass/traj={len(passed)/max(1,len(per)):.3f} TTL={len(ttl)} steps>=512={len(ge512)}")
        for label, grp in (("all", per), ("DONE", done), ("PASS", passed), ("TTL", ttl)):
            if not grp:
                continue
            print(f"  {label:5s} steps p50={pct([r['steps'] for r in grp],0.5):6.0f} "
                  f"think p50={pct([r['think_p50'] for r in grp],0.5):7.0f} "
                  f"cmd/step={statistics.mean([r['cmd_per_step'] for r in grp]):.2f} "
                  f"dur>=30s={statistics.mean([r['dur_ge30'] for r in grp])*100:5.1f}% "
                  f"compactions={statistics.mean([r['compactions'] for r in grp]):.1f} "
                  f"declare/traj={statistics.mean([r['declares'] for r in grp]):.2f} "
                  f"agent_min p50={pct([r['agent_min'] for r in grp],0.5):6.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])