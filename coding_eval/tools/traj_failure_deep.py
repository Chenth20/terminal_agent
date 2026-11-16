#!/usr/bin/env python3
"""Deep behavioural profile of TB2.1 trajectories, per job, split by outcome.

For every task dir <job>/tasks/<env>/s<k> (trajectory.jsonl + task.json [+ tbench_test.log]) compute:
  outcome   PASS | TTL | CAP512 | NO_DECLARE | DECLARED_NOTEST | DECLARED_PARTIAL | DECLARED_ALLFAIL | NONDONE
  steps, agent_min, compactions, cmds/step
  think_p50, think_max, frac_think_gt20k, total_think_chars, declare_think (think length at the declaring step)
  repeat_ratio   fraction of bash commands identical (normalised) to an earlier command in the same task
  err_ratio      fraction of observations containing an error marker
  sleep_cmds     commands that are only sleep/wait
  test_cmds      commands that run pytest / test scripts / the task's own verification
  explore_ratio  fraction of commands that are pure inspection (ls/cat/head/find/grep/pwd/which)
  write_cmds     commands that write files (cat >, tee, python - <<, sed -i, echo >)
  long_steps     steps whose completion_tokens >= 16k
  no_tool_steps  agent steps without any tool call (format break)
  compl_p50      completion tokens p50
  last_obs_err   the last observation contains an error marker
  declared_after_test  declared complete AND ran a test command before declaring

Usage: traj_failure_deep.py <job_dir> [...]  -> writes <job>/traj_deep.json and prints a per-outcome summary.
"""

from __future__ import annotations

import json
import re
import statistics as S
import sys
from collections import Counter
from pathlib import Path

ERR = re.compile(r"Traceback|Error:|error:|command not found|No such file|Permission denied|Segmentation|FAILED|failed", re.I)
SLEEP = re.compile(r"^\s*(sleep\s+\d|wait\b)\s*$")
TEST = re.compile(r"pytest|python[3]? -m unittest|/tests?/|test_|\bmake test\b|npm test|cargo test|go test|\bctest\b")
EXPLORE = re.compile(r"^\s*(ls|cat|head|tail|find|grep|rg|pwd|which|file|wc|stat|tree|echo \$|env|whoami|du|df|type)\b")
WRITE = re.compile(r"cat\s*>|tee\s|python3? - <<|<<\s*'?(PY|EOF)|sed -i|>\s*\S+\.(py|sh|txt|json|c|cpp|rs|go|js|ts|toml|yaml|yml|cfg|ini)\b|cp |mv |mkdir|git (checkout|commit|apply)|patch ")
PASSED = re.compile(r"(\d+) passed")
FAILED = re.compile(r"(\d+) failed")
ERRORS = re.compile(r"(\d+) error")


def norm_cmd(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())[:400]


def test_counts(log: str) -> tuple[int, int, int]:
    tail = log[-4000:]
    g = lambda rx: int(rx.search(tail).group(1)) if rx.search(tail) else 0  # noqa: E731
    return g(PASSED), g(FAILED), g(ERRORS)


def analyse(d: Path) -> dict | None:
    meta_p = d / "task.json"
    tr = d / "trajectory.jsonl"
    if not meta_p.exists() or not tr.exists():
        return None
    meta = json.loads(meta_p.read_text())
    rows = []
    for line in tr.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    ag = [r for r in rows if r.get("source") == "agent"]
    if not ag:
        return None
    think, compl = [], []
    cmds, declare_idx, no_tool, err_obs, n_obs = [], None, 0, 0, 0
    sleep_cmds = test_cmds = explore = write = 0
    test_before_declare = False
    for i, r in enumerate(ag):
        t = r.get("reasoning_content") or ""
        think.append(len(t))
        m = r.get("metrics") or {}
        if isinstance(m, dict) and m.get("completion_tokens") is not None:
            compl.append(int(m["completion_tokens"]))
        calls = r.get("tool_calls") or []
        if not calls:
            no_tool += 1
        for c in calls:
            fn = c.get("function_name")
            if fn == "mark_task_complete" and declare_idx is None:
                declare_idx = i
                test_before_declare = test_cmds > 0
            if fn == "bash_command":
                ks = (c.get("arguments") or {}).get("keystrokes") or ""
                cmds.append(norm_cmd(ks))
                if SLEEP.match(ks):
                    sleep_cmds += 1
                if TEST.search(ks):
                    test_cmds += 1
                if EXPLORE.match(ks):
                    explore += 1
                if WRITE.search(ks):
                    write += 1
        obs = r.get("observation")
        if obs:
            n_obs += 1
            txt = json.dumps(obs, ensure_ascii=False)[:20000]
            if ERR.search(txt):
                err_obs += 1
    seen, rep = set(), 0
    for c in cmds:
        if c in seen:
            rep += 1
        seen.add(c)
    compactions = sum(1 for r in rows if r.get("source") == "system")
    agent_min = (meta.get("timings") or {}).get("agent_ms", 0) / 60000
    status = meta.get("status")
    score = meta.get("score")
    log = (d / "tbench_test.log").read_text(errors="replace") if (d / "tbench_test.log").exists() else ""
    p, f, e = test_counts(log)
    if status != "DONE":
        cat = "NONDONE"
    elif score == 1:
        cat = "PASS"
    elif agent_min >= 235:
        cat = "TTL"
    elif len(ag) >= 500:
        cat = "CAP512"
    elif declare_idx is None:
        cat = "NO_DECLARE"
    elif p + f + e == 0 or "no tests ran" in log or "ERROR collecting" in log or "ModuleNotFoundError" in log[-3000:]:
        cat = "DECLARED_NOTEST"
    elif p > 0:
        cat = "DECLARED_PARTIAL"
    else:
        cat = "DECLARED_ALLFAIL"
    last_obs = json.dumps(ag[-1].get("observation") or {}, ensure_ascii=False)[:20000]
    return dict(
        env=meta["env_key"].replace("tb2-", ""), s=meta.get("sample_index"), status=status, score=score, cat=cat,
        steps=len(ag), agent_min=round(agent_min, 1), compactions=compactions,
        cmds=len(cmds), cmds_per_step=round(len(cmds) / max(1, len(ag)), 2),
        think_p50=S.median(think) if think else 0, think_max=max(think) if think else 0,
        frac_think_gt20k=round(sum(1 for x in think if x > 20000) / max(1, len(think)), 3),
        total_think=sum(think), declare_step=declare_idx,
        declare_think=(think[declare_idx] if declare_idx is not None else None),
        repeat_ratio=round(rep / max(1, len(cmds)), 3), err_ratio=round(err_obs / max(1, n_obs), 3),
        sleep_cmds=sleep_cmds, test_cmds=test_cmds, explore_ratio=round(explore / max(1, len(cmds)), 3), write_cmds=write,
        long_steps=sum(1 for x in compl if x >= 16000), no_tool_steps=no_tool,
        compl_p50=S.median(compl) if compl else 0, compl_max=max(compl) if compl else 0,
        last_obs_err=bool(ERR.search(last_obs)), declared_after_test=test_before_declare,
        tests=(p, f, e), last_fail=re.findall(r"FAILED [^\n]{0,110}", log[-3000:])[:2],
    )


def summarise(job: Path, recs: list[dict]) -> None:
    cats = Counter(r["cat"] for r in recs)
    n = len(recs)
    print(f"\n== {job.name}: tasks={n} " + " ".join(f"{k}={v}" for k, v in cats.most_common()))
    hdr = "cat               n  steps  min  comp  cmd/st  think50  thkmax  >20k  decl_thk  rep   err  sleep test  expl  write  long  notool  compl50"
    print(hdr)
    for k in ["PASS", "TTL", "CAP512", "NO_DECLARE", "DECLARED_NOTEST", "DECLARED_PARTIAL", "DECLARED_ALLFAIL", "NONDONE"]:
        g = [r for r in recs if r["cat"] == k]
        if not g:
            continue
        med = lambda key: S.median(r[key] for r in g)  # noqa: E731
        mean = lambda key: S.mean(r[key] for r in g)  # noqa: E731
        dt = [r["declare_think"] for r in g if r["declare_think"] is not None]
        print(f"{k:16s} {len(g):3d} {med('steps'):6.0f} {med('agent_min'):4.0f} {mean('compactions'):5.1f} {med('cmds_per_step'):7.2f} "
              f"{med('think_p50'):8.0f} {med('think_max'):7.0f} {mean('frac_think_gt20k'):5.2f} {S.median(dt) if dt else 0:9.0f} "
              f"{mean('repeat_ratio'):5.2f} {mean('err_ratio'):5.2f} {mean('sleep_cmds'):5.1f} {mean('test_cmds'):4.1f} "
              f"{mean('explore_ratio'):5.2f} {mean('write_cmds'):6.1f} {mean('long_steps'):5.1f} {mean('no_tool_steps'):7.1f} {med('compl_p50'):8.0f}")


def main() -> None:
    for jd in sys.argv[1:]:
        job = Path(jd)
        recs = [r for r in (analyse(d) for d in sorted(job.glob("tasks/*/s*")) if d.is_dir()) if r]
        (job / "traj_deep.json").write_text(json.dumps(recs, indent=0, ensure_ascii=False))
        summarise(job, recs)


if __name__ == "__main__":
    main()
