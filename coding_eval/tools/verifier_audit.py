#!/usr/bin/env python3
"""Verifier (reward-stage) bootstrap audit for TB2.1 jobs: which 0-score tasks never ran pytest, and why.

Reads <job_dir>/tasks/*/s*/tbench_test.log (written by eval/pull_trajectories.sh or `arena task download`).
A task "ran pytest" iff the log contains "test session starts" anywhere (a later crash or hang of the
test process is then a property of the deliverable, not of the reward bootstrap).
Otherwise the last 3 kB are matched against known bootstrap-failure signatures.

Usage: verifier_audit.py <job_dir> [...]
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

RAN = re.compile(r"test session starts")  # pytest session started = reward bootstrap succeeded; searched over the WHOLE log
SIGS = [
    ("no_workdir", ("No working directory set",)),
    ("uv_installer_unreachable", ("uvx: command not found", "uv: command not found", "astral.sh")),
    ("pypi_fetch_timeout", ("operation timed out", "Failed to fetch", "Request failed after", "error sending request")),
    ("uv_resolve_failed", ("No solution found", "which depends on", "depends on `")),
    ("uv_io_error", ("No such file or directory (os error 2)",)),
    ("hf_xet_403", ("xethub", "403 Forbidden")),
    ("apt_failed", ("Unable to fetch some archives", "Temporary failure resolving", "E: Failed to fetch")),
    ("oom", ("Killed", "MemoryError", "out of memory")),
    ("reward_timeout", ("shared reward exec (timeout)", "timed out after")),
    ("collect_error", ("ERROR collecting", "ModuleNotFoundError", "ImportError")),
]


def classify(text: str) -> str:
    if not text.strip():
        return "empty_log"
    tail = text[-3000:]
    for name, pats in SIGS:
        if any(p in tail for p in pats):
            return name
    return "other"


def main() -> None:
    for jd in sys.argv[1:]:
        job = Path(jd)
        n = ran = 0
        sig = collections.Counter()
        envs = collections.defaultdict(set)
        passed_cond = [0, 0]
        for log in sorted(job.glob("tasks/*/s*/tbench_test.log")):
            meta = json.loads((log.parent / "task.json").read_text())
            if meta.get("status") != "DONE":
                continue
            n += 1
            t = log.read_text(errors="replace")
            if RAN.search(t):
                ran += 1
                passed_cond[0] += 1 if meta.get("score") == 1 else 0
                passed_cond[1] += 1
                continue
            k = classify(t)
            sig[k] += 1
            envs[k].add(meta["env_key"].replace("tb2-", ""))
        if not n:
            print(f"{job.name}: no tbench_test.log")
            continue
        boot = n - ran
        print(f"\n{job.name[:60]}: DONE-with-log={n} pytest-ran={ran} bootstrap-fail={boot} ({boot / n:.1%}) "
              f"cond-acc(pytest ran)={passed_cond[0]}/{passed_cond[1]}={passed_cond[0] / max(1, passed_cond[1]):.3f}")
        for k, v in sig.most_common():
            print(f"  {k:26s} {v:3d}  envs: {', '.join(sorted(envs[k]))[:150]}")


if __name__ == "__main__":
    main()