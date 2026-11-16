#!/usr/bin/env python3
"""Offline master table of every arena eval job from local job.json snapshots (no network).

Reads /storage/openpsi/users/chen/output/<job>/job.json (last `arena job get` snapshot written by the puller)
and emits one row per job: benchmark, harness, model, status, n tasks, DONE, pass, pass@all, TTL, non-DONE breakdown,
median agent minutes, snapshot time. Also emits per-env pass vectors for the TB2.1 jobs for cross-model comparison.

Usage: offline_results.py [--out <dir>]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
from datetime import datetime

OUT_ROOT = "/storage/openpsi/users/chen/output"
TTL_MS = 235 * 60000


def load(job_dir: str):
    p = os.path.join(job_dir, "job.json")
    try:
        j = json.load(open(p))
    except Exception:
        return None
    j = j.get("data", j)
    jb = j.get("job") or {}
    ts = j.get("tasks") or []
    if not jb.get("job_id") or not ts:
        return None
    return jb, ts


def row(jb, ts, snap_mtime):
    st = collections.Counter(t["status"] for t in ts)
    done = [t for t in ts if t["status"] == "DONE"]
    passed = [t for t in done if (t.get("score") or 0) >= 1]
    ttl = [t for t in done if (t.get("timings") or {}).get("agent_ms", 0) >= TTL_MS]
    mins = [(t.get("timings") or {}).get("agent_ms", 0) / 60000 for t in done if (t.get("timings") or {}).get("agent_ms")]
    envs = collections.defaultdict(list)
    for t in done:
        envs[t["env_key"]].append(1 if (t.get("score") or 0) >= 1 else 0)
    n_env = len({t["env_key"] for t in ts})
    env1 = sum(1 for v in envs.values() if any(v))
    nond = {k: v for k, v in st.items() if k != "DONE"}
    return dict(
        job=jb["job_id"], status=jb.get("status"),
        benchmark=f'{jb["benchmark_ref"]["key"]}@{jb["benchmark_ref"]["version"]}',
        harness=f'{jb["harness_ref"]["key"]}@{jb["harness_ref"]["version"]}',
        model=jb["model_ref"]["key"], samples=jb.get("sample_numbers"), tasks=len(ts), envs=n_env,
        done=len(done), passed=len(passed), pass_at_all=round(len(passed) / len(ts), 4) if ts else None,
        pass_at_done=round(len(passed) / len(done), 4) if done else None,
        env_ge1=env1, ttl=len(ttl), agent_min_p50=round(statistics.median(mins)) if mins else None,
        non_done=json.dumps(nond, sort_keys=True), created=jb.get("created_at", "")[:16].replace("T", " "),
        snapshot=datetime.fromtimestamp(snap_mtime).strftime("%Y-%m-%d %H:%M"),
        per_env={k: v for k, v in envs.items()},
    )


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/storage/openpsi/users/chen/output/_summary")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(OUT_ROOT, "*/"))):
        r = load(d)
        if not r:
            continue
        jb, ts = r
        rows.append(row(jb, ts, os.path.getmtime(os.path.join(d, "job.json"))))
    rows.sort(key=lambda r: r["created"])
    cols = ["created", "job", "status", "benchmark", "harness", "model", "samples", "tasks", "envs", "done", "passed",
            "pass_at_all", "pass_at_done", "env_ge1", "ttl", "agent_min_p50", "non_done", "snapshot"]
    with open(os.path.join(a.out, "all_jobs.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    json.dump(rows, open(os.path.join(a.out, "all_jobs.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(rows)} jobs -> {a.out}/all_jobs.tsv")


if __name__ == "__main__":
    main()
