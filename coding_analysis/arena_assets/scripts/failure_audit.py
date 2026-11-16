#!/usr/bin/env python3
"""Classify every task of pulled EnvArena jobs into model vs non-model failure causes.

Reads the layout written by pull_trajectories.sh:
    <out_root>/<job_id>/summary.tsv            env_key sample status score agent_ms eval_ms
    <out_root>/<job_id>/tasks/<env>/s<i>/{trajectory.jsonl,harness.log,reward.log}

Usage:
    python3 failure_audit.py <job_id> [<job_id> ...] [--out-root DIR] [--floor-jobs JOB,JOB,...]
                             [--floor-threshold 0.03] [--markdown FILE]

Per task it reports
  * end      how the trajectory ended: self-complete | max-turns | deadline | silent-end | not-done
  * reward   test.sh exit code from reward.log, plus whether pytest actually ran (from tbench_test.log when the
             puller saved it): reward-stage bootstrap failures (uv installer unreachable, pypi timeout, no WORKDIR)
             score 0 without ever running the verifier and are NOT model failures
  * tags     non-model signals seen in the terminal output or harness log:
             pip-net (pypi ReadTimeout / connection retries), pip-nover (no matching distribution),
             dns, apt, hf (huggingface fetch errors), oom (Killed/MemoryError), disk, gateway (LLM api retry/5xx),
             gave-up-on-network (model text says it is skipping runtime verification because of connectivity)
and per job a breakdown of failed tasks by cause plus a "floor task" split: envs whose solve rate across
--floor-jobs (default: the audited jobs) is below --floor-threshold are reported separately so
model-to-model comparisons are not compressed by tasks nobody solves.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict

DEFAULT_OUT_ROOT = "/storage/openpsi/users/chen/output"
DEADLINE_MIN = 235  # agent cap is 240 min; >=235 counts as hitting the deadline
MAX_TURNS = 500  # TERMINUS_MAX_TURNS=512 minus initial/summary rows

OBS_PATTERNS = {
    "pip-net": re.compile(
        r"ReadTimeoutError|NewConnectionError|Failed to establish a new connection|Max retries exceeded"
        r"|Could not fetch URL|ProxyError|SSLError|connection broken by",
        re.I,
    ),
    "pip-nover": re.compile(r"No matching distribution found|Could not find a version that satisfies", re.I),
    "dns": re.compile(
        r"Temporary failure in name resolution|Could not resolve host|Name or service not known|getaddrinfo failed", re.I
    ),
    "apt": re.compile(r"E: Unable to locate package|E: Failed to fetch|Err:\d+ https?://", re.I),
    "hf": re.compile(r"huggingface\.co.*(timed out|Connection|Max retries)|OSError: .*(connect|offline|huggingface)", re.I),
    "oom": re.compile(r"\bKilled\b|MemoryError|Cannot allocate memory|Out of memory", re.I),
    "disk": re.compile(r"No space left on device|Disk quota exceeded", re.I),
}
GAVE_UP = re.compile(
    r"(poor|slow|no|without|limited) (network|connectivity|internet)|network (issues|timeouts?|problems)|"
    r"(rely|relying) on (syntax|static|code review)|cannot (install|download)|skip(ping)? (runtime|the) (test|verification)",
    re.I,
)
GATEWAY = re.compile(r"RateLimit|APIConnectionError|retry attempts? exhausted|RetryError|HTTP/1\.1 5\d\d|InternalServerError", re.I)
REGISTER_MODEL_NOISE = re.compile(r".*register_model.*\n")
REWARD_EXIT = re.compile(r"test\.sh exit=(\d+)")
TESTS_RAN = re.compile(r"\d+ passed|\d+ failed|passed in|collected \d+ item|no tests ran")
REWARD_BOOTSTRAP = {
    "reward:uv-installer": re.compile(r"curl: \(22\).*502|uvx: command not found|\.local/bin/env: No such file"),
    "reward:pypi-timeout": re.compile(r"Failed to (download|fetch)|operation timed out|Request failed after \d+ retries|ReadTimeoutError"),
    "reward:no-workdir": re.compile(r"No working directory set"),
    "reward:apt": re.compile(r"E: Unable to locate package|E: Failed to fetch"),
}


def read_summary(job_dir: str) -> list[dict]:
    path = os.path.join(job_dir, "summary.tsv")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def audit_task(task_dir: str, status: str, score: float, agent_ms: int) -> dict:
    row = {"end": "not-done" if status != "DONE" else "unknown", "steps": 0, "reward_exit": None, "tags": []}
    reward_log = os.path.join(task_dir, "reward.log")
    if os.path.exists(reward_log):
        m = REWARD_EXIT.search(open(reward_log, errors="ignore").read())
        if m:
            row["reward_exit"] = int(m.group(1))
    row["tests_ran"] = None
    row["reward_bootstrap"] = None
    test_log = os.path.join(task_dir, "tbench_test.log")
    if os.path.exists(test_log):
        text = open(test_log, errors="ignore").read()
        row["tests_ran"] = bool(TESTS_RAN.search(text))
        if not row["tests_ran"]:
            row["reward_bootstrap"] = next((k for k, p in REWARD_BOOTSTRAP.items() if p.search(text)), "reward:unknown-no-tests")
    harness_log = os.path.join(task_dir, "harness.log")
    if os.path.exists(harness_log):
        text = REGISTER_MODEL_NOISE.sub("", open(harness_log, errors="ignore").read())
        if GATEWAY.search(text):
            row["tags"].append("gateway")
    traj = os.path.join(task_dir, "trajectory.jsonl")
    if not os.path.exists(traj):
        if status == "DONE":
            row["end"] = "no-trajectory"
        return row
    steps: list[dict] = []
    with open(traj, errors="ignore") as fh:
        for line in fh:
            if not line.startswith("{"):
                continue
            try:
                steps.append(json.loads(line, strict=False))
            except json.JSONDecodeError:
                continue
    row["steps"] = len(steps)
    tags: set[str] = set(row["tags"])
    for step in steps:
        obs = json.dumps(step.get("observation", ""), ensure_ascii=False)
        for name, pat in OBS_PATTERNS.items():
            if pat.search(obs):
                tags.add(name)
        if step.get("source") == "agent" and GAVE_UP.search(step.get("message") or ""):
            tags.add("gave-up-on-network")
    row["tags"] = sorted(tags)
    if status != "DONE":
        return row
    last = steps[-1] if steps else {}
    calls = [c.get("function_name") for c in (last.get("tool_calls") or [])]
    if "mark_task_complete" in calls:
        row["end"] = "self-complete"
    elif len(steps) >= MAX_TURNS:
        row["end"] = "max-turns"
    elif agent_ms >= DEADLINE_MIN * 60_000:
        row["end"] = "deadline"
    else:
        row["end"] = "silent-end"
    return row


def audit_job(job_id: str, out_root: str) -> list[dict]:
    job_dir = os.path.join(out_root, job_id)
    rows = []
    for rec in read_summary(job_dir):
        status = rec["status"]
        score = float(rec["score"] or 0)
        agent_ms = int(rec["agent_ms"] or 0)
        task_dir = os.path.join(job_dir, "tasks", rec["env_key"], f"s{rec['sample']}")
        info = audit_task(task_dir, status, score, agent_ms)
        rows.append(
            {
                "job": job_id,
                "env": rec["env_key"],
                "sample": rec["sample"],
                "status": status,
                "score": score,
                "agent_min": agent_ms // 60_000,
                "eval_s": int(rec["eval_ms"] or 0) // 1000,
                **info,
            }
        )
    return rows


def classify_failure(row: dict) -> str:
    """One primary cause per failed task, most 'non-model' first."""
    if row["status"] != "DONE":
        return f"platform:{row['status']}"
    if row.get("reward_bootstrap"):
        return row["reward_bootstrap"]
    if row["reward_exit"] not in (None, 0):
        return "reward:test.sh-exit!=0"
    if row["end"] == "deadline":
        return "limit:agent-deadline"
    if row["end"] == "max-turns":
        return "limit:max-turns"
    if row["end"] == "silent-end":
        return "harness:silent-end"
    tags = set(row["tags"])
    if "gateway" in tags:
        return "infra:llm-gateway"
    if "oom" in tags or "disk" in tags:
        return "infra:oom-or-disk"
    if "gave-up-on-network" in tags and tags & {"pip-net", "dns", "apt", "hf"}:
        return "infra:network-gave-up"
    if tags & {"pip-net", "dns", "apt", "hf", "pip-nover"}:
        return "model:wrong(network-noise-present)"
    return "model:wrong"


def floor_envs(rows_by_job: dict[str, list[dict]], floor_jobs: list[str], threshold: float) -> set[str]:
    solved: Counter = Counter()
    attempts: Counter = Counter()
    for job in floor_jobs:
        for r in rows_by_job.get(job, []):
            if r["status"] != "DONE":
                continue
            attempts[r["env"]] += 1
            solved[r["env"]] += r["score"] > 0
    return {e for e in attempts if solved[e] / attempts[e] < threshold}


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def fmt(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def report(rows_by_job: dict[str, list[dict]], floors: set[str]) -> str:
    out = ["# TB2.1 failure audit", ""]
    out.append("| job | n | mean(fail=0) | mean excl. floor envs | acc given tests ran | reward bootstrap fail (tests never ran) | self-complete→0 | deadline | max-turns | silent-end | reward exit≠0 | infra (oom/disk/gateway/network) | platform |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for job, rows in rows_by_job.items():
        n = len(rows)
        m_all = sum(r["score"] for r in rows) / n if n else None
        non_floor = [r for r in rows if r["env"] not in floors]
        m_nf = sum(r["score"] for r in non_floor) / len(non_floor) if non_floor else None
        failed = [r for r in rows if r["score"] == 0]
        causes = Counter(classify_failure(r) for r in failed)
        infra = sum(v for k, v in causes.items() if k.startswith("infra:"))
        platform = sum(v for k, v in causes.items() if k.startswith("platform:"))
        boot = sum(1 for r in failed if r.get("reward_bootstrap"))
        ran = [r for r in rows if r.get("tests_ran")]
        acc_ran = sum(r["score"] > 0 for r in ran) / len(ran) if ran else None
        known = sum(1 for r in rows if r.get("tests_ran") is not None)
        out.append(
            f"| {job} | {n} | {fmt(m_all)} | {fmt(m_nf)} ({len(non_floor)}) | {fmt(acc_ran)} ({len(ran)}/{known}) | {boot} | "
            f"{sum(1 for r in failed if r['end']=='self-complete' and r['reward_exit'] in (None,0))} | "
            f"{causes.get('limit:agent-deadline',0)} | {causes.get('limit:max-turns',0)} | {causes.get('harness:silent-end',0)} | "
            f"{causes.get('reward:test.sh-exit!=0',0)} | {infra} | {platform} |"
        )
    out += ["", "## Failed tasks by primary cause", ""]
    all_causes = sorted({classify_failure(r) for rows in rows_by_job.values() for r in rows if r["score"] == 0})
    out.append("| cause | " + " | ".join(rows_by_job) + " |")
    out.append("|---|" + "---|" * len(rows_by_job))
    for cause in all_causes:
        cells = [str(sum(1 for r in rows if r["score"] == 0 and classify_failure(r) == cause)) for rows in rows_by_job.values()]
        out.append(f"| {cause} | " + " | ".join(cells) + " |")
    out += ["", "## Network signals (tasks containing the signal: failed / passed)", ""]
    out.append("| signal | " + " | ".join(rows_by_job) + " |")
    out.append("|---|" + "---|" * len(rows_by_job))
    for sig in list(OBS_PATTERNS) + ["gave-up-on-network", "gateway"]:
        cells = []
        for rows in rows_by_job.values():
            f = sum(1 for r in rows if sig in r["tags"] and r["score"] == 0)
            p = sum(1 for r in rows if sig in r["tags"] and r["score"] > 0)
            cells.append(f"{f} / {p}")
        out.append(f"| {sig} | " + " | ".join(cells) + " |")
    if floors:
        out += ["", f"## Floor envs (solve rate below threshold across reference jobs): {len(floors)}", ""]
        out += [f"- {e}" for e in sorted(floors)]
    out += ["", "## Non-model failures (task list)", ""]
    out.append("| job | env | sample | cause | end | steps | agent min | tags |")
    out.append("|---|---|---|---|---|---|---|---|")
    for job, rows in rows_by_job.items():
        for r in rows:
            if r["score"] > 0:
                continue
            cause = classify_failure(r)
            if cause.startswith("model:wrong") and cause != "model:wrong(network-noise-present)":
                continue
            out.append(f"| {job} | {r['env']} | {r['sample']} | {cause} | {r['end']} | {r['steps']} | {r['agent_min']} | {','.join(r['tags'])} |")
    out += ["", "Note: `acc given tests ran` conditions on tbench_test.log showing pytest output; tasks whose reward stage failed to bootstrap (uv installer / pypi / workdir) are excluded from that denominator. tbench_test.log is only present for jobs pulled after 10-31 (pull_trajectories.sh downloads it); older jobs show (0/0)."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--floor-jobs", default="", help="comma-separated jobs used to compute floor envs (default: the audited jobs)")
    ap.add_argument("--floor-threshold", type=float, default=0.03)
    ap.add_argument("--markdown", default="", help="also write the report to this file")
    ap.add_argument("--json", dest="json_out", default="", help="write per-task rows to this JSON file")
    args = ap.parse_args()

    rows_by_job = {j: audit_job(j, args.out_root) for j in args.jobs}
    floor_jobs = [j for j in args.floor_jobs.split(",") if j] or args.jobs
    for j in floor_jobs:
        rows_by_job.setdefault(j, audit_job(j, args.out_root))
    floors = floor_envs(rows_by_job, floor_jobs, args.floor_threshold)
    text = report({j: rows_by_job[j] for j in args.jobs}, floors)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump([r for j in args.jobs for r in rows_by_job[j]], fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
