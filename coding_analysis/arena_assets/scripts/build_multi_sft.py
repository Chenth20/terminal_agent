#!/usr/bin/env python3
"""Build TerminalHorizon-3K style SFT data from multiple EnvArena jobs / samples.

Collects every tasks/<env_key>/s<idx>/trajectory.jsonl from the given job dirs
(all sample indices, not just s0) and converts each one with the same mapping as
build_th500_sft.py, emitting {"conversations": [{role, content}, ...]} lines.

Compaction: Terminus 2 replaces the chat after each summarization, so a
trajectory with k compactions is k+1 independent model contexts. By default each
is written as its own conversation (segment 0 = original prompt ... last step
before compaction; segment i>0 = original first user prompt + handoff answers +
continuation). --linear keeps the whole trajectory as one conversation instead.

Usage:
  python3 build_multi_sft.py --out-dir /storage/.../code_sft_data \
      --dataset-name TerminalHorizon-dsv41-96k-SFT \
      <job_dir> [<job_dir> ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from build_th500_sft import convert


def split_segments(lines: list[dict[str, Any]], stats: dict[str, int]) -> list[list[dict[str, str]]]:
    """Split a trajectory at compaction boundaries (source == "system").

    After compaction the harness replaces the chat with [original first user
    message, questions, answers-handoff]; segment k>0 is therefore re-prefixed
    with the original first user line so each conversation stands alone.
    """
    first_user = next((l for l in lines if l.get("source") == "user"), None)
    segments: list[list[dict[str, Any]]] = [[]]
    for line in lines:
        if line.get("source") == "system":
            segments.append([])
            continue
        segments[-1].append(line)
    out: list[list[dict[str, str]]] = []
    for i, seg in enumerate(segments):
        if i > 0 and first_user is not None:
            seg = [first_user] + seg
        convs = convert(seg, stats)
        if len(convs) >= 2:
            out.append(convs)
    return out


def new_stats() -> dict[str, int]:
    return {
        "user_turns": 0,
        "assistant_turns": 0,
        "bash_commands": 0,
        "assistant_turns_no_commands": 0,
        "assistant_turns_without_output": 0,
        "system_lines": 0,
        "skipped_empty_user": 0,
        "merged_users": 0,
        "dropped_parse_error_turns": 0,
        "dropped_leading_assistant": 0,
        "dropped_trailing_user": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dirs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--done-only", action="store_true",
                    help="only convert tasks whose platform status is DONE (skip HARNESS_FAILED/TIMEOUT/... trajectories)")
    ap.add_argument("--linear", action="store_true",
                    help="keep compacted trajectories as one conversation (handoff text inline) instead of splitting at compaction boundaries")
    ap.add_argument("--allow-manifest", default="",
                    help="manifest jsonl (job / env_key / sample per line): only convert trajectories listed in it "
                         "(e.g. the KEEP manifest written by apply_judge.py)")
    args = ap.parse_args()
    allow: set[tuple[str, str, str]] | None = None
    if args.allow_manifest:
        allow = set()
        with open(args.allow_manifest) as f:
            for raw in f:
                m = json.loads(raw)
                allow.add((m["job"], m["env_key"], m["sample"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "jobs": {},
        "stats": new_stats(),
        "missing": [],
    }

    for job_dir in args.job_dirs:
        job_path = Path(job_dir)
        job_id = job_path.name
        jsum = {"tasks_seen": 0, "trajectories": 0, "rows": 0, "per_status": {}, "compacted_trajectories": 0}
        # status/score from the latest `arena job get` snapshot; the per-task task.json written by the
        # puller can be stale (e.g. tasks marked CANCELLED by a job stop that later re-ran to DONE)
        platform: dict[tuple[str, int], dict[str, Any]] = {}
        jj = job_path / "job.json"
        if jj.exists():
            jd = json.loads(jj.read_text())
            jd = jd.get("data", jd)
            for t in jd.get("tasks", []):
                platform[(t["env_key"], int(t["sample_index"]))] = t
        for sdir in sorted(job_path.glob("tasks/*/s*")):
            if not sdir.is_dir():
                continue
            jsum["tasks_seen"] += 1
            env_key = sdir.parent.name
            sample = sdir.name
            traj = sdir / "trajectory.jsonl"
            meta_path = sdir / "task.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            pt = platform.get((env_key, int(sample[1:])))
            if pt is not None:
                meta = pt
            status = meta.get("status", "UNKNOWN")
            score = meta.get("score", 0) or 0
            jsum["per_status"][status] = jsum["per_status"].get(status, 0) + 1
            if args.done_only and status != "DONE":
                jsum["skipped_not_done"] = jsum.get("skipped_not_done", 0) + 1
                continue
            if allow is not None and (job_id, env_key, sample) not in allow:
                jsum["skipped_not_allowed"] = jsum.get("skipped_not_allowed", 0) + 1
                continue
            if not traj.exists():
                summary["missing"].append({"job": job_id, "env_key": env_key, "sample": sample, "status": status})
                continue
            lines = []
            for raw in traj.read_text(errors="replace").splitlines():
                if not raw.strip():
                    continue
                try:
                    lines.append(json.loads(raw, strict=False))
                except json.JSONDecodeError:
                    continue
            if not lines:
                summary["missing"].append({"job": job_id, "env_key": env_key, "sample": sample,
                                           "status": status, "reason": "empty_file"})
                continue
            jsum["trajectories"] += 1
            n_compactions = sum(1 for l in lines if l.get("source") == "system")
            if args.linear:
                segs = [convert(lines, summary["stats"])]
                segs = [c for c in segs if len(c) >= 2]
            else:
                segs = split_segments(lines, summary["stats"])
            if not segs:
                summary["missing"].append({"job": job_id, "env_key": env_key, "sample": sample,
                                           "status": status, "reason": "no_turns"})
                continue
            if n_compactions:
                jsum["compacted_trajectories"] += 1
            jsum["rows"] += len(segs)
            for si, convs in enumerate(segs):
                rows.append({
                    "job": job_id,
                    "env_key": env_key,
                    "sample": sample,
                    "segment": si,
                    "n_segments": len(segs),
                    "status": status,
                    "score": score,
                    "n_compactions": n_compactions,
                    "n_messages": len(convs),
                    "n_chars": sum(len(m["content"]) for m in convs),
                    "conversations": convs,
                })
        summary["jobs"][job_id] = jsum

    def write(subset: list[dict[str, Any]], path: Path) -> None:
        with path.open("w") as f:
            for row in subset:
                f.write(json.dumps({"conversations": row["conversations"]}, ensure_ascii=False) + "\n")

    path_all = out_dir / f"{args.dataset_name}.jsonl"
    write(rows, path_all)
    solved = [r for r in rows if r["status"] == "DONE" and r["score"] > 0]
    path_solved = out_dir / f"{args.dataset_name}-solved.jsonl"
    write(solved, path_solved)

    manifest = out_dir / f"{args.dataset_name}.manifest.jsonl"
    with manifest.open("w") as f:
        for row in rows:
            f.write(json.dumps({k: row[k] for k in
                                ("job", "env_key", "sample", "segment", "n_segments", "status", "score",
                                 "n_compactions", "n_messages", "n_chars")}, ensure_ascii=False) + "\n")

    if rows:
        chars = sorted(r["n_chars"] for r in rows)
        msgs = sorted(r["n_messages"] for r in rows)
        summary["size"] = {
            "rows": len(rows),
            "solved_rows": len(solved),
            "trajectories": sum(j["trajectories"] for j in summary["jobs"].values()),
            "rows_from_compacted": sum(1 for r in rows if r["n_compactions"]),
            "median_chars": chars[len(chars) // 2],
            "mean_chars": int(statistics.mean(chars)),
            "total_chars": sum(chars),
            "median_messages": msgs[len(msgs) // 2],
            "max_messages": msgs[-1],
        }
    summary["written"] = {"all": str(path_all), "solved": str(path_solved), "manifest": str(manifest)}
    (out_dir / f"{args.dataset_name}.build_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({"jobs": summary["jobs"], "size": summary.get("size"),
                      "missing": len(summary["missing"]), "written": summary["written"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
