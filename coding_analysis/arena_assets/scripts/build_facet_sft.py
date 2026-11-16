#!/usr/bin/env python3
"""Convert the FACET delivery (ATIF-v1.7 ``trajectories.jsonl``) into the linear SFT layout used by
the rest of the TerminalHorizon cleaning chain.

Input : <release>/trajectories.jsonl, one record per canonical attempt:
        {task_id, attempt_id, instruction, metadata, trajectory{schema_version, session_id, agent,
         steps[], final_metrics}}. Each step is already shaped like one harness line
        (source=user|agent, message, reasoning_content, tool_calls[{function_name, arguments}],
        observation{results[{content}]}), so the lineage converter ``build_th500_sft.convert`` is
        applied unchanged - exactly the function the dsv41 / flash sets were built with.

FACET ran no compaction (zero ``source == "system"`` steps in the whole release), so every
trajectory is a single linear conversation: segment 0, n_segments 1, n_compactions 0.

Output (streaming, one pass; nothing larger than the running stats list is held in memory):
  <out>/<name>.jsonl                 {"conversations": [...]} per trajectory
  <out>/<name>.manifest.jsonl        job / env_key / sample / segment / status / score / n_chars ...
  <out>/<name>-solved.jsonl          score == 1 subset (same order)
  <out>/<name>.build_stats.json

Status is derived from the release metadata so the merged/judge steps can filter on it:
  HARNESS_FAILED    metadata.harness_error
  INFRA_FAILED      metadata.infrastructure_error
  TIMEOUT           metadata.timed_out
  DONE              otherwise
``--exclude-status`` (default HARNESS_FAILED,INFRA_FAILED) drops environment breakage only; a
timed-out trajectory is still a valid partial model run and stays, labelled TIMEOUT, because the
judge is asked to keep honest partial work (judge.md / TH App. J.1) and never sees the label.

Usage:
  python3 build_facet_sft.py \
      --release /storage/.../facet_raw/extracted/facet-deepseek-v4-1-6020-20261112-v1 \
      --out-dir /storage/.../code_sft_data/facet_linear --name facet-linear
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_th500_sft import convert  # noqa: E402


def derive_status(md: dict) -> str:
    if md.get("harness_error"):
        return "HARNESS_FAILED"
    if md.get("infrastructure_error"):
        return "INFRA_FAILED"
    if md.get("timed_out"):
        return "TIMEOUT"
    return "DONE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="extracted release dir (holds trajectories.jsonl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="facet-linear")
    ap.add_argument("--exclude-status", default="HARNESS_FAILED,INFRA_FAILED",
                    help="comma list of derived statuses to drop; '' keeps everything")
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude_status.split(",") if s.strip()}
    src = Path(args.release) / "trajectories.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "records_seen": 0, "written": 0, "skipped_status": 0, "skipped_no_turns": 0, "skipped_empty": 0,
        "solved": 0, "unscored": 0, "per_status": {}, "jobs": {},
        "conv_stats": {k: 0 for k in ("user_turns", "assistant_turns", "bash_commands",
                                      "assistant_turns_no_commands", "assistant_turns_without_output",
                                      "system_lines", "skipped_empty_user", "merged_users",
                                      "dropped_parse_error_turns", "dropped_leading_assistant",
                                      "dropped_trailing_user")},
    }
    chars: list[int] = []
    msgs: list[int] = []
    skipped: list[dict] = []

    base = out_dir / args.name
    with open(src, encoding="utf-8") as f, \
            open(str(base) + ".jsonl", "w") as dj, \
            open(str(base) + ".manifest.jsonl", "w") as mj, \
            open(str(base) + "-solved.jsonl", "w") as sj, \
            open(str(base) + "-solved.manifest.jsonl", "w") as smj:
        for raw in f:
            if not raw.strip():
                continue
            stats["records_seen"] += 1
            rec = json.loads(raw)
            md = rec.get("metadata") or {}
            status = derive_status(md)
            stats["per_status"][status] = stats["per_status"].get(status, 0) + 1
            if status in exclude:
                stats["skipped_status"] += 1
                skipped.append({"task_id": rec.get("task_id"), "status": status})
                continue
            steps = (rec.get("trajectory") or {}).get("steps") or []
            if not steps:
                stats["skipped_empty"] += 1
                skipped.append({"task_id": rec.get("task_id"), "status": status, "reason": "no_steps"})
                continue
            convs = convert(steps, stats["conv_stats"])
            if len(convs) < 2:
                stats["skipped_no_turns"] += 1
                skipped.append({"task_id": rec.get("task_id"), "status": status, "reason": "no_turns"})
                continue
            score = md.get("score")
            if score is None:
                stats["unscored"] += 1
            job = md.get("source_job") or ""
            row_meta = {
                "job": job, "env_key": rec["task_id"], "sample": "s0", "segment": 0, "n_segments": 1,
                "status": status, "score": -1 if score is None else int(score),
                "n_compactions": 0, "n_messages": len(convs),
                "n_chars": sum(len(m["content"]) for m in convs),
            }
            conv_line = json.dumps({"conversations": convs}, ensure_ascii=False) + "\n"
            meta_line = json.dumps(row_meta, ensure_ascii=False) + "\n"
            dj.write(conv_line)
            mj.write(meta_line)
            if row_meta["score"] == 1:
                stats["solved"] += 1
                sj.write(conv_line)
                smj.write(meta_line)
            stats["written"] += 1
            stats["jobs"][job] = stats["jobs"].get(job, 0) + 1
            chars.append(row_meta["n_chars"])
            msgs.append(len(convs))

    def q(vals: list[int], p: float) -> int:
        vals = sorted(vals)
        return vals[min(len(vals) - 1, int(len(vals) * p))]

    if chars:
        stats["size"] = {
            "rows": len(chars), "solved_rows": stats["solved"],
            "chars_p50": q(chars, .5), "chars_p90": q(chars, .9), "chars_max": max(chars),
            "chars_total": sum(chars),
            "messages_p50": q(msgs, .5), "messages_p90": q(msgs, .9), "messages_max": max(msgs),
        }
    stats["written_files"] = {
        "all": str(base) + ".jsonl", "manifest": str(base) + ".manifest.jsonl",
        "solved": str(base) + "-solved.jsonl", "solved_manifest": str(base) + "-solved.manifest.jsonl",
    }
    stats["skipped"] = skipped[:50]
    Path(str(base) + ".build_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("skipped",)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()