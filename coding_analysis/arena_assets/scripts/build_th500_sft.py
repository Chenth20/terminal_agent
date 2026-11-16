#!/usr/bin/env python3
"""
Convert one EnvArena Terminus-2-stream eval job into TerminalHorizon-3K style SFT data.

Input : <job_dir>/tasks/<env_key>/s0/{trajectory.jsonl,task.json}
        trajectory line = {"source": "user"|"agent"|"system", "message": str,
                           "reasoning_content": str, "tool_calls": [...],
                           "observation": {"results": [{"content": str}]}}
Output: {"conversations": [{"role": "user"|"assistant", "content": str}, ...]}

Mapping (mirrors /storage/openpsi/users/chen/code_data/terminal_horizon/TerminalHorizon-3K):
  * first `user` line  -> user message (harness system prompt + task + current terminal state)
  * `system` line      -> prefixed onto the next `user` message (TH-3K has no system role)
  * `agent` line       -> assistant message =  thinking{reasoning_content}{JSON}
        JSON = {"analysis", "plan", "commands": [{"keystrokes","duration"}], ["task_complete"]}
        analysis/plan come from the agent text (either raw JSON or "Analysis: ... Plan: ...")
        commands come from tool_calls[function_name == bash_command]
        task_complete=True when a mark_task_complete tool call is present
  * that agent line's observation -> the following user message ("New Terminal Output: ...")
  * agent lines whose observation is a harness JSON parse error are dropped together with
    their observation (they contain no usable assistant target and no terminal output)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

PARSE_ERROR_MARK = "Previous response had parsing errors"


# First delimiter after the analysis text in any of the raw tool-argument syntaxes the model emits:
#   JSON            ...", "plan": "...      (possibly double-escaped: \", \"plan\": \")
#   XML-ish         <plan> / <commands> / <parameter name="plan">
#   DeepSeek DSML   <｜DSML｜ parameter name="plan" ...>
_JSONISH = re.compile(
    r'(\\*"\s*,?\s*\\*"?(?:plan|commands|task_complete)\\*"?\s*[:>]'
    r'|<plan>|<commands>|<parameter name="(?:plan|commands)"[^>]*>'
    r'|(?:</｜DSML｜>\s*)?<｜DSML｜\s*(?:parameter|invoke) name="(?:plan|commands|task_complete)"(?:"?\s*:|[^>\n]*>)'
    r'|(?:\\*",?\s*)?\n?\s*<｜DSML｜\s*calls>'
    r'|\\",\s*\\"(?:plan|commands)\\")',
    re.S,
)
_PLAN_JUNK = re.compile(
    r'(\\*"\s*,\s*\\*"?(?:commands|task_complete)\\*"?.*$'
    r'|\s*<commands>.*$|\s*<parameter name="commands".*$|\\*",?\s*\n?\s*<｜DSML｜.*$|\s*<｜DSML｜\s*(?:parameter|invoke) name="(?:commands|task_complete)".*$'
    r'|\\*"\s*,?\s*$)',
    re.S,
)
_TAIL_PLAN = re.compile(r"\n\s*\**Plan\**:\s*(?P<plan>.*)$", re.S | re.IGNORECASE)


def _unescape_json_string(s: str) -> str:
    # may be single- or double-escaped (tool-call arguments re-serialised)
    for _ in range(2):
        try:
            s = json.loads('"' + s + '"')
        except json.JSONDecodeError:
            break
    return s


def parse_analysis_plan(message: str) -> tuple[str, str]:
    """Recover (analysis, plan) from an agent message.

    Shapes produced by the Terminus harness:
      1. raw JSON  {"analysis": ..., "plan": ..., ...}
      2. "Analysis: ...\nPlan: ..." text (harness re-render of the parsed response)
      3. tool-call mode (cth-terminus-2-stream 1.4.x): the harness re-render is
         'Analysis: <raw function arguments with the leading {"analysis": " removed>'
         followed by '\nPlan: <parsed plan>'. The raw arguments may be single/double
         escaped JSON or XML-ish (<plan>, <parameter name="plan">). We keep the
         analysis up to the first plan delimiter and take the plan from the trailing
         'Plan:' line, which comes from the successfully parsed response.
    """
    msg = (message or "").strip()
    if not msg:
        return "", ""

    if msg.startswith("{"):
        try:
            obj = json.loads(msg)
            if isinstance(obj, dict):
                return str(obj.get("analysis") or "").strip(), str(obj.get("plan") or "").strip()
        except json.JSONDecodeError:
            pass

    head = re.match(r"^\**Analysis\**:\s*", msg, re.IGNORECASE)
    body = msg[head.end():] if head else msg

    tail = _TAIL_PLAN.search(body)
    plan_tail = tail.group("plan").strip() if tail else ""
    if tail:
        body = body[: tail.start()]
    plan_tail = _PLAN_JUNK.sub("", plan_tail).strip()

    cut = _JSONISH.search(body)
    if cut:
        analysis = _unescape_json_string(body[: cut.start()])
        analysis = re.sub(r'(\\*"\s*,?|</｜DSML｜>)\s*$', "", analysis.rstrip())
        plan = plan_tail
        if not plan:
            seg = body[cut.start():]
            pm = (re.search(r'(?:parameter|invoke) name="plan"(?:"?\s*:\s*"|[^>\n]*>)(.*?)(?:</｜DSML｜|\\*",?\s*\n?\s*(?:</｜DSML｜>\s*)?<｜DSML｜|$)', seg, re.S)
                  or re.search(r'plan\\*"?\s*[:>]\s*\\*"?(.*?)(?:\\*"\s*,\s*\\*"commands|<commands>|$)', seg, re.S))
            plan = _unescape_json_string(pm.group(1)) if pm else ""
        plan = re.sub(r'(\\+"\s*,?\s*(?:\\n)?\s*)+$', "", _PLAN_JUNK.sub("", plan).rstrip())
        return analysis.strip(), plan.strip()

    return body.strip(), plan_tail


def build_assistant_content(line: dict[str, Any]) -> tuple[str, int]:
    """Return (content, n_bash_commands) for one agent line."""
    analysis, plan = parse_analysis_plan(line.get("message", ""))

    commands: list[dict[str, Any]] = []
    task_complete = False
    for call in line.get("tool_calls") or []:
        fname = call.get("function_name")
        args = call.get("arguments") or {}
        if fname == "bash_command":
            cmd: dict[str, Any] = {"keystrokes": args.get("keystrokes", "")}
            duration = args.get("duration")
            cmd["duration"] = 1.0 if duration is None else duration
            commands.append(cmd)
        elif fname == "mark_task_complete":
            task_complete = True

    payload: dict[str, Any] = {"analysis": analysis, "plan": plan, "commands": commands}
    if task_complete:
        payload["task_complete"] = True
    body = json.dumps(payload, ensure_ascii=False, indent=2)

    reasoning = (line.get("reasoning_content") or "").strip()
    tag_open = "<" + "think>"
    tag_close = "</" + "think>"
    content = tag_open + reasoning + tag_close + body if reasoning else body
    return content, len(commands)


def observation_text(line: dict[str, Any]) -> str:
    obs = line.get("observation") or {}
    parts: list[str] = []
    for res in obs.get("results") or []:
        text = (res or {}).get("content") or ""
        if text.strip():
            parts.append(text)
    return "".join(parts)


def convert(lines: list[dict[str, Any]], stats: dict[str, Any]) -> list[dict[str, str]]:
    convs: list[dict[str, str]] = []
    pending_system: str | None = None

    for line in lines:
        source = line.get("source")

        if source == "system":
            pending_system = (line.get("message") or "").strip()
            stats["system_lines"] += 1
            continue

        if source == "user":
            content = (line.get("message") or "").strip()
            if pending_system:
                content = f"{pending_system}\n\n{content}"
                pending_system = None
            if not content:
                stats["skipped_empty_user"] += 1
                continue
            if convs and convs[-1]["role"] == "user":
                convs[-1]["content"] += "\n\n" + content
                stats["merged_users"] += 1
                continue
            convs.append({"role": "user", "content": content})
            stats["user_turns"] += 1
            continue

        if source != "agent":
            key = f"skipped_source_{source}"
            stats[key] = stats.get(key, 0) + 1
            continue

        obs = observation_text(line)
        has_calls = bool(line.get("tool_calls"))
        if PARSE_ERROR_MARK in obs and not has_calls:
            stats["dropped_parse_error_turns"] += 1
            continue
        # Tool-call mode (harness 1.4.x) silent failure: the model's function arguments were
        # not valid JSON, so the harness executed nothing and returned a bare prompt, but it did
        # NOT send a parse-error message. The text still contains the commands the model meant
        # to run. Keeping such a turn would teach "emit analysis/plan with commands=[]" for a
        # step where the model clearly intended to act, so drop it (and its empty observation)
        # exactly like an explicit parse-error turn. The next turn is the model's own retry.
        if not has_calls and '"keystrokes"' in (line.get("message") or ""):
            stats["dropped_silent_fail_turns"] = stats.get("dropped_silent_fail_turns", 0) + 1
            continue

        content, n_cmds = build_assistant_content(line)
        stats["assistant_turns"] += 1
        stats["bash_commands"] += n_cmds
        if n_cmds == 0:
            stats["assistant_turns_no_commands"] += 1
        convs.append({"role": "assistant", "content": content})

        if obs:
            convs.append({"role": "user", "content": obs})
            stats["user_turns"] += 1
        else:
            stats["assistant_turns_without_output"] += 1

    if convs and convs[0]["role"] != "user":
        stats["dropped_leading_assistant"] += 1
        convs = convs[1:]
    # TH-3K conversations end on an assistant turn: drop the trailing terminal-output
    # message that follows the final agent step (or the final unanswered observation).
    if convs and convs[-1]["role"] == "user":
        stats["dropped_trailing_user"] += 1
        convs = convs[:-1]
    return convs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True, help="<output>/<job_id> containing tasks/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--only-solved", action="store_true")
    args = ap.parse_args()

    job_dir = Path(args.job_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "job_dir": str(job_dir),
        "dataset": args.dataset_name,
        "tasks_total": 0,
        "trajectories_found": 0,
        "trajectories_missing": [],
        "conversations_written": 0,
        "per_status": {},
        "stats": {
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
        },
    }

    for task_dir in sorted(p for p in (job_dir / "tasks").iterdir() if p.is_dir()):
        summary["tasks_total"] += 1
        s0 = task_dir / "s0"
        traj_path = s0 / "trajectory.jsonl"
        task_path = s0 / "task.json"
        meta = json.loads(task_path.read_text()) if task_path.exists() else {}
        status = meta.get("status", "UNKNOWN")
        score = meta.get("score", 0) or 0
        summary["per_status"][status] = summary["per_status"].get(status, 0) + 1

        if not traj_path.exists():
            summary["trajectories_missing"].append(
                {"env_key": task_dir.name, "status": status, "score": score}
            )
            continue

        lines = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
        if not lines:
            summary["trajectories_missing"].append(
                {"env_key": task_dir.name, "status": status, "score": score, "reason": "empty_file"}
            )
            continue
        summary["trajectories_found"] += 1

        convs = convert(lines, summary["stats"])
        if len(convs) < 2:
            summary["trajectories_missing"].append(
                {"env_key": task_dir.name, "status": status, "score": score, "reason": "no_turns"}
            )
            continue

        rows.append(
            {
                "env_key": task_dir.name,
                "status": status,
                "score": score,
                "n_messages": len(convs),
                "n_chars": sum(len(m["content"]) for m in convs),
                "conversations": convs,
            }
        )

    def write(subset: list[dict[str, Any]], path: Path) -> None:
        with path.open("w") as f:
            for row in subset:
                f.write(json.dumps({"conversations": row["conversations"]}, ensure_ascii=False) + "\n")

    path_all = out_dir / f"{args.dataset_name}.jsonl"
    write(rows, path_all)
    written = {"all": str(path_all), "all_rows": len(rows)}

    solved = [r for r in rows if r["status"] == "DONE" and r["score"] > 0]
    if solved:
        path_solved = out_dir / f"{args.dataset_name}-solved.jsonl"
        write(solved, path_solved)
        written["solved"] = str(path_solved)
        written["solved_rows"] = len(solved)

    if rows:
        chars = sorted(r["n_chars"] for r in rows)
        turns = sorted(r["n_messages"] for r in rows)
        summary["size"] = {
            "median_chars": chars[len(chars) // 2],
            "mean_chars": int(statistics.mean(chars)),
            "median_messages": turns[len(turns) // 2],
            "mean_messages": round(statistics.mean(turns), 1),
            "max_messages": turns[-1],
            "total_chars": sum(chars),
        }
    summary["conversations_written"] = len(rows)
    summary["written"] = written
    (out_dir / "build_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({k: summary[k] for k in ("tasks_total", "trajectories_found", "per_status", "size", "written")}, indent=2, ensure_ascii=False))
    print(json.dumps(summary["stats"], indent=2))
    print("missing:", len(summary["trajectories_missing"]))


if __name__ == "__main__":
    main()