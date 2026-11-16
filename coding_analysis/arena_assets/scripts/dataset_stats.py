#!/usr/bin/env python3
"""Uniform statistics for Terminus-2 SFT datasets (whole-trajectory jsonl, {"conversations": [...]}).

Everything is computed from the conversations themselves so datasets with and without a manifest
are comparable. Token counts use the Qwen3.5 tokenizer when --tokenizer is given, else chars/2.1.

Per dataset:
  trajectories, envs (from manifest or row2env map), solved (manifest score / external map)
  length: chars, tokens, messages, assistant turns  (p50 / p90 / max / total)
  assistant turn: think chars (p50/mean), action JSON chars, commands per turn, duration per command
  behaviour: wait turns, commands:[] turns, task_complete claims, compaction handoffs per trajectory
  noise: assistant mojibake / unparseable JSON / markup leak, user mojibake, harness parse-error msgs,
         identical consecutive observations, short-think (<20) turns, huge-think (>30k) turns
  command mix: top verbs of the first token of each keystroke line
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_bad_actions import parse_action, MOJIBAKE, MARKUP, PARSE_ERR  # noqa: E402

HANDOFF = re.compile(r"Here are the answers the other agent provided|handoff|Summary of previous", re.I)
OBS_SPLIT = re.compile(r"New Terminal Output:|Current Terminal Screen:|Current terminal state:")
VERB = re.compile(r"^\s*(?:cd\s+\S+\s*(?:&&|;)\s*)?(?:sudo\s+)?([A-Za-z0-9_./-]+)")
WRITE_FILE = re.compile(r"cat\s*>+\s*\S+\s*<<|tee\s+\S+|sed\s+-i|python3?\s+-\s*<<|>\s*\S+\.(py|sh|c|cpp|rs|go|js|ts|json|yaml|yml|toml|md|txt)\b")
TEST_RUN = re.compile(r"\b(pytest|cargo test|go test|npm test|make test|ctest|unittest|\./test|bash .*test)\b")


def q(v, p):
    if not v:
        return 0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def fmt_k(x):
    return f"{x/1000:.0f}k" if x >= 10000 else f"{x:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path without .jsonl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--row2env", default="", help="th3k: json list [{row, env}] for env count")
    ap.add_argument("--scores", default="", help="th3k: replay_scores.json {env/sN: score} (reference only)")
    ap.add_argument("--out", required=True, help="json output path")
    args = ap.parse_args()

    manifest = None if args.no_manifest else [json.loads(l) for l in open(args.base + ".manifest.jsonl")]
    row2env = None
    if args.row2env:
        row2env = {r["row"]: r["env"] for r in json.load(open(args.row2env))}

    S = Counter()
    chars, msgs, asst_n, think_len, act_len, cmds_per_turn, durations = [], [], [], [], [], [], []
    handoffs_per_traj, wait_frac, short_frac = [], [], []
    verbs = Counter()
    envs = set()
    solved = 0
    with open(args.base + ".jsonl") as f:
        for i, l in enumerate(f):
            conv = json.loads(l)["conversations"]
            S["traj"] += 1
            if manifest:
                m = manifest[i]
                envs.add(m["env_key"])
                solved += m.get("score") == 1
            elif row2env:
                envs.add(row2env.get(i, f"row{i}"))
            text_chars = sum(len(x["content"]) for x in conv)
            chars.append(text_chars)
            msgs.append(len(conv))
            n_a = 0
            n_wait = n_short = 0
            n_handoff = 0
            prev_body = None
            for j, x in enumerate(conv):
                c = x["content"]
                if x["role"] == "assistant":
                    n_a += 1
                    S["asst"] += 1
                    think, obj, rest, dup = parse_action(c)
                    if MOJIBAKE.search(c):
                        S["asst_mojibake"] += 1
                    if think is None:
                        S["no_think"] += 1
                    else:
                        think_len.append(len(think))
                        if len(think.strip()) < 20:
                            S["short_think"] += 1
                            n_short += 1
                        if len(think) > 30000:
                            S["huge_think"] += 1
                    if obj is None:
                        S["bad_json"] += 1
                        continue
                    act_len.append(len(rest))
                    if MARKUP.search(rest):
                        S["markup_leak"] += 1
                    if dup:
                        S["dup_key"] += 1
                    cmds = [cc for cc in (obj.get("commands") or []) if isinstance(cc, dict)]
                    keys = [cc.get("keystrokes", "") for cc in cmds]
                    cmds_per_turn.append(len(cmds))
                    for cc in cmds:
                        d = cc.get("duration")
                        if isinstance(d, (int, float)):
                            durations.append(float(d))
                    if obj.get("task_complete"):
                        S["complete_claims"] += 1
                    if not keys and not obj.get("task_complete"):
                        S["empty_cmd"] += 1
                    elif keys and not any(k.strip() for k in keys):
                        S["wait"] += 1
                        n_wait += 1
                    for k in keys:
                        if not k.strip():
                            continue
                        S["keystrokes"] += 1
                        mv = VERB.match(k)
                        if mv:
                            verbs[mv.group(1)] += 1
                        if WRITE_FILE.search(k):
                            S["write_file_cmds"] += 1
                        if TEST_RUN.search(k):
                            S["test_cmds"] += 1
                else:
                    if j == 0:
                        continue
                    S["user"] += 1
                    if MOJIBAKE.search(c):
                        S["user_mojibake"] += 1
                    if PARSE_ERR.search(c):
                        S["parse_err_msg"] += 1
                    if HANDOFF.search(c) and "Are you sure" not in c:
                        n_handoff += 1
                    body = OBS_SPLIT.split(c)[-1].strip()
                    if prev_body is not None and body == prev_body:
                        S["identical_obs"] += 1
                    prev_body = body
            asst_n.append(n_a)
            handoffs_per_traj.append(n_handoff)
            wait_frac.append(n_wait / max(1, n_a))
            short_frac.append(n_short / max(1, n_a))
            if n_handoff:
                S["traj_with_handoff"] += 1
            if n_wait / max(1, n_a) > 0.25:
                S["traj_poll_heavy"] += 1

    n = S["traj"]
    A = max(1, S["asst"])
    U = max(1, S["user"])
    out = {
        "name": args.name,
        "trajectories": n,
        "envs": len(envs) if envs else None,
        "solved": solved if manifest else None,
        "chars": {"p50": q(chars, .5), "p90": q(chars, .9), "max": max(chars), "total": sum(chars)},
        "tokens_est": {"p50": q(chars, .5) / 2.1, "p90": q(chars, .9) / 2.1, "max": max(chars) / 2.1, "total": sum(chars) / 2.1},
        "messages": {"p50": q(msgs, .5), "p90": q(msgs, .9), "max": max(msgs)},
        "assistant_turns": {"p50": q(asst_n, .5), "p90": q(asst_n, .9), "max": max(asst_n), "total": S["asst"], "mean": S["asst"] / n},
        "think_chars": {"p50": q(think_len, .5), "p90": q(think_len, .9), "mean": sum(think_len) / max(1, len(think_len))},
        "action_json_chars": {"p50": q(act_len, .5), "mean": sum(act_len) / max(1, len(act_len))},
        "commands_per_turn": {"mean": sum(cmds_per_turn) / max(1, len(cmds_per_turn)), "p90": q(cmds_per_turn, .9)},
        "duration_per_cmd": {"p50": q(durations, .5), "p90": q(durations, .9), "mean": sum(durations) / max(1, len(durations))},
        "compaction": {"traj_with_handoff_frac": S["traj_with_handoff"] / n, "handoffs_per_traj_mean": sum(handoffs_per_traj) / n},
        "behaviour_turn_frac": {
            "wait": S["wait"] / A, "empty_cmd": S["empty_cmd"] / A, "short_think": S["short_think"] / A,
            "huge_think": S["huge_think"] / A, "no_think": S["no_think"] / A,
            "complete_claim_traj_frac": None,
        },
        "behaviour_traj_frac": {"poll_heavy": S["traj_poll_heavy"] / n, "wait_frac_p50": q(wait_frac, .5), "wait_frac_p90": q(wait_frac, .9),
                                "short_think_frac_p90": q(short_frac, .9)},
        "noise_turn_frac": {
            "asst_mojibake": S["asst_mojibake"] / A, "bad_json": S["bad_json"] / A, "markup_leak": S["markup_leak"] / A,
            "dup_key": S["dup_key"] / A, "user_mojibake": S["user_mojibake"] / U, "parse_err_msg": S["parse_err_msg"] / U,
            "identical_obs": S["identical_obs"] / U,
        },
        "commands": {"total_keystrokes": S["keystrokes"], "write_file_frac": S["write_file_cmds"] / max(1, S["keystrokes"]),
                     "test_run_frac": S["test_cmds"] / max(1, S["keystrokes"]),
                     "top_verbs": verbs.most_common(25)},
        "complete_claims": S["complete_claims"],
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "commands"}, indent=1))
    print("top verbs:", verbs.most_common(20))


if __name__ == "__main__":
    main()