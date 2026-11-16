#!/usr/bin/env python3
"""Fingerprint an SFT trajectory jsonl (TerminalHorizon format): think length, cmds/step, declared durations,
steps/sample, fraction ending with task_complete, fraction whose first user turn is a compaction handoff.

Usage: sft_data_fingerprint.py <name>=<path.jsonl> [...]   (samples 250 rows per file, seed 0)
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys

THINK = re.compile(r"<think>(.*?)</think>", re.S)


def fingerprint(path: str, n: int = 250) -> dict:
    random.seed(0)
    with open(path) as f:
        lines = f.readlines()
    idx = random.sample(range(len(lines)), min(n, len(lines)))
    think, cmds, durs, steps = [], [], [], []
    k = tail = noact = handoff = 0
    for i in idx:
        d = json.loads(lines[i])
        conv = d.get("conversations") or d.get("messages") or []
        a = [m for m in conv if m["role"] == "assistant"]
        if not a:
            continue
        k += 1
        steps.append(len(a))
        for m in a:
            c = m["content"]
            t = THINK.search(c)
            think.append(len(t.group(1)) if t else 0)
            body = c[t.end():] if t else c
            try:
                j = json.loads(body)
            except json.JSONDecodeError:
                continue
            cl = j.get("commands", []) or []
            cmds.append(len(cl))
            if not cl:
                noact += 1
            for x in cl:
                try:
                    durs.append(float(x.get("duration", 0)))
                except (TypeError, ValueError):
                    pass
        last = a[-1]["content"]
        tail += ('"task_complete": true' in last) or ('"task_complete":true' in last)
        u = [m for m in conv if m["role"] == "user"]
        if u and "Here are the answers the other agent provided" in u[0]["content"]:
            handoff += 1
    return dict(
        n=k,
        think_p50=statistics.median(think),
        think_mean=round(statistics.mean(think)),
        cmd_per_step=round(statistics.mean(cmds), 2) if cmds else 0,
        dur_p50=statistics.median(durs) if durs else 0,
        dur_ge30=round(sum(x >= 30 for x in durs) / max(1, len(durs)), 3),
        steps_p50=statistics.median(steps),
        ends_task_complete=round(tail / k, 2),
        starts_with_handoff=round(handoff / k, 2),
        no_action_steps=round(noact / max(1, len(think)), 3),
    )


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        try:
            print(name, fingerprint(path))
        except FileNotFoundError as e:
            print(name, "MISSING", e)
