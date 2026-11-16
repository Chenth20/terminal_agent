#!/usr/bin/env python3
"""Classify tb21 eval failures as infra vs model-side by tailing each task's harness log.

infra  = 'no healthy compatible upstream endpoint' (503), or 'upstream stream request failed'
         with no max_tokens/context-limit evidence in the same log
model  = 'max_tokens', 'exceeds the maximum allowed length', 'no server running', 'embedded null byte',
         'failed to send non-blocking keys' (tmux), context-limit errors

Usage: classify_tb21_failures.py <job_id> [...]     (needs arena CLI on PATH + ~/.secrets.env sourced)
Writes /tmp/opencode/class_<job>.tsv: task_id <TAB> status <TAB> class <TAB> evidence
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys

MODEL_MARKERS = [
    ("max_tokens", "MODEL_maxtok"),
    ("exceeds the maximum allowed length", "MODEL_ctx"),
    ("no server running", "MODEL_noserver"),
    ("embedded null byte", "MODEL_nullbyte"),
    ("failed to send non-blocking keys", "MODEL_tmux"),
    ("context length", "MODEL_ctx"),
]
INFRA_MARKERS = [
    ("no healthy compatible upstream endpoint", "INFRA_503"),
    ("upstream stream request failed", "INFRA_stream"),
]


def job_json(job: str) -> dict:
    out = subprocess.run(["arena", "job", "get", job], capture_output=True, text=True,
                         env={**__import__("os").environ, "ANT_COOKIE": ""}).stdout
    return json.JSONDecoder().raw_decode(out[out.find("{"):])[0]


def classify(job: str) -> None:
    d = job_json(job)
    rows = []
    counts = collections.Counter()
    for t in d["tasks"]:
        if t["status"] == "DONE":
            continue
        log = subprocess.run(
            ["arena", "job", "log", job, t["task_id"], "harness", "--limit", "262144"],
            capture_output=True, text=True, env={**__import__("os").environ, "ANT_COOKIE": ""},
        ).stdout
        cls, ev = "UNKNOWN", ""
        for mk, name in MODEL_MARKERS:
            if mk in log:
                cls, ev = name, mk
                break
        if cls == "UNKNOWN":
            for mk, name in INFRA_MARKERS:
                if mk in log:
                    cls, ev = name, mk
                    break
        if cls == "UNKNOWN":
            if "Terminus-2 runner failed" in log or "Traceback" in log:
                cls, ev = "MODEL_runner_crash", "runner failed"
            elif not log.strip():
                cls, ev = "NO_LOG", ""
        counts[cls] += 1
        rows.append(f"{t['task_id']}\t{t['status']}\t{cls}\t{ev}")
    path = f"/tmp/opencode/class_{job}.tsv"
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(job[19:48], dict(counts), "->", path)


if __name__ == "__main__":
    for j in sys.argv[1:]:
        classify(j)