#!/usr/bin/env python3
"""Process-quality LLM judge for whole Terminus-2 trajectories (TerminalHorizon App F / J.1).

Implements coding_sft/experiment_analysis/judge.md verbatim, with the adaptations A1-A5:
  A1  judge = matrixllm claude-opus-4-8 (strongest available), reasoning effort high
  A2  the judge never sees score / status / job id; record_id = sha1(env_key|sample)[:16]
  A3  one extra line in "Understanding the trajectory" about compaction handoff summaries
  A4  trajectories above --max-chars (or rejected for context length) -> verdict OVERLONG_UNJUDGED
  A5  one trajectory per request, temperature 0, strict JSON, <=2 retries -> JUDGE_ERROR;
      a gateway safety refusal (finish_reason=content_filter) -> CONTENT_FILTERED (deterministic, excluded)

Input : <base>.jsonl (+ line-aligned <base>.manifest.jsonl) in the dsv41_merged format.
Output: <out>/reviews.jsonl  one line per record (resumable: existing record_ids are skipped),
        <out>/summary.json   verdict counts, DROP reason buckets, token/latency stats.

The trajectory is rendered as plain text, one block per message:
    === USER (turn k) ===\n<content>\n
    === ASSISTANT (turn k) ===\n<content>\n
Nothing is truncated or summarised.

Usage:
  python3 llm_judge.py --base /storage/.../dsv41_merged/dsv41_merged --out /storage/.../dsv41_merged/judge_opus48 \
      [--limit N] [--workers 8] [--max-chars 1000000] [--only-ids id,id]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://matrixllm.alipay.com/v1/chat/completions"
MODEL = "claude-opus-4-8"
CHARS_PER_TOKEN = 2.0           # measured 1.95-2.29 on this corpus with the model's own counter

# ----------------------------------------------------------------------------------------- prompt
# Appendix J.1, verbatim; the single A3 addition is marked.
JUDGE_PROMPT = """You are an expert reviewer of terminal-agent trajectories for supervised fine-tuning. Your task is to
determine whether each complete multi-turn trajectory provides a reliable and useful training example.

Read the entire conversation. Do not substitute excerpts, keyword counts, command counts, or
programmatic summaries for the original trajectory. Base your judgment exclusively on the trajectory
itself. Do not access task packages, private verifiers, external benchmarks, or the internet.

Understanding the trajectory
- The initial user message typically contains the task description and the agent interaction protocol.
- Subsequent user messages usually contain terminal observations produced by preceding commands,
rather than new user requests.
- Assistant messages contain reasoning, plans, and terminal actions. One assistant turn may contain
multiple commands.
- Reasoning, including <think>, analysis, and plan content, is not itself evidence of execution.
Evaluate it together with the actual actions and subsequent observations.
- A Terminus-2 trajectory may end with the terminal observation from its final command batch. This is
a normal ending and does not require an additional assistant confirmation.
- Final success or failure is not sufficient to determine training value. An incomplete trajectory or an
honest stopping point at an unresolved obstacle may still provide a useful demonstration.
- This harness may compact earlier context into a handoff summary when the context window fills. A handoff summary followed by continued work is a normal harness event, not context truncation or a defect.

Evaluate only the following three dimensions.

1. Trajectory validity
Is the trajectory coherent and correctly ordered? Can actions be reliably associated with their terminal
observations? Check for genuine context truncation, severe format corruption, execution in the wrong
container, infrastructure failures, explicit refusal, or private-answer leakage that compromises the
demonstration.
Do not reject an entire trajectory because of one or two isolated invalid JSON responses, malformed
commands, or parser errors. If the agent recognizes the problem, restores valid interaction, and continues
with a clear action–observation chain, this constitutes a learnable recovery process. Treat protocol errors
as an overall validity defect when they recur without reliable recovery or make action–observation
correspondence uninterpretable.
Protocol recovery does not excuse independent integrity violations. Reject trajectories whose core
solution relies on private answers or hidden-test leakage, internal-state tampering, deleting runtime or
database state to reset the task, or bypassing required task interfaces.
Distinguish failure to finish from lack of training value. An unfinished trajectory can be retained
when its main content demonstrates reliable environment understanding, root-cause diagnosis, justified
implementation, feedback-based elimination of alternatives, correction of assumptions, or identification
of unresolved obstacles.
An isolated incorrect final judgment or unsupported completion claim does not automatically invalidate
all preceding work. Determine whether the error compromises the core action chain. Retention is
possible when the main process remains reliable and useful. Reject the trajectory when an incorrect
strategy dominates the solution, central conclusions contradict observations, or the overall example
primarily demonstrates an invalid way to complete the task.

2. Environment-grounded progress
Are important actions supported by actual terminal feedback? Does the agent use files, logs, command
outputs, errors, and test results to advance the task and adjust its approach when necessary?
Identify persistent disregard of feedback, repeated ineffective commands, random attempts, unmotivated
minor edits, and unsupported completion claims. Evaluate these behaviors in context: unsuccessful
attempts can be valuable when their observations inform subsequent diagnosis, revision, or recovery.

3. Multi-turn training value
Does the trajectory demonstrate a multi-turn terminal problem-solving process worth learning? Do
interactions contribute meaningful evidence, implementation, state changes, decisions, verification, or
recovery?
Consider whether the apparent interaction could be trivially replaced by one obvious edit, a fixed script,
or a mechanical loop. Do not count SSH, Docker, or Harbor infrastructure operations, waiting, repeated
polling, or irrelevant directory browsing as evidence of task difficulty.
Do not judge training value solely from trajectory length, command count, the presence of mistakes, or
final task success.

Decision
KEEP: The trajectory is valid, grounded in the environment, and provides sufficient multi-turn training
value. Final failure, incomplete execution, or an honest stopping point does not preclude KEEP when the
main process is reliable and instructive.
DROP: The trajectory is globally invalid; its core solution relies on leakage, state tampering, or interface
bypass; its behavior substantially degenerates or lacks environmental grounding; or errors dominate the
main action chain with little reliable multi-turn learning value.

Do not assign DROP solely because the task was not completed.

Disputed cases
If prior review claims are provided, treat them as hypotheses to verify, not as facts or votes. Return to the
original conversation to establish whether the relevant commands were executed, whether observations
support the claims, whether the issue was subsequently repaired, and whether it compromises the core
solution. Explain how the trajectory evidence resolves the disagreement.

Output
Return exactly one review for each input record:
{
"record_id": "<input record ID>",
"verdict": "KEEP or DROP",
"reason": "<concise justification>",
"evidence": [
"<specific evidence from the trajectory>",
"<specific evidence from the trajectory>"
]
}
Copy record id exactly. Provide two to four concrete pieces of evidence. Do not omit, duplicate, or
invent records.
"""

FORBIDDEN_IN_INPUT = ("score", "reward", "solved", "status", "success", "cth-thfull14", "cth-thredo")


def record_id(job: str, env_key: str, sample: str) -> str:
    """Opaque id. The job is hashed in so that a re-run trajectory replacing the same (env, sample)
    is judged again instead of being skipped as done; the hash reveals nothing about the outcome."""
    return hashlib.sha1(f"{job}|{env_key}|{sample}".encode()).hexdigest()[:16]


def render(conversations: list[dict]) -> str:
    parts = []
    for i, m in enumerate(conversations):
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        parts.append(f"=== {role} (turn {i}) ===\n{m['content']}\n")
    return "\n".join(parts)


def build_user_message(rid: str, traj_text: str) -> str:
    return (f"Input record\nrecord_id: {rid}\n\n"
            f"<trajectory>\n{traj_text}\n</trajectory>\n\n"
            f"Return the review JSON for record_id {rid}.")


class Client:
    def __init__(self, key: str, timeout: int):
        self.key = key
        self.timeout = timeout

    def post(self, body: dict) -> dict:
        req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def count_tokens(self, text: str) -> int:
        d = self.post({"model": MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": text}]})
        return int(d["usage"]["prompt_tokens"])

    def judge(self, rid: str, traj_text: str, effort: str, max_tokens: int):
        body = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": effort,
            "messages": [{"role": "system", "content": JUDGE_PROMPT},
                         {"role": "user", "content": build_user_message(rid, traj_text)}],
        }
        d = self.post(body)
        ch = d["choices"][0]
        content = (ch.get("message") or {}).get("content") or ""
        return content, d.get("usage", {}), ch.get("finish_reason")


JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_review(text: str, rid: str) -> dict | None:
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("record_id") != rid:
        return None
    if obj.get("verdict") not in ("KEEP", "DROP"):
        return None
    ev = obj.get("evidence")
    if not isinstance(ev, list) or len(ev) < 2 or not all(isinstance(e, str) and e.strip() for e in ev):
        return None
    if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
        return None
    return {"record_id": rid, "verdict": obj["verdict"], "reason": obj["reason"], "evidence": ev[:4]}


def iter_records(base: str):
    """Yield (manifest, byte_offset) per record. The trajectory line itself is NOT read here: a
    merged dataset is ~3 GB and holding every pending line in memory (the previous behaviour)
    OOM-killed two concurrent judges inside a 16 GiB cgroup. Workers re-read their line lazily."""
    with open(base + ".manifest.jsonl") as mf, open(base + ".jsonl", "rb") as df:
        off = 0
        for ml in mf:
            dl = df.readline()
            if not dl:
                break
            yield json.loads(ml), off
            off += len(dl)


def read_line_at(path: str, off: int) -> str:
    with open(path, "rb") as f:
        f.seek(off)
        return f.readline().decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-ids", default="", help="comma list of record_ids (smoke tests)")
    ap.add_argument("--max-chars", type=int, default=1_900_000,
                    help="A4 gate on the rendered trajectory. Bedrock Opus 4.8 hard limit is 1,000,000 prompt "
                         "tokens; at the measured 1.95-2.29 chars/token 1.9M chars stays under it (probe: 1.88M "
                         "chars = 812k tokens, needle at the end retrieved; 2.4M chars = 1.03M tokens rejected). "
                         "A context-length HTTP 400 also maps to OVERLONG_UNJUDGED")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    key = os.environ.get("MATRIXLLM_API_KEY")
    if not key:
        sys.exit("MATRIXLLM_API_KEY not set (source ~/.secrets.env)")
    os.makedirs(args.out, exist_ok=True)
    reviews_path = os.path.join(args.out, "reviews.jsonl")
    done = set()
    if os.path.exists(reviews_path):
        for l in open(reviews_path):
            done.add(json.loads(l)["record_id"])
    id_map_path = os.path.join(args.out, "record_ids.jsonl")   # record_id -> env/sample (kept OUTSIDE the judge input)
    only = set(args.only_ids.split(",")) if args.only_ids else None

    todo = []
    with open(id_map_path, "w") as idm:
        seen_ids = set()
        for m, off in iter_records(args.base):
            rid = record_id(m["job"], m["env_key"], m["sample"])
            if rid not in seen_ids:
                idm.write(json.dumps({"record_id": rid, "job": m["job"], "env_key": m["env_key"], "sample": m["sample"]}) + "\n")
                seen_ids.add(rid)
            if only and rid not in only:
                continue
            if rid in done:
                continue
            todo.append((rid, m["n_chars"], off))
            if args.limit and len(todo) >= args.limit:
                break
    print(f"to judge: {len(todo)} (already done: {len(done)})", flush=True)
    data_path = args.base + ".jsonl"

    client = Client(key, args.timeout)
    lock = threading.Lock()
    stats = {"KEEP": 0, "DROP": 0, "OVERLONG_UNJUDGED": 0, "JUDGE_ERROR": 0, "CONTENT_FILTERED": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "latency_s": []}

    def write(rec: dict):
        with lock:
            with open(reviews_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[rec["verdict"]] += 1
            stats["prompt_tokens"] += rec.get("prompt_tokens", 0)
            stats["completion_tokens"] += rec.get("completion_tokens", 0)
            if "latency_s" in rec:
                stats["latency_s"].append(rec["latency_s"])

    def work(item):
        rid, n_chars, off = item
        d = json.loads(read_line_at(data_path, off))
        text = render(d["conversations"])
        del d                      # keep only the rendered text alive during the (minutes-long) request
        # A2 assertion: nothing in the judge input can reveal outcome. The trajectory itself is
        # allowed to contain the words (task output); the check is on our wrapper only.
        wrapper = build_user_message(rid, "")
        assert not any(w in wrapper.lower() for w in FORBIDDEN_IN_INPUT), wrapper
        if len(text) > args.max_chars:
            write({"record_id": rid, "verdict": "OVERLONG_UNJUDGED",
                   "reason": f"{len(text)} chars > {args.max_chars} (~{int(len(text) / CHARS_PER_TOKEN)} tokens)",
                   "evidence": [], "n_chars": len(text)})
            return
        last_err = ""
        for attempt in range(args.retries + 1):
            t0 = time.time()
            try:
                content, usage, finish = client.judge(rid, text, args.effort, args.max_tokens)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                last_err = f"HTTP {e.code}: {body}"
                if e.code in (400, 413) and ("too long" in body.lower() or "maximum" in body.lower()):
                    write({"record_id": rid, "verdict": "OVERLONG_UNJUDGED", "reason": last_err, "evidence": [],
                           "n_chars": len(text)})
                    return
                time.sleep(5 * (attempt + 1) + random.random() * 3)
                continue
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(5 * (attempt + 1) + random.random() * 3)
                continue
            if finish == "content_filter" and not content.strip():
                # gateway safety filter refused (seen on malware / firmware-exploit / outbreak tasks);
                # deterministic, so do not retry and do not default to KEEP
                write({"record_id": rid, "verdict": "CONTENT_FILTERED", "reason": "gateway finish_reason=content_filter, empty content",
                       "evidence": [], "n_chars": len(text), "prompt_tokens": usage.get("prompt_tokens", 0)})
                return
            rev = parse_review(content, rid)
            if rev is None:
                last_err = "unparseable/invalid review: " + content[:300].replace("\n", " ")
                time.sleep(3 * (attempt + 1) + random.random() * 3)  # gateway sometimes returns a canned reply
                continue
            rev.update({"prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "latency_s": round(time.time() - t0, 1), "n_chars": len(text), "attempt": attempt})
            write(rev)
            return
        write({"record_id": rid, "verdict": "JUDGE_ERROR", "reason": last_err, "evidence": [], "n_chars": len(text)})

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 10 == 0 or i == len(futs):
                with lock:
                    el = time.time() - t_start
                    print(f"[{i}/{len(futs)}] {el/60:.1f} min  KEEP={stats['KEEP']} DROP={stats['DROP']} "
                          f"OVERLONG={stats['OVERLONG_UNJUDGED']} ERR={stats['JUDGE_ERROR']} "
                          f"prompt_tok={stats['prompt_tokens']/1e6:.1f}M", flush=True)

    # summary over the whole reviews file
    allrev = [json.loads(l) for l in open(reviews_path)]
    from collections import Counter
    summ = {"n": len(allrev), "verdicts": dict(Counter(r["verdict"] for r in allrev)),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in allrev),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in allrev)}
    lat = sorted(r["latency_s"] for r in allrev if "latency_s" in r)
    if lat:
        summ["latency_p50_s"] = lat[len(lat) // 2]
        summ["latency_p90_s"] = lat[int(len(lat) * 0.9)]
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()