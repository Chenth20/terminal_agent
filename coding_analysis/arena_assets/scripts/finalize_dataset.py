#!/usr/bin/env python3
"""Final cleanup of the judged KEEP set -> dsv41_final.

Trajectory-level DROP (whole trajectory removed):
  short-think   fraction of assistant turns whose <think> is < 20 chars  > --short-think-max (0.20)
  form-A heavy  >25% of steps are foreground-blocking waits with an identical observation
  form-C heavy  >10% of steps are `commands: []` without task_complete (no duration given)

Turn-level collapse (trajectory kept, redundant pairs removed, action-observation chain stays coherent):
  form A  a run of consecutive (wait, observation) pairs whose observation is byte-identical to the
          previous observation: keep the FIRST pair of the run, drop the rest. The next pair (the one
          whose observation finally changed) is kept, so the sequence reads
          "wait -> nothing -> wait -> finished -> act" instead of 17x "wait -> nothing".
  form C  a run of consecutive `commands: []` turns (>= --collapse-min, default 3): keep the first
          and the last pair of the run.
A wait = all keystrokes empty (duration-only). Compaction handoff turns are never touched.

Outputs <out>/dsv41_final.jsonl + .manifest.jsonl (+ -solved.*), drops.tsv, collapse.tsv, build_stats.json.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_bad_actions import parse_action  # noqa: E402

OBS_SPLIT = re.compile(r"New Terminal Output:|Current Terminal Screen:|Current terminal state:")


def obs_body(text):
    parts = OBS_SPLIT.split(text)
    return parts[-1].strip()


def classify(conv):
    """Per assistant index k (pair k = conv[2k+1], conv[2k+2]): kind in {wait, empty, act, complete, badjson}."""
    kinds, short = [], []
    for j, m in enumerate(conv):
        if m["role"] != "assistant":
            continue
        think, obj, rest, dup = parse_action(m["content"])
        short.append(think is not None and len(think.strip()) < 20)
        if obj is None:
            kinds.append("badjson")
            continue
        keys = [c.get("keystrokes", "") for c in (obj.get("commands") or []) if isinstance(c, dict)]
        if obj.get("task_complete"):
            kinds.append("complete")
        elif not keys:
            kinds.append("empty")
        elif not any(k.strip() for k in keys):
            kinds.append("wait")
        else:
            kinds.append("act")
    return kinds, short


def pairs_of(conv):
    """[(asst_idx, user_idx_or_None)] in order; conv[0] is the task prompt."""
    out, i = [], 1
    while i < len(conv):
        if conv[i]["role"] != "assistant":
            i += 1
            continue
        u = i + 1 if i + 1 < len(conv) and conv[i + 1]["role"] == "user" else None
        out.append((i, u))
        i = i + 2 if u else i + 1
    return out


def collapse(conv, kinds, min_run):
    pairs = pairs_of(conv)
    assert len(pairs) == len(kinds)
    drop = set()
    stats = {"A_pairs": 0, "A_runs": 0, "C_pairs": 0, "C_runs": 0}

    # form A: identical-observation wait runs
    prev_body = obs_body(conv[0]["content"]) if conv else ""
    run = []
    for k, (a, u) in enumerate(pairs):
        body = obs_body(conv[u]["content"]) if u is not None else None
        same = kinds[k] == "wait" and body is not None and body == prev_body
        if same:
            run.append(k)
        else:
            if len(run) >= 2:
                stats["A_runs"] += 1
                for kk in run[1:]:
                    drop.add(kk)
                    stats["A_pairs"] += 1
            run = []
        if body is not None:
            prev_body = body
    if len(run) >= 2:
        stats["A_runs"] += 1
        for kk in run[1:]:
            drop.add(kk)
            stats["A_pairs"] += 1

    # form C: commands=[] runs
    run = []
    for k in range(len(pairs) + 1):
        if k < len(pairs) and kinds[k] == "empty":
            run.append(k)
            continue
        if len(run) >= min_run:
            stats["C_runs"] += 1
            for kk in run[1:-1]:
                if kk not in drop:
                    drop.add(kk)
                    stats["C_pairs"] += 1
        run = []

    keep_idx = {0}
    for k, (a, u) in enumerate(pairs):
        if k in drop:
            continue
        keep_idx.add(a)
        if u is not None:
            keep_idx.add(u)
    new = [conv[i] for i in range(len(conv)) if i in keep_idx]
    return new, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/storage/openpsi/users/chen/code_sft_data/dsv41_merged/dsv41_merged_judged-keep")
    ap.add_argument("--out", default="/storage/openpsi/users/chen/code_sft_data/dsv41_final")
    ap.add_argument("--short-think-max", type=float, default=0.20)
    ap.add_argument("--formA-max", type=float, default=0.25)
    ap.add_argument("--formC-max", type=float, default=0.10)
    ap.add_argument("--collapse-min", type=int, default=3)
    ap.add_argument("--name", default="", help="output basename (default: basename of --out, e.g. dsv41_final)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    o = os.path.join(args.out, args.name or os.path.basename(os.path.normpath(args.out)))
    drops, collapses = [], []
    n_in = n_out = n_solved = 0
    tot = {"A_pairs": 0, "A_runs": 0, "C_pairs": 0, "C_runs": 0, "turns_in": 0, "turns_out": 0}
    with open(args.base + ".manifest.jsonl") as mf, open(args.base + ".jsonl") as df, \
         open(o + ".jsonl", "w") as oj, open(o + ".manifest.jsonl", "w") as om, \
         open(o + "-solved.jsonl", "w") as sj, open(o + "-solved.manifest.jsonl", "w") as sm:
        for ml, dl in zip(mf, df):
            m = json.loads(ml)
            conv = json.loads(dl)["conversations"]
            n_in += 1
            kinds, short = classify(conv)
            n = len(kinds) or 1
            short_frac = sum(short) / n
            # form-A heavy: waits whose obs is identical to previous
            prev = obs_body(conv[0]["content"])
            a_turns = 0
            for k, (a, u) in enumerate(pairs_of(conv)):
                body = obs_body(conv[u]["content"]) if u is not None else None
                if kinds[k] == "wait" and body is not None and body == prev:
                    a_turns += 1
                if body is not None:
                    prev = body
            a_frac = a_turns / n
            c_frac = kinds.count("empty") / n
            why = []
            if short_frac > args.short_think_max:
                why.append(f"short-think {short_frac:.2f}")
            if a_frac > args.formA_max:
                why.append(f"formA {a_frac:.2f}")
            if c_frac > args.formC_max:
                why.append(f"formC {c_frac:.2f}")
            if why:
                drops.append((m["record_id"], m["env_key"], m["sample"], m["score"], n, "; ".join(why)))
                continue
            new, st = collapse(conv, kinds, args.collapse_min)
            tot["turns_in"] += n
            tot["turns_out"] += sum(1 for x in new if x["role"] == "assistant")
            for k in ("A_pairs", "A_runs", "C_pairs", "C_runs"):
                tot[k] += st[k]
            if st["A_pairs"] or st["C_pairs"]:
                collapses.append((m["record_id"], m["env_key"], m["sample"], n, st["A_runs"], st["A_pairs"], st["C_runs"], st["C_pairs"]))
            m2 = {**m, "n_messages": len(new), "n_chars": sum(len(x["content"]) for x in new),
                  "collapsed_pairs": st["A_pairs"] + st["C_pairs"]}
            line = json.dumps({"conversations": new}, ensure_ascii=False) + "\n"
            oj.write(line)
            om.write(json.dumps(m2, ensure_ascii=False) + "\n")
            n_out += 1
            if m["score"] == 1:
                n_solved += 1
                sj.write(line)
                sm.write(json.dumps(m2, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "drops.tsv"), "w") as f:
        f.write("record_id\tenv_key\tsample\tscore\tn_steps\treason\n")
        for r in drops:
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(args.out, "collapse.tsv"), "w") as f:
        f.write("record_id\tenv_key\tsample\tn_steps\tA_runs\tA_pairs_dropped\tC_runs\tC_pairs_dropped\n")
        for r in collapses:
            f.write("\t".join(map(str, r)) + "\n")
    stats = {"in": n_in, "out": n_out, "solved": n_solved, "dropped": len(drops),
             "dropped_reasons": {"short-think": sum("short-think" in r[5] for r in drops),
                                 "formA": sum("formA" in r[5] for r in drops),
                                 "formC": sum("formC" in r[5] for r in drops)},
             "collapsed_trajectories": len(collapses), **tot,
             "thresholds": {"short_think_max": args.short_think_max, "formA_max": args.formA_max,
                            "formC_max": args.formC_max, "collapse_min": args.collapse_min}}
    json.dump(stats, open(os.path.join(args.out, "build_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()