#!/usr/bin/env python3
"""Build a system-error-clean whole-trajectory set from one or more linear datasets.

Rules
  * rows are keyed by (env_key, sample); later --src datasets REPLACE earlier rows on the same key
    (use this for reruns: original first, rerun last)
  * inside one dataset, duplicate (env, sample) across jobs (e.g. the same long128 env sampled by two
    jobs) are all kept by suffixing the sample with the job (s0@<job>), so nothing is lost
  * a row is dropped when (job, env_key, sample) appears in any --polluted file
    (audit_env_pollution.py *_traj.json; keys "job|env|sample" or legacy "env|sample" = any job)

Usage
  build_merged.py --out /storage/.../flash_merged/flash_merged \\
      --src /storage/.../flash_linear/flash-linear \\
      --polluted /path/flash_polluted_traj.json [--polluted JOB=/path/other.json ...]
"""
import argparse
import json
import os
from collections import Counter


def load_polluted(specs):
    out = set()
    for spec in specs:
        job_prefix, path = (spec.split("=", 1) if "=" in spec else (None, spec))
        for k in json.load(open(path)):
            parts = k.split("|")
            if len(parts) == 3:
                job, env, sample = parts
                out.add((job_prefix or job or "*", env, sample))
            else:
                env, sample = parts
                out.add((job_prefix or "*", env, sample))
    return out


def is_polluted(polluted, job, env, sample):
    return (job, env, sample) in polluted or ("*", env, sample) in polluted or ("", env, sample) in polluted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="linear dataset bases (without .jsonl), in precedence order")
    ap.add_argument("--polluted", nargs="*", default=[])
    ap.add_argument("--out", required=True, help="output base (without .jsonl)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    polluted = load_polluted(args.polluted)

    rows = {}
    dropped, replaced = [], 0
    for si, base in enumerate(args.src):
        seen_in_src = Counter()
        with open(base + ".manifest.jsonl") as mf, open(base + ".jsonl") as df:
            for ml, dl in zip(mf, df):
                m = json.loads(ml)
                env, sample, job = m["env_key"], m["sample"], m.get("job", "")
                if is_polluted(polluted, job, env, sample):
                    dropped.append(m)
                    continue
                k = (env, sample)
                if seen_in_src[k]:
                    k = (env, f"{sample}@{job}")   # same env/sample sampled by another job in this dataset
                seen_in_src[(env, sample)] += 1
                if k in rows and si > 0:
                    replaced += 1
                rows[k] = (m, dl)

    n_solved = 0
    with open(args.out + ".jsonl", "w") as dj, open(args.out + ".manifest.jsonl", "w") as mj, \
         open(args.out + "-solved.jsonl", "w") as sj, open(args.out + "-solved.manifest.jsonl", "w") as smj:
        for k in sorted(rows):
            m, dl = rows[k]
            dj.write(dl)
            mj.write(json.dumps(m, ensure_ascii=False) + "\n")
            if m.get("score") == 1:
                n_solved += 1
                sj.write(dl)
                smj.write(json.dumps(m, ensure_ascii=False) + "\n")
    stats = {"rows": len(rows), "solved": n_solved, "dropped_polluted": len(dropped), "replaced": replaced,
             "envs": len({k[0] for k in rows}),
             "dropped_by_job": dict(Counter(m.get("job", "") for m in dropped)),
             "rows_by_job": dict(Counter(m.get("job", "") for m, _ in rows.values())),
             "dropped_list": [{"job": m.get("job"), "env_key": m["env_key"], "sample": m["sample"], "score": m.get("score")} for m in dropped]}
    json.dump(stats, open(args.out + ".build_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped_list"}, indent=1))


if __name__ == "__main__":
    main()