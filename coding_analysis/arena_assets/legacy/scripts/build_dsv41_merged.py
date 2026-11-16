#!/usr/bin/env python3
"""Build dsv41_merged: the system-error-clean union of
  - cth-thfull14-dsv41-2610270211 linear trajectories (th14-linear)
  - cth-thredo14-dsv41-2611011220-2611010420 replacements (already merged in dsv41_history_linear)
minus the 63 th14 trajectories whose DONE content carries environment pollution
(/tmp/opencode/sft_env_pollution_traj.json <- coding_arena/code/sft/audit_env_pollution.py;
archived copy: coding_arena/work/manifests/th14_polluted_envs.tsv).

Rule: pollution lists are per job (--polluted JOB=path ...); a row is dropped when its
(job, env_key, sample) is listed. The redo job is audited too: harness 1.7.96 fixes pip/OOM, but
ARCA's eperm on task-private networks and github reachability are NOT fixed by a rerun, so redo
rows can be polluted as well (19 of 182).
When the pollution-rerun job (cth-thredopol-dsv41-2611061153) finishes, build its linear set,
audit it, and re-run this script with --extra thredopol-linear (later rows replace same
(env, sample)) plus its own --polluted entry.

Input/output are line-aligned (<name>.jsonl + <name>.manifest.jsonl).
"""
import argparse
import json
import os

SRC_DIR = "/storage/openpsi/users/chen/code_sft_data/dsv41_history_linear"
OUT_DIR = "/storage/openpsi/users/chen/code_sft_data/dsv41_merged"


def load_polluted(specs):
    out = set()
    for spec in specs:
        job, path = spec.split("=", 1)
        for k in json.load(open(path)):
            env, sample = k.split("|")
            out.add((job, env, sample))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polluted", nargs="+", required=True, metavar="JOB=traj.json",
                    help="audit_env_pollution.py *_traj.json per source job")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional linear jsonl basenames (without .jsonl) whose rows REPLACE "
                         "same (env,sample) rows already present")
    args = ap.parse_args()

    polluted = load_polluted(args.polluted)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = {}   # (env, sample) -> (manifest, data_line)
    dropped = []
    base = os.path.join(SRC_DIR, "dsv41_history_linear")
    with open(base + ".manifest.jsonl") as mf, open(base + ".jsonl") as df:
        for ml, dl in zip(mf, df):
            m = json.loads(ml)
            k = (m["env_key"], m["sample"])
            if (m["job"], *k) in polluted:
                dropped.append(m)
                continue
            rows[k] = (m, dl)

    replaced = 0
    for name in args.extra:
        eb = os.path.join(SRC_DIR, name)
        with open(eb + ".manifest.jsonl") as mf, open(eb + ".jsonl") as df:
            for ml, dl in zip(mf, df):
                m = json.loads(ml)
                k = (m["env_key"], m["sample"])
                if (m["job"], *k) in polluted:
                    dropped.append(m)
                    continue
                if k in rows:
                    replaced += 1
                rows[k] = (m, dl)

    out = os.path.join(OUT_DIR, "dsv41_merged")
    n_solved = 0
    with open(out + ".jsonl", "w") as dj, open(out + ".manifest.jsonl", "w") as mj, \
         open(out + "-solved.jsonl", "w") as sj, open(out + "-solved.manifest.jsonl", "w") as smj:
        for k in sorted(rows):
            m, dl = rows[k]
            dj.write(dl)
            mj.write(json.dumps(m, ensure_ascii=False) + "\n")
            if m["score"] == 1:
                n_solved += 1
                sj.write(dl)
                smj.write(json.dumps(m, ensure_ascii=False) + "\n")

    stats = {"rows": len(rows), "solved": n_solved, "dropped_polluted": len(dropped),
             "replaced_by_extra": replaced,
             "dropped_by_job": {j: sum(1 for m in dropped if m["job"] == j) for j in {m["job"] for m in dropped}},
             "dropped_list": [{"job": m["job"], "env_key": m["env_key"], "sample": m["sample"], "score": m["score"]}
                              for m in dropped]}
    json.dump(stats, open(out + ".build_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped_list"}, indent=1))


if __name__ == "__main__":
    main()