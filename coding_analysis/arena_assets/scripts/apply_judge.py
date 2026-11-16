#!/usr/bin/env python3
"""Apply llm_judge.py reviews to a dataset: produce the KEEP set + audit tables.

  KEEP              -> <out-prefix>-keep.jsonl / .manifest.jsonl
  DROP              -> listed in drops.tsv (with reason), excluded
  OVERLONG_UNJUDGED -> listed separately (P2 long-task arm), excluded from KEEP
  JUDGE_ERROR       -> listed; re-run llm_judge.py --only-ids on them first
  CONTENT_FILTERED  -> gateway refused to review (safety filter); listed in errors.tsv, excluded

Usage:
  python3 apply_judge.py --base .../dsv41_merged/dsv41_merged --judge .../dsv41_merged/judge_opus48 \
      --out-prefix .../dsv41_merged/dsv41_merged_judged
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_judge import record_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    reviews = {}
    for l in open(os.path.join(args.judge, "reviews.jsonl")):
        r = json.loads(l)
        reviews[r["record_id"]] = r   # later lines win (reruns)

    counts = Counter()
    drops, overlong, errors, missing = [], [], [], []
    n_keep_solved = 0
    with open(args.base + ".manifest.jsonl") as mf, open(args.base + ".jsonl") as df, \
         open(args.out_prefix + "-keep.jsonl", "w") as kj, \
         open(args.out_prefix + "-keep.manifest.jsonl", "w") as km:
        for ml, dl in zip(mf, df):
            m = json.loads(ml)
            rid = record_id(m["job"], m["env_key"], m["sample"])
            r = reviews.get(rid)
            if r is None:
                counts["MISSING"] += 1
                missing.append(m)
                continue
            counts[r["verdict"]] += 1
            row = (rid, m["env_key"], m["sample"], str(m["score"]), r.get("reason", "").replace("\t", " ")[:400])
            if r["verdict"] == "KEEP":
                kj.write(dl)
                km.write(json.dumps({**m, "record_id": rid}, ensure_ascii=False) + "\n")
                n_keep_solved += m["score"] == 1
            elif r["verdict"] == "DROP":
                drops.append(row)
            elif r["verdict"] == "OVERLONG_UNJUDGED":
                overlong.append(row)
            else:
                errors.append((r["verdict"],) + row)

    def dump(name, rows):
        with open(args.out_prefix + f"-{name}.tsv", "w") as f:
            f.write("record_id\tenv_key\tsample\tscore\treason\n")
            for r in rows:
                f.write("\t".join(r) + "\n")

    dump("drops", drops)
    dump("overlong", overlong)
    with open(args.out_prefix + "-errors.tsv", "w") as f:
        f.write("verdict\trecord_id\tenv_key\tsample\tscore\treason\n")
        for r in errors:
            f.write("\t".join(r) + "\n")
    judged = counts["KEEP"] + counts["DROP"]
    summary = {"counts": dict(counts), "keep_rate_among_judged": round(counts["KEEP"] / judged, 4) if judged else None,
               "keep_solved": n_keep_solved, "missing": len(missing)}
    json.dump(summary, open(args.out_prefix + "-summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()