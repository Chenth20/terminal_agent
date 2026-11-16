#!/usr/bin/env python3
"""Simulate "plan D" (trim trailing dead suffix, keep failure->recovery chain) on dsv41_final.

A trailing dead suffix is defined from the last PRODUCTIVE turn:
  productive = turn issuing (a) a write into /app (cat >, tee, sed -i, heredoc, cp/mv into /app,
               python - << etc writing a file), or (b) a task-specific verification command
               (pytest/test_outputs/check.py/CLI verifier/diff/md5/sha/gmx check/... ), or (c) task_complete.
Everything after the last productive turn is the dead suffix. We report how much it is, what it
contains, and what the dataset looks like if trimmed (keeping a small grace window).
"""
import json
import re
import statistics as st

FIN = "/storage/openpsi/users/chen/code_sft_data/dsv41_final/dsv41_final.jsonl"
MAN = "/storage/openpsi/users/chen/code_sft_data/dsv41_final/dsv41_final.manifest.jsonl"

WRITE = re.compile(r"(cat\s*>\s*/app|tee\s+/app|sed -i|>>?\s*/app/|cp .*/app/|mv .*/app/|"
                   r"python3? - <<|cat > )")
TEST = re.compile(r"(pytest|test_outputs|check\.py|/tests/|md5sum|sha256sum|diff |"
                  r"gmx (check|energy|traj)|verify|validate|clients? |ctl )")
THINK_OPEN = chr(96) * 3 + "think"
BT = chr(96) * 3


def split(content):
    """Return (think, action_text). Handles  thinking, <thinking>, and backtick-think markers."""
    m = re.match(r"\s*<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)(.*)", content, re.S)
    if m:
        return m.group(1), m.group(2)
    tag = chr(96) * 3
    i = content.find(tag + "think")
    if i != -1:
        j = content.find(tag, i + len(tag) + 5)
        if j != -1:
            return content[i + len(tag) + 5:j], content[j + len(tag):]
        return content[i + len(tag) + 5:], ""
    return "", content


def action(rest):
    s = rest.strip()
    i = s.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(s[i:j + 1])
                    except Exception:
                        o = None
                    if isinstance(o, dict) and "commands" in o:
                        return [c.get("keystrokes", "") or "" for c in (o.get("commands") or [])
                                if isinstance(c, dict)], bool(o.get("task_complete"))
                    break
        i = s.find("{", i + 1)
    return [], False


def main():
    mf = open(MAN)
    rows = []
    for line in open(FIN):
        meta = json.loads(mf.readline())
        row = json.loads(line)
        turns = [m["content"] for m in row["conversations"] if m.get("role") == "assistant"]
        n = len(turns)
        prod, decl, think, cms = [], False, [], []
        for t in turns:
            th, rest = split(t)
            cs, tc = action(rest)
            decl = decl or tc
            think.append(len(th))
            joined = "\n".join(cs)
            p = bool(WRITE.search(joined)) or bool(TEST.search(joined))
            prod.append(p)  # task_complete does NOT count as progress (false declarations are the norm in failures)
        last = max((i for i, p in enumerate(prod) if p), default=-1)
        tail = n - 1 - last if last >= 0 else n
        tail_think = think[last + 1:] if last + 1 <= n else []
        pre_think = think[:last + 1]
        # declaration position: last turn with task_complete
        rows.append(dict(
            n=n, score=meta.get("score"), env=meta.get("env_key"), decl=decl,
            last_prod=last, tail=tail,
            tail_chars=sum(tail_think), pre_chars=sum(pre_think),
            tail_think_mean=(st.mean(tail_think) if tail_think else 0),
            tail_think_p50=(st.median(tail_think) if tail_think else 0),
            pre_think_mean=(st.mean(pre_think) if pre_think else 0),
            tail_gt20k=sum(1 for x in tail_think if x > 20000),
            tail_wait_only=sum(1 for x in tail_think if x < 100),
            chars=meta.get("n_chars") or 0,
        ))
    print(f"trajectories {len(rows)}")
    for sc in (1, 0):
        rs = [r for r in rows if r["score"] == sc]
        print(f"\n== score={sc}: n={len(rs)}")
        print(f"  turns p50 {st.median(r['n'] for r in rs):.0f}; with dead tail>0: "
              f"{sum(1 for r in rs if r['tail'] > 0) / len(rs):.0%}; tail turns: mean {st.mean(r['tail'] for r in rs):.1f}"
              f" p50 {st.median(r['tail'] for r in rs):.0f} p90 {sorted(r['tail'] for r in rs)[int(.9 * (len(rs) - 1))]}")
        print(f"  tail >= 20 turns: {sum(1 for r in rs if r['tail'] >= 20) / len(rs):.0%}; "
              f"tail >= 50: {sum(1 for r in rs if r['tail'] >= 50) / len(rs):.0%}")
        tot_t = sum(r['tail_chars'] for r in rs)
        tot_c = sum(r['pre_chars'] + r['tail_chars'] for r in rs)
        print(f"  assistant think chars in tail: {tot_t:,} = {tot_t / max(1, tot_c):.1%} of assistant think chars")
        th = [r['tail_think_mean'] for r in rs if r['tail_chars']]
        if th:
            print(f"  tail think mean {st.mean(th):.0f} vs prefix think mean {st.mean(r['pre_think_mean'] for r in rs):.0f}")
        print(f"  tails >20k-think turns total {sum(r['tail_gt20k'] for r in rs)}; "
              f"tails made only of <100-char thinks (poll tails) {sum(1 for r in rs if r['tail'] and r['tail_wait_only'] == r['tail'])}")
        print(f"  trajectories would fall below 5 turns after trim: "
              f"{sum(1 for r in rs if 0 <= r['last_prod'] + 1 < 5) / len(rs):.1%}")
        print(f"  trajectories that declared task_complete (whole traj): {sum(1 for r in rs if r['decl']) / len(rs):.0%}")
    # after-trim distribution
    keep = [r for r in rows if r["last_prod"] >= 0]
    new_turns = [max(1, r["last_prod"] + 1) for r in keep]
    print(f"\n== after trim (keep through last productive turn): trajectories {len(keep)}/{len(rows)}")
    print(f"  turns: mean {st.mean(new_turns):.1f} p50 {st.median(new_turns):.0f} p90 {sorted(new_turns)[int(.9*(len(new_turns)-1))]}"
          f"  (now mean {st.mean(r['n'] for r in rows):.1f} p50 {st.median(r['n'] for r in rows):.0f})")
    print(f"  assistant think chars total: {sum(r['pre_chars'] for r in keep):,} vs {sum(r['pre_chars']+r['tail_chars'] for r in keep):,} "
          f"(-{sum(r['tail_chars'] for r in keep)/max(1,sum(r['pre_chars']+r['tail_chars'] for r in keep)):.1%})")
    print(f"  think mean over all turns: {st.mean(r['pre_think_mean'] for r in keep):.0f} vs {st.mean((r['pre_think_mean']*max(1,r['last_prod']+1)+r['tail_think_mean']*r['tail'])/max(1,r['n']) for r in keep):.0f}")
    # solved trajectories hurt?
    s1 = [r for r in rows if r["score"] == 1]
    print(f"\n  solved trajectories with tail>=20 turns (their final verification/declaration is far from last write/test): "
          f"{sum(1 for r in s1 if r['tail'] >= 20)} ({sum(1 for r in s1 if r['tail'] >= 20)/len(s1):.0%} of solved)")
    print(f"  solved with tail>=50: {sum(1 for r in s1 if r['tail'] >= 50)}")
    # distribution of tail sizes for score 0
    s0 = [r for r in rows if r["score"] == 0]
    buckets = [(0, 0), (1, 9), (10, 29), (30, 59), (60, 119), (120, 10 ** 9)]
    for a, b in buckets:
        c = sum(1 for r in s0 if a <= r["tail"] <= b)
        print(f"  score=0 tail in [{a},{b if b < 10**9 else 'inf'}]: {c} ({c/len(s0):.0%})")


if __name__ == "__main__":
    main()