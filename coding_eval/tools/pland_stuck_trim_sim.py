#!/usr/bin/env python3
"""Plan-D simulation v2 on dsv41_final.

Rule D' (adaptation of Terminal-Universe "trim trailing failed rounds"):
  stuck turn = repeated command batch OR error observation OR empty/wait command OR env-fix command
  onset       = first start of a run of >= L consecutive stuck turns
  cut         = onset - grace  (keep the diagnosis chain that led into it)
  keep        = [0, cut) ; mark trajectory as "honest stop at unresolved obstacle"
Signals computed per trajectory, plus segment-level impact (segments fully inside the cut region).
"""
import json
import re
import statistics as st

FIN = "/storage/openpsi/users/chen/code_sft_data/dsv41_final/dsv41_final.jsonl"
MAN = "/storage/openpsi/users/chen/code_sft_data/dsv41_final/dsv41_final.manifest.jsonl"
ENVFIX = re.compile(r"(sources\.list|/etc/apt/|update-alternatives|dpkg --|dpkg -i|\.deb|apt-get install|"
                    r"apt-get -f|apt --fix|ldconfig|CC=)")


def split(content):
    m = re.match(r"\s*<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)(.*)", content, re.S)
    if m:
        return m.group(1), m.group(2)
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


def norm(c):
    return re.sub(r"\s+", " ", c.strip())[:200]


ERR = re.compile(r"(Traceback \(most recent|[A-Za-z_.]*Error:|error:|ERROR|command not found|No such file|"
                 r"Permission denied|Killed|E: |Unable to locate|Cannot open|failed)", re.M)
HANDOFF = "answers the other agent"


def analyze(L=20, grace=3):
    mf = open(MAN)
    out = []
    for line in open(FIN):
        meta = json.loads(mf.readline())
        row = json.loads(line)
        msgs = row["conversations"]
        turns, obs, seg = [], [], 0
        for i, m in enumerate(msgs):
            if m.get("role") == "user":
                if HANDOFF in m["content"][:200]:
                    seg += 1
                continue
            th, rest = split(m["content"])
            cs, tc = action(rest)
            nxt = msgs[i + 1]["content"] if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user" else ""
            turns.append(dict(think=len(th), cmds=[norm(c) for c in cs if c.strip()], tc=tc,
                              seg=seg, obs=nxt[-1200:]))
            obs.append(nxt)
        n = len(turns)
        seen, stuck, rpt = set(), [], []
        for t in turns:
            key = tuple(sorted(t["cmds"]))
            repeated = bool(key) and key in seen
            err = bool(ERR.search(t["obs"]))
            wait = not t["cmds"]
            envfix = any(ENVFIX.search(c) for c in t["cmds"])
            stuck.append(repeated or err or wait or envfix)
            rpt.append(repeated)
            if key:
                seen.add(key)
        # first onset of >=L consecutive stuck turns
        onset, run = -1, 0
        for i, s in enumerate(stuck):
            run = run + 1 if s else 0
            if run >= L:
                onset = i - L + 1
                break
        cut = max(1, onset - grace) if onset >= 0 else n
        out.append(dict(n=n, score=meta.get("score"), nseg=meta.get("n_segments") or 1,
                        onset=onset, cut=cut, segs=[t["seg"] for t in turns],
                        think=[t["think"] for t in turns],
                        stuck=sum(stuck) / max(1, n), repeat=sum(rpt) / max(1, n),
                        tc_end=bool(turns and turns[-1]["tc"]), chars=meta.get("n_chars") or 0,
                        decl=any(t["tc"] for t in turns)))
    return out


def report(rows, L):
    print(f"\n### rule D' with L={L}")
    for sc in (0, 1):
        rs = [r for r in rows if r["score"] == sc]
        cut_rs = [r for r in rs if r["cut"] < r["n"]]
        if not rs:
            continue
        print(f"== score={sc}  n={len(rs)}")
        print(f"  truncated: {len(cut_rs)} ({len(cut_rs)/len(rs):.0%}); median turns cut {st.median([r['n']-r['cut'] for r in cut_rs]):.0f}"
              f" / kept {st.median([r['cut'] for r in cut_rs]):.0f}")
        tt = sum(sum(r["think"][r["cut"]:]) for r in cut_rs)
        tp = sum(sum(r["think"]) for r in rs)
        print(f"  think chars removed {tt/1e6:.1f}M = {tt/max(1,tp):.1%} of this group's think; "
              f"turns removed {sum(r['n']-r['cut'] for r in cut_rs)/max(1,sum(r['n'] for r in rs)):.1%}")
        print(f"  stuck-turn share before trim {st.mean(r['stuck'] for r in rs):.1%} -> after "
              f"{sum(r['stuck']*r['n'] - 0.0 for r in rs)/max(1,sum(r['n'] for r in rs)):.1%} (measure below)")
        # post-trim think mean
        pre = [x for r in rs for x in r["think"]]
        post = [x for r in rs for x in (r["think"][:r["cut"]] if r["cut"] < r["n"] else r["think"])]
        print(f"  think mean {st.mean(pre):.0f} -> {st.mean(post):.0f}; p50 {st.median(pre):.0f} -> {st.median(post):.0f}; "
              f"share >20k {sum(1 for x in pre if x>20000)/len(pre):.1%} -> {sum(1 for x in post if x>20000)/len(post):.1%}")
        print(f"  trajectories kept with <5 turns: {sum(1 for r in cut_rs if r['cut']<5)}")
        # segment impact
        seg_total = sum(r["nseg"] for r in rs)
        seg_cut = sum(len({s for s in r["segs"][r["cut"]:]}) for r in cut_rs)
        seg_partial = sum(len({s for s in r["segs"][:r["cut"]]} & {s for s in r["segs"][r["cut"]:]}) for r in cut_rs)

        print(f"  segments: total {seg_total}; fully inside cut region {seg_cut} ({seg_cut/max(1,seg_total):.0%} of group); partially cut {seg_partial}")


if __name__ == "__main__":
    for L in (10, 20, 40):
        rows = analyze(L=L, grace=3)
        report(rows, L)
        if L == 20:
            rs = rows
            print("\n   -- false-positive check on solved:")
            s = [r for r in rs if r["score"] == 1]
            bad = [r for r in s if r["cut"] < r["n"]]
            print(f"   solved truncated {len(bad)}/{len(s)} ({len(bad)/len(s):.0%}); "
                  f"of them median kept {st.median([r['cut'] for r in bad]):.0f} of {st.median([r['n'] for r in bad]):.0f} turns")
            print("\n   -- distribution of stuck-run onset for failures (score=0):")
            f0 = [r for r in rs if r["score"] == 0 and r["onset"] >= 0]
            print(f"   failures with an onset: {len(f0)}/{sum(1 for r in rs if r['score']==0)}; "
                  f"onset position p50 {st.median([r['onset'] for r in f0]):.0f} of p50 n {st.median([r['n'] for r in rs if r['score']==0]):.0f}")