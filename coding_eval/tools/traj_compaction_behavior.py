#!/usr/bin/env python3
"""Per-trajectory behaviour distribution + compaction-window behaviour for TB2.1 eval jobs.

Reads output/<job>/tasks/*/s*/{trajectory.jsonl,task.json}; prints two tables per job:
  1. behaviour distribution (steps, think, cmds, sleep, clear, waits, tests, declare)
  2. compaction analysis: pre/post window behaviour, re-exploration, handoff quality,
     and outcome split by number of compactions.
"""
import ast
import json
import re
import statistics as st
import sys
from pathlib import Path

OUT = Path("/storage/openpsi/users/chen/output")
JOBS = {
    "th3k c312": "cth-tb21m-mask-const-c312-2611051616",
    "dsv41 armA c224": "cth-tb21m-dsv41j-a-c224-2611051554",
    "dsvh const c336": "cth-tb21m-dsvh-rand-c336-2611060357",
    "qf solved c105": "cth-tb21-96-abl-qf-solved-c105-2610301043-2610300243",
    "qf const c129": "cth-tb21-96-abl-qf-const-c129-2611010046-2610311646",
}
EXPLORE = re.compile(r"^\s*(ls|cat|head|tail|sed -n|nl|grep|find|wc|tree|pwd|file|stat)\b")
TEST = re.compile(r"(pytest|test_outputs|check\.py|/tests/|python3? -m unittest|npm test|cargo test|make test|go test)")
WRITE = re.compile(r"(cat >|<<\s*['\"]?EOF|tee |sed -i|> /app|>> /app|python3? - <<)")


def p(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else 0


def parse_tool_calls(tc):
    if tc is None:
        return []
    if isinstance(tc, str):
        try:
            tc = ast.literal_eval(tc)
        except Exception:
            try:
                tc = json.loads(tc)
            except Exception:
                return []
    out = []
    for c in tc or []:
        a = c.get("arguments", {}) if isinstance(c, dict) else {}
        if c.get("function_name") == "bash_command":
            out.append((a.get("keystrokes", "") or "", a.get("duration", 1.0) or 0))
        elif c.get("function_name") in ("task_complete", "mark_task_complete"):
            out.append(("__TASK_COMPLETE__", 0))
    return out


def obs_text(r):
    o = r.get("observation")
    if isinstance(o, str):
        try:
            o = ast.literal_eval(o)
        except Exception:
            return o
    try:
        return "\n".join(x.get("content", "") for x in o["results"])
    except Exception:
        return str(o)


def scan(path):
    recs = [json.loads(l) for l in open(path)]
    task = json.load(open(path.parent / "task.json"))
    steps = []  # list of dicts for agent steps, in order with segment index
    seg = 0
    handoffs = []
    for r in recs:
        if r.get("source") == "system" and "compaction" in str(r.get("extra")):
            seg += 1
            continue
        if r.get("source") == "user" and seg > 0 and r.get("message", "").startswith("Here are the answers"):
            handoffs.append(len(r["message"]))
            continue
        if r.get("source") != "agent":
            continue
        think = r.get("reasoning_content") or ""
        cmds = parse_tool_calls(r.get("tool_calls"))
        keys = [k for k, _ in cmds if k != "__TASK_COMPLETE__"]
        ne = [k for k in keys if k.strip()]
        sleep = sum(int(m.group(1)) for k in keys for m in re.finditer(r"\bsleep\s+(\d+)", k))
        decl = any(k == "__TASK_COMPLETE__" for k, _ in cmds) or bool(re.search(r'"task_complete"\s*:\s*true', r.get("message", "")))
        obs = obs_text(r)
        steps.append(dict(
            seg=seg, think=len(think), ncmd=len(ne), wait=(len(ne) == 0),
            sleep=sleep, long=sum(1 for _, d in cmds if isinstance(d, (int, float)) and d >= 30),
            clear=sum(1 for k in keys if k.strip().startswith("clear")),
            explore=sum(1 for k in ne if EXPLORE.match(k)),
            test=sum(1 for k in ne if TEST.search(k)),
            write=sum(1 for k in ne if WRITE.search(k)),
            cc=sum(1 for k in keys if k.strip() == "C-c"),
            decl=decl, obs_err=bool(re.search(r"(Error|error:|Traceback|not found|No such file)", obs[-800:])),
            compl=(r.get("metrics") or {}).get("completion_tokens", 0) if isinstance(r.get("metrics"), dict) else 0,
        ))
    return dict(env=task["env_key"], s=task["sample_index"], status=task["status"], score=task.get("score") or 0,
                agent_min=task["timings"]["agent_ms"] / 60000, ncomp=seg, handoffs=handoffs, steps=steps)


def behaviour_table(trajs):
    rows = []
    steps_all = [s for t in trajs for s in t["steps"]]
    n = len(trajs)
    tot_steps = [len(t["steps"]) for t in trajs]
    passed = sum(1 for t in trajs if t["score"] == 1)
    ttl = sum(1 for t in trajs if t["agent_min"] >= 235)
    declared = [t for t in trajs if any(s["decl"] for s in t["steps"])]
    decl_pass = sum(1 for t in declared if t["score"] == 1)
    th = [s["think"] for s in steps_all]
    rows.append(("trajectories / pass@all / TTL(>=235min)", f"{n} / {passed/n:.3f} / {ttl} ({ttl/n:.0%})"))
    rows.append(("declared task_complete / pass after declare", f"{len(declared)} ({len(declared)/n:.0%}) / {decl_pass/max(1,len(declared)):.2f}"))
    rows.append(("steps p50 / p90 / mean", f"{p(tot_steps,.5)} / {p(tot_steps,.9)} / {st.mean(tot_steps):.0f}"))
    rows.append(("agent min p50 / mean", f"{p([t['agent_min'] for t in trajs],.5):.0f} / {st.mean(t['agent_min'] for t in trajs):.0f}"))
    rows.append(("think chars p50 / p90 / mean", f"{p(th,.5)} / {p(th,.9)} / {st.mean(th):.0f}"))
    rows.append(("think <100 / >20k / >50k (step %)", f"{sum(1 for x in th if x<100)/len(th):.1%} / {sum(1 for x in th if x>20000)/len(th):.1%} / {sum(1 for x in th if x>50000)/len(th):.1%}"))
    rows.append(("cmds per step (non-wait)", f"{st.mean(s['ncmd'] for s in steps_all if not s['wait']):.2f}"))
    rows.append(("wait steps (empty keystrokes) %", f"{sum(s['wait'] for s in steps_all)/len(steps_all):.1%}"))
    rows.append(("sleep sec / traj mean ; cmds >=30s / traj", f"{st.mean(sum(s['sleep'] for s in t['steps']) for t in trajs):.0f} ; {st.mean(sum(s['long'] for s in t['steps']) for t in trajs):.1f}"))
    rows.append(("clear / traj ; C-c / traj", f"{st.mean(sum(s['clear'] for s in t['steps']) for t in trajs):.1f} ; {st.mean(sum(s['cc'] for s in t['steps']) for t in trajs):.1f}"))
    rows.append(("explore cmd share ; write cmd share", f"{sum(s['explore'] for s in steps_all)/max(1,sum(s['ncmd'] for s in steps_all)):.1%} ; {sum(s['write'] for s in steps_all)/max(1,sum(s['ncmd'] for s in steps_all)):.1%}"))
    rows.append(("test cmds / traj ; traj with >=1 test", f"{st.mean(sum(s['test'] for s in t['steps']) for t in trajs):.1f} ; {sum(1 for t in trajs if any(s['test'] for s in t['steps']))/n:.0%}"))
    rows.append(("compactions / traj mean ; traj with >=1", f"{st.mean(t['ncomp'] for t in trajs):.2f} ; {sum(1 for t in trajs if t['ncomp'])/n:.0%}"))
    rows.append(("completion tok / step p50", f"{p([s['compl'] for s in steps_all if s['compl']],.5)}"))
    return rows


def compaction_table(trajs):
    rows = []
    n = len(trajs)
    by_k = {}
    for t in trajs:
        k = min(t["ncomp"], 4)
        by_k.setdefault(k, []).append(t)
    for k in sorted(by_k):
        ts = by_k[k]
        rows.append((f"ncomp={'>=4' if k==4 else k}: n / pass / TTL / declared",
                     f"{len(ts)} / {sum(t['score']==1 for t in ts)/len(ts):.2f} / {sum(t['agent_min']>=235 for t in ts)} / {sum(any(s['decl'] for s in t['steps']) for t in ts)/len(ts):.0%}"))
    # window behaviour: 8 steps before vs 8 steps after each compaction
    pre, post, first = [], [], []
    rexp, think_pre, think_post, wait_pre, wait_post, cmd_pre, cmd_post = [], [], [], [], [], [], []
    for t in trajs:
        steps = t["steps"]
        for seg in range(1, t["ncomp"] + 1):
            idx = [i for i, s in enumerate(steps) if s["seg"] == seg]
            if not idx:
                continue
            a = idx[0]
            w_pre = steps[max(0, a - 8):a]
            w_post = steps[a:a + 8]
            if not w_pre or not w_post:
                continue
            def share(w):
                c = sum(s["ncmd"] for s in w)
                return sum(s["explore"] for s in w) / c if c else 0
            rexp.append((share(w_pre), share(w_post)))
            think_pre.append(st.mean(s["think"] for s in w_pre)); think_post.append(st.mean(s["think"] for s in w_post))
            wait_pre.append(st.mean(s["wait"] for s in w_pre)); wait_post.append(st.mean(s["wait"] for s in w_post))
            cmd_pre.append(st.mean(s["ncmd"] for s in w_pre)); cmd_post.append(st.mean(s["ncmd"] for s in w_post))
            first.append(steps[a]["think"])
    if rexp:
        rows.append(("compaction windows (8 steps pre / post)", f"{len(rexp)}"))
        rows.append(("explore cmd share pre -> post", f"{st.mean(x for x,_ in rexp):.2f} -> {st.mean(y for _,y in rexp):.2f}"))
        rows.append(("think mean pre -> post ; first step after", f"{st.mean(think_pre):.0f} -> {st.mean(think_post):.0f} ; {p(first,.5)}"))
        rows.append(("wait share pre -> post", f"{st.mean(wait_pre):.2f} -> {st.mean(wait_post):.2f}"))
        rows.append(("cmds/step pre -> post", f"{st.mean(cmd_pre):.2f} -> {st.mean(cmd_post):.2f}"))
    hl = [h for t in trajs for h in t["handoffs"]]
    if hl:
        rows.append(("handoff msg chars p50 / p90", f"{p(hl,.5)} / {p(hl,.9)}"))
    # steps per segment / segment-level pass structure
    seglen = [sum(1 for s in t["steps"] if s["seg"] == 0) for t in trajs if t["ncomp"]]
    if seglen:
        rows.append(("steps before 1st compaction p50 (for ncomp>=1)", f"{p(seglen,.5)}"))
    # declared inside a post-compaction segment
    decl_after = sum(1 for t in trajs if t["ncomp"] and any(s["decl"] and s["seg"] > 0 for s in t["steps"]))
    rows.append(("ncomp>=1 trajs that still declared", f"{decl_after} / {sum(1 for t in trajs if t['ncomp'])}"))
    return rows


def main():
    jobs = JOBS if len(sys.argv) == 1 else {a: a for a in sys.argv[1:]}
    results = {}
    for name, job in jobs.items():
        trajs = []
        for tj in sorted((OUT / job / "tasks").glob("*/s*/trajectory.jsonl")):
            try:
                t = scan(tj)
            except Exception as e:  # noqa: BLE001
                print(f"skip {tj}: {e}", file=sys.stderr)
                continue
            if t["steps"]:
                trajs.append(t)
        results[name] = trajs
        print(f"\n### {name}  ({job})")
        for k, v in behaviour_table(trajs):
            print(f"| {k} | {v} |")
        print()
        for k, v in compaction_table(trajs):
            print(f"| {k} | {v} |")
    # paired env comparison across jobs on compaction
    names = list(results)
    if len(names) >= 2:
        a, b = names[0], names[1]
        ka = {(t["env"], t["s"]): t for t in results[a]}
        kb = {(t["env"], t["s"]): t for t in results[b]}
        common = sorted(set(ka) & set(kb))
        print(f"\n### paired {a} vs {b}: {len(common)} common (env,sample)")
        tab = {}
        for key in common:
            x, y = ka[key], kb[key]
            tab.setdefault((x["score"], y["score"]), []).append((x["ncomp"], y["ncomp"]))
        for (sx, sy), v in sorted(tab.items()):
            print(f"| score {sx}->{sy} | n={len(v)} | ncomp mean {st.mean(i for i,_ in v):.2f} -> {st.mean(j for _,j in v):.2f} |")
    json.dump({k: [{kk: vv for kk, vv in t.items() if kk != 'steps'} | {'nsteps': len(t['steps'])} for t in v] for k, v in results.items()},
              open("/tmp/opencode/traj_compaction_behavior.json", "w"))


if __name__ == "__main__" and "--ext" not in sys.argv and "--tbl" not in sys.argv:
    main()


# ---- extended compaction diagnostics (progress loss / handoff adherence) ----
def cmd_norm(k):
    return re.sub(r"\s+", " ", k.strip())[:80]


def scan_ext(path):
    recs = [json.loads(l) for l in open(path)]
    task = json.load(open(path.parent / "task.json"))
    seg, out, handoff_cmds, prev_cmds, written = 0, [], None, [], set()
    results = dict(env=task["env_key"], s=task["sample_index"], score=task.get("score") or 0,
                   agent_min=task["timings"]["agent_ms"] / 60000, comps=[])
    cur = None
    for r in recs:
        if r.get("source") == "system" and "compaction" in str(r.get("extra")):
            seg += 1
            cur = dict(seg=seg, step_at=len(out), pre_written=set(written), post_steps=0, redo_writes=0,
                       dup_cmds=0, post_cmds=0, handoff_follow=None, post_writes=0, first_cmds=None)
            results["comps"].append(cur)
            continue
        if r.get("source") == "user" and cur and cur["handoff_follow"] is None and r.get("message", "").startswith("Here are the answers"):
            m = re.search(r'"commands"\s*:\s*\[(.*?)\]\s*,?\s*"?task_complete"?|"commands"\s*:\s*\[(.*)\]', r["message"], re.S)
            hc = set()
            for mm in re.finditer(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', r["message"]):
                hc.add(cmd_norm(mm.group(1).encode().decode("unicode_escape", "ignore")))
            handoff_cmds = hc
            continue
        if r.get("source") != "agent":
            continue
        cmds = [cmd_norm(k) for k, _ in parse_tool_calls(r.get("tool_calls")) if k != "__TASK_COMPLETE__" and k.strip()]
        out.append(cmds)
        wr = set(m.group(1) for k in cmds for m in re.finditer(r"(?:cat >|tee|> )\s*(/app/\S+)", k))
        if cur and len(out) - cur["step_at"] <= 15:
            cur["post_steps"] += 1
            cur["post_cmds"] += len(cmds)
            cur["dup_cmds"] += sum(1 for c in cmds if c in prev_cmds)
            cur["redo_writes"] += len(wr & cur["pre_written"])
            cur["post_writes"] += len(wr)
            if cur["first_cmds"] is None and cmds:
                cur["first_cmds"] = cmds
                if handoff_cmds:
                    cur["handoff_follow"] = len(set(cmds) & handoff_cmds) / len(set(cmds))
        prev_cmds.extend(cmds)
        written |= wr
    for c in results["comps"]:
        c.pop("pre_written", None); c.pop("first_cmds", None)
    results["nsteps"] = len(out)
    return results


def ext_main():
    for name, job in JOBS.items():
        rows = []
        for tj in sorted((OUT / job / "tasks").glob("*/s*/trajectory.jsonl")):
            try:
                rows.append(scan_ext(tj))
            except Exception:
                continue
        comps = [c for t in rows for c in t["comps"] if c["post_steps"] >= 5]
        if not comps:
            continue
        hf = [c["handoff_follow"] for c in comps if c["handoff_follow"] is not None]
        print(f"\n### {name}: {len(comps)} compaction windows (15 steps post)")
        print(f"| first-step cmds overlapping handoff plan (mean Jaccard-ish) | {st.mean(hf):.2f} (n={len(hf)}) ; =0 share {sum(1 for x in hf if x==0)/len(hf):.0%} |")
        print(f"| post cmds that exactly repeat an earlier cmd | {sum(c['dup_cmds'] for c in comps)/sum(c['post_cmds'] for c in comps):.1%} |")
        print(f"| windows re-writing an /app file already written before compaction | {sum(1 for c in comps if c['redo_writes'])/len(comps):.0%} |")
        print(f"| windows with zero /app writes in 15 post steps | {sum(1 for c in comps if c['post_writes']==0)/len(comps):.0%} |")
        # by outcome
        for lab, f in (("pass", lambda t: t["score"] == 1), ("TTL", lambda t: t["agent_min"] >= 235)):
            cs = [c for t in rows if f(t) for c in t["comps"] if c["post_steps"] >= 5]
            if cs:
                print(f"| [{lab}] windows {len(cs)}: dup-cmd share / zero-write share | {sum(c['dup_cmds'] for c in cs)/max(1,sum(c['post_cmds'] for c in cs)):.1%} / {sum(1 for c in cs if c['post_writes']==0)/len(cs):.0%} |")


if __name__ == "__main__" and "--ext" in sys.argv and "--tbl" not in sys.argv:
    ext_main()


# ---- tb_local layout support ----
TBL = Path("/storage/openpsi/users/chen/tb_local/runs")
TBL_JOBS = {
    "dsv41_judged 4B (dsvj4b-local)": "dsvj4b-local",
    "dsv41_judged 9B (dsvj9b-local)": "dsvj9b-local",
    "th3k c312 local (mask-c312-96k-local)": "mask-c312-96k-local",
}


def scan_tbl(out_dir):
    res = json.load(open(out_dir.parent / "result.json"))
    recs = [json.loads(l) for l in open(out_dir / "trajectory.jsonl")]
    seg = 0
    steps = []
    handoffs = []
    for r in recs:
        if r.get("source") == "system" and "compaction" in str(r.get("extra")):
            seg += 1
            continue
        if r.get("source") == "user" and seg > 0 and r.get("message", "").startswith("Here are the answers"):
            handoffs.append(len(r["message"]))
            continue
        if r.get("source") != "agent":
            continue
        think = r.get("reasoning_content") or ""
        cmds = parse_tool_calls(r.get("tool_calls"))
        keys = [k for k, _ in cmds if k != "__TASK_COMPLETE__"]
        ne = [k for k in keys if k.strip()]
        sleep = sum(int(m.group(1)) for k in keys for m in re.finditer(r"\bsleep\s+(\d+)", k))
        decl = any(k == "__TASK_COMPLETE__" for k, _ in cmds) or bool(re.search(r'"task_complete"\s*:\s*true', r.get("message", "")))
        obs = obs_text(r)
        met = r.get("metrics") or {}
        steps.append(dict(seg=seg, think=len(think), ncmd=len(ne), wait=(len(ne) == 0), sleep=sleep,
                          long=sum(1 for _, d in cmds if isinstance(d, (int, float)) and d >= 30),
                          clear=sum(1 for k in keys if k.strip().startswith("clear")),
                          explore=sum(1 for k in ne if EXPLORE.match(k)), test=sum(1 for k in ne if TEST.search(k)),
                          write=sum(1 for k in ne if WRITE.search(k)), cc=sum(1 for k in keys if k.strip() == "C-c"),
                          decl=decl, obs_err=bool(re.search(r"(Error|error:|Traceback|not found|No such file)", obs[-800:])),
                          compl=met.get("completion_tokens", 0) if isinstance(met, dict) else 0))
    return dict(env=res["task"], s=res["sample"], status=res["harness_status"], score=res.get("score") or 0,
                agent_min=res["agent_sec"] / 60, ncomp=seg, handoffs=handoffs, steps=steps,
                cap512=(len(steps) >= 505), contract=res.get("contract", "v1"))


def tbl_main():
    results = {}
    for name, run in TBL_JOBS.items():
        trajs = []
        for tj in sorted((TBL / run).glob("*/s*/out/trajectory.jsonl")):
            try:
                t = scan_tbl(tj.parent)
            except Exception as e:  # noqa: BLE001
                print(f"skip {tj}: {e}", file=sys.stderr)
                continue
            if t["steps"]:
                trajs.append(t)
        results[name] = trajs
        print(f"\n### {name}  (tb_local {run}, n={len(trajs)})")
        cap = sum(1 for t in trajs if t["cap512"])
        print(f"| CAP512 (>=505 steps) | {cap} ({cap/len(trajs):.0%}) |")
        print(f"| contracts | {collections_counter(t['contract'] for t in trajs)} |")
        for k, v in behaviour_table(trajs):
            print(f"| {k} | {v} |")
        print()
        for k, v in compaction_table(trajs):
            print(f"| {k} | {v} |")
    names = list(results)
    a, b = names[0], names[2]
    ka = {(t["env"], t["s"]): t for t in results[a]}
    kb = {(t["env"], t["s"]): t for t in results[b]}
    common = sorted(set(ka) & set(kb))
    print(f"\n### paired {a} vs {b}: {len(common)}")
    tab = {}
    for key in common:
        x, y = ka[key], kb[key]
        tab.setdefault((x["score"], y["score"]), []).append((x["ncomp"], y["ncomp"]))
    for (sx, sy), v in sorted(tab.items()):
        print(f"| judged {sx} / th3k {sy} | n={len(v)} | ncomp {st.mean(i for i,_ in v):.2f} vs {st.mean(j for _,j in v):.2f} |")
    json.dump({k: [{kk: vv for kk, vv in t.items() if kk != 'steps'} | {'nsteps': len(t['steps'])} for t in v] for k, v in results.items()},
              open("/tmp/opencode/tbl_behavior.json", "w"))


def collections_counter(it):
    import collections
    return dict(collections.Counter(it))


if __name__ == "__main__" and "--tbl" in sys.argv:
    tbl_main()
