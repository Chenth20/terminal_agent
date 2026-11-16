#!/usr/bin/env python3
"""High-precision audit of environment-caused defects inside the dsv41_history SFT set.

The dataset keeps only DONE trajectories, but the source job ran harness 1.4.96: the agent shell
had NO pypi/HF mirrors (added 1.5/1.6.96), only 8Gi memory (16Gi from 1.6.96), and ARCA's network
policy denies connections into sandbox-created private networks. A DONE trajectory can therefore
carry environment faults that are harmful SFT signal.

Signals (user turns = terminal output, unless noted):
  pip-net    package-index host + network exception in one window, or pip's own
             "Could not install packages due to an OSError" banner
  hf-net     huggingface.co unreachable / xet CAS failure
  git-net    git/curl to github failing to connect / resolve
  oom        a bare "Killed" line not explained by the model's own `timeout -s KILL`/`kill -9`
             in that step, or a kernel oom-kill line (8Gi cap in harness 1.4.96)
  eperm-net  "connect(): operation not permitted"-family: ARCA policy denying traffic into
             private bridges/netns the task created (kind, k3s CNI, iperf veth). NOT fixable
             by a rerun - listed separately.
  gave-up    assistant abandons runtime verification citing network; counted as env fault
             ONLY when the same trajectory also has pip/hf/git-net evidence (otherwise it is
             model speculation or task-designed offline setup).

Known-noise exclusions, each verified against samples: Go runtime string tables, kernel/JBD2
source code shown in editors, quota tests whose *expected* output is ENOSPC/EDQUOT,
`timeout -s KILL` kills, std::bad_alloc in model-authored C++.

Alignment: <base>.manifest.jsonl is line-aligned with <base>.jsonl.
Outputs: /tmp/opencode/sft_env_pollution.json (segments), _traj.json (trajectories), printed table.
"""
import argparse, json, re
from collections import Counter, defaultdict

BASE = "/storage/openpsi/users/chen/code_sft_data/dsv41_history/TerminalHorizon-dsv41-96k-tool-SFT"
IDX = r"(?:files\.pythonhosted\.org|pypi\.org|registry\.npmjs\.org|crates\.io|static\.crates\.io)"
NETEX = (r"(?:ReadTimeoutError|NewConnectionError|ConnectTimeoutError|Max retries exceeded"
         r"|Temporary failure in name resolution|Could not fetch URL|SSLError|ProxyError"
         r"|ConnectionResetError|Connection refused|Read timed out)")

PAT = {
    "pip-net": [
        re.compile(NETEX + r"[\s\S]{0,300}" + IDX, re.I),
        re.compile(IDX + r"[\s\S]{0,300}" + NETEX, re.I),
        re.compile(r"ERROR: Could not install packages due to an (?:OSError|EnvironmentError)", re.I),
    ],
    "hf-net": [
        re.compile(r"couldn'?t (?:connect|reach) (?:to )?'?https://huggingface\.co", re.I),
        re.compile(r"cas-server\.xethub\.hf\.co", re.I),
        re.compile(r"huggingface\.co[\s\S]{0,200}(?:timed out|Max retries|Connection refused)", re.I),
    ],
    "git-net": [
        re.compile(r"fatal: unable to access 'https://github\.com|Could not resolve host: github\.com"
                   r"|Failed to connect to github\.com", re.I),
        re.compile(r"raw\.githubusercontent\.com[\s\S]{0,200}(?:timed out|Failed to connect|Could not resolve)", re.I),
    ],
    "eperm-net": [
        re.compile(r"(?:connect|dial tcp [\d.:]+|sendto|connect to [\w./:-]+)"
                   r"[\s\S]{0,80}?[Oo]peration not permitted"),
        re.compile(r"[Oo]peration not permitted[\s\S]{0,60}(?:connect|dial)", re.I),
    ],
}
_TK = r"timeout\s+(?:[^\n|;&]{0,40}?\s)?(?:-s\s*(?:SIG)?KILL|--signal[= ](?:SIG)?KILL|-k\s*\S+|--kill-after[= ]\S+)"
TIMEOUT_KILL = re.compile(_TK, re.I)
# the model itself killing a process explains a "Killed" line in the next observation
MODEL_KILL = re.compile(r"(?:kill|pkill|killall)\s+(?:-9|-KILL|-SIGKILL|-s\s*(?:SIG)?KILL)|" + _TK, re.I)
OOM_BARE = re.compile(r"^\s*Killed\s*$|^(?!.*timeout).{0,200}:\s+Killed\s*$", re.M)
OOM_KERNEL = re.compile(r"oom-kill:constraint=|Out of memory: Killed process|invoked oom-killer", re.I)
GAVE_UP = re.compile(
    r"(?:no|without|unable to access the|can'?t access the|no access to the)\s+(?:internet|network|external network)"
    r"|network (?:is )?(?:unavailable|restricted|not available|blocked)"
    r"|give up on (?:pip|install|download)"
    r"|rely(?:ing)? on (?:syntax|static|code review) (?:instead|rather)", re.I)


def detect_oom(text, prev_assistant):
    """Bare 'Killed' from the shell = SIGKILL of the agent's own foreground command. Count it only
    when neither the observation nor the preceding assistant turn shows the model killing on
    purpose (timeout -s KILL / kill -9); kernel oom-kill lines always count. 'Cannot allocate
    memory' is NOT used: in this corpus it comes from Go runtime string tables, source code and
    tc-netem socket-buffer errors (task semantics), not from the 8Gi cap."""
    n = len(OOM_KERNEL.findall(text))
    if OOM_BARE.search(text) and not TIMEOUT_KILL.search(text) and not MODEL_KILL.search(prev_assistant or ""):
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE, help="<base>.jsonl + line-aligned <base>.manifest.jsonl")
    ap.add_argument("--out", default="/tmp/opencode/sft_env_pollution", help="output prefix")
    args = ap.parse_args()
    base, outp = args.base, args.out
    seg = []
    traj = defaultdict(lambda: {"s": Counter(), "gave": 0, "score": None, "status": None})
    with open(base + ".manifest.jsonl") as mf, open(base + ".jsonl") as df:
        for i, (ml, dl) in enumerate(zip(mf, df)):
            m = json.loads(ml)
            d = json.loads(dl)
            sig, gave = Counter(), 0
            prev_assistant = ""
            for msg in d["conversations"]:
                c = msg["content"]
                if msg["role"] == "user":
                    for k, pats in PAT.items():
                        n = sum(len(p.findall(c)) for p in pats)
                        if n:
                            sig[k] += n
                    n = detect_oom(c, prev_assistant)
                    if n:
                        sig["oom"] += n
                else:
                    prev_assistant = c
                    if GAVE_UP.search(c):
                        gave += 1
            if sig or gave:
                seg.append({"line": i, "env_key": m["env_key"], "sample": m["sample"],
                            "segment": m["segment"], "score": m["score"],
                            "signals": dict(sig), "gave_up_turns": gave})
                t = traj[(m.get("job", ""), m["env_key"], m["sample"])]
                t["s"].update(sig)
                t["gave"] += gave
                t["score"] = m["score"]

    # gave-up counts as env fault only with network evidence in the same trajectory
    NET = ("pip-net", "hf-net", "git-net")
    out = {}
    for (j, e, s), t in traj.items():
        sigs = dict(t["s"])
        if t["gave"] and any(k in sigs for k in NET):
            sigs["gave-up+net"] = t["gave"]
        if sigs:
            # key = job|env|sample (job may be empty for manifests without it); consumers split on "|"
            out[f"{j}|{e}|{s}"] = {"signals": sigs, "score": t["score"]}
    json.dump(seg, open(outp + ".json", "w"), indent=1)
    json.dump(out, open(outp + "_traj.json", "w"), indent=1)

    print(f"rows flagged (any raw signal): {len(seg)}")
    print(f"trajectories with env-fault signal: {len(out)}")
    c_all, c_bad, envs = Counter(), Counter(), defaultdict(set)
    for key, t in out.items():
        e = key.split("|")[-2]
        for k in t["signals"]:
            c_all[k] += 1
            envs[k].add(e)
            if t["score"] != 1:
                c_bad[k] += 1
    print(f"{'signal':12s} {'traj':>5s} {'score=0':>8s} {'score=1':>8s} {'envs':>5s}")
    for k, v in c_all.most_common():
        print(f"{k:12s} {v:5d} {c_bad[k]:8d} {v - c_bad[k]:8d} {len(envs[k]):5d}")


if __name__ == "__main__":
    main()