#!/usr/bin/env python3
"""Score SFT trajectory datasets with the TerminalHorizon-paper teacher criteria.

Criteria implemented (paper refs):
  - TH3K Table 3: mean turns, mean feedback chain, exact-repeat %, pass rate
  - TH3K App. F judge: (1) validity: bad-JSON / no-action turns, truncation-without-task_complete
    (2) environment-grounded progress: after an error observation, share of next turns that
        repeat the identical command batch (feedback ignored) vs revise
    (3) multi-turn training value: wait/poll turns, mechanical loops (>=3 identical batches),
        browse-only trajectories
  - Data Recipes Sec 3.6: min-turns filter (share of trajectories with <5 assistant turns)
"""
import json
import re
import sys

import statistics as st

def split_think(content):
    """Split assistant content into (think, action_text). Handles backtick-think tags."""
    bt = chr(96) * 3
    open_t, close_t = bt + "think", bt
    i = content.find(open_t)
    if i != -1:
        j = content.find(close_t, i + len(open_t))
        if j != -1:
            return content[i + len(open_t):j], content[j + len(close_t):]
        return content[i + len(open_t):], ""
    m = re.match(r"\s*<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)(.*)", content, re.S)
    if m:
        return m.group(1), m.group(2)
    return "", content


def parse_action(rest):
    """Return (list_of_keystrokes, task_complete, parsed_ok) for a Terminus JSON action."""
    s = rest.strip()
    i = s.find("{")
    while i != -1:
        depth, end = 0, None
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end:
            try:
                obj = json.loads(s[i:end])
            except Exception:
                i = s.find("{", i + 1)
                continue
            if isinstance(obj, dict) and ("commands" in obj or "analysis" in obj or "plan" in obj):
                cmds = []
                for c in obj.get("commands") or []:
                    if isinstance(c, dict):
                        k = c.get("keystrokes", "") or ""
                        if k.strip():
                            cmds.append(k)
                return cmds, bool(obj.get("task_complete")), True
        i = s.find("{", i + 1)
    return [], False, False


ERR = re.compile(r"(Traceback \(most recent|^[A-Za-z_]*Error\b|error:|ERROR|command not found|No such file|"
                 r"Permission denied|not found|cannot|failed|Killed)", re.M)
EXPLORE = re.compile(r"^\s*(ls|cat|head|tail|sed -n|nl|grep|find|wc|tree|pwd|file|stat|less|more)\b")
WRITE = re.compile(r"(cat >|<<\s*['\"]?\w+|tee |sed -i|>>? */app|python3? - <<|mkdir|touch|cp |mv )")


def norm_cmds(cmds):
    return tuple(sorted(re.sub(r"\s+", " ", c.strip())[:200] for c in cmds))


def analyze(row, meta):
    msgs = row["conversations"]
    turns = [(m["content"] or "") for m in msgs if m.get("role") == "assistant"]
    obs = [(m["content"] or "") for m in msgs if m.get("role") == "user"]
    n = len(turns)
    think_lens, batch_keys, ncmd, waits, bads = [], [], [], 0, 0
    expl = wr = 0
    total_cmds = 0
    err_after_repeat, err_after_total = 0, 0
    decl = False
    for i, t in enumerate(turns):
        th, rest = split_think(t)
        think_lens.append(len(th.strip()))
        cmds, tc, ok = parse_action(rest)
        decl = decl or tc
        if not ok:
            bads += 1
        key = norm_cmds(cmds)
        batch_keys.append(key)
        ncmd.append(len(cmds))
        total_cmds += len(cmds)
        if not cmds:
            waits += 1
        for c in cmds:
            if EXPLORE.match(c):
                expl += 1
            if WRITE.search(c):
                wr += 1
        # environment-grounding: previous observation had an error -> did this turn repeat the exact batch?
        if i > 0 and i - 1 < len(obs) - 1 + 1:
            prev_obs = obs[i] if i < len(obs) else ""
            if prev_obs and ERR.search(prev_obs[-1500:]):
                err_after_total += 1
                if key and key == batch_keys[i - 1]:
                    err_after_repeat += 1
    # command-level exact repeats (TH3K Table 3 "exact repeats" metric)
    cmd_seen, cmd_total, cmd_rep = set(), 0, 0
    for k in batch_keys:
        for c in k:
            cmd_total += 1
            if c in cmd_seen:
                cmd_rep += 1
            cmd_seen.add(c)
    # exact repeats: batch already seen earlier in this trajectory
    seen, repeats = set(), 0
    for k in batch_keys:
        if not k:
            continue
        if k in seen:
            repeats += 1
        seen.add(k)
    # mechanical loops: >=3 identical consecutive non-empty batches
    loop_turns = 0
    run = 0
    for i, k in enumerate(batch_keys):
        if k and i and k == batch_keys[i - 1]:
            run += 1
        else:
            run = 0
        if run >= 2:
            loop_turns += 1
    # feedback chain: maximal run of consecutive turns with >=1 command
    chains, cur = [], 0
    for c in ncmd:
        if c:
            cur += 1
        else:
            if cur:
                chains.append(cur)
            cur = 0
    if cur:
        chains.append(cur)
    nonempty = sum(1 for c in ncmd if c)
    return dict(
        turns=n,
        chain_mean=(st.mean(chains) if chains else 0),
        chain_max=(max(chains) if chains else 0),
        repeat=(repeats / nonempty if nonempty else 0),
        repeat_cmd=(cmd_rep / cmd_total if cmd_total else 0),
        loop_turns=loop_turns,
        wait=(waits / n if n else 0),
        bad=(bads / n if n else 0),
        think_p50=(st.median(think_lens) if think_lens else 0),
        think_mean=(st.mean(think_lens) if think_lens else 0),
        cmds_per_turn=(total_cmds / n if n else 0),
        short_think=(sum(1 for x in think_lens if x < 20) / n if n else 0),
        explore=(expl / total_cmds if total_cmds else 0),
        write=(wr / total_cmds if total_cmds else 0),
        wrote=wr > 0,
        decl=decl,
        err_ignored=(err_after_repeat / err_after_total if err_after_total else None),
        err_share=(err_after_total / max(1, n - 1)),
        score=meta.get("score"),
        env=meta.get("env_key"),
    )


def run(path, manifest=None, cap=700):
    nlines = sum(1 for _ in open(path, "rb"))
    stride = 1 if cap <= 0 else max(1, nlines // cap)
    out = []
    mf = open(manifest) if manifest else None
    for i, line in enumerate(open(path)):
        m = json.loads(mf.readline() or "{}") if mf else {}
        if i % stride:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not row.get("conversations"):
            continue
        out.append(analyze(row, m))
    return out, nlines


def summarize(name, rs):
    def m(k):
        v = [r[k] for r in rs if r[k] is not None]
        return st.mean(v) if v else float("nan")

    def q(k, p):
        v = sorted(r[k] for r in rs if r[k] is not None)
        return v[int(p * (len(v) - 1))] if v else 0

    sc = [r["score"] for r in rs if r["score"] is not None]
    rows = [
        ("trajectories scored (sample)", str(len(rs))),
        ("pass rate (manifest solved)", f"{st.mean(sc):.2f}" if sc else "n/a"),
        ("turns per trajectory: mean / p50 / p90", f"{m('turns'):.1f} / {q('turns', .5):.0f} / {q('turns', .9):.0f}"),
        ("turns < 5 (Data Recipes min-turns floor)", f"{sum(1 for r in rs if r['turns'] < 5) / len(rs):.1%}"),
        ("feedback chain: mean run / mean longest", f"{m('chain_mean'):.1f} / {m('chain_max'):.1f}"),
        ("exact-repeat batch share", f"{m('repeat'):.1%}"),
        ("exact-repeat command share", f"{m('repeat_cmd'):.1%}"),
        ("mechanical-loop turns (3+ identical) share", f"{m('loop_turns'):.1%}"),
        ("wait/poll turns share", f"{m('wait'):.1%}"),
        ("bad/unparseable action share", f"{m('bad'):.1%}"),
        ("trajectories ending with task_complete", f"{sum(1 for r in rs if r['decl']) / len(rs):.0%}"),
        ("think chars: p50 / mean", f"{m('think_p50'):.0f} / {m('think_mean'):.0f}"),
        ("short think (<20 chars) share", f"{m('short_think'):.1%}"),
        ("commands per turn", f"{m('cmds_per_turn'):.2f}"),
        ("explore / write cmd share", f"{m('explore'):.1%} / {m('write'):.1%}"),
        ("trajectories with >=1 write", f"{sum(1 for r in rs if r['wrote']) / len(rs):.0%}"),
        ("after-error turn repeats same batch (feedback ignored)", f"{m('err_ignored'):.1%}" if any(r['err_ignored'] is not None for r in rs) else "n/a"),
        ("turns following an error observation", f"{m('err_share'):.1%}"),
    ]
    print(f"\n#### {name}")
    for k, v in rows:
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    D = "/storage/openpsi/users/chen/data"
    sets = [
        ("TH3K (th3k_mask, 2441)", f"{D}/terminal_horizon_th3k_mask/th3k_mask.jsonl", None),
        ("dsv41_judged (8876 segments)", f"{D}/terminal_horizon_dsv41_judged/dsv41_judged.jsonl",
         f"{D}/terminal_horizon_dsv41_judged/dsv41_judged.manifest.jsonl"),
        ("qwen3.8-flash (11001 segments)", f"{D}/terminal_horizon_qwen38_flash/qwen38_flash.jsonl",
         f"{D}/terminal_horizon_qwen38_flash/qwen38_flash.manifest.jsonl"),
    ]
    if len(sys.argv) > 1:
        sets = [s for s in sets if sys.argv[1] in s[0]]
    for name, p, mf in sets:
        rs, n = run(p, mf)
        print(f"\n# {name}: file rows={n}")
        summarize(name, rs)

# ---------- discount-adjusted metrics (judge criterion 3: waiting/polling earns no credit) ----------
POLL = re.compile(r"^\s*(clear\s*$|sleep\s+\d+|tail\s|gmx\s+check|ps\s|cat\s+\S*(status|log)|"
                  r"ls\s+-la\s+\S+\.(xtc|log|cpt|out)|echo\s+(CHECK|AFTER|PING|READY)|wc\s+-l)")


def discount_row(row):
    msgs = row["conversations"]
    turns = [(m["content"] or "") for m in msgs if m.get("role") == "assistant"]
    n = len(turns)
    cmd_n = clear_n = sleep_n = long_n = 0
    wait_turn = 0
    wait_sec = sleep_sec = 0.0
    eff_turn = 0
    eff_think = []
    for t in turns:
        th, rest = split_think(t)
        i0 = rest.find("{")
        obj = None
        while i0 != -1:
            depth = 0
            for j in range(i0, len(rest)):
                if rest[j] == "{":
                    depth += 1
                elif rest[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            o = json.loads(rest[i0:j + 1])
                        except Exception:
                            o = None
                        if isinstance(o, dict) and "commands" in o:
                            obj = o
                        break
            if obj:
                break
            i0 = rest.find("{", i0 + 1)
        if not obj:
            continue
        productive = False
        any_cmd = False
        for c in obj.get("commands") or []:
            k = (c.get("keystrokes") or "")
            d = c.get("duration", 1.0) or 0
            d = float(d) if isinstance(d, (int, float)) else 0.0
            if not k.strip():
                wait_sec += d
                continue
            any_cmd = True
            cmd_n += 1
            if k.strip().startswith("clear"):
                clear_n += 1
                continue
            if re.match(r"^\s*sleep\s+\d+", k):
                sleep_n += 1
                m = re.match(r"^\s*sleep\s+(\d+)", k)
                sleep_sec += int(m.group(1))
                continue
            if d >= 30:
                long_n += 1
            if not POLL.match(k):
                productive = True
        if not any_cmd:
            wait_turn += 1
        if productive:
            eff_turn += 1
            eff_think.append(len(th.strip()))
    eff_cmd = cmd_n - clear_n - sleep_n
    return dict(turns=n, cmd_n=cmd_n, clear=clear_n / cmd_n if cmd_n else 0,
                sleep=sleep_n / cmd_n if cmd_n else 0, long30=long_n / cmd_n if cmd_n else 0,
                wait_turn=wait_turn / n if n else 0,
                wait_min=(wait_sec + sleep_sec) / 60.0,
                eff_turn_share=eff_turn / n if n else 0,
                eff_cmd_per_turn=eff_cmd / n if n else 0,
                eff_cmd_per_eff_turn=eff_cmd / eff_turn if eff_turn else 0,
                eff_think_p50=(st.median(eff_think) if eff_think else 0))


def discount_summary(name, path, manifest=None, cap=500):
    nlines = sum(1 for _ in open(path, "rb"))
    stride = 1 if cap <= 0 else max(1, nlines // cap)
    rs = []
    mf = open(manifest) if manifest else None
    for i, line in enumerate(open(path)):
        if mf:
            mf.readline()
        if i % stride:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not row.get("conversations"):
            continue
        rs.append(discount_row(row))
    if not rs:
        return

    def m(k):
        return st.mean(r[k] for r in rs)

    print(f"\n#### {name} (n={len(rs)})")
    print(f"| turns / traj | {m('turns'):.0f} |")
    print(f"| clear share of cmds | {m('clear'):.1%} |")
    print(f"| sleep share of cmds | {m('sleep'):.1%} |")
    print(f"| cmds with duration>=30s | {m('long30'):.1%} |")
    print(f"| pure-wait turns (empty keystrokes) | {m('wait_turn'):.1%} |")
    print(f"| declared wait+sleep min / traj | {m('wait_min'):.1f} |")
    print(f"| productive turns share | {m('eff_turn_share'):.1%} |")
    print(f"| effective cmds / turn | {m('eff_cmd_per_turn'):.2f} |")
    print(f"| effective cmds / productive turn | {m('eff_cmd_per_eff_turn'):.2f} |")
    print(f"| effective think p50 | {m('eff_think_p50'):.0f} |")
