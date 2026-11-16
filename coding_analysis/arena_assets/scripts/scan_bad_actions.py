#!/usr/bin/env python3
"""Scan a Terminus-2 SFT jsonl for bad-action patterns that hurt SFT but are not environment faults
and that the process judge may let through as "isolated" errors:

  per assistant turn
    empty-cmd        commands == []  and no task_complete (model emitted nothing executable)
    bad-json         assistant content after </think> is not a parseable Terminus JSON object
    dup-key          duplicated top-level keys in the JSON (tool-mode leak: "analysis" twice ...)
    markup-leak      DSML / <parameter / <plan> / "Analysis:" residue in analysis/plan text
    think-empty      <think></think> with < 20 chars
    think-trunc      think content ends mid-token (no closing tag)
    huge-think       think > 30k chars
    repeat-cmd       exact same NON-EMPTY command batch as the previous turn
    repeat-cmd3      same non-empty batch 3+ times in a row
    poll             turn whose only command is an empty keystroke (duration wait) or sleep
    poll-heavy       trajectory where > 25% of assistant turns are polls
    mojibake-asst    mojibake inside an ASSISTANT turn (= would enter the loss)
    mojibake-think   ... inside <think>;  mojibake-cmd ... inside keystrokes
  per user turn
    mojibake         replacement char U+FFFD or classic mojibake sequences (Ã©, â€™, ï¿½)
    control-chars    raw C0 controls other than \\t \\n \\r (ANSI escape residue)
    parse-error-msg  harness "Previous response had parsing errors" / "Command N should end with newline"
    huge-obs         observation > 40k chars (should have been truncated by harness)
    identical-obs    same observation text as the previous user turn (terminal stuck / nothing happened)

Reports per-pattern: turns, trajectories, and lists the worst trajectories.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

THINK = re.compile(r"<think>(.*?)</think>", re.S)
THINK_OPEN = re.compile(r"<think>")
MOJIBAKE = re.compile(r"\ufffd|Ã[©¨ ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]|â€[™œ\x9d\"]|ï¿½")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PARSE_ERR = re.compile(r"Previous response had parsing errors|should end with a newline|WARNINGS:", re.I)
MARKUP = re.compile(r"<｜DSML｜|<parameter name=|</?plan>|<invoke name=|\"analysis\": \"Analysis:|\\n\\nPlan:")
POLL_CMD = re.compile(r"^\s*(sleep\s+[\d.]+;?\s*|\s*)$")


def parse_action(content):
    """Return (think, json_obj_or_None, raw_after_think, dup_keys)."""
    m = THINK.search(content)
    think = m.group(1) if m else None
    rest = content[m.end():] if m else content
    rest = rest.strip()
    dup = False
    obj = None
    try:
        def hook(pairs):
            nonlocal dup
            keys = [k for k, _ in pairs]
            if len(keys) != len(set(keys)):
                dup = True
            return dict(pairs)
        obj = json.loads(rest, object_pairs_hook=hook)
    except Exception:  # noqa: BLE001
        obj = None
    if not isinstance(obj, dict):
        obj = None
    return think, obj, rest, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--no-manifest", action="store_true", help="dataset has no .manifest.jsonl (e.g. th3k)")
    args = ap.parse_args()

    turn_c = Counter()
    traj_c = Counter()
    worst = defaultdict(list)   # pattern -> [(count, key)]
    n_traj = 0
    n_asst = n_user = 0
    import contextlib, itertools
    mf = open(args.base + ".manifest.jsonl") if not args.no_manifest else itertools.repeat(None)
    with contextlib.ExitStack() as es, open(args.base + ".jsonl") as df:
        if not args.no_manifest:
            es.enter_context(mf)
        for i, (ml, dl) in enumerate(zip(mf, df)):
            if args.limit and i >= args.limit:
                break
            m = json.loads(ml) if ml else {"env_key": "       row", "sample": "-", "score": "?"}
            key = f"{m['env_key'][7:62]}|{m['sample']}|sc{m['score']}|L{i}"
            conv = json.loads(dl)["conversations"]
            n_traj += 1
            local = Counter()
            prev_batch = None
            run = 0
            prev_obs = None
            for idx, msg in enumerate(conv):
                c = msg["content"]
                if msg["role"] == "assistant":
                    n_asst += 1
                    think, obj, rest, dup = parse_action(c)
                    if MOJIBAKE.search(c):
                        local["mojibake-asst"] += 1
                        if think and MOJIBAKE.search(think):
                            local["mojibake-think"] += 1
                        if obj and any(MOJIBAKE.search(x.get("keystrokes", "")) for x in (obj.get("commands") or []) if isinstance(x, dict)):
                            local["mojibake-cmd"] += 1
                    if think is None:
                        if THINK_OPEN.search(c):
                            local["think-trunc"] += 1
                    else:
                        if len(think.strip()) < 20:
                            local["think-empty"] += 1
                        if len(think) > 30000:
                            local["huge-think"] += 1
                    if obj is None:
                        local["bad-json"] += 1
                        prev_batch = None
                        continue
                    if dup:
                        local["dup-key"] += 1
                    if MARKUP.search(rest):
                        local["markup-leak"] += 1
                    cmds = obj.get("commands") or []
                    keys = [x.get("keystrokes", "") for x in cmds if isinstance(x, dict)]
                    if not keys and not obj.get("task_complete"):
                        local["empty-cmd"] += 1
                    if keys and all(POLL_CMD.match(k) for k in keys):
                        local["poll"] += 1
                    # a wait batch ("" keystrokes) is legitimate polling, not a repeated command
                    batch = tuple(k for k in keys if k.strip())
                    if batch and batch == prev_batch:
                        run += 1
                        local["repeat-cmd"] += 1
                        if run >= 2:
                            local["repeat-cmd3"] += 1
                    else:
                        run = 0
                    prev_batch = batch or None
                else:
                    n_user += 1
                    if idx == 0:
                        continue
                    if MOJIBAKE.search(c):
                        local["mojibake"] += 1
                    if CTRL.search(ANSI.sub("", c)):
                        local["control-chars"] += 1
                    if PARSE_ERR.search(c):
                        local["parse-error-msg"] += 1
                    if len(c) > 40000:
                        local["huge-obs"] += 1
                    body = c.split("New Terminal Output:", 1)[-1].strip()
                    if prev_obs is not None and body and body == prev_obs:
                        local["identical-obs"] += 1
                    prev_obs = body
            n_asst_t = sum(1 for x in conv if x["role"] == "assistant")
            if n_asst_t and local["poll"] / n_asst_t > 0.25:
                local["poll-heavy"] += 1
            for k, v in local.items():
                turn_c[k] += v
                traj_c[k] += 1
                worst[k].append((v, key))

    print(f"trajectories {n_traj}  assistant turns {n_asst}  user turns {n_user}\n")
    print(f"{'pattern':16s} {'turns':>7s} {'%asst/user':>10s} {'traj':>5s} {'%traj':>6s}")
    for k, v in sorted(turn_c.items(), key=lambda x: -x[1]):
        denom = n_user if k in ("mojibake", "control-chars", "parse-error-msg", "huge-obs", "identical-obs") else n_asst
        print(f"{k:16s} {v:7d} {100*v/denom:9.2f}% {traj_c[k]:5d} {100*traj_c[k]/n_traj:5.1f}%")
    for k in sorted(worst):
        ws = sorted(worst[k], reverse=True)[: args.top]
        print(f"\n--- worst {k}: " + "; ".join(f"{n}x {key}" for n, key in ws))


if __name__ == "__main__":
    main()