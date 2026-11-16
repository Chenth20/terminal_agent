#!/usr/bin/env python3
"""Build the B-arm (trunk + branches tree) tokenized cache from the A-arm (perdh) cache.

One row per trajectory::

    input_ids    = trunk ++ branch_0 ++ branch_1 ++ ... (branches in step order)
    trunk        = the whole trajectory with EVERY assistant turn think-stripped (the visible stream the
                   generator actually saw), encoded message-by-message with the same template code path
                   LlamaFactory used for the perdh rows (``ReasoningTemplate._encode``)
    fork_k       = len(trunk prefix through source_k)  (right after user turn k incl. the assistant header)
    branch_k     = the supervised tail of perdh row (traj, k) = <think>think_k</think> + output_k + <|im_end|>\\n
    labels       = -100 on the trunk, branch token ids on the branches
    position_ids = trunk t -> t ; branch_k token j -> fork_k + j   (identical to the perdh row positions)
    tree_forks / tree_branch_lens / trunk_len = tree metadata for the collator / attention masks

Equivalence with arm A is asserted token-for-token:
  * trunk[:fork_k] == perdh_row(traj,k).input_ids[:prompt_len_k]   for every step k
  * branch_k       == perdh_row(traj,k).input_ids[prompt_len_k:]   (its supervised tail; labels == ids there)
  * concat_k branch_k == baseline cache supervised tokens of that trajectory
  * len(input_ids) == trunk_len + sum(branch_lens)
  * per dbg step: supervised(tree) == supervised(base dbg) == supervised(perdh)

Steps whose assistant turn is flagged ``loss: 0`` have no supervised tokens in any arm; they get NO branch
(the trunk still carries the stripped visible output), which is listed in the manifest.

Run inside the training container (CPU only, ~20 GB RAM):

    PYTHONPATH=src python coding_sft/prep/prepare_tree.py [--limit 3]
"""

import argparse
import json
import os
import sys
import time


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from build_arm_caches import perdh_index, read_cache
from datasets import Dataset, DatasetDict, Features, Value
from prepare_perdh import strip_think
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments

try:
    from datasets import List as SeqFeature  # datasets >= 4
except ImportError:  # datasets 3.x
    from datasets import Sequence as SeqFeature

DATA = "/storage/openpsi/users/chen/data"
FEATURES = Features(
    {
        "input_ids": SeqFeature(Value("int32")),
        "attention_mask": SeqFeature(Value("int32")),
        "labels": SeqFeature(Value("int64")),
        "position_ids": SeqFeature(Value("int32")),
        "tree_forks": SeqFeature(Value("int64")),
        "tree_branch_lens": SeqFeature(Value("int64")),
        "trunk_len": Value("int64"),
        "images": Value("null"),
        "videos": Value("null"),
        "audios": Value("null"),
    }
)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def encode_trunk(template, tokenizer, conv: list[dict]) -> tuple[list[int], list[int]]:
    """Trunk ids of a trajectory + fork offsets (one per step: trunk length through source_k).

    Assistant content goes through exactly the perdh pipeline for HISTORY turns: jsonl-level ``strip_think``
    (prepare_perdh.py) followed by ``ReasoningTemplate.remove_thought`` (encode_multiturn with
    discarding_history_cot=True). No empty-cot is inserted (encode_multiturn only does that for the LAST
    turn, i.e. the branch, never for history turns).
    """
    msgs = []
    for i, m in enumerate(conv):
        if i % 2 == 0:
            assert m["role"] == "user", (i, m["role"])
            msgs.append({"role": "user", "content": m["content"]})
        else:
            assert m["role"] == "assistant", (i, m["role"])
            msgs.append({"role": "assistant", "content": template.remove_thought(strip_think(m["content"]))})
    encoded = template._encode(tokenizer, msgs, system=None, tools=None)
    assert len(encoded) == len(msgs)
    ids: list[int] = []
    forks: list[int] = []
    for i, e in enumerate(encoded):
        ids.extend(e)
        if i % 2 == 0:
            forks.append(len(ids))
    return ids, forks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--jsonl", default=f"{DATA}/th3k_align/th3k_align.jsonl")
    ap.add_argument("--base-cache", default=f"{DATA}/th3k_align/tokenized_qwen3_5_text_262144")
    ap.add_argument("--base-dbg-cache", default=f"{DATA}/th3k_align/tokenized_qwen3_5_text_262144_dbg")
    ap.add_argument("--perdh-cache", default=f"{DATA}/th3k_align_perdh/tokenized_qwen3_5_text_262144")
    ap.add_argument("--perdh-jsonl", default=f"{DATA}/th3k_align_perdh/th3k_perdh.jsonl")
    ap.add_argument("--schedule-dbg", default=f"{DATA}/th3k_align/group_schedule_dbg.json")
    ap.add_argument("--out", default=f"{DATA}/th3k_align_tree/tokenized_qwen3_5_text_262144")
    ap.add_argument("--out-dbg", default=f"{DATA}/th3k_align_tree/tokenized_qwen3_5_text_262144_dbg")
    ap.add_argument("--cutoff", type=int, default=262144)
    ap.add_argument("--limit", type=int, default=0, help="only build the first N trajectories (no caches written)")
    ap.add_argument("--context", type=int, default=24, help="tokens of decoded context around a mismatch")
    args = ap.parse_args()
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    data_args = DataArguments(template="qwen3_5_text", cutoff_len=args.cutoff, enable_thinking=True)
    template = get_template_and_fix_tokenizer(tok, data_args)
    log(f"template {type(template).__name__} enable_thinking={template.enable_thinking} "
        f"preserve_thinking={template.preserve_thinking} thought_words={template.thought_words}", t0)

    with open(args.jsonl, encoding="utf-8") as f:
        convs = [json.loads(line)["conversations"] for line in f]
    n_traj_all = len(convs)
    n_traj = min(n_traj_all, args.limit) if args.limit else n_traj_all
    log(f"{n_traj_all} trajectories in {args.jsonl}; building {n_traj}", t0)

    rows = perdh_index(args.perdh_jsonl)
    perdh = read_cache(args.perdh_cache, ["input_ids", "labels"])
    assert len(perdh["input_ids"]) == len(rows), (len(perdh["input_ids"]), len(rows))
    by_traj: dict[int, list[int]] = {}
    for r, (t, k) in enumerate(rows):
        by_traj.setdefault(t, []).append(r)
    for t, rs in by_traj.items():
        assert [rows[r][1] for r in rs] == list(range(len(rs))), f"traj {t}: perdh steps not contiguous"
    log(f"perdh cache: {len(rows)} rows, {len(by_traj)} trajectories", t0)

    base = read_cache(args.base_cache, ["labels"])
    assert len(base["labels"]) == n_traj_all, (len(base["labels"]), n_traj_all)
    log(f"base cache: {len(base['labels'])} rows", t0)

    failures: list[dict] = []
    no_branch: list[tuple[int, int]] = []
    out = {k: [] for k in ("input_ids", "attention_mask", "labels", "position_ids", "tree_forks", "tree_branch_lens", "trunk_len")}
    sup_by_traj: list[int] = []
    tot_trunk = tot_branch = 0

    for t in range(n_traj):
        conv = convs[t]
        n_steps = len(conv) // 2
        assert len(conv) == 2 * n_steps and len(by_traj[t]) == n_steps, (t, len(conv), len(by_traj[t]))
        trunk, forks = encode_trunk(template, tok, conv)
        trunk = np.asarray(trunk, dtype=np.int32)
        assert len(forks) == n_steps
        assert len(trunk) <= args.cutoff, (t, len(trunk))

        b_forks: list[int] = []
        b_ids: list[np.ndarray] = []
        traj_ok = True
        for k in range(n_steps):
            r = by_traj[t][k]
            p_ids, p_lab = perdh["input_ids"][r], perdh["labels"][r]
            sup = np.flatnonzero(p_lab != IGNORE_INDEX)
            fork = forks[k]
            # 1. trunk prefix == perdh prompt (what the branch conditions on)
            pre = p_ids[:fork]
            cmp_len = min(len(pre), len(trunk))
            neq = np.flatnonzero(trunk[:cmp_len] != pre[:cmp_len])
            if len(neq) or len(pre) != fork or len(trunk) < fork:
                pos = int(neq[0]) if len(neq) else cmp_len
                lo, hi = max(0, pos - args.context), pos + args.context
                failures.append(
                    {
                        "traj": t, "step": k, "kind": "trunk_prefix", "pos": pos, "fork": fork, "perdh_len": len(p_ids),
                        "trunk_ctx": tok.decode(trunk[lo:hi].tolist()), "perdh_ctx": tok.decode(p_ids[lo:hi].tolist()),
                    }
                )
                traj_ok = False
                continue
            if len(sup) == 0:
                no_branch.append((t, k))  # loss=0 turn: nothing supervised in any arm
                continue
            prompt_len = int(sup[0])
            # 2. supervised tail is contiguous and starts right at the fork (= directly after the assistant header)
            if prompt_len != fork or int(sup[-1]) != len(p_ids) - 1 or len(sup) != len(p_ids) - prompt_len:
                failures.append(
                    {
                        "traj": t, "step": k, "kind": "branch_span", "pos": prompt_len, "fork": fork, "perdh_len": len(p_ids),
                        "n_sup": int(len(sup)), "trunk_ctx": tok.decode(trunk[max(0, fork - args.context): fork].tolist()),
                        "perdh_ctx": tok.decode(p_ids[max(0, prompt_len - args.context): prompt_len + args.context].tolist()),
                    }
                )
                traj_ok = False
                continue
            br = p_ids[prompt_len:].astype(np.int32)
            assert np.array_equal(br.astype(np.int64), p_lab[prompt_len:]), (t, k, "labels != ids on supervised tail")
            b_forks.append(fork)
            b_ids.append(br)

        if not traj_ok:
            continue

        # 3. branches concatenated == baseline supervised tokens of this trajectory
        b_lab = base["labels"][t]
        b_sup = b_lab[b_lab != IGNORE_INDEX]
        cat = np.concatenate(b_ids).astype(np.int64) if b_ids else np.zeros(0, dtype=np.int64)
        assert b_sup.shape == cat.shape and np.array_equal(b_sup, cat), (
            f"traj {t}: branch tokens ({cat.shape[0]}) != baseline supervised ({b_sup.shape[0]})"
        )

        lens = [len(x) for x in b_ids]
        ids = np.concatenate([trunk] + b_ids) if b_ids else trunk.copy()
        labels = np.concatenate([np.full(len(trunk), IGNORE_INDEX, dtype=np.int64)] + [x.astype(np.int64) for x in b_ids])
        pos = np.concatenate([np.arange(len(trunk), dtype=np.int32)] + [np.arange(f, f + n, dtype=np.int32) for f, n in zip(b_forks, lens)])
        assert len(ids) == len(trunk) + sum(lens) == len(labels) == len(pos), (t, len(ids), len(trunk), sum(lens))
        assert int((labels != IGNORE_INDEX).sum()) == sum(lens) == int(b_sup.shape[0])
        assert all(0 < f <= len(trunk) for f in b_forks) and b_forks == sorted(b_forks)
        assert len(ids) <= args.cutoff, (t, len(ids))

        out["input_ids"].append(ids)
        out["attention_mask"].append(np.ones(len(ids), dtype=np.int32))
        out["labels"].append(labels)
        out["position_ids"].append(pos)
        out["tree_forks"].append(np.asarray(b_forks, dtype=np.int64))
        out["tree_branch_lens"].append(np.asarray(lens, dtype=np.int64))
        out["trunk_len"].append(len(trunk))
        sup_by_traj.append(sum(lens))
        tot_trunk += len(trunk)
        tot_branch += sum(lens)
        if t < 5 or (t + 1) % 25 == 0:
            log(f"traj {t:3d}: {n_steps:2d} steps, trunk {len(trunk):7d}, {len(b_ids)} branches {sum(lens):7d} tok, total {len(ids):7d}", t0)

    if failures:
        log(f"FAILED: {len(failures)} (traj, step) pairs could not be reconciled with the perdh cache", t0)
        for f in failures:
            print(f"  traj {f['traj']} step {f['step']} [{f['kind']}] first mismatch @ {f['pos']} (fork {f['fork']}, perdh row len {f['perdh_len']})")
            print(f"     trunk: {f['trunk_ctx']!r}")
            print(f"     perdh: {f['perdh_ctx']!r}")
        sys.exit(1)

    perdh_sup_total = sum(int((perdh["labels"][r] != IGNORE_INDEX).sum()) for t in range(n_traj) for r in by_traj[t])
    base_sup_total = sum(int((base["labels"][t] != IGNORE_INDEX).sum()) for t in range(n_traj))
    assert tot_branch == perdh_sup_total == base_sup_total, (tot_branch, perdh_sup_total, base_sup_total)
    log(f"ALL {n_traj} trajectories reconciled: trunk prefix == perdh prompt and branch == perdh supervised tail for every step", t0)
    log(f"supervised tokens: tree {tot_branch:,} == perdh {perdh_sup_total:,} == base {base_sup_total:,}", t0)
    log(f"tokens: trunk {tot_trunk:,} + branches {tot_branch:,} = {tot_trunk + tot_branch:,} "
        f"(perdh rows total {sum(len(perdh['input_ids'][r]) for t in range(n_traj) for r in by_traj[t]):,})", t0)
    log(f"steps without a branch (loss=0 turns): {len(no_branch)} -> {no_branch}", t0)

    if args.limit:
        log("--limit set: not writing caches", t0)
        return

    n = len(out["input_ids"])
    assert n == n_traj_all, (n, n_traj_all)
    cols = dict(out)
    cols["images"] = [None] * n; cols["videos"] = [None] * n; cols["audios"] = [None] * n
    ds = Dataset.from_dict(cols, features=FEATURES)
    DatasetDict({"train": ds}).save_to_disk(args.out)
    log(f"full cache: {len(ds)} rows (trajectory order) -> {args.out}", t0)

    # ---------------- dbg cache: 64 rows in group_schedule_dbg order, row 8s+j = groups[s][j] ----------------
    with open(args.schedule_dbg, encoding="utf-8") as f:
        sched = json.load(f)
    groups: list[list[int]] = sched["groups"]
    TPS = sched["trajs_per_step"]
    assert all(len(g) == TPS for g in groups), [len(g) for g in groups]
    order = [t for g in groups for t in g]
    base_dbg = read_cache(args.base_dbg_cache, ["labels"])
    assert len(base_dbg["labels"]) == len(order), (len(base_dbg["labels"]), len(order))

    dcols = {k: [out[k][t] for t in order] for k in out}
    dcols["images"] = [None] * len(order); dcols["videos"] = [None] * len(order); dcols["audios"] = [None] * len(order)
    manifest = {
        "schedule": args.schedule_dbg, "trajs_per_step": TPS, "n_steps": len(groups), "rows": len(order), "cutoff": args.cutoff,
        "source_perdh_cache": args.perdh_cache, "source_base_dbg_cache": args.base_dbg_cache,
        "steps_without_branch": no_branch, "steps": [],
    }
    for s, g in enumerate(groups):
        tree_sup = sum(sup_by_traj[t] for t in g)
        base_sup = sum(int((base_dbg["labels"][TPS * s + j] != IGNORE_INDEX).sum()) for j in range(TPS))
        perdh_sup = sum(int((perdh["labels"][r] != IGNORE_INDEX).sum()) for t in g for r in by_traj[t])
        assert tree_sup == base_sup == perdh_sup, (s, tree_sup, base_sup, perdh_sup)
        tokens = sum(len(out["input_ids"][t]) for t in g)
        trunk_tok = sum(out["trunk_len"][t] for t in g)
        manifest["steps"].append(
            {"step": s, "trajs": g, "rows": [TPS * s + j for j in range(TPS)], "supervised": tree_sup, "tokens": tokens,
             "trunk_tokens": trunk_tok, "branch_tokens": tokens - trunk_tok, "n_branches": int(sum(len(out["tree_forks"][t]) for t in g)),
             "max_row_tokens": max(len(out["input_ids"][t]) for t in g)}
        )
        log(f"dbg step {s}: supervised tree {tree_sup:,} == base_dbg == perdh; {tokens:,} tokens (trunk {trunk_tok:,}), max row {manifest['steps'][-1]['max_row_tokens']:,}", t0)
    dds = Dataset.from_dict(dcols, features=FEATURES)
    DatasetDict({"train": dds}).save_to_disk(args.out_dbg)
    with open(os.path.join(args.out_dbg, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    log(f"dbg cache: {len(dds)} rows -> {args.out_dbg} (+ manifest.json)", t0)
    log("DONE", t0)


if __name__ == "__main__":
    main()
