#!/usr/bin/env python3
"""Build an arm-B (trunk + branches tree) tokenized cache directly from a sharegpt jsonl.

Self-contained generalisation of ``prepare_tree.py`` (which reconciles against an existing perdh cache): there is
no A-arm cache here, so the per-step prompt is re-derived from scratch and asserted against the trunk prefix.

One row per conversation::

    input_ids    = trunk ++ branch_0 ++ branch_1 ++ ...   (branches in step order)
    trunk        = the whole conversation with EVERY assistant turn think-stripped (``template.remove_thought``),
                   encoded message-by-message with LlamaFactory's own ``ReasoningTemplate._encode``
                   (qwen3_5_text, enable_thinking=True) = the visible stream the generator actually saw
    fork_k       = len(trunk prefix through user turn k)  (right after the assistant header of step k)
    branch_k     = token ids of assistant turn k WITH its think block, encoded exactly as the template renders a
                   live (last) turn = ``<think>\\n{reasoning}\\n</think>\\n\\n{output}<|im_end|>\\n``
    labels       = -100 on the trunk, branch token ids on the branches
    position_ids = trunk t -> t ; branch_k token j -> fork_k + j
    tree_forks / tree_branch_lens / trunk_len = tree metadata for the collator / attention masks

Schema (FEATURES) is identical to ``prepare_tree.py`` / th3k_align_tree so the model patch consumes it unchanged.

Asserted per conversation:
  * every row: ``encode_oneturn(messages)`` on the FULL conversation (the public per-step path, history think
    stripped by the template itself) gives prompt == trunk[:fork_last] and response == last branch;
  * first ``--verify-first`` conversations + a seeded ``--verify-frac`` sample (``--verify-all`` for all): the same
    from-scratch check for EVERY step k (prompt_k = encode_oneturn(messages[:2k+2]) prompt, NOT a slice of the
    trunk) == trunk[:fork_k], and its response == branch_k;
  * the same rows: the actual arm-A row of step k (prepare_perdh.strip_think on history + ``encode_multiturn(...,
    discarding_history_cot=True)``, i.e. SupervisedDatasetProcessor with mask_history=True) == trunk[:fork_k] ++ branch_k
    with its supervised tail == branch_k;
  * every row: len(input_ids) == trunk_len + sum(branch_lens); supervised count == sum(branch_lens); labels on
    branches == ids; forks strictly increasing within (0, trunk_len]; every branch alone <= --cutoff.
Messages are tokenized ONCE each in the build (prompt_k by concatenation, exact because ``_encode`` tokenizes
messages independently); the ``encode_oneturn`` re-encode is the independent check.

Input normalisation follows ``prepare_terminal_horizon.py`` (points 2 and 3 of its docstring): user content
``strip()``-ed, every assistant turn rewritten to ``<think>\\n{reasoning}\\n</think>\\n\\n{json}`` via
``normalize_assistant`` (JSON-action validation; leniently-parsed actions re-serialised; a turn without a think
block gets an empty one). Malformed actions: ``--bad-pair mask`` (default, the th3k_mask policy) keeps the turn in
the trunk verbatim with ``loss: 0`` -> NO branch; ``--bad-pair drop`` drops the conversation.

Over-long handling (prepare_terminal_horizon.py rules, point 3/6 of its docstring): an (observation, action) pair that
alone would not fit ``--cutoff`` (``tok_lens[0] + obs + act + 24 > cutoff``) is treated like a malformed action
(mask -> no branch, or drop); ``--overlong-conv drop`` drops a conversation whose think-inline whole-trajectory token
estimate (``sum(tok_lens) + 16 * pairs``) exceeds ``--cutoff`` (= ``--no-chunk --overlong drop``, the th3k_mask
policy). Independently, a conversation whose TREE row exceeds ``--max-tree-tokens`` (0 = unlimited) is DROPPED whole:
never truncated mid-branch. Every drop is counted, logged and listed in the manifest; the number of kept rows whose
tree exceeds ``--cutoff`` is reported too (``rows_over_cutoff``) so the consumer can filter.

``--split-tokens N`` (the production setting): a tree row longer than N is split into consecutive-branch sub-rows
``trunk[:fork_last] ++ branches[a:b]``, each <= N tokens. This is EXACT, not a truncation: branch k only ever attends
to trunk[:fork_k] and to itself, so cutting the trunk at the last fork of the group changes nothing any branch in the
group can see, and every supervised token is kept (asserted). The manifest records ``src_row`` / ``part`` /
``n_parts`` / ``branch_range`` per row so sub-rows can be traced back to their conversation.

Run inside the training container (CPU only):

    PYTHONPATH=src python coding_sft/prep/prepare_tree_generic.py --input X.jsonl --out DIR --name NAME [--limit 20]
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, DatasetInfo, Features, Value
from datasets.table import InMemoryTable
from prepare_perdh import strip_think
from prepare_terminal_horizon import THINK_CLOSE, THINK_OPEN, normalize_assistant, validate_schema

from llamafactory.extras.constants import IGNORE_INDEX

try:
    from datasets import List as SeqFeature  # datasets >= 4
except ImportError:  # datasets 3.x
    from datasets import Sequence as SeqFeature

# MUST stay identical to prepare_tree.py (= th3k_align_tree/tokenized_qwen3_5_text_262144/train/dataset_info.json)
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
LIST_COLS = ("input_ids", "attention_mask", "labels", "position_ids", "tree_forks", "tree_branch_lens")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


class Dropped(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def load_template(model: str, template_name: str, cutoff: int):
    from transformers import AutoTokenizer

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    tok = AutoTokenizer.from_pretrained(model)
    data_args = DataArguments(template=template_name, cutoff_len=cutoff, enable_thinking=True)
    template = get_template_and_fix_tokenizer(tok, data_args)
    assert template.enable_thinking is True and template.thought_words == (THINK_OPEN, THINK_CLOSE), template.thought_words
    assert not template.efficient_eos and not template.preserve_thinking
    return tok, template


def normalize_conv(conv: list[dict], tokenizer, cutoff: int, bad_pair: str, overlong_conv: str) -> tuple[list[dict], dict]:
    """prepare_terminal_horizon.py rules: user strip(), assistant -> template think layout + JSON validation,
    oversize (observation, action) pairs -> bad pair, whole think-inline conversation > cutoff -> drop (if enabled)."""
    if not validate_schema(conv):
        raise Dropped("schema")
    fixed: list[dict] = []
    bad_pairs: set[int] = set()
    n_bad_action = 0
    for i, m in enumerate(conv):
        if i % 2 == 0:
            fixed.append({"role": "user", "content": m["content"].strip()})
            continue
        res = normalize_assistant(m["content"])
        if res is None:
            bad_pairs.add(i // 2)
            n_bad_action += 1
            fixed.append({"role": "assistant", "content": m["content"]})  # kept verbatim (mask) / never emitted (drop)
            continue
        content, _has_think = res
        assert content.startswith(THINK_OPEN) and THINK_CLOSE in content, (i, content[:80])
        fixed.append({"role": "assistant", "content": content})
    # prepare_terminal_horizon.py: raw content token lengths (no template), slack 8 per message
    tok_lens = [len(x) for x in tokenizer([m["content"] for m in fixed], add_special_tokens=False)["input_ids"]]
    n_oversize = 0
    for k in range(0, len(fixed), 2):
        if k // 2 not in bad_pairs and tok_lens[0] + tok_lens[k] + tok_lens[k + 1] + 24 > cutoff:
            bad_pairs.add(k // 2)
            n_oversize += 1
    if len(bad_pairs) * 2 >= len(fixed):
        raise Dropped("all_pairs_bad", f"{n_bad_action} bad actions, {n_oversize} oversize pairs")
    if bad_pairs and bad_pair == "drop":
        raise Dropped("bad_action", f"pairs {sorted(bad_pairs)}: {n_bad_action} bad actions, {n_oversize} oversize")
    approx = sum(tok_lens) + 8 * len(fixed)
    if overlong_conv == "drop" and approx > cutoff:
        raise Dropped("overlong_conv", f"~{approx} think-inline tokens > {cutoff}")
    for k in bad_pairs:
        fixed[2 * k + 1]["loss"] = 0  # mask: in the trunk verbatim, unsupervised -> no branch
    return fixed, {"bad_action_turns": n_bad_action, "oversize_pairs": n_oversize, "approx_conv_tokens": approx}


def encode_messages(template, tokenizer, msgs: list[dict]) -> list[list[int]]:
    """Per-message token ids via LlamaFactory's own template path (what SupervisedDatasetProcessor calls)."""
    encoded = template._encode(tokenizer, msgs, system=None, tools=None)
    assert len(encoded) == len(msgs)
    return encoded


def live_turn_ids(template, tokenizer, content: str) -> list[int]:
    """Assistant turn encoded as the LAST turn of a conversation (think kept; empty cot added if none)."""
    ids = encode_messages(template, tokenizer, [{"role": "user", "content": "x"}, {"role": "assistant", "content": content}])[1]
    if template.thought_words[0].strip() not in content and template.thought_words[1].strip() not in content:
        ids = template.get_thought_word_ids(tokenizer) + ids  # mirrors ReasoningTemplate.encode_multiturn (enable_thinking)
    return ids


def build_tree(template, tokenizer, conv: list[dict], cutoff: int):
    """Return (trunk, forks, b_forks, branches, no_branch_steps); every message tokenized once per view."""
    n_steps = len(conv) // 2
    hist = [
        {"role": "user", "content": m["content"]} if i % 2 == 0
        else {"role": "assistant", "content": template.remove_thought(m["content"])}
        for i, m in enumerate(conv)
    ]
    trunk: list[int] = []
    forks: list[int] = []
    for i, e in enumerate(encode_messages(template, tokenizer, hist)):
        trunk.extend(e)
        if i % 2 == 0:
            forks.append(len(trunk))
    assert len(forks) == n_steps

    branches: list[list[int]] = []
    b_forks: list[int] = []
    no_branch: list[int] = []
    for k in range(n_steps):
        m = conv[2 * k + 1]
        if m.get("loss", True) in (0, False):
            no_branch.append(k)  # loss=0 turn: unsupervised in every arm -> no branch
            continue
        ids = live_turn_ids(template, tokenizer, m["content"])
        if len(ids) > cutoff:
            raise Dropped("branch_gt_cutoff", f"step {k}: {len(ids)} > {cutoff}")
        branches.append(ids)
        b_forks.append(forks[k])
    return trunk, forks, b_forks, branches, no_branch


def _first_mismatch(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def verify_from_scratch(template, tokenizer, conv, trunk, forks, b_forks, branches, steps, ctx: int = 24) -> None:
    """Independent re-encode: ``encode_oneturn(messages[:2k+2])`` from scratch (history think stripped by the
    template itself, NOT by this script) must give prompt == trunk[:fork_k] and response == branch_k."""
    bidx = {f: i for i, f in enumerate(b_forks)}
    for k in steps:
        msgs = [{"role": m["role"], "content": m["content"]} for m in conv[: 2 * k + 2]]
        p_ids, r_ids = template.encode_oneturn(tokenizer, msgs, system=None, tools=None)
        fork = forks[k]
        if p_ids != trunk[:fork]:
            pos = _first_mismatch(p_ids, trunk)
            raise AssertionError(
                f"step {k}: prompt_k ({len(p_ids)}) != trunk[:{fork}] first mismatch @ {pos}\n"
                f"  scratch: {tokenizer.decode(p_ids[max(0, pos - ctx): pos + ctx])!r}\n"
                f"  trunk  : {tokenizer.decode(trunk[max(0, pos - ctx): pos + ctx])!r}"
            )
        if conv[2 * k + 1].get("loss", True) in (0, False):
            continue
        br = branches[bidx[fork]]
        if r_ids != br:
            pos = _first_mismatch(r_ids, br)
            raise AssertionError(
                f"step {k}: branch_k ({len(br)}) != scratch response ({len(r_ids)}) first mismatch @ {pos}\n"
                f"  scratch: {tokenizer.decode(r_ids[max(0, pos - ctx): pos + ctx])!r}\n"
                f"  branch : {tokenizer.decode(br[max(0, pos - ctx): pos + ctx])!r}"
            )


def verify_arm_a(template, tokenizer, conv, trunk, forks, b_forks, branches, steps) -> None:
    """The ACTUAL arm-A row of step k: prepare_perdh.py builds messages[:2k+2] with history assistant turns
    ``strip_think``-ed, then SupervisedDatasetProcessor (mask_history=True -> discarding_history_cot=True) calls
    ``encode_multiturn``; the row is the concatenation of all (source, target) pairs, supervised = last target.
    Must equal trunk[:fork_k] ++ branch_k exactly (so the branch NLL is numerically arm A's per-step NLL)."""
    bidx = {f: i for i, f in enumerate(b_forks)}
    for k in steps:
        if conv[2 * k + 1].get("loss", True) in (0, False):
            continue
        msgs = []
        for j in range(k):
            msgs.append({"role": "user", "content": conv[2 * j]["content"]})
            msgs.append({"role": "assistant", "content": strip_think(conv[2 * j + 1]["content"])})
        msgs.append({"role": "user", "content": conv[2 * k]["content"]})
        msgs.append({"role": "assistant", "content": conv[2 * k + 1]["content"]})
        pairs = template.encode_multiturn(tokenizer, msgs, system=None, tools=None, discarding_history_cot=True)
        row: list[int] = []
        for src, tgt in pairs:
            row += src + tgt
        prompt_len = len(row) - len(pairs[-1][1])
        fork = forks[k]
        assert prompt_len == fork, f"step {k}: arm-A prompt_len {prompt_len} != fork {fork}"
        assert row[:fork] == trunk[:fork], f"step {k}: arm-A row prefix != trunk[:fork] @ {_first_mismatch(row, trunk)}"
        assert pairs[-1][1] == branches[bidx[fork]], f"step {k}: arm-A supervised tail != branch_k"


# ---------------------------------------------------------------- worker ----------------------------------------
_W: dict = {}


def _init_worker(model: str, template_name: str, cutoff: int, max_tree: int, bad_pair: str, overlong_conv: str,
                 verify_first: int, verify_frac: float, verify_all: bool, seed: int, split_tokens: int) -> None:
    tok, template = load_template(model, template_name, cutoff)
    _W.update(tok=tok, template=template, cutoff=cutoff, max_tree=max_tree, bad_pair=bad_pair, overlong_conv=overlong_conv,
              verify_first=verify_first, verify_frac=verify_frac, verify_all=verify_all, seed=seed, split_tokens=split_tokens)


def split_groups(b_forks: list[int], lens: list[int], budget: int) -> list[tuple[int, int]]:
    """Greedy left-to-right split of the branches into consecutive groups [a, b) such that every group's row
    (trunk[:fork_{b-1}] ++ branches a..b-1) has at most ``budget`` tokens. Exact, not an approximation: branch k
    only ever attends to trunk[:fork_k] and to itself, so a group that stops at fork_{b-1} drops nothing any of
    its branches could see. Every single branch must fit (fork_k + len_k <= budget), else ValueError."""
    groups: list[tuple[int, int]] = []
    a = 0
    while a < len(lens):
        b = a
        acc = 0
        while b < len(lens) and b_forks[b] + acc + lens[b] <= budget:
            acc += lens[b]
            b += 1
        if b == a:
            raise ValueError(f"branch {a}: fork {b_forks[a]} + len {lens[a]} > budget {budget}")
        groups.append((a, b))
        a = b
    return groups


def assemble_row(trunk: list[int], b_forks: list[int], branches: list[list[int]], a: int, b: int) -> dict:
    """Arrays for the (sub-)row made of trunk[:fork_{b-1}] ++ branches[a:b]."""
    lens = [len(x) for x in branches[a:b]]
    forks = b_forks[a:b]
    tl = forks[-1]
    trunk_a = np.asarray(trunk[:tl], dtype=np.int32)
    b_arr = [np.asarray(x, dtype=np.int32) for x in branches[a:b]]
    ids = np.concatenate([trunk_a] + b_arr)
    labels = np.concatenate([np.full(tl, IGNORE_INDEX, dtype=np.int64)] + [x.astype(np.int64) for x in b_arr])
    pos = np.concatenate([np.arange(tl, dtype=np.int32)] + [np.arange(fk, fk + n, dtype=np.int32) for fk, n in zip(forks, lens)])
    total = tl + sum(lens)
    assert len(ids) == total == len(labels) == len(pos)
    assert int((labels != IGNORE_INDEX).sum()) == sum(lens)
    assert np.array_equal(labels[tl:], ids[tl:].astype(np.int64))
    assert all(0 < fk <= tl for fk in forks) and forks == sorted(forks) and len(set(forks)) == len(forks)
    return {
        "input_ids": ids, "labels": labels, "position_ids": pos,
        "tree_forks": np.asarray(forks, dtype=np.int64), "tree_branch_lens": np.asarray(lens, dtype=np.int64),
        "trunk_len": tl, "tokens": total, "branch_tokens": sum(lens), "max_branch": max(lens), "n_branches": b - a,
    }


def process_conv(job: tuple[int, int, str]) -> dict:
    """job = (read_idx, src_row, raw line). Returns {'ok': True, arrays..., meta} or {'ok': False, reason, detail}."""
    ri, li, line = job
    tok, template, cutoff = _W["tok"], _W["template"], _W["cutoff"]
    try:
        conv = json.loads(line).get("conversations")
        if not isinstance(conv, list):
            raise Dropped("schema")
        conv, norm_info = normalize_conv(conv, tok, cutoff, _W["bad_pair"], _W["overlong_conv"])
        trunk, forks, b_forks, branches, no_branch = build_tree(template, tok, conv, cutoff)
        n_steps = len(conv) // 2
        lens = [len(b) for b in branches]
        total = len(trunk) + sum(lens)
        if len(trunk) > cutoff:
            raise Dropped("trunk_gt_cutoff", f"{len(trunk)} > {cutoff}")
        if _W["max_tree"] and total > _W["max_tree"]:
            raise Dropped("tree_gt_max", f"{total} > {_W['max_tree']} (trunk {len(trunk)}, {len(branches)} branches {sum(lens)})")
        if not branches:
            raise Dropped("no_branch", "every assistant turn flagged loss=0")
    except Dropped as e:
        return {"ok": False, "src_row": li, "reason": e.reason, "detail": e.detail}

    # ---- independent re-encode (deterministic choice per row) ----
    verify_all_steps = _W["verify_all"] or ri < _W["verify_first"] or random.Random(_W["seed"] * 1000003 + li).random() < _W["verify_frac"]
    steps = list(range(n_steps)) if verify_all_steps else [n_steps - 1]
    verify_from_scratch(template, tok, conv, trunk, forks, b_forks, branches, steps)
    verify_arm_a(template, tok, conv, trunk, forks, b_forks, branches, steps)

    # ---- structural asserts on every row ----
    assert len(trunk) <= cutoff, (li, len(trunk))
    assert all(0 < fk <= len(trunk) for fk in b_forks) and b_forks == sorted(b_forks) and len(set(b_forks)) == len(b_forks)
    assert all(n <= cutoff for n in lens)
    split = _W["split_tokens"]
    if split and total > split:
        try:
            groups = split_groups(b_forks, lens, split)
        except ValueError as e:
            return {"ok": False, "src_row": li, "reason": "branch_gt_split", "detail": str(e)}
    else:
        groups = [(0, len(branches))]
    rows = []
    for pi, (a, b) in enumerate(groups):
        arr = assemble_row(trunk, b_forks, branches, a, b)
        rows.append({
            **arr,
            "meta": {"src_row": li, "part": pi, "n_parts": len(groups), "branch_range": [a, b], "steps": n_steps,
                     "n_branches": arr["n_branches"], "no_branch_steps": no_branch if pi == 0 else [],
                     "trunk_tokens": arr["trunk_len"], "full_trunk_tokens": len(trunk), "branch_tokens": arr["branch_tokens"],
                     "tokens": arr["tokens"], "supervised": arr["branch_tokens"], "max_branch": arr["max_branch"],
                     "verified_all_steps": bool(verify_all_steps), **norm_info},
        })
    assert sum(r["branch_tokens"] for r in rows) == sum(lens)  # splitting never loses a supervised token
    return {"ok": True, "src_row": li, "rows": rows, "n_parts": len(groups), "full_tokens": total}


# ---------------------------------------------------------------- arrow ------------------------------------------
def list_column(arrays: list[np.ndarray], dtype: pa.DataType, rows_per_chunk: int = 128) -> pa.ChunkedArray:
    chunks = []
    for s in range(0, len(arrays), rows_per_chunk):
        part = arrays[s: s + rows_per_chunk]
        offsets = np.zeros(len(part) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum([len(a) for a in part])
        values = pa.array(np.concatenate(part) if part else np.zeros(0, dtype=dtype.to_pandas_dtype()), type=dtype)
        chunks.append(pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values))
    return pa.chunked_array(chunks, type=pa.list_(dtype))


def build_dataset(cols: dict[str, list]) -> Dataset:
    n = len(cols["input_ids"])
    schema = FEATURES.arrow_schema
    arrays = {
        "input_ids": list_column(cols["input_ids"], pa.int32()),
        "attention_mask": list_column([np.ones(len(a), dtype=np.int32) for a in cols["input_ids"]], pa.int32()),
        "labels": list_column(cols["labels"], pa.int64()),
        "position_ids": list_column(cols["position_ids"], pa.int32()),
        "tree_forks": list_column(cols["tree_forks"], pa.int64()),
        "tree_branch_lens": list_column(cols["tree_branch_lens"], pa.int64()),
        "trunk_len": pa.array(cols["trunk_len"], type=pa.int64()),
        "images": pa.nulls(n), "videos": pa.nulls(n), "audios": pa.nulls(n),
    }
    table = pa.Table.from_arrays([arrays[f.name] for f in schema], schema=schema)
    return Dataset(InMemoryTable(table), info=DatasetInfo(features=FEATURES))


# ---------------------------------------------------------------- main -------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="sharegpt jsonl ({'conversations': [{role, content}, ...]})")
    ap.add_argument("--out", required=True, help="cache dir (DatasetDict.save_to_disk)")
    ap.add_argument("--name", required=True, help="dataset name (manifest / logs)")
    ap.add_argument("--model", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--template", default="qwen3_5_text")
    ap.add_argument("--cutoff", type=int, default=262144, help="every single branch (and the trunk) must fit this")
    ap.add_argument("--max-tree-tokens", type=int, default=0, help="drop rows whose tree row exceeds this (0 = unlimited)")
    ap.add_argument("--split-tokens", type=int, default=0,
                    help="a tree row longer than this is split into consecutive-branch sub-rows (trunk[:fork_last] ++ branches), "
                         "each <= this many tokens; exact (a branch never sees anything past its fork). 0 = never split")
    ap.add_argument("--solved-only", default=None,
                    help="jsonl whose lines select the subset of --input to build (exact line match; --input order kept)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N (selected) conversations")
    ap.add_argument("--bad-pair", choices=["mask", "drop"], default="mask",
                    help="malformed action JSON / oversize pair: mask = keep in trunk, loss=0, no branch (th3k_mask policy); drop = drop row")
    ap.add_argument("--overlong-conv", choices=["keep", "drop"], default="keep",
                    help="drop = prepare_terminal_horizon --no-chunk --overlong drop: think-inline conversation estimate > --cutoff -> drop")
    ap.add_argument("--pad-token-id", type=int, default=None, help="asserted == tokenizer.pad_token_id; recorded in manifest")
    ap.add_argument("--verify-first", type=int, default=3, help="every-step from-scratch re-encode on the first N conversations")
    ap.add_argument("--verify-frac", type=float, default=0.01, help="... plus this fraction of the rest (seeded per row)")
    ap.add_argument("--verify-all", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    max_tree = args.max_tree_tokens

    tok, template = load_template(args.model, args.template, args.cutoff)
    log(f"[{args.name}] template {type(template).__name__} enable_thinking={template.enable_thinking} "
        f"preserve_thinking={template.preserve_thinking} thought_words={template.thought_words} "
        f"eos={tok.eos_token!r}/{tok.eos_token_id} pad={tok.pad_token!r}/{tok.pad_token_id}; cutoff {args.cutoff} "
        f"max_tree {max_tree or 'unlimited'} bad_pair={args.bad_pair} overlong_conv={args.overlong_conv} workers={args.workers}", t0)
    if args.pad_token_id is not None:
        assert args.pad_token_id == tok.pad_token_id, (args.pad_token_id, tok.pad_token_id)

    select: set[str] | None = None
    if args.solved_only:
        with open(args.solved_only, "rb") as f:
            select = {hashlib.sha1(line.rstrip(b"\n")).hexdigest() for line in f}
        log(f"[{args.name}] subset: {len(select)} distinct lines from {args.solved_only}", t0)

    def jobs():
        n_sel = 0
        with open(args.input, "rb") as f:
            for li, raw in enumerate(f):
                if select is not None and hashlib.sha1(raw.rstrip(b"\n")).hexdigest() not in select:
                    continue
                n_sel += 1
                if args.limit and n_sel > args.limit:
                    return
                yield (n_sel - 1, li, raw.decode("utf-8"))

    cols = {k: [] for k in ("input_ids", "labels", "position_ids", "tree_forks", "tree_branch_lens", "trunk_len")}
    rows_meta: list[dict] = []
    dropped: list[dict] = []
    reasons = Counter()
    n_read = n_verified = 0
    tot_trunk = tot_branch = tot_branches = tot_no_branch = 0
    init = (args.model, args.template, args.cutoff, max_tree, args.bad_pair, args.overlong_conv, args.verify_first, args.verify_frac,
            args.verify_all, args.seed, args.split_tokens)
    n_split_convs = 0
    last_log = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
        for res in pool.imap(process_conv, jobs(), chunksize=1):
            n_read += 1
            if not res["ok"]:
                dropped.append({"src_row": res["src_row"], "reason": res["reason"], "detail": res["detail"]})
                reasons[res["reason"]] += 1
                if len(dropped) <= 5 or len(dropped) % 100 == 0:
                    log(f"[{args.name}] src row {res['src_row']}: DROPPED {res['reason']} {res['detail']}", t0)
                continue
            n_split_convs += res["n_parts"] > 1
            for r in res["rows"]:
                for k in ("input_ids", "labels", "position_ids", "tree_forks", "tree_branch_lens", "trunk_len"):
                    cols[k].append(r[k])
                m = r["meta"]
                m["row"] = len(rows_meta)
                rows_meta.append(m)
                n_verified += m["verified_all_steps"] and m["part"] == 0
                tot_trunk += m["trunk_tokens"]; tot_branch += m["branch_tokens"]; tot_branches += m["n_branches"]; tot_no_branch += len(m["no_branch_steps"])
                if len(rows_meta) <= 5 or len(rows_meta) % 100 == 0 or time.time() - last_log > 300:
                    last_log = time.time()
                    log(f"[{args.name}] row {m['row']} (src {m['src_row']} part {m['part'] + 1}/{m['n_parts']}): {m['steps']:3d} steps, "
                        f"trunk {m['trunk_tokens']:7d}, {m['n_branches']} branches {m['branch_tokens']:7d} tok, total {m['tokens']:7d}, "
                        f"no-branch {len(m['no_branch_steps'])}, verified_all={m['verified_all_steps']} | read {n_read} built "
                        f"{len(rows_meta)} dropped {len(dropped)} tokens {tot_trunk + tot_branch:,}", t0)

    n_built = len(rows_meta)
    n_convs = len({r["src_row"] for r in rows_meta})
    assert n_convs + len(dropped) == n_read
    log(f"[{args.name}] read {n_read} conversations: built {n_convs} ({n_split_convs} split by --split-tokens {args.split_tokens or 'off'} "
        f"-> {n_built} rows), dropped {len(dropped)} {dict(reasons)}", t0)
    if select is not None:
        assert n_read == (min(len(select), args.limit) if args.limit else len(select)), (n_read, len(select))
    log(f"[{args.name}] ASSERT OK: encode_oneturn(full conv) prompt == trunk[:fork_last] and response == last branch, and "
        f"arm-A row (strip_think + encode_multiturn discarding_history_cot) == trunk[:fork_last] ++ last branch on every row; "
        f"the same for EVERY step k on {n_verified}/{n_convs} conversations ({n_verified / max(n_convs, 1):.1%}); "
        f"len == trunk + sum(branches), supervised == sum(branches), labels == ids on branches on every row; "
        f"every branch and trunk <= {args.cutoff}; tree <= {max_tree or 'unlimited'}", t0)
    over = [r for r in rows_meta if r["tokens"] > args.cutoff]
    log(f"[{args.name}] rows whose tree row exceeds cutoff {args.cutoff}: {len(over)}/{n_built} "
        f"({sum(r['tokens'] for r in over):,} tokens; consumer must filter or raise the budget)", t0)
    masked_rows = sum(1 for r in rows_meta if r["no_branch_steps"])
    log(f"[{args.name}] bad-action turns {sum(r['bad_action_turns'] for r in rows_meta)}, oversize pairs "
        f"{sum(r['oversize_pairs'] for r in rows_meta)} -> {masked_rows} rows with >=1 loss=0 step (no branch)", t0)
    sup = sorted(r["supervised"] for r in rows_meta)
    tt = sorted(r["tokens"] for r in rows_meta)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    log(f"[{args.name}] tokens: trunk {tot_trunk:,} + branches {tot_branch:,} = {tot_trunk + tot_branch:,}; "
        f"{tot_branches} branches, {tot_no_branch} loss=0 steps without branch", t0)
    log(f"[{args.name}] tree tokens/row p50/p90/max {q(tt, .5):,}/{q(tt, .9):,}/{tt[-1] if tt else 0:,}; "
        f"supervised/row p50/p90/max {q(sup, .5):,}/{q(sup, .9):,}/{sup[-1] if sup else 0:,}", t0)

    ds = build_dataset(cols)
    assert ds.features == FEATURES, ds.features
    os.makedirs(args.out, exist_ok=True)
    DatasetDict({"train": ds}).save_to_disk(args.out)
    manifest = {
        "name": args.name, "input": args.input, "solved_only": args.solved_only, "model": args.model, "template": args.template,
        "cutoff": args.cutoff, "max_tree_tokens": max_tree, "bad_pair": args.bad_pair, "limit": args.limit,
        "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
        "read": n_read, "conversations": n_convs, "split_tokens": args.split_tokens, "split_conversations": n_split_convs,
        "rows": n_built, "dropped": len(dropped), "dropped_reasons": dict(reasons),
        "tokens": tot_trunk + tot_branch, "trunk_tokens": tot_trunk, "branch_tokens": tot_branch, "supervised_tokens": tot_branch,
        "n_branches": tot_branches, "no_branch_steps": tot_no_branch, "rows_verified_all_steps": n_verified,
        "rows_over_cutoff": len(over), "tokens_in_rows_over_cutoff": sum(r["tokens"] for r in over),
        "bad_action_turns": sum(r["bad_action_turns"] for r in rows_meta), "oversize_pairs": sum(r["oversize_pairs"] for r in rows_meta),
        "overlong_conv": args.overlong_conv,
        "wall_clock_s": round(time.time() - t0, 1),
        "per_row": rows_meta, "dropped_rows": dropped,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    log(f"[{args.name}] cache: {len(ds)} rows -> {args.out} (+ manifest.json); wall-clock {time.time() - t0:.0f}s", t0)
    log("DONE", t0)


if __name__ == "__main__":
    main()
