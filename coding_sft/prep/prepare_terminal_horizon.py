# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare TerminalHorizon-3K (Terminus-2 terminal-agent trajectories) for LlamaFactory SFT.

Input rows look like ``{"conversations": [{"role": "user"|"assistant", "content": str}, ...]}``.
Assistant content is ``<think>...</think>{json action}``.

What this script does, and why (each item is a known failure mode of agent-trajectory SFT):

1. Role alternation / schema checks. LlamaFactory's sharegpt converter silently drops a row that
   does not strictly alternate user/assistant and end with assistant; we validate up front so the
   drop count is visible instead of hidden in a rank-0 warning.
2. Whitespace / thinking-block normalisation. User content is ``strip()``-ed because the Qwen3.5
   chat template renders ``content|trim`` at inference, while LlamaFactory encodes the string
   verbatim; without this every training prefix differs from the deployment prefix. The dataset writes ``<think>X</think>{...}`` with no newlines; the
   ``qwen3_5`` ReasoningTemplate uses ``("<think>\\n", "\\n</think>\\n\\n")`` as its thought words.
   If the strings differ the template cannot locate the thought and *prepends an empty thought*,
   so the model would learn ``<think>\\n\\n</think>\\n\\n<think>X</think>{...}`` (double think).
   We rewrite every assistant turn to exactly the template's layout.
3. Action JSON validation. Assistant actions are kept verbatim when they parse as JSON; actions
   that only parse leniently (literal control chars / stray text around ``{...}``) are
   re-serialised; unparseable ones are dropped together with their observation and force a
   window boundary (so a window never skips a step of the interaction). The same applies to
   (observation, action) pairs that alone exceed ``--max-tokens`` - LF would otherwise truncate
   the action mid-JSON.
4. Long-horizon chunking. Trajectories are 18-806 turns and ~140K tokens at the median; LF's
   ``cutoff_len`` truncation keeps only the first window and discards the rest. We cut each
   trajectory into windows of at most ``--max-tokens`` tokens made of contiguous (user, assistant)
   pairs. Continuation windows re-use the first user turn (agent instructions + task) merged into
   their first terminal observation, so each window is a valid alternating conversation that still
   carries the task. Windows never overlap, so no assistant turn is supervised twice.
5. ``task_complete`` only at the end. The final assistant turn of the *original* trajectory is
   the only one allowed to contain ``"task_complete": true``; windows that end mid-trajectory
   are left as is (their last turn naturally has task_complete false). Violations are counted.
6. No silent tail loss in ``--no-chunk`` mode. A trajectory that does not fit ``--max-tokens`` is
   isolated into ``<name>_overlong.jsonl`` (``--overlong drop``, default) instead of being cut: a
   truncated trajectory would keep only exploration and lose implementation / verification /
   delivery. ``--overlong truncate`` restores the old prefix-keeping behaviour.

Outputs ``<out_dir>/<name>.jsonl`` (role/content sharegpt, plus ``src_row`` / ``target_tokens``
meta that LlamaFactory ignores) and ``<out_dir>/dataset_info.json``.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from tokenizers import Tokenizer


THINK_RE = re.compile(r"^\s*<think>(.*?)</think>(.*)$", re.DOTALL)
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"
# Prepended to the first observation of a continuation window (see chunk()).
CONTINUATION_SEP = (
    "\n\nThe task above is already in progress. Earlier commands have been executed; "
    "continue from the current terminal state shown below.\n\n"
)


@dataclass
class Stats:
    rows: int = 0
    think_capped_turns: int = 0
    dropped_schema: int = 0
    dropped_bad_turn: int = 0
    dropped_overlong: int = 0
    bad_turns: int = 0
    oversize_pairs: int = 0
    trajs_with_bad: int = 0
    truncated_trajs: int = 0
    tail_trajs: int = 0
    cut_at_bad_pair: int = 0
    masked_turns: int = 0
    kept_pairs: int = 0
    total_pairs: int = 0
    assistant_turns: int = 0
    assistant_with_think: int = 0
    empty_think: int = 0
    task_complete_mid: int = 0
    task_complete_missing_end: int = 0
    windows: int = 0
    windows_single_turn_dropped: int = 0
    window_tokens: list = field(default_factory=list)
    target_tokens: list = field(default_factory=list)
    think_tokens: int = 0
    action_tokens: int = 0
    obs_tokens: int = 0
    multi_window_trajs: int = 0


def parse_action(action: str) -> Optional[tuple[dict, bool]]:
    """Parse a Terminus action. Returns (obj, was_strict) or None."""
    try:
        return json.loads(action), True
    except json.JSONDecodeError:
        pass
    # Lenient: literal control chars inside strings (the harness accepts them), or stray text
    # around the outermost {...}.
    i, j = action.find("{"), action.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(action[i : j + 1], strict=False), False
    except json.JSONDecodeError:
        return None


def normalize_assistant(content: str) -> Optional[tuple[str, bool]]:
    """Return (normalised_content, has_nonempty_think) or None if the turn is malformed."""
    m = THINK_RE.match(content)
    if m is None:
        thought, action = "", content
    else:
        thought, action = m.group(1), m.group(2)
        if "<think>" in action or "</think>" in action:
            return None  # nested / duplicated think tags
    action = action.strip()
    parsed = parse_action(action)
    if parsed is None:
        return None
    obj, strict = parsed
    if not isinstance(obj, dict) or "commands" not in obj:
        return None
    if not strict:
        # re-serialise so the supervised target is valid JSON (literal newlines -> \n etc.)
        action = json.dumps(obj, ensure_ascii=False, indent=2)
    thought = thought.strip()
    return THINK_OPEN + thought + THINK_CLOSE + action, bool(thought)


def validate_schema(conv: list[dict]) -> bool:
    if len(conv) < 2 or len(conv) % 2 != 0:
        return False
    for i, m in enumerate(conv):
        if not isinstance(m, dict) or set(m) != {"role", "content"} or not isinstance(m["content"], str):
            return False
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return False
        if not m["content"].strip():
            return False
    return True


def chunk(conv: list[dict], tok_lens: list[int], max_tokens: int, bad_pairs: set[int], stats: Stats) -> list[list[dict]]:
    """Split into windows of contiguous (user, assistant) pairs, each <= max_tokens.

    Pair k covers conv[2k], conv[2k+1]; pair 0 is (task prompt, first action). Pairs in
    ``bad_pairs`` (malformed action or oversize) are dropped and force a window boundary, so no
    window ever contains a gap in the interaction history.
    tok_lens[i] is the token count of conv[i] content only; template overhead is ``slack``/turn.
    """
    slack = 8
    head_len = tok_lens[0] + slack
    pair_len = [tok_lens[i] + tok_lens[i + 1] + 2 * slack for i in range(0, len(conv), 2)]

    windows: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    for k in range(len(pair_len)):
        if k in bad_pairs:
            if cur:
                windows.append(cur)
            cur, cur_len = [], 0
            continue
        start_len = pair_len[k] if k == 0 else head_len + pair_len[k]
        if cur and cur_len + pair_len[k] > max_tokens:
            windows.append(cur)
            cur, cur_len = [k], start_len
        elif not cur:
            cur, cur_len = [k], start_len
        else:
            cur.append(k)
            cur_len += pair_len[k]
    if cur:
        windows.append(cur)

    out = []
    for w in windows:
        if w[0] == 0:
            msgs, toks = [], 0
        else:
            # Continuation window: the task prompt (conv[0]) and the first terminal observation of
            # the window are both user content; merge them into ONE user turn so the window still
            # alternates user/assistant (LF drops rows with two consecutive user turns).
            first_user = conv[2 * w[0]]
            msgs = [{"role": "user", "content": conv[0]["content"] + CONTINUATION_SEP + first_user["content"]}]
            toks = tok_lens[0] + tok_lens[2 * w[0]]
            msgs.append(conv[2 * w[0] + 1])
            toks += tok_lens[2 * w[0] + 1]
            w = w[1:]
        for k in w:
            msgs += [conv[2 * k], conv[2 * k + 1]]
            toks += tok_lens[2 * k] + tok_lens[2 * k + 1]
        out.append(msgs)
        stats.window_tokens.append(toks)
    if len(out) > 1:
        stats.multi_window_trajs += 1
    return out


def whole_trajectory(conv, tok_lens, max_tokens, bad_pairs, stats, overlong: str, bad_pair: str):
    """One sample = the whole trajectory (no chunking).

    ``overlong`` decides what happens when the trajectory does not fit ``max_tokens``:
      * ``drop``: return "overlong" so the caller isolates the whole trajectory (no silent truncation).
      * ``truncate``: keep the longest PREFIX of complete (observation, action) pairs that fits
        (loses implementation / verification / delivery).
      * ``tail``: keep the task prompt + the longest SUFFIX of complete pairs that fits; the first
        observation of the suffix is merged with the task prompt like a chunk() continuation window
        (loses early exploration, keeps the delivery phase).
    ``bad_pair`` decides what happens with malformed actions (unparseable JSON):
      * ``drop``: isolate the whole trajectory (returns "bad").
      * ``cut``: the sample ends right before the first malformed action.
      * ``mask``: keep the malformed action verbatim in the context (the harness' error reply and the
        model's fix depend on it) but flag it ``loss: 0`` so it is never a training target.
      * ``keep``: keep it verbatim and supervise it like any other turn (no flag).
    """
    slack = 8
    if bad_pair == "drop" and bad_pairs:
        return "bad"
    if bad_pair == "cut":
        n_pairs = min(bad_pairs) if bad_pairs else len(conv) // 2
        stats.cut_at_bad_pair += bool(bad_pairs) and n_pairs > 0
    else:
        n_pairs = len(conv) // 2
    if n_pairs == 0:
        return None

    pair_len = [tok_lens[2 * k] + tok_lens[2 * k + 1] + 2 * slack for k in range(n_pairs)]
    if sum(pair_len) <= max_tokens:
        first, last = 0, n_pairs  # whole thing fits
    elif overlong == "drop":
        return "overlong"
    elif overlong == "truncate":
        toks, last = 0, 0
        while last < n_pairs and toks + pair_len[last] <= max_tokens:
            toks += pair_len[last]
            last += 1
        first = 0
        stats.truncated_trajs += 1
    else:  # tail: task prompt (pair 0's user turn) + suffix
        head = tok_lens[0] + slack
        toks, first = head, n_pairs
        while first > 0 and toks + pair_len[first - 1] <= max_tokens:
            toks += pair_len[first - 1]
            first -= 1
        last = n_pairs
        if first == 0:  # suffix happens to be everything (cannot be, but keep the invariant)
            pass
        stats.tail_trajs += 1

    if last - first <= 0:
        return None
    if first == 0:
        msgs = []
        for k in range(last):
            msgs += [conv[2 * k], conv[2 * k + 1]]
    else:
        first_user = conv[2 * first]
        msgs = [{"role": "user", "content": conv[0]["content"] + CONTINUATION_SEP + first_user["content"]}, conv[2 * first + 1]]
        for k in range(first + 1, last):
            msgs += [conv[2 * k], conv[2 * k + 1]]
    if bad_pair == "mask":
        msgs = [dict(m) for m in msgs]
        for mi in range(1, len(msgs), 2):
            k = first + mi // 2
            if k in bad_pairs:
                msgs[mi]["loss"] = 0
                stats.masked_turns += 1
    toks = sum(tok_lens[2 * k] + tok_lens[2 * k + 1] + 2 * slack for k in range(first, last)) + (tok_lens[0] if first else 0)
    stats.kept_pairs += last - first
    stats.total_pairs += len(conv) // 2
    stats.window_tokens.append(toks)
    return [msgs]


CONFIRM_PROMPT = "Are you sure you want to mark the task as complete"


def check_task_complete(windows: list[list[dict]], stats: Stats) -> None:
    """Terminus-2 ends with: action{task_complete:true} -> user "Are you sure...?" -> confirming action.

    So `task_complete: true` is expected in the last action and in the one right before the confirmation
    prompt. Any other occurrence means the agent declared completion and then kept working
    (reconsidered after the prompt) - legal, but counted so it can be audited.
    """
    for wi, w in enumerate(windows):
        for mi in range(1, len(w), 2):
            is_last = wi == len(windows) - 1 and mi == len(w) - 1
            flagged = '"task_complete": true' in w[mi]["content"]
            followed_by_confirm = mi + 1 < len(w) and CONFIRM_PROMPT in w[mi + 1]["content"]
            if flagged and not is_last and not followed_by_confirm:
                stats.task_complete_mid += 1
            if is_last and not flagged:
                stats.task_complete_missing_end += 1


def accumulate_token_stats(windows: list[list[dict]], tok, stats: Stats) -> list[dict]:
    """Per-window meta (total / target tokens) + global think/action/observation token split.

    Message token counts come from the first pass (``_ntok``, set in process_row); only the think prefix
    of every assistant turn is encoded here (a second full encode of the corpus was ~75% of the run time).
    """
    metas = []
    for w in windows:
        tgt = 0
        for mi, m in enumerate(w):
            n = m["_ntok"] if "_ntok" in m else len(tok.encode(m["content"]).ids)
            if mi % 2 == 0:
                stats.obs_tokens += n
            else:
                tgt += n
                i = m["content"].find(THINK_CLOSE)
                th = len(tok.encode(m["content"][: i + len(THINK_CLOSE)]).ids) if i >= 0 else 0
                stats.think_tokens += th
                stats.action_tokens += n - th
        stats.target_tokens.append(tgt)
        metas.append({"turns": len(w), "target_tokens": tgt})
    return metas


def merge_stats(dst: Stats, src: Stats) -> None:
    for k, v in vars(src).items():
        if isinstance(v, list):
            getattr(dst, k).extend(v)
        else:
            setattr(dst, k, getattr(dst, k) + v)


_W: dict = {}  # worker globals (tokenizer + args), set by _init_worker


def _init_worker(tokenizer_path: str, args) -> None:
    _W["tok"] = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
    _W["args"] = args


def process_row(item: tuple[int, str]):
    """One source row -> ("traj", li, windows) | ("overlong"|"badaction", li, conv, ntok) | ("drop", reason);
    plus the per-row Stats delta. Runs in a worker process (tokenizer + args from _init_worker)."""
    li, line = item
    tok, args = _W["tok"], _W["args"]
    stats = Stats()
    stats.rows = 1
    conv = json.loads(line).get("conversations")
    if not isinstance(conv, list) or not validate_schema(conv):
        stats.dropped_schema += 1
        return ("drop", "schema"), stats

    fixed: list[dict] = []
    bad_pairs: set[int] = set()
    for i, m in enumerate(conv):
        if m["role"] == "user":
            # The model's HF chat_template renders `content|trim`; terminal screens end with ~40
            # blank lines. Strip here so the training prefix == the deployment prefix token-for-token
            # (verified by check_prefix_consistency.py).
            fixed.append({"role": "user", "content": m["content"].strip()})
            continue
        res = normalize_assistant(m["content"])
        if res is None:
            bad_pairs.add(i // 2)
            stats.bad_turns += 1
            fixed.append({"role": "assistant", "content": m["content"]})  # placeholder, never emitted
            continue
        content, has_think = res
        stats.assistant_turns += 1
        stats.assistant_with_think += has_think
        stats.empty_think += not has_think
        fixed.append({"role": "assistant", "content": content})

    tok_lens = [len(e.ids) for e in tok.encode_batch([m["content"] for m in fixed])]
    for m, n in zip(fixed, tok_lens):
        m["_ntok"] = n
    for k in range(0, len(fixed), 2):
        if k // 2 not in bad_pairs and tok_lens[0] + tok_lens[k] + tok_lens[k + 1] + 24 > args.max_tokens:
            bad_pairs.add(k // 2)
            stats.oversize_pairs += 1
    if len(bad_pairs) * 2 >= len(fixed):
        stats.dropped_bad_turn += 1
        return ("drop", "all_pairs_bad"), stats
    stats.trajs_with_bad += bool(bad_pairs)
    if args.no_chunk:
        windows = whole_trajectory(fixed, tok_lens, args.max_tokens, bad_pairs, stats, args.overlong, args.bad_pair)
        if windows is None:
            stats.dropped_bad_turn += 1
            return ("drop", "no_chunk_first_pair_bad"), stats
        if windows in ("bad", "overlong"):
            for m in fixed:
                m.pop("_ntok", None)
            if windows == "bad":
                stats.dropped_bad_turn += 1
                return ("badaction", li, fixed, sum(tok_lens) + 8 * len(fixed)), stats
            stats.dropped_overlong += 1
            return ("overlong", li, fixed, sum(tok_lens) + 8 * len(fixed)), stats
    else:
        windows = chunk(fixed, tok_lens, args.max_tokens, bad_pairs, stats)
    if args.bad_pair == "mask":
        # emit the flag on EVERY message: `datasets` infers one arrow struct type for the column, and a
        # mixed struct<role,content,loss> / struct<role,content> cannot be cast (build_tokenized failed on
        # the qwen38 flash set where the first `loss: 0` only appears ~1300 rows in).
        for w in windows:
            for m in w:
                m.setdefault("loss", 1)
        if args.think_cap > 0:
            # think-cap: over-long reasoning turns are kept as context (the student must still condition on
            # them) but not supervised, so the model does not learn to emit 30k-char thoughts.
            for w in windows:
                for m in w:
                    if m["role"] != "assistant" or m.get("loss") == 0:
                        continue
                    tm = THINK_RE.match(m["content"])
                    if tm is not None and len(tm.group(1)) > args.think_cap:
                        m["loss"] = 0
                        stats.think_capped_turns += 1
    check_task_complete(windows, stats)
    metas = accumulate_token_stats(windows, tok, stats)
    for w in windows:
        for m in w:
            m.pop("_ntok", None)
    return ("traj", li, windows, metas), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/storage/openpsi/users/chen/code_data/terminal_horizon/TerminalHorizon-3K/data/TerminalHorizon-3K-SFT.jsonl")
    ap.add_argument("--tokenizer", default="/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B")
    ap.add_argument("--out-dir", default="/storage/openpsi/users/chen/data/terminal_horizon")
    ap.add_argument("--name", default="terminal_horizon_3k")
    ap.add_argument("--max-tokens", type=int, default=32768, help="window budget; set to training cutoff_len")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: only read the first N rows")
    ap.add_argument("--no-chunk", action="store_true",
                    help="one sample per trajectory (long-context training); see --overlong for what happens "
                         "to trajectories that do not fit --max-tokens")
    ap.add_argument("--overlong", choices=["drop", "truncate", "tail"], default="drop",
                    help="--no-chunk only. drop: isolate trajectories > --max-tokens into <name>_overlong.jsonl "
                         "(default). truncate: keep the longest fitting PREFIX. tail: task prompt + longest fitting "
                         "SUFFIX (keeps the delivery phase).")
    ap.add_argument("--bad-pair", choices=["drop", "cut", "mask", "keep"], default="drop",
                    help="--no-chunk only. drop: isolate trajectories containing a malformed action into "
                         "<name>_badaction.jsonl (default). cut: keep the prefix before the first malformed action. "
                         "mask: keep the malformed action in the context verbatim, flag it loss=0 (needs the "
                         "`loss_tag` in dataset_info.json, written automatically). keep: keep it verbatim AND "
                         "supervise it (the model learns the broken output too; ablation baseline).")
    ap.add_argument("--think-cap", type=int, default=0,
                    help="--no-chunk + --bad-pair mask only. Assistant turns whose <think> is longer than N characters "
                         "stay in the context verbatim but get loss=0 (not supervised). 0 = off.")
    ap.add_argument("--val-ratio", type=float, default=0.0, help="hold out this fraction of *trajectories*")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="row-parallel worker processes")
    args = ap.parse_args()

    stats = Stats()
    reasons = Counter()
    trajectories: list[tuple[int, list[list[dict]], list[dict]]] = []  # (source row index, windows, metas)
    overlong: list[tuple[int, list[dict], int]] = []  # (row index, fixed conv, token count)
    badaction: list[tuple[int, list[dict], int]] = []

    def rows():
        with open(args.input, encoding="utf-8") as f:
            for li, line in enumerate(f):
                if args.max_rows and li >= args.max_rows:
                    break
                yield li, line

    from multiprocessing import Pool

    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer, args)) as pool:
        for n_done, (res, st) in enumerate(pool.imap(process_row, rows(), chunksize=4), 1):
            merge_stats(stats, st)
            if res[0] == "drop":
                reasons[res[1]] += 1
            elif res[0] == "badaction":
                reasons["bad_action"] += 1
                badaction.append(res[1:])
            elif res[0] == "overlong":
                reasons["overlong"] += 1
                overlong.append(res[1:])
            else:
                trajectories.append(res[1:])
            if n_done % 1000 == 0:
                print(f"[{time.time() - t0:.0f}s] {n_done} rows", flush=True)
    print(f"[{time.time() - t0:.0f}s] processed {stats.rows} rows with {args.workers} workers", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(trajectories)
    n_val = int(len(trajectories) * args.val_ratio)
    splits = {args.name: trajectories[n_val:]}
    if n_val:
        splits[f"{args.name}_val"] = trajectories[:n_val]

    os.makedirs(args.out_dir, exist_ok=True)
    info = {}
    for split_name, trajs in splits.items():
        path = os.path.join(args.out_dir, f"{split_name}.jsonl")
        n = 0
        with open(path, "w", encoding="utf-8") as fo:
            for ti, (src_row, windows, metas) in enumerate(trajs):
                for wi, (w, meta) in enumerate(zip(windows, metas)):
                    row = {"conversations": w, "traj_id": ti, "src_row": src_row, "window": wi, **meta}
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
        stats.windows += n
        info[split_name] = {
            "file_name": os.path.basename(path),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                     "loss_tag": "loss"},
        }
        print(f"[write] {path}: {n} windows from {len(trajs)} trajectories")

    for tag, isolated, why in (("overlong", overlong, f"> {args.max_tokens} tokens"), ("badaction", badaction, "with a malformed action")):
        if isolated:
            path = os.path.join(args.out_dir, f"{args.name}_{tag}.jsonl")
            with open(path, "w", encoding="utf-8") as fo:
                for src_row, conv, ntok in isolated:
                    fo.write(json.dumps({"conversations": conv, "src_row": src_row, "approx_tokens": ntok}, ensure_ascii=False) + "\n")
            print(f"[write] {path}: {len(isolated)} isolated trajectories {why} (NOT registered for training)")

    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as fo:
        json.dump(info, fo, indent=2)

    wt = sorted(stats.window_tokens)
    tt = sorted(stats.target_tokens)
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    think_rate = stats.assistant_with_think / max(stats.assistant_turns, 1)
    print("---- stats ----")
    print(f"rows read                 : {stats.rows}")
    print(f"dropped (schema)          : {stats.dropped_schema}")
    print(f"dropped (bad action)      : {stats.dropped_bad_turn}  (no-chunk --bad-pair drop: isolated to *_badaction.jsonl)")
    print(f"dropped (overlong)        : {stats.dropped_overlong}  (isolated to *_overlong.jsonl)")
    print(f"bad assistant turns       : {stats.bad_turns} (pair dropped, window split)  oversize pairs: {stats.oversize_pairs}")
    print(f"trajectories with >=1 bad : {stats.trajs_with_bad}")
    print(f"trajectories kept         : {len(trajectories)}  (multi-window: {stats.multi_window_trajs})")
    print(f"windows written           : {stats.windows}")
    if args.no_chunk:
        print(f"no-chunk (overlong={args.overlong}, bad-pair={args.bad_pair}): prefix-truncated {stats.truncated_trajs}, "
              f"tail-windowed {stats.tail_trajs}, cut before a bad action {stats.cut_at_bad_pair}, masked turns "
              f"{stats.masked_turns}; pairs kept {stats.kept_pairs}/{stats.total_pairs} ({stats.kept_pairs / max(stats.total_pairs, 1):.1%})")
    print(f"assistant turns kept      : {stats.assistant_turns}  with-think rate: {think_rate:.3f}  empty-think: {stats.empty_think}")
    print(f"task_complete             : declared-then-continued {stats.task_complete_mid} turns, missing at end {stats.task_complete_missing_end} trajs")
    print(f"window tokens p50/p90/p99/max : {q(wt, 0.5)} / {q(wt, 0.9)} / {q(wt, 0.99)} / {wt[-1] if wt else 0}  (budget {args.max_tokens})")
    print(f"target tokens p50/p90/p99/max : {q(tt, 0.5)} / {q(tt, 0.9)} / {q(tt, 0.99)} / {tt[-1] if tt else 0}  (sum {sum(tt)})")
    tot = max(stats.think_tokens + stats.action_tokens + stats.obs_tokens, 1)
    print(f"token split think/action/obs  : {stats.think_tokens / tot:.1%} / {stats.action_tokens / tot:.1%} / {stats.obs_tokens / tot:.1%}")
    print(f"reasons                   : {dict(reasons)}")
    if think_rate < 0.5:
        print("WARNING: <50% of assistant turns have thinking; use the *_nothink template instead.", file=sys.stderr)


if __name__ == "__main__":
    main()
