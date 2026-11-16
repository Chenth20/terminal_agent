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

"""Train/deploy prefix consistency check for multi-turn agent SFT.

For every assistant turn t of a trajectory, the token prefix the model is trained on
(LlamaFactory template, concatenated encode_multiturn pairs) must equal the token prefix the
deployment harness will build for that turn. Two deployment conventions are compared:

  A. HF chat_template with history reasoning KEPT (messages carry `reasoning_content`, and the
     jinja is patched so it does not strip history <think>). This is what the data implies (every
     turn has real reasoning) and what the harness must do at inference.
  B. HF chat_template as shipped (history <think> dropped, only the last turn's is kept).

Also checks: labels are exactly the assistant spans, each span == template-rendered assistant
message, exactly one <think> per span, number of supervised spans == number of assistant turns.

    python coding_sft/checks/check_prefix_consistency.py --data <jsonl> --rows 0 5 --template qwen3_5_text
"""

import argparse
import json
import os
import sys


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments


def split_think(content: str) -> tuple[str, str]:
    i = content.find("</think>")
    if i < 0:
        return "", content
    reasoning = content[:i].split("<think>", 1)[-1].strip("\n")
    return reasoning, content[i + len("</think>") :].lstrip("\n")


def hf_messages(conv: list[dict], keep_history_reasoning: bool) -> list[dict]:
    out = []
    for m in conv:
        if m["role"] == "assistant":
            r, c = split_think(m["content"])
            if keep_history_reasoning:
                out.append({"role": "assistant", "content": c, "reasoning_content": r})
            else:
                out.append({"role": "assistant", "content": m["content"]})
        else:
            out.append(m)
    return out


def hf_prefix_ids(tok, conv, upto_turn: int, keep_history_reasoning: bool, jinja: str) -> list[int]:
    """Deployment prefix for assistant turn `upto_turn` (0-based): messages[: 2*upto_turn+1] + generation prompt."""
    msgs = hf_messages(conv[: 2 * upto_turn + 1], keep_history_reasoning)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=jinja)
    return tok.encode(text, add_special_tokens=False)


def patch_jinja_keep_history_reasoning(jinja: str) -> str:
    # The shipped template only renders <think> for messages after the last user query
    # (loop.index0 > ns.last_query_index). For agent trajectories with reasoning in every turn,
    # the harness must render reasoning for all assistant messages.
    needle = "{%- if loop.index0 > ns.last_query_index %}"
    if needle not in jinja:
        assert "{%- if true %}" in jinja, "unexpected chat template layout"
        print("NOTE: model chat_template already keeps history reasoning (patched); SHIPPED == KEPT below")
    return jinja.replace(needle, "{%- if true %}")


def first_diff(a: list[int], b: list[int], tok) -> tuple[int, str, str]:
    k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return k, tok.decode(a[max(0, k - 20) : k + 20]), tok.decode(b[max(0, k - 20) : k + 20])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--model", default="storage/models/Qwen__Qwen3.5-4B-text")
    ap.add_argument("--template", default="qwen3_5_text")
    ap.add_argument("--max-turns", type=int, default=0, help="only check the first N assistant turns (0 = all)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    data_args = DataArguments(template=args.template, cutoff_len=10**9, enable_thinking=True)
    template = get_template_and_fix_tokenizer(tok, data_args)
    print(
        f"template={args.template} preserve_thinking={template.preserve_thinking} enable_thinking={template.enable_thinking} "
        f"thought_words={template.thought_words!r} efficient_eos={template.efficient_eos} default_system={template.default_system!r}"
    )
    jinja_shipped = tok.chat_template
    jinja_keep = patch_jinja_keep_history_reasoning(jinja_shipped)
    opener = tok.encode(template.thought_words[0], add_special_tokens=False)

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    bad = 0
    for ri in args.rows:
        conv = rows[ri]["conversations"]
        pairs = template.encode_multiturn(tok, conv, system=None, tools=None)
        n_turns = len(pairs)
        ids, labels = [], []  # exactly as SupervisedDatasetProcessor builds it (no cutoff)
        for s, t in pairs:
            ids += s + t
            labels += [IGNORE_INDEX] * len(s) + t
        spans, p = [], 0
        while p < len(labels):
            if labels[p] != IGNORE_INDEX:
                q = p
                while q < len(labels) and labels[q] != IGNORE_INDEX:
                    q += 1
                spans.append((p, q))
                p = q
            else:
                p += 1
        ok_spans = len(spans) == n_turns == len(conv) // 2
        print(
            f"\n=== row {ri} (src_row={rows[ri].get('src_row')}): {len(conv)} msgs, {n_turns} assistant turns, "
            f"{len(ids)} tokens, supervised {sum(l != IGNORE_INDEX for l in labels)}  spans_ok={ok_spans}"
        )
        bad += not ok_spans

        turns = range(n_turns) if not args.max_turns else range(min(n_turns, args.max_turns))
        mm_keep = mm_ship = 0
        d_keep = d_ship = None
        for t in turns:
            train_prefix = ids[: spans[t][0]]
            span_txt = tok.decode(ids[spans[t][0] : spans[t][1]])
            exp = template.format_assistant.apply(content=conv[2 * t + 1]["content"])[0]
            if span_txt != exp:
                bad += 1
                print(f"  turn {t}: supervised span != rendered assistant message\n    got: {span_txt[:120]!r}\n    exp: {exp[:120]!r}")
            if span_txt.count("<think>") != 1:
                bad += 1
                print(f"  turn {t}: {span_txt.count('<think>')} <think> in span")
            if not tok.decode(train_prefix).endswith("<|im_start|>assistant\n"):
                bad += 1
                print(f"  turn {t}: train prefix does not end with assistant header: {tok.decode(train_prefix)[-60:]!r}")
            # HF generation prompt ends with '<think>\n', which in LF is the first supervised tokens.
            dep = hf_prefix_ids(tok, conv, t, True, jinja_keep)
            if train_prefix + opener != dep:
                mm_keep += 1
                d_keep = d_keep or (t, *first_diff(train_prefix + opener, dep, tok))
            dep = hf_prefix_ids(tok, conv, t, False, jinja_shipped)
            if train_prefix + opener != dep:
                mm_ship += 1
                d_ship = d_ship or (t, *first_diff(train_prefix + opener, dep, tok))
        print(f"  prefix vs HF template, history reasoning KEPT   : {mm_keep}/{len(turns)} turns mismatch")
        if d_keep:
            bad += 1
            print(f"    first diff turn {d_keep[0]} tok {d_keep[1]}:\n      train : {d_keep[2]!r}\n      deploy: {d_keep[3]!r}")
        print(f"  prefix vs HF template as SHIPPED (history dropped): {mm_ship}/{len(turns)} turns mismatch")
        if d_ship:
            print(f"    first diff turn {d_ship[0]} tok {d_ship[1]}:\n      train : {d_ship[2]!r}\n      deploy: {d_ship[3]!r}")

    print(f"\nproblems={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
