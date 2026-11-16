#!/usr/bin/env bash
# Qwen/Qwen3.5-4B (VLM checkpoint)  ->  Qwen__Qwen3.5-4B-text  (what every training yaml points at)
#
#   bash deploy/model/make_text_only.sh <VLM dir> <text-only dst dir> [--verify]
#   e.g. bash deploy/model/make_text_only.sh ${ST}/models/Qwen__Qwen3.5-4B ${ST}/models/Qwen__Qwen3.5-4B-text --verify
#
# Two steps, both are needed (see coding_sft/docs/SUMMARY.md 1 "model" / "chat template"):
#   1. export the language model as Qwen3_5ForCausalLM (model_type qwen3_5_text):
#      - drops the 334M vision tower and 121M MTP head (never used in text SFT, but ZeRO-3 would shard them)
#      - the text class accepts loss kwargs -> true token-mean loss across the effective batch
#        (the VLM class forces per-micro-batch mean -> every trajectory weighted equally, wrong)
#      - language weights are bit-identical; --verify compares logits + greedy decode on a GPU
#   2. patch chat_template.jinja: the shipped template renders <think> only for turns AFTER the last
#      user message, i.e. it drops history reasoning at inference while training keeps it in every
#      assistant turn. One-line change `{%- if loop.index0 > ns.last_query_index %}` -> `{%- if true %}`
#      makes train prefix == deploy prefix token-for-token. Original is kept as chat_template.jinja.orig.
#      The inference harness must then pass reasoning_content (or keep <think> in content) for history turns.
#
# Same procedure for 9B: Qwen__Qwen3.5-9B -> Qwen__Qwen3.5-9B-text.
set -euo pipefail
SRC=${1:?VLM checkpoint dir (huggingface-cli download Qwen/Qwen3.5-4B --local-dir ...)}
DST=${2:?destination dir}
VERIFY=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

[ -f "$SRC/config.json" ] || { echo "no config.json in $SRC"; exit 1; }
python - "$SRC" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]+'/config.json'))
assert 'Qwen3_5ForConditionalGeneration' in c.get('architectures',[]), f"expected the VLM checkpoint, got {c.get('architectures')}"
PY

echo "== 1/2 export language model (Qwen3_5ForCausalLM, bf16, 5GB shards)"
python "$REPO/coding_sft/prep/export_qwen3_5_text_only.py" --src "$SRC" --dst "$DST" ${VERIFY:+--verify}

echo "== 2/2 patch chat template (history <think> rendered for every assistant turn)"
T="$DST/chat_template.jinja"
[ -f "$T" ] || { echo "missing $T"; exit 1; }
if grep -q "{%- if loop.index0 > ns.last_query_index %}" "$T"; then
    cp -n "$T" "$T.orig"
    sed -i 's/{%- if loop.index0 > ns.last_query_index %}/{%- if true %}/' "$T"
    echo "patched; original kept as chat_template.jinja.orig"
elif grep -q "{%- if true %}" "$T"; then
    echo "already patched"
else
    echo "ERROR: template line not found - Qwen changed the template; patch by hand, reference copy: $HERE/chat_template.patched.jinja"; exit 1
fi
# the shipped reference is the exact template every checkpoint in this project was trained/served with
if cmp -s "$T" "$HERE/chat_template.patched.jinja"; then echo "template == project reference (byte-identical)"; else echo "WARN: template differs from $HERE/chat_template.patched.jinja (new Qwen release?) - diff before training"; fi

python - "$DST" <<'PY'
import json,sys,os
d=sys.argv[1]; c=json.load(open(d+'/config.json'))
assert c['architectures']==['Qwen3_5ForCausalLM'] and c['model_type']=='qwen3_5_text', c['architectures']
need=['model.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','chat_template.jinja.orig','generation_config.json']
missing=[f for f in need if not os.path.exists(os.path.join(d,f))]
assert not missing, f"missing {missing}"
print("ok:", d, "| layers", c['num_hidden_layers'], "| hidden", c['hidden_size'], "| dtype", c.get('dtype'))
PY
echo "done. point model_name_or_path at $DST (template qwen3_5_text)."