#!/usr/bin/env bash
# Publish @1.3.0 for the 87 flattened TerminalHorizon envs: replace hooks.setup_script with
# th_flat_setup_hook.sh (chroot sidecars into the main netns, write /opt/flatten/ready.json, restart
# arca_envd) and re-upload query+oracle, then create/version-create/publish.
#
# Why 1.3.0 and not an in-place edit: published versions are immutable, and the 87 envs keep their
# @1.2.0 spec (image, workdir, timeouts, resources) byte-identical - only the hook changes.
#
# Usage (from coding_arena/):
#   bash code/reg/register_flat_1.3.0.sh            # all flat envs (handoff_meta flat: true)
#   ONLY="task1 task2" bash code/reg/register_flat_1.3.0.sh
#   DRY_RUN=1 bash code/reg/register_flat_1.3.0.sh  # print the patched spec for the first task, register nothing
#
# Idempotent: skips a key whose @1.3.0 is already PUBLISHED (state/1.3.0/<task>.ok records key@ver).
set -euo pipefail
export PATH=/storage/openpsi/users/hechuyi.hcy/tools:$PATH
[[ -f ~/.secrets.env ]] && { set -a; source ~/.secrets.env; set +a; }
HERE=$(cd "$(dirname "$0")" && pwd)
CA_ROOT=$(cd "$HERE/../.." && pwd)
WORK=${TH_WORK:-$CA_ROOT/work}
HOOK=$HERE/th_flat_setup_hook.sh
VER=${VER:-1.3.0}
cd "$WORK"
[[ -f $HOOK ]] || { echo "missing hook: $HOOK" >&2; exit 1; }

# task list: handoff_meta flat:true, deduplicated, only tasks that have a local spec
python3 - > /tmp/flat_tasks_130.txt <<'PY'
import json, os
meta = json.load(open("manifests/handoff_meta.json"))
for t, m in sorted(meta.items()):
    if m.get("flat") and os.path.exists(f"env-specs/{t}.json"):
        print(t)
PY
TASKS=${ONLY:-$(cat /tmp/flat_tasks_130.txt)}
echo "flat tasks: $(wc -l < /tmp/flat_tasks_130.txt), selected: $(echo $TASKS | wc -w)"

patch_spec() {  # task -> env-specs/1.3.0/<task>.json
  python3 - "$1" "$HOOK" <<'PY'
import json, os, sys
task, hook = sys.argv[1], sys.argv[2]
src = json.load(open(f"env-specs/{task}.json"))
src["hooks"]["setup_script"] = open(hook).read()      # full replacement (the flat hook also starts envd)
src.setdefault("metadata", {})["cth_1_3_0"] = ("setup hook starts the flattened sidecars via chroot in the main "
                                               "netns and writes /opt/flatten/ready.json; the platform never runs "
                                               "the image ENTRYPOINT, so the original supervisor never starts")
os.makedirs("env-specs/1.3.0", exist_ok=True)
json.dump(src, open(f"env-specs/1.3.0/{task}.json", "w"), ensure_ascii=False, indent=1)
PY
}

if [[ ${DRY_RUN:-0} == 1 ]]; then
  t=$(echo $TASKS | awk '{print $1}')
  patch_spec "$t"
  python3 - "env-specs/1.3.0/$t.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print("image:", s["runtime"]["image"])
print("hook bytes:", len(s["hooks"]["setup_script"]), "| first line:",
      s["hooks"]["setup_script"].splitlines()[1][:90])
print("metadata keys:", sorted(s["metadata"]))
PY
  echo "DRY_RUN: wrote env-specs/1.3.0/$t.json, nothing registered"
  exit 0
fi

published() {
  env -u ANT_COOKIE arena env version-get "$1" "$VER" --json 2>/dev/null \
    | python3 -c 'import json,sys;sys.exit(0 if json.load(sys.stdin).get("data",{}).get("status")=="PUBLISHED" else 1)'
}

register_one() {
  local t=$1
  local A=(env -u ANT_COOKIE arena)
  local key="cth-th-$t"
  local state="state/$VER"
  [[ -f "$state/$t.ok" ]] && { echo "skip $t"; return 0; }
  published "$key" && { echo "$key@$VER" > "$state/$t.ok"; echo "skip $key@$VER (published)"; return 0; }
  patch_spec "$t"
  local q o
  q=$("${A[@]}" upload "${ASSETS:-/storage/openpsi/users/chen/code_data/terminal_horizon/env_assets}/$t/instruction.md" --json \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"]["uri"])')
  o=$("${A[@]}" upload "${ASSETS:-/storage/openpsi/users/chen/code_data/terminal_horizon/env_assets}/$t/oracle" --json \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"]["uri"])')
  python3 - "$t" "$q" "$o" <<'PY'
import json, sys
t, q, o = sys.argv[1:4]
p = f"env-specs/1.3.0/{t}.json"; s = json.load(open(p))
s["query"]["uri"] = q; s["oracle"]["uri"] = o
json.dump(s, open(p, "w"), ensure_ascii=False, indent=1)
PY
  "${A[@]}" env create --key "$key" --name "TerminalHorizon: $t (cth)" \
      --tags "source:terminal-horizon,domain:terminal,chen,th-flat130" --json >/dev/null 2>&1 || true
  "${A[@]}" env version-create "$key" "$VER" --file "env-specs/1.3.0/$t.json" --json \
    | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("ok") else (print(d.get("error"),file=sys.stderr) or 1))' \
    || "${A[@]}" env version-update "$key" "$VER" --file "env-specs/1.3.0/$t.json" --json \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("ok") else (print(d.get("error"),file=sys.stderr) or 1))' \
    || { echo "FAILED version $key@$VER"; return 1; }
  "${A[@]}" env publish "$key" "$VER" --yes --json \
    | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("ok") else (print(d.get("error"),file=sys.stderr) or 1))' \
    || { echo "FAILED publish $key@$VER"; return 1; }
  published "$key" || { echo "FAILED verify $key@$VER"; return 1; }
  echo "$key@$VER" > "$state/$t.ok"
  echo "published $key@$VER"
}

mkdir -p "state/$VER" logs
fail=0
for t in $TASKS; do
  register_one "$t" 2>&1 | tee -a "logs/register_$VER.log" || fail=1
done
[[ $fail == 0 ]] || { echo "some envs failed; re-run to retry" >&2; exit 1; }

echo "done. Rebuild the benchmark so these envs point at @$VER (env_refs read from state/$VER/*.ok, see register_batch.sh tail)."
