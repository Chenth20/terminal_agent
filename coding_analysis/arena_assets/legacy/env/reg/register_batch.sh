#!/usr/bin/env bash
# Batch-register TerminalHorizon Envs (create -> version-create -> publish) in parallel, then publish a Benchmark.
#
# Usage:
#   bash register_batch.sh <env_version> <bench_key> <bench_version> [tasks_file]
#   e.g. bash register_batch.sh 1.1.0 cth-th-500 1.0.0 tasks500.txt
#
# Idempotent: a task whose "<env_version> PUBLISHED" marker exists in state/ is skipped,
# so the script can be re-run after a partial failure.
set -euo pipefail
export PATH=/storage/openpsi/users/hechuyi.hcy/tools:$PATH
HERE=$(cd "$(dirname "$0")" && pwd)
CA_ROOT=$(cd "$HERE/../.." && pwd)
WORK=${TH_WORK:-$CA_ROOT/work}                     # manifests/ benchmarks/ env-specs/ state/ logs/
export ASSETS=${TH_ASSETS:-/storage/openpsi/users/chen/code_data/terminal_horizon/env_assets}
cd "$WORK"

VER=${1:?env version}
BKEY=${2:?benchmark key}
BVER=${3:?benchmark version}
TASKS_FILE=${4:-manifests/tasks500.txt}
REWARD=${REWARD_REF:-tb2-reward@4.0.0}
PAR=${PAR:-8}
TAG=${BATCH_TAG:-th-500}
PREFIX=${KEY_PREFIX:-cth-th-}   # soft-deleted keys cannot be reused; override e.g. KEY_PREFIX=cth-th2-
STATE=state/$VER
mkdir -p "$STATE" logs

ok_json() { python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("ok") else (print(d.get("error"),file=sys.stderr) or 1))'; }
export -f ok_json

register_one() {
  local t=$1 ver=$2 tag=$3 state=$4 prefix=$5
  local A=(env -u ARENA_OPENAPI_TOKEN -u ANT_COOKIE arena)
  local key="$prefix$t"
  [[ -f "$state/$t.ok" ]] && { echo "skip $t"; return 0; }
  local q o
  q=$("${A[@]}" upload "$ASSETS/$t/instruction.md" --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"]["uri"])')
  o=$("${A[@]}" upload "$ASSETS/$t/oracle" --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"]["uri"])')
  python3 - "$t" "$q" "$o" <<'PY'
import json,sys
t,q,o=sys.argv[1:4]
p=f"env-specs/{t}.json"; s=json.load(open(p))
s["query"]["uri"]=q; s["oracle"]["uri"]=o
json.dump(s,open(p,"w"),ensure_ascii=False,indent=1)
PY
  "${A[@]}" env create --key "$key" --name "TerminalHorizon: $t (cth)" \
      --tags "source:terminal-horizon,domain:terminal,chen,$tag" --json >/dev/null 2>&1 || true   # may already exist
  "${A[@]}" env version-create "$key" "$ver" --file "env-specs/$t.json" --json | ok_json \
    || "${A[@]}" env version-update "$key" "$ver" --file "env-specs/$t.json" --json | ok_json \
    || { echo "FAILED version-create/update $key@$ver"; return 1; }   # draft left by an earlier failed run -> update it
  "${A[@]}" env publish "$key" "$ver" --yes --json | ok_json || { echo "FAILED publish $key@$ver"; return 1; }
  "${A[@]}" env version-get "$key" "$ver" --json | python3 -c 'import json,sys;sys.exit(0 if json.load(sys.stdin).get("data",{}).get("status")=="PUBLISHED" else 1)' \
      || { echo "FAILED verify $key@$ver"; return 1; }
  echo "$key@$ver" > "$state/$t.ok"
  echo "published $key@$ver"
}
export -f register_one

xargs -P "$PAR" -I{} bash -c 'register_one "$@" 2>&1 | tee -a logs/register_'"$VER"'.log' _ {} "$VER" "$TAG" "$STATE" "$PREFIX" < "$TASKS_FILE"

# verify every task is published before building the benchmark
missing=0
while read -r t; do [[ -f "$STATE/$t.ok" ]] || { echo "NOT REGISTERED: $t"; missing=1; }; done < "$TASKS_FILE"
[[ $missing == 0 ]] || { echo "some envs failed; re-run to retry"; exit 1; }

if [[ "${SKIP_BENCH:-0}" == "1" ]]; then
  echo "SKIP_BENCH=1: envs registered, benchmark skipped"
  exit 0
fi

# env keys are read from the state markers (a task may have been registered under a different prefix)
python3 - "$VER" "$REWARD" "$TASKS_FILE" "$BKEY" "$TAG" "$STATE" <<'PY'
import json,sys
ver,reward,tf,bkey,tag,state=sys.argv[1:7]
tasks=[l.strip() for l in open(tf) if l.strip()]
rkey,rver=reward.split("@")
def ref(t):
    k,v=open(f"{state}/{t}.ok").read().strip().split("@"); return {"env_ref":{"key":k,"version":v},"weight":1}
b={"reward_ref":{"key":rkey,"version":rver},
   "env_refs":[ref(t) for t in tasks],
   "aggregation":{"method":"mean","failed_task_policy":"failed_as_zero"},
   "tags":["source:terminal-horizon","domain:terminal","chen",tag],
   "metadata":{"purpose":f"TerminalHorizon {len(tasks)} tasks (hcr-images)","count":str(len(tasks))}}
json.dump(b,open(f"benchmarks/bench-{bkey}.json","w"),ensure_ascii=False,indent=1)
print(f"benchmarks/bench-{bkey}.json:",len(tasks),"envs")
PY
A=(env -u ARENA_OPENAPI_TOKEN -u ANT_COOKIE arena)
"${A[@]}" benchmark create --key "$BKEY" --name "TerminalHorizon $(wc -l < "$TASKS_FILE") tasks (cth)" \
    --tags "source:terminal-horizon,chen,$TAG" --json >/dev/null 2>&1 || echo "(benchmark exists)"
"${A[@]}" benchmark version-create "$BKEY" "$BVER" --file "bench-$BKEY.json" --json | ok_json \
  || "${A[@]}" benchmark version-update "$BKEY" "$BVER" --file "bench-$BKEY.json" --json | ok_json   # draft already exists -> update
"${A[@]}" benchmark publish "$BKEY" "$BVER" --yes --json | ok_json || { echo "FAILED to publish benchmark $BKEY@$BVER"; exit 1; }
echo "published benchmark $BKEY@$BVER"
