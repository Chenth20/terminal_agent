#!/usr/bin/env bash
# Continuously pull EnvArena job trajectories + logs + results to local storage.
# Polls until the job reaches a terminal state; each finished task is pulled once.
#
#   bash pull_trajectories.sh <job_id> [out_root]        # default out_root=/storage/openpsi/users/chen/output
#   ONCE=1 bash pull_trajectories.sh <job_id>            # single pass, no polling
#
# Layout:  <out_root>/<job_id>/
#            job.json                 latest `arena job get`
#            result.json              `arena job result` (written when terminal)
#            summary.tsv              env_key  status  score  agent_ms  eval_ms
#            tasks/<env_key>/s<idx>/  task.json  trajectory.jsonl  harness.log  runner.log  reward.log
set -uo pipefail
export PATH=/storage/openpsi/users/hechuyi.hcy/tools:$PATH
[[ -f ~/.secrets.env ]] && { set -a; source ~/.secrets.env; set +a; }   # ARENA_OPENAPI_TOKEN / AISTUDIO_TOKEN live here, never in scripts
ARENA=(env -u ANT_COOKIE arena)

JOB=${1:?job_id}
OUT=${2:-/storage/openpsi/users/chen/output}/$JOB
INTERVAL=${INTERVAL:-120}
mkdir -p "$OUT/tasks"
LOG=$OUT/pull.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

pull_log() {  # $1=task_id $2=kind $3=dest  (pages through --offset/--limit)
  local tid=$1 kind=$2 dest=$3 off=0 lim=65536
  : > "$dest.tmp"
  while :; do
    local page
    page=$("${ARENA[@]}" job log "$JOB" "$tid" "$kind" --json --offset "$off" --limit "$lim" 2>/dev/null) || break
    python3 -c '
import json,sys
d=json.load(sys.stdin)
if not d.get("ok"): sys.exit(2)
dd=d["data"]; sys.stdout.write(dd.get("content") or "")
sys.exit(0 if dd.get("has_more") else 1)' <<<"$page" >> "$dest.tmp"
    local rc=$?
    (( rc == 2 )) && { rm -f "$dest.tmp"; return 1; }
    (( rc == 1 )) && break
    off=$((off + lim))
  done
  mv "$dest.tmp" "$dest"
}

pull_task() {  # $1 = task json (one line)
  local tj=$1 tid env idx d
  tid=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["task_id"])' <<<"$tj")
  env=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["env_key"])' <<<"$tj")
  idx=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("sample_index",0))' <<<"$tj")
  d=$OUT/tasks/$env/s$idx
  [[ -f $d/.done ]] && return 0
  mkdir -p "$d"
  echo "$tj" | python3 -m json.tool > "$d/task.json"
  "${ARENA[@]}" job trajectory "$JOB" "$tid" --format raw > "$d/trajectory.jsonl.tmp" 2>/dev/null \
    && [[ -s $d/trajectory.jsonl.tmp ]] && mv "$d/trajectory.jsonl.tmp" "$d/trajectory.jsonl" || rm -f "$d/trajectory.jsonl.tmp"
  for k in harness runner reward; do pull_log "$tid" "$k" "$d/$k.log"; done
  # official verifier output (apt/uv bootstrap + pytest); tells whether tests ran at all
  "${ARENA[@]}" task download "$tid" tbench_test.log -o "$d/tbench_test.log" --json >/dev/null 2>&1 || rm -f "$d/tbench_test.log"
  local st; st=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<<"$tj")
  if [[ -f $d/trajectory.jsonl ]]; then
    touch "$d/.done"
    log "pulled $env/s$idx score=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("score","?"))' <<<"$tj") events=$(wc -l < "$d/trajectory.jsonl")"
  elif [[ $st != DONE && -s $d/harness.log ]]; then
    # failed tasks may have no trajectory; logs are enough. Do NOT mark .done: a later
    # `arena job retry` turns these into DONE and the trajectory must be re-pulled.
    log "pulled $env/s$idx status=$st (no trajectory, logs only; will re-check)"
  else
    log "WARN $env/s$idx: trajectory not available yet (will retry)"
  fi
}

while :; do
  JOBJSON=$("${ARENA[@]}" job get "$JOB" --json 2>/dev/null)
  if ! python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("ok") else 1)' <<<"$JOBJSON"; then
    log "job get failed; retry in ${INTERVAL}s"; sleep "$INTERVAL"; continue
  fi
  echo "$JOBJSON" > "$OUT/job.json"
  STATUS=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["job"]["status"])' <<<"$JOBJSON")
  COUNTS=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["job"]["task_counts"])' <<<"$JOBJSON")
  log "job status=$STATUS counts=$COUNTS"

  # Tasks come from `job get` (full list). `job tasks` is paginated at 100 and would truncate big jobs.
  # finished tasks -> pull; also write summary.tsv
  python3 -c '
import json,sys
items=json.load(sys.stdin)["data"]["tasks"]
fin=[t for t in items if t["status"] in ("DONE","FAILED","CANCELLED","TIMEOUT","STOPPED","HARNESS_FAILED","REWARD_FAILED","EVAL_FAILED")]
with open(sys.argv[1],"w") as f:
    f.write("env_key\tsample\tstatus\tscore\tagent_ms\teval_ms\n")
    for t in sorted(items,key=lambda x:(x["env_key"],x.get("sample_index",0))):
        tm=t.get("timings") or {}
        f.write("%s\t%s\t%s\t%s\t%s\t%s\n"%(t["env_key"],t.get("sample_index",0),t["status"],t.get("score",""),tm.get("agent_ms",""),tm.get("eval_ms","")))
for t in fin: print(json.dumps(t))
' "$OUT/summary.tsv" <<<"$JOBJSON" | while IFS= read -r tj; do pull_task "$tj"; done

  case $STATUS in
    DONE|FAILED|TIMEOUT|CANCELLED|STOPPED)
      "${ARENA[@]}" job result "$JOB" --json > "$OUT/result.json" 2>/dev/null
      log "job terminal ($STATUS). result -> $OUT/result.json"
      python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d.get("data",d),ensure_ascii=False,indent=1)[:1500])' "$OUT/result.json" | tee -a "$LOG"
      n=$(ls "$OUT"/tasks/*/*/.done 2>/dev/null | wc -l); log "pulled $n task dirs under $OUT/tasks"
      break ;;
  esac
  [[ ${ONCE:-0} == 1 ]] && break
  sleep "$INTERVAL"
done
