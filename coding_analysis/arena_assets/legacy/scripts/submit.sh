#!/usr/bin/env bash
# Submit a benchmark job (benchmark + harness + model) and start a trajectory puller.
#
#   bash eval/submit.sh --benchmark cth-tbench-filtered-100@1.0.0 \
#        --harness terminus-2-tb21@1.0.12 --model qwen3-8-27b-router-xhigh-chen@1.0.3 \
#        --samples 2 --parallel 32 --job-id cth-tb100-terminus-qwen
#
#   bash eval/submit.sh --model <ref> --benchmark cth-tb21-full89-mirror@1.0.1 --samples 2   # DEFAULT harness: cth-terminus-2-stream@1.6.96
#   bash eval/submit.sh --harness all --model <ref> --benchmark <ref>    # all three, for comparison
#        (--harness all = cth-terminus-2-stream@1.0.0, cth-mimo-agent@1.0.3, cth-claude-code-swe@1.0.3)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../lib/common.sh"

BENCH=${BENCHMARK:-cth-tbench-filtered-100@1.0.0}
MODEL=${MODEL:-}
HARNESSES=${HARNESS:-}
DEFAULT_HARNESS=cth-terminus-2-stream@1.6.96   # streaming Terminus 2, 96k trigger, pypi/HF mirrors in the task shell, 16Gi (see docs/tb21-failure-audit.md); TB2.1 pairs it with cth-tb21-full89-mirror@1.0.1
SAMPLES=${SAMPLES:-2}
PARALLEL=${PARALLEL:-32}
RETRIES=${RETRIES:-1}
BACKOFF=${BACKOFF:-60}
JOB_ID=${JOB_ID:-}
TAG=${TAG:-chen,tbench-100}
OUT_ROOT=${OUT_ROOT:-/storage/openpsi/users/chen/output}
POLL=${POLL:-1}

while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark) BENCH=$2; shift 2 ;;
    --harness)   HARNESSES=$2; shift 2 ;;
    --model)     MODEL=$2; shift 2 ;;
    --samples)   SAMPLES=$2; shift 2 ;;
    --parallel)  PARALLEL=$2; shift 2 ;;
    --retries)   RETRIES=$2; shift 2 ;;
    --job-id)    JOB_ID=$2; shift 2 ;;
    --tag)       TAG=$2; shift 2 ;;
    --no-poll)   POLL=0; shift ;;
    -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n $MODEL ]] || die "--model is required (e.g. qwen3-8-27b-router-xhigh-chen@1.0.3)"

if [[ -z $HARNESSES || $HARNESSES == default ]]; then
  HARNESSES=$DEFAULT_HARNESS
elif [[ $HARNESSES == all ]]; then
  HARNESSES="$DEFAULT_HARNESS cth-mimo-agent@1.0.3 cth-claude-code-swe@1.0.3"
fi

short() {  # harness ref -> short tag for the job id
  case $1 in
    terminus*|cth-terminus*) echo terminus ;;
    cth-mimo-agent*)  echo mimo ;;
    cth-claude-code*) echo claude ;;
    *)                echo "${1%%@*}" | tr -cd 'a-z0-9-' ;;
  esac
}

for h in $HARNESSES; do
  short_name=$(short "$h")
  model_short=$(echo "${MODEL%%@*}" | sed -E 's/^qwen3-8-([a-z0-9]+).*/\1/; s/^deepseek.*/dsv4/')
  id=${JOB_ID:-cth-tb100-$short_name-$model_short}
  log "submitting harness=$h model=$MODEL bench=$BENCH samples=$SAMPLES par=$PARALLEL"
  job=$(arena job create --benchmark "$BENCH" --harness "$h" --model-ref "$MODEL" \
          --samples "$SAMPLES" --max-parallel "$PARALLEL" --max-retries "$RETRIES" --retry-backoff "$BACKOFF" \
          --job-id "$id" --tags "$TAG,$short_name,$model_short" --json \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"]["job_id"] if d.get("ok") else "")')
  [[ -n $job ]] || die "job create failed for $h"
  log "  job_id=$job"
  if (( POLL )); then
    mkdir -p "$OUT_ROOT/$job"
    nohup bash "$HERE/pull_trajectories.sh" "$job" "$OUT_ROOT" > "$OUT_ROOT/$job/poller.out" 2>&1 &
    log "  puller started -> $OUT_ROOT/$job/"
  fi
done
