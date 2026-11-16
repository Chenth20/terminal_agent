#!/bin/bash
# Watcher: wait INITIAL_WAIT, then every INTERVAL look for a reservation with >= NODES fully idle
# GPU nodes and re-submit the mask_tail 4-node resume there (run_ablation_multi.slurm resumes from
# checkpoint-96 by itself). Exits after the first successful sbatch, or if the job is already queued.
#   setsid nohup bash coding_sft/launch/slurm/watch_resume_mask_tail.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
VARIANT=${VARIANT:-mask_tail}
NODES=${NODES:-4}
INITIAL_WAIT=${INITIAL_WAIT:-18000}   # 5 h
INTERVAL=${INTERVAL:-3600}            # 1 h
REPO=/storage/openpsi/users/chen/llama-factory
JOBNAME=chenth_abl_${VARIANT}
LOG=/storage/openpsi/users/chen/runs/ablation/watch_resume_${VARIANT}.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

free_nodes_in() {  # prints the number of nodes of reservation $1 that are up and have no GPU allocated
    local nodes n st alloc cnt=0
    nodes=$(scontrol show res -o "$1" | grep -o 'Nodes=[^ ]*' | cut -d= -f2)
    for n in $(scontrol show hostnames "$nodes"); do
        info=$(scontrol show node -o "$n" | tr ' ' '\n')
        st=$(echo "$info" | grep '^State=' | cut -d= -f2)
        alloc=$(echo "$info" | grep '^AllocTRES=' | cut -d= -f2-)
        case "$st" in *DOWN*|*DRAIN*|*NOT_RESPONDING*) continue ;; esac
        [[ "$alloc" == *gpu* ]] && continue
        cnt=$((cnt + 1))
    done
    echo "$cnt"
}

log "watcher started pid $$: sleeping ${INITIAL_WAIT}s, then checking every ${INTERVAL}s for ${NODES} idle nodes"
sleep "$INITIAL_WAIT"
while true; do
    if squeue -h -o %j | grep -qx "$JOBNAME"; then
        log "$JOBNAME already in queue; exiting"; exit 0
    fi
    picked=""
    for res in $(scontrol show res -o | grep -o 'ReservationName=[^ ]*' | cut -d= -f2); do
        cnt=$(free_nodes_in "$res")
        log "  $res: $cnt idle"
        if [ "$cnt" -ge "$NODES" ] && [ -z "$picked" ]; then picked=$res; fi
    done
    if [ -n "$picked" ]; then
        out=$(cd "$REPO" && VARIANT=$VARIANT sbatch --nodes="$NODES" --reservation="$picked" --job-name="$JOBNAME" \
              coding_sft/launch/slurm/run_ablation_multi.slurm 2>&1)
        rc=$?
        log "sbatch on $picked -> rc=$rc: $out"
        [ "$rc" -eq 0 ] && exit 0
    else
        log "no reservation with ${NODES} idle nodes; retry in ${INTERVAL}s"
    fi
    sleep "$INTERVAL"
done
