#!/bin/bash
# One-shot: submit the 4-node resume of an ablation on ANY reservation with >= 4 idle healthy nodes.
#   VARIANT=mask_tail bash coding_sft/launch/slurm/resume_when_free.sh
# Exit 0 = submitted (or a job for this variant already exists); exit 3 = no reservation has 4 idle nodes.
# Idempotent: refuses to submit if a chenth_abl_<VARIANT> job is already PENDING/RUNNING.
set -uo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
VARIANT=${VARIANT:?}
NEED=${NEED:-4}
JOB=chenth_abl_${VARIANT}
LOG=/storage/openpsi/users/chen/runs/ablation/resume_${VARIANT}_watch.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

existing=$(squeue -h -n "$JOB" -o "%i %T" | head -1)
if [ -n "$existing" ]; then log "already have $JOB: $existing -> nothing to do"; exit 0; fi

best=""; bestn=0
for r in $(scontrol show res -o | grep -o 'ReservationName=[^ ]*' | cut -d= -f2); do
    nodes=$(scontrol show res -o "$r" | grep -o ' Nodes=[^ ]*' | cut -d= -f2)
    free=0; list=""
    for n in $(scontrol show hostnames "$nodes"); do
        info=$(scontrol show node -o "$n")
        state=$(echo "$info" | grep -o ' State=[^ ]*' | cut -d= -f2)
        alloc=$(echo "$info" | grep -o ' AllocTRES=[^ ]*' | cut -d= -f2)
        case "$state" in *DOWN*|*DRAIN*|*NOT_RESPONDING*|*FAIL*|*INVAL*) continue ;; esac
        [ -z "$alloc" ] && { free=$((free + 1)); list="$list $n"; }
    done
    log "reservation $r: $free idle [${list# }]"
    [ "$free" -gt "$bestn" ] && { bestn=$free; best=$r; }
done

if [ "$bestn" -lt "$NEED" ]; then log "no reservation has $NEED idle nodes (best: $best=$bestn)"; exit 3; fi

cd "$REPO" || exit 1
out=$(VARIANT=$VARIANT sbatch --nodes="$NEED" --reservation="$best" --job-name="$JOB" \
      coding_sft/launch/slurm/run_ablation_multi.slurm 2>&1)
rc=$?
log "sbatch on $best (rc=$rc): $out"
[ $rc -eq 0 ] || exit $rc
jid=$(echo "$out" | grep -o '[0-9]*$')
sleep 90
log "job $jid state: $(squeue -h -j "$jid" -o '%T %N' 2>/dev/null || echo gone)"
exit 0
