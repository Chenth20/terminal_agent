#!/bin/bash
# One-shot status of every ablation: trainer_log step/loss, pod heartbeat, AIS job state, Slurm jobs.
#   bash coding_sft/launch/status.sh            # all known runs
#   bash coding_sft/launch/status.sh keep base  # subset
CK=/storage/openpsi/users/chen/checkpoints; LOGS=/storage/openpsi/users/chen/runs/ais
date '+== %F %T'
for v in ${@:-base mask keep mask_cut mask_tail}; do
    for d in "$CK"/abl_${v}_ais2n "$CK"/abl_${v}_slurm; do
        [ -f "$d/trainer_log.jsonl" ] || continue
        r=$(basename "$d")
        last=$(tail -n1 "$d/trainer_log.jsonl")
        step=$(echo "$last" | grep -o '"current_steps": [0-9]*' | grep -o '[0-9]*$'); tot=$(echo "$last" | grep -o '"total_steps": [0-9]*' | grep -o '[0-9]*$')
        loss=$(echo "$last" | grep -o '"loss": [0-9.]*' | grep -o '[0-9.]*$' | cut -c1-6); ep=$(echo "$last" | grep -o '"epoch": [0-9.]*' | grep -o '[0-9.]*$' | cut -c1-4)
        rem=$(echo "$last" | grep -o '"remaining_time": "[^"]*"' | cut -d'"' -f4)
        upd=$(stat -c %y "$d/trainer_log.jsonl" | cut -c12-16)
        hb=$(ls -t "$LOGS"/${r}_*.log 2>/dev/null | head -2 | xargs -r -I{} sh -c 'grep -a "^\[hb" {} | tail -1 | cut -c5-9' | tr '\n' ' ')
        wb=$(cat "$d/wandb_run_id" 2>/dev/null)
        printf '%-20s step %3s/%-3s loss %-6s ep %-4s rem %-8s last_log %s  pods(UTC) %-12s wandb %s\n' "$r" "$step" "$tot" "$loss" "$ep" "$rem" "$upd" "$hb" "$wb"
    done
done
if [ -z "${NO_AIS:-}" ]; then
    export LAUNCH_CONTAINER_MODE=dev_local
    ids=$(grep -h "record id=" /storage/openpsi/users/chen/runs/ais_submit_abl_*.log 2>/dev/null | grep -o '[0-9]*$' | sort -u | tail -8 | tr '\n' ' ')
    timeout 120 python3 - $ids <<'EOF' 2>/dev/null | grep -v INFO
import sys
from aistudio_common.rest import job as J
print("AIS:", " ".join(f"{i}={J.query_job_status(int(i))}" for i in sys.argv[1:]))
EOF
fi
squeue -u "$(whoami)" -h -o "slurm %i %j %T %M %N" 2>/dev/null | grep chenth
