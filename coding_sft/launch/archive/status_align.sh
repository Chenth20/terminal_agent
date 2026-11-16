#!/bin/bash
# One-shot status of the alignment suite (base / perdh / tree) on AIS:
#   bash coding_sft/launch/status_align.sh
CK=/storage/openpsi/users/chen/checkpoints; LOGS=/storage/openpsi/users/chen/runs/ais
date '+== %F %T'
for a in ${@:-base perdh tree}; do
    r=align_$a; d=$CK/$r
    hb=$(ls -t "$LOGS"/${r}_*.log 2>/dev/null | head -2 | xargs -r -I{} sh -c 'grep -a "^\[hb" {} | tail -1 | cut -c5-9' | tr '\n' ' ')
    last_line=$(ls -t "$LOGS"/${r}_*.log 2>/dev/null | head -1 | xargs -r tail -n1 | cut -c1-110)
    if [ -f "$d/trainer_log.jsonl" ]; then
        last=$(tail -n1 "$d/trainer_log.jsonl")
        step=$(echo "$last" | grep -o '"current_steps": [0-9]*' | grep -o '[0-9]*$'); tot=$(echo "$last" | grep -o '"total_steps": [0-9]*' | grep -o '[0-9]*$')
        loss=$(echo "$last" | grep -o '"loss": [0-9.]*' | grep -o '[0-9.]*$' | cut -c1-6)
        rem=$(echo "$last" | grep -o '"remaining_time": "[^"]*"' | cut -d'"' -f4)
        upd=$(stat -c %y "$d/trainer_log.jsonl" | cut -c12-16)
        printf '%-12s step %2s/%-2s loss %-6s rem %-8s last_log %s  pods(UTC) %s\n' "$r" "$step" "$tot" "$loss" "$rem" "$upd" "$hb"
    else
        printf '%-12s (no trainer_log yet) pods(UTC) %s | %s\n' "$r" "$hb" "$last_line"
    fi
done
if [ -z "${NO_AIS:-}" ]; then
    export LAUNCH_CONTAINER_MODE=dev_local
    ids=$(grep -h "record id=" /storage/openpsi/users/chen/runs/ais_submit_align_*.log 2>/dev/null | grep -o '[0-9]*$' | sort -u | tr '\n' ' ')
    timeout 120 python3 - $ids <<'EOF' 2>/dev/null | grep -v INFO
import sys
from aistudio_common.rest import job as J
print("AIS:", " ".join(f"{i}={J.query_job_status(int(i))}" for i in sys.argv[1:]))
EOF
fi
