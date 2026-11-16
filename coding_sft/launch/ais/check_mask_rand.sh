#!/bin/bash
# One supervision check for the AIS run abl_mask_rand_const_e8_ais2n. Usage: bash check_mask_rand.sh <record_id>
RID=${1:?record id}
CK=/storage/openpsi/users/chen/checkpoints/abl_mask_rand_const_e8_ais2n
set -a; source ~/.secrets.env 2>/dev/null; set +a; export LAUNCH_CONTAINER_MODE=dev_local
STATUS=$(python3 - "$RID" <<'EOF' 2>/dev/null
import logging, sys; logging.disable(logging.CRITICAL)
from aistudio_common.rest import job as J
rid = int(sys.argv[1])
try:
    print(J.query_job_status(rid))
except Exception as e:
    print('ERR ' + str(e)[:120])
EOF
)
CKPTS=$(ls -d $CK/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tr '\n' ' ')
LAST=$(tail -1 $CK/trainer_log.jsonl 2>/dev/null)
STEP=$(echo "$LAST" | python3 -c "import json,sys; d=json.loads(sys.stdin.read() or '{}'); print(d.get('current_steps','-'), d.get('loss','-'))" 2>/dev/null)
NLOGS=$(ls /storage/openpsi/users/chen/runs/ais/abl_mask_rand_const_e8_ais2n_*.log 2>/dev/null | wc -l)
DONE=$([ -f $CK/train_results.json ] && echo yes || echo no)
LINE="$(date '+%F %T'), AIS($RID)=$STATUS, step/loss=$STEP /304, ckpts=$CKPTS, pod_logs=$NLOGS, train_results=$DONE"
echo "$LINE" | tee -a /storage/openpsi/users/chen/runs/supervise_mask_rand.log
LOG=$(ls -t /storage/openpsi/users/chen/runs/ais/abl_mask_rand_const_e8_ais2n_*.log 2>/dev/null | head -1)
[ -n "$LOG" ] && grep -E "Traceback|NCCL error|SIGTERM|resum" "$LOG" | tail -4 | cut -c1-200
