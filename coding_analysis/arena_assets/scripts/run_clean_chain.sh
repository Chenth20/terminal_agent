#!/usr/bin/env bash
# Post-judge tail of the terminal_horizon.md cleaning chain for one merged dataset:
#   wait for llm_judge.py -> retry JUDGE_ERROR (2 passes) -> apply_judge (KEEP set)
#   -> finalize_dataset (form A/C + short-think) -> scan_bad_actions + dataset_stats
#   -> pollution re-audit on the KEEP set (must be 0 hits)
# Usage: run_clean_chain.sh <merged_dir> <merged_name> <final_dir> <final_name>
#   e.g. run_clean_chain.sh /storage/.../flash_merged flash_merged /storage/.../flash_final flash_final
set -euo pipefail
MDIR=$1; MNAME=$2; FDIR=$3; FNAME=$4
S=$(cd "$(dirname "$0")" && pwd)
BASE=$MDIR/$MNAME
OUT=$MDIR/judge_opus48
[[ -f ~/.secrets.env ]] && { set -a; source ~/.secrets.env; set +a; }

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

while pgrep -f "llm_judge.py.*${MNAME}/judge_opus48" >/dev/null 2>&1; do sleep 120; done
log "main judge finished for $MNAME"

for pass in 1 2; do
  n_err=$(grep -c '"verdict": "JUDGE_ERROR"' "$OUT/reviews.jsonl" || true)
  log "pass $pass: $n_err JUDGE_ERROR records"
  [[ $n_err -eq 0 ]] && break
  grep -v '"verdict": "JUDGE_ERROR"' "$OUT/reviews.jsonl" > "$OUT/reviews.tmp"
  mv "$OUT/reviews.tmp" "$OUT/reviews.jsonl"
  python3 "$S/llm_judge.py" --base "$BASE" --out "$OUT" --workers 4 --retries 4 >> "$OUT/run.log" 2>&1
done

python3 "$S/apply_judge.py" --base "$BASE" --judge "$OUT" --out-prefix "${BASE}_judged" | tee "$OUT/apply.log"
log "apply done"

mkdir -p "$FDIR"
python3 "$S/finalize_dataset.py" --base "${BASE}_judged-keep" --out "$FDIR" --name "$FNAME" | tee "$FDIR/finalize.log"
log "finalize done"

python3 "$S/scan_bad_actions.py" --base "$FDIR/$FNAME" > "$FDIR/scan_bad_actions.txt" 2>&1 || log "scan_bad_actions failed"
python3 "$S/dataset_stats.py" --base "$FDIR/$FNAME" --name "$FNAME" --out "$FDIR/stats_$FNAME.json" > "$FDIR/dataset_stats.log" 2>&1 || log "dataset_stats failed"
python3 "$S/audit_env_pollution.py" --base "$FDIR/$FNAME" --out "$FDIR/pollution_recheck" > "$FDIR/pollution_recheck.log" 2>&1 || log "pollution recheck failed"
log "all done for $FNAME"
