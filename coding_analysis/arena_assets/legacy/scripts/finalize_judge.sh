#!/usr/bin/env bash
# Wait for the running llm_judge.py to finish, retry JUDGE_ERROR records (up to 2 passes),
# then apply the reviews to the dataset. Safe to re-run.
set -euo pipefail
REPO=/home/admin/chentinhong.cth/Chenth20/LlamaFactory/coding_arena
BASE=/storage/openpsi/users/chen/code_sft_data/dsv41_merged/dsv41_merged
OUT=/storage/openpsi/users/chen/code_sft_data/dsv41_merged/judge_opus48
[[ -f ~/.secrets.env ]] && { set -a; source ~/.secrets.env; set +a; }

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

# 1. wait for any running judge on this output dir
while pgrep -f "llm_judge.py.*judge_opus48" >/dev/null 2>&1; do sleep 120; done
log "main judge finished"

# 2. retry errors (strip JUDGE_ERROR lines so the resume logic re-queues them)
for pass in 1 2; do
  n_err=$(grep -c '"verdict": "JUDGE_ERROR"' "$OUT/reviews.jsonl" || true)
  log "pass $pass: $n_err JUDGE_ERROR records"
  [[ $n_err -eq 0 ]] && break
  grep -v '"verdict": "JUDGE_ERROR"' "$OUT/reviews.jsonl" > "$OUT/reviews.tmp"
  mv "$OUT/reviews.tmp" "$OUT/reviews.jsonl"
  python3 "$REPO/code/sft/llm_judge.py" --base "$BASE" --out "$OUT" --workers 4 --retries 4 \
    >> "$OUT/run.log" 2>&1
done

# 3. apply
python3 "$REPO/code/sft/apply_judge.py" --base "$BASE" --judge "$OUT" \
  --out-prefix "${BASE}_judged" | tee "$OUT/apply.log"
log "done"