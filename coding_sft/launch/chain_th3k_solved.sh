#!/usr/bin/env bash
# Chain: wait for the TH-3K replay job -> keep solved trajectories -> prep + schedule -> Slurm 2-node training.
#   setsid nohup bash coding_sft/launch/chain_th3k_solved.sh > /storage/openpsi/users/chen/runs/chain_th3k_solved.log 2>&1 &
# Idempotent: every stage is skipped if its output already exists.
set -uo pipefail
export PATH=/storage/openpsi/users/hechuyi.hcy/tools:$PATH
REPO=/storage/openpsi/users/chen/llama-factory
JOBS="cth-th3k-replay2-2610311834 cth-th3k-replay1-2610311834"
RD=/storage/openpsi/users/chen/code_sft_data/th3k_replay
DD=/storage/openpsi/users/chen/data/terminal_horizon_th3k_solved
log() { echo "[$(date '+%F %T')] $*"; }

# ---- 1. wait for BOTH replay jobs -------------------------------------------------------------
for JOB in $JOBS; do
  while true; do
    ST=$(env -u ARENA_OPENAPI_TOKEN -u ANT_COOKIE timeout 120 arena job get "$JOB" --json 2>/dev/null \
         | python3 -c "import json,sys; d=json.load(sys.stdin); j=d['data']['job']; print(j['status'], json.dumps(j.get('task_counts')))" 2>/dev/null) || ST=""
    log "replay job $JOB: $ST"
    case "$ST" in
        DONE*|FAILED*|CANCELLED*|TIMEOUT*) break ;;
    esac
    sleep 600
  done
  log "replay job $JOB finished: $ST"
done

# ---- 2. scores -> solved row list ------------------------------------------------------------
if [ ! -s "$RD/replay_scores.json" ]; then
    python3 - "$RD" $JOBS <<'EOF'
import json, os, sys, collections
rd = sys.argv[1]; jobs = sys.argv[2:]
scores = {}
for job in jobs:
    tasks_dir = os.path.join("/storage/openpsi/users/chen/output", job, "tasks")
    if not os.path.isdir(tasks_dir):
        continue
    for env in os.listdir(tasks_dir):
        for s in os.listdir(os.path.join(tasks_dir, env)):
            tj = os.path.join(tasks_dir, env, s, "task.json")
            if not os.path.isfile(tj):
                continue
            t = json.load(open(tj))
            sc = t.get("score")
            if sc is None:
                continue
            scores[f"{env}/{s}"] = float(sc)
json.dump(scores, open(os.path.join(rd, "replay_scores.json"), "w"), indent=0)
solved = {k for k, v in scores.items() if v == 1.0}
print(f"scored {len(scores)} tasks, solved {len(solved)}")
# rows: replay_meta has env_ref + sample per source row
rows = []
for line in open(os.path.join(rd, "replay_meta.jsonl")):
    m = json.loads(line)
    if m.get("status") != "ok":
        continue
    key = f"{m['env_ref'].split('@')[0]}/s{m['sample']}"
    if key in solved:
        rows.append(m["row"])
json.dump(sorted(rows), open(os.path.join(rd, "solved_rows.json"), "w"))
print(f"solved source rows: {len(rows)}")
EOF
fi
N_SOLVED=$(python3 -c "import json; print(len(json.load(open('$RD/solved_rows.json'))))")
log "solved trajectories: $N_SOLVED"
[ "$N_SOLVED" -ge 64 ] || { log "ERROR: fewer than 64 solved trajectories; aborting"; exit 1; }

# ---- 3. filtered jsonl + manifest ------------------------------------------------------------
mkdir -p "$DD"
if [ ! -s "$DD/th3k_solved_src.jsonl" ]; then
    python3 - "$RD" "$DD" <<'EOF'
import json, sys
rd, dd = sys.argv[1], sys.argv[2]
rows = set(json.load(open(f"{rd}/solved_rows.json")))
meta = {}
for line in open(f"{rd}/replay_meta.jsonl"):
    m = json.loads(line)
    if m.get("status") == "ok":
        meta[m["row"]] = m
src = open("/storage/openpsi/users/chen/code_data/terminal_horizon/TerminalHorizon-3K/data/TerminalHorizon-3K-SFT.jsonl")
out = open(f"{dd}/th3k_solved_src.jsonl", "w")
man = open(f"{dd}/th3k_solved_src.manifest.jsonl", "w")
kept = 0
for i, line in enumerate(src):
    if i not in rows:
        continue
    out.write(line)
    m = meta[i]
    man.write(json.dumps({"job": "th3k-replay", "env_key": m["env_ref"].split("@")[0],
                          "sample": f"s{m['sample']}", "segment": 0, "n_segments": 1,
                          "status": "DONE", "score": 1, "src_row": i}) + "\n")
    kept += 1
out.close(); man.close()
print(f"wrote {kept} solved trajectories")
EOF
fi

# ---- 4. prep + tokenize + schedule (CPU slurm job) -------------------------------------------
if [ ! -f "$DD/tokenized_qwen3_5_text_262144_sched16_gsort/manifest.json" ]; then
    JID=$(SRC=$DD/th3k_solved_src.jsonl MANIFEST=$DD/th3k_solved_src.manifest.jsonl NAME=th3k_solved \
          sbatch --parsable "$REPO/coding_sft/launch/slurm/prep_and_sched.slurm")
    log "prep_and_sched submitted: $JID"
    while squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep 60; done
    [ -f "$DD/tokenized_qwen3_5_text_262144_sched16_gsort/manifest.json" ] \
        || { log "ERROR: prep_and_sched $JID finished without a schedule"; exit 1; }
fi
log "schedule ready: $(python3 -c "import json; m=json.load(open('$DD/tokenized_qwen3_5_text_262144_sched16_gsort/manifest.json')); print(m['n_trajs'],'trajs,',m['steps'],'steps,',m['steps_per_epoch'],'per epoch, ga',m['G'])")"

# ---- 5. training: Slurm 2 nodes, same recipe as abl_mask_const_e8 + warmup 0.3 epoch ----------
if ! squeue -h -n chenth_abl_th3k_solved 2>/dev/null | grep -q .; then
    JID=$(VARIANT=th3k_solved_const_e8_w03ep sbatch --parsable --nodes=2 --job-name=chenth_abl_th3k_solved \
          ${TRAIN_RESERVATION:+--reservation=$TRAIN_RESERVATION} \
          "$REPO/coding_sft/launch/slurm/run_ablation_multi.slurm")
    log "training submitted: Slurm $JID (run abl_th3k_solved_const_e8_w03ep_slurm2n)"
    echo "$JID" > /storage/openpsi/users/chen/runs/th3k_solved_train_jobid.txt
fi
log "chain complete"
