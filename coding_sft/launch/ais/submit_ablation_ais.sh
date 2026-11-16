#!/bin/bash
# Submit one ablation variant to AIStudio:   VARIANT=<name> NODES=2 bash coding_sft/launch/ais/submit_ablation_ais.sh
#
# NODES x 8 GPU, gradient_accumulation_steps = 64 / (NODES*8) -> effective batch 64 on every cluster.
# Prerequisite: the variant's dataset AND its tokenized cache exist (coding_sft/launch/ablation.sh
# prints the dataset dir; cache = <dir>/tokenized_qwen3_5_text_262144) - otherwise 16 ranks tokenize
# independently and dead-lock on the rank-0 save barrier.
# The job command runs coding_sft/launch/ais_train.sh on every node; it auto-resumes from the newest
# complete checkpoint after every pre-emption and keeps appending to the same wandb run.
export AISTUDIO_LOGIN_NAME="aistudio"; export AISTUDIO_USERNUMBER="492801"; export AISTUDIO_TOKEN="${AISTUDIO_TOKEN:?Set AISTUDIO_TOKEN before submitting}"
REPO=/storage/openpsi/users/chen/llama-factory          # the checkout the pods see (afsync mirror of /home/admin)
VARIANT=${VARIANT:?}; NODES=${NODES:-2}; GA=$((64 / (NODES * 8)))
# CONFIG=<yaml> selects the model recipe; variants whose name contains "_9b" default to the 9B yaml
# (identical hparams, model_name_or_path = the 9B text-only export). Passed through to ais_train.sh.
CONFIG=${CONFIG:-$( case "$VARIANT" in *_9b*) echo coding_sft/configs/qwen3_5_9b_terminal_agent_sft_256k.yaml ;; *) echo coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml ;; esac )}
[ -f "$REPO/$CONFIG" ] || { echo "ERROR: CONFIG $REPO/$CONFIG missing" >&2; exit 1; }
# SCHED=rand|gsort selects a non-default trajectory schedule (<dir>/tokenized_..._sched<ranks>_<SCHED>) and tags the run name
SCHED=${SCHED:-}
RUN=${RUN_NAME:-abl_${VARIANT%%_const*}${SCHED:+_$SCHED}${VARIANT#${VARIANT%%_const*}}_ais${NODES}n}   # RUN_NAME=... to start a fresh run
read -r DD DS OV < <(VARIANT=$VARIANT bash $REPO/coding_sft/launch/ablation.sh)
[ -f "$DD/tokenized_qwen3_5_text_262144/dataset_dict.json" ] || echo "WARNING: tokenized cache missing ($DD/tokenized_qwen3_5_text_262144); node 0 will build it before training (~10 min)" >&2
# checkpoint policy (same everywhere, see src/llamafactory/train/callbacks.py): a FULL resume state every
# step (save_steps=1) and KeepOptimizerOnlyLastCallback reduces every earlier step to nothing, except steps
# that are a multiple of one epoch (keep_weights_every = steps/epoch) which are kept as weights-only.
# Effective batch = 64 COMPLETE TRAJECTORIES. A trajectory written in compaction mode is 1..14 cache rows
# (one per compaction segment), so the rows of one trajectory must land in ONE optimizer step and `ga` stops
# being 64/GPUs: the launcher reads the trajectory schedule (build_traj_sched.py) - G micro-batches per rank,
# max_steps, steps/epoch. Set SCHED_TOK to the sched cache; empty = fall back to plain per-row batching.
SCHED_TOK=${SCHED_TOK:-}
if [ -z "$SCHED_TOK" ] && [ -n "$SCHED" ]; then
    SCHED_TOK=$DD/tokenized_qwen3_5_text_262144_sched$((NODES * 8))_$SCHED
    [ -f "$SCHED_TOK/manifest.json" ] || { echo "ERROR: SCHED=$SCHED but $SCHED_TOK/manifest.json missing" >&2; exit 1; }
fi
# auto-detect a schedule built by prep/prep_and_sched.py: prefer the global-sort one (_gsort: one global length
# sort, 64 consecutive trajectories per step, identical every epoch - the 2026-10-31 recipe), else the legacy one.
for cand in "$DD/tokenized_qwen3_5_text_262144_sched$((NODES * 8))_gsort" "$DD/tokenized_qwen3_5_text_262144_sched$((NODES * 8))"; do
    if [ -z "$SCHED_TOK" ] && [ -f "$cand/manifest.json" ] && [ -f "$DD/$DS.manifest.jsonl" ]; then
        SCHED_TOK=$cand; echo "found trajectory schedule: $SCHED_TOK"
    fi
done
if [ -n "$SCHED_TOK" ]; then
    read -r GA STEPS SPE < <(python3 -c "import json,sys; m=json.load(open(sys.argv[1]+'/manifest.json')); print(m['G'], m['steps'], m['steps_per_epoch'])" "$SCHED_TOK")
    MAXSTEPS="max_steps=$STEPS disable_shuffling=true"   # sched cache has all 8 epochs baked in; num_train_epochs is ignored
    TP=$SCHED_TOK
    echo "trajectory schedule: $STEPS steps ($SPE/epoch = 64 trajectories each), ga=$GA (max rows in a step / $((NODES * 8)))"
else
    N_ROWS=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['splits']['train']['num_examples'])" "$DD/tokenized_qwen3_5_text_262144/train/dataset_info.json" 2>/dev/null || echo 0)
    SPE=$(( (N_ROWS + 63) / 64 )); [ "$SPE" -gt 0 ] || SPE=1
    MAXSTEPS=""
    TP=$DD/tokenized_qwen3_5_text_262144
    echo "row schedule (one row = one trajectory): steps/epoch=$SPE"
fi
echo "checkpoint policy: save_steps=1 keep_weights_every=$SPE (one full ckpt + one weights-only ckpt per epoch)"
export WORKER_NUM=$((NODES - 1)); export JOB_TAG=""; export JOB_NAME="chenth_${RUN}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/llamafactory-agent-sft:tf5.12.1"
WANDB_KEY=$(awk '/8.150.1.98/{f=1} f&&/password/{print $2; exit}' /root/.netrc 2>/dev/null)
[ -n "$WANDB_KEY" ] || WANDB_KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key 2>/dev/null)
[ -n "$WANDB_KEY" ] || echo "WARNING: no wandb key found (/root/.netrc, /storage/.../.secrets/wandb_key); run will not be logged" >&2
export JOB_COMMAND="\
cd $REPO \
&& N_NODES=$NODES CONFIG=$CONFIG WANDB_API_KEY=${WANDB_KEY} RUN_NAME=${RUN} \
   EXTRA='dataset_dir=$DD dataset=$DS tokenized_path=$TP output_dir=/storage/openpsi/users/chen/checkpoints/${RUN} overwrite_output_dir=false overwrite_cache=true gradient_accumulation_steps=$GA $MAXSTEPS save_steps=1 keep_weights_every=$SPE save_total_limit=1000 ${OV:-}' \
   bash coding_sft/launch/ais_train.sh"
echo "JOB_COMMAND: $JOB_COMMAND"
pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ >/dev/null
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python3 /storage/openpsi/users/public/projects/AReaL-miumiu/submit_template.py
