#!/bin/bash
# Submit one arm-B (tree) run to AIStudio:
#   NAME=flash NODES=4 bash coding_sft/launch/ais/submit_tree_ais.sh
#   NAME=flash_solved NODES=2 bash coding_sft/launch/ais/submit_tree_ais.sh
#   NAME=th3k        NODES=2 bash coding_sft/launch/ais/submit_tree_ais.sh
# Effective batch = 64 complete trajectories: ga = 64 / (NODES*8). Saves once per epoch.
export AISTUDIO_LOGIN_NAME="aistudio"; export AISTUDIO_USERNUMBER="492801"; export AISTUDIO_TOKEN="${AISTUDIO_TOKEN:?Set AISTUDIO_TOKEN before submitting}"
REPO=/storage/openpsi/users/chen/llama-factory
NAME=${NAME:?flash|flash_solved|th3k}; NODES=${NODES:-2}; WORLD=$((NODES * 8))
case "$NAME" in
  flash)        TOK=$REPO/../data/flash_full_tree/tokenized_qwen3_5_text_262144_sched${WORLD};;
  flash_solved) TOK=$REPO/../data/flash_solved_tree/tokenized_qwen3_5_text_262144_sched${WORLD};;
  th3k)         TOK=$REPO/../data/th3k_tree/tokenized_qwen3_5_text_262144_sched${WORLD};;
  *) echo "unknown NAME $NAME" >&2; exit 1;;
esac
TOK=$(readlink -f "$TOK")
[ -f "$TOK/dataset_dict.json" ] || { echo "ERROR: sched tree cache missing: $TOK (build_tree_sched.py --world $WORLD)" >&2; exit 1; }
# the sched cache fixes everything: step s = 64 trajectories, ga = G from its manifest, one row per micro-batch
read -r GA SPE STEPS N_ROWS < <(python3 -c "import json; m=json.load(open('$TOK/manifest.json')); assert m['world']==$WORLD, m['world']; print(m['G'], m['steps_per_epoch'], m['steps'], m['rows'])")
# save EVERY step with full resume state (optimizer/scheduler/DeepSpeed shards); after each save
# KeepOptimizerOnlyLastCallback deletes the previous step's checkpoint entirely unless its step is a
# multiple of SPE (= steps/epoch), in which case it is stripped to a weights-only checkpoint for eval.
# Net disk: 1 full ckpt (newest) + one weights-only ckpt per epoch. A pre-emption now loses at most 1 step.
# save_total_limit is raised so HF's own rotation never touches the epoch checkpoints.
SAVE=1
echo "sched cache rows=$N_ROWS -> $STEPS steps ($SPE/epoch), ga=$GA (save_steps=$SAVE, keep_weights_every=$SPE)"
RUN=${RUN_NAME:-tree_${NAME}_ais${NODES}n_v4}; OUT=/storage/openpsi/users/chen/checkpoints/${RUN}
export WORKER_NUM=$((NODES - 1)); export JOB_TAG=""; export JOB_NAME="chenth_${RUN}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/llamafactory-agent-sft:tf5.12.1"
WANDB_KEY=$(awk '/8.150.1.98/{f=1} f&&/password/{print $2; exit}' /root/.netrc 2>/dev/null)
[ -n "$WANDB_KEY" ] || WANDB_KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key 2>/dev/null)
export JOB_COMMAND="\
cd $REPO \
&& N_NODES=$NODES CONFIG=coding_sft/configs/qwen3_5_tree_sft.yaml WANDB_API_KEY=${WANDB_KEY} RUN_NAME=${RUN} \
   TREE_ENTRY=coding_sft/patches/train_tree.py \
   EXTRA='tokenized_path=$TOK output_dir=$OUT gradient_accumulation_steps=$GA max_steps=$STEPS save_steps=$SAVE keep_weights_every=$SPE save_total_limit=1000 overwrite_output_dir=false' \
   bash coding_sft/launch/ais_train.sh"
echo "JOB_COMMAND: $JOB_COMMAND"
pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ >/dev/null
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python3 /storage/openpsi/users/public/projects/AReaL-miumiu/submit_template.py
