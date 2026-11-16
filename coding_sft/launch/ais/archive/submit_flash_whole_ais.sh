#!/bin/bash
# Submit experiment 2 ("flash full, reasoning visible, whole trajectory") to AIStudio:
#   NODES=2 bash coding_sft/launch/ais/submit_flash_whole_ais.sh
#
# Copy of submit_ablation_ais.sh pinned to the flash_whole dataset / config. NODES x 8 GPU,
# gradient_accumulation_steps = 64 / (NODES*8) -> effective batch 64 complete trajectories.
# "Per epoch" checkpoints: save_steps = keep_weights_every = ceil(n_rows / 64), computed from the
# dataset here so the yaml never goes stale. Prerequisite: dataset + tokenized cache exist
# (coding_sft/prep/build_flash_whole.py, then coding_sft/prep/build_tokenized.py <config>).
# The job command runs coding_sft/launch/ais_train.sh on every node; it auto-resumes from the newest
# complete checkpoint after every pre-emption and keeps appending to the same wandb run.
export AISTUDIO_LOGIN_NAME="aistudio"; export AISTUDIO_USERNUMBER="492801"; export AISTUDIO_TOKEN="${AISTUDIO_TOKEN:?Set AISTUDIO_TOKEN before submitting}"
REPO=/storage/openpsi/users/chen/llama-factory          # the checkout the pods see (afsync mirror of /home/admin)
NODES=${NODES:-2}; GA=$((64 / (NODES * 8)))
CONFIG=${CONFIG:-coding_sft/configs/qwen3_5_flash_whole.yaml}
DD=/storage/openpsi/users/chen/data/flash_full_whole; DS=flash_whole
TP=$DD/tokenized_qwen3_5_text_262144
RUN=${RUN_NAME:-flash_whole_ais${NODES}n}
OUT=${OUT_DIR:-/storage/openpsi/users/chen/checkpoints/$RUN}
[ -f "$DD/$DS.jsonl" ] || { echo "ERROR: dataset missing ($DD/$DS.jsonl)" >&2; exit 1; }
[ -f "$TP/dataset_dict.json" ] || echo "WARNING: tokenized cache missing ($TP); node 0 will build it before training" >&2
N_ROWS=$(wc -l < "$DD/$DS.jsonl"); SPE=$(( (N_ROWS + 63) / 64 ))
echo "dataset rows=$N_ROWS -> steps/epoch=$SPE (save_steps=keep_weights_every=$SPE)"
export WORKER_NUM=$((NODES - 1)); export JOB_TAG=""; export JOB_NAME=${JOB_NAME:-chenth_flash_whole}
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/llamafactory-agent-sft:tf5.12.1"
WANDB_KEY=$(awk '/8.150.1.98/{f=1} f&&/password/{print $2; exit}' /root/.netrc 2>/dev/null)
[ -n "$WANDB_KEY" ] || WANDB_KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key 2>/dev/null)
[ -n "$WANDB_KEY" ] || echo "WARNING: no wandb key found (/root/.netrc, /storage/.../.secrets/wandb_key); run will not be logged" >&2
export JOB_COMMAND="\
cd $REPO \
&& N_NODES=$NODES WANDB_API_KEY=${WANDB_KEY} CONFIG=$CONFIG RUN_NAME=$RUN \
   EXTRA='dataset_dir=$DD dataset=$DS tokenized_path=$TP output_dir=$OUT overwrite_output_dir=false gradient_accumulation_steps=$GA save_steps=$SPE keep_weights_every=$SPE ${EXTRA_OV:-}' \
   bash coding_sft/launch/ais_train.sh"
echo "JOB_COMMAND: $JOB_COMMAND"
pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ >/dev/null
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python3 /storage/openpsi/users/public/projects/AReaL-miumiu/submit_template.py
