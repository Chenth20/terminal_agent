#!/bin/bash
# Submit one alignment-suite arm to AIStudio (always 2 nodes x 8 GPU, one container per node):
#   ARM=base  bash coding_sft/launch/ais/submit_align_ais.sh
#   ARM=perdh bash coding_sft/launch/ais/submit_align_ais.sh
#   ARM=tree  bash coding_sft/launch/ais/submit_align_ais.sh
# The arm's yaml (coding_sft/configs/qwen3_5_align_<arm>.yaml) carries the schedule-ordered cache,
# ga and disable_shuffling; this script only adds output_dir / wandb and checks the cache exists.
# Re-submitting the same ARM resumes from the newest checkpoint and appends to the same wandb run.
export AISTUDIO_LOGIN_NAME="aistudio"; export AISTUDIO_USERNUMBER="492801"; export AISTUDIO_TOKEN="${AISTUDIO_TOKEN:?Set AISTUDIO_TOKEN before submitting}"
REPO=/storage/openpsi/users/chen/llama-factory          # the checkout the pods see (afsync mirror of /home/admin)
ARM=${ARM:?base|perdh|tree}; NODES=2
CFG=coding_sft/configs/qwen3_5_align_${ARM}.yaml
[ -f "$REPO/$CFG" ] || { echo "ERROR: $CFG not found" >&2; exit 1; }
TOK=$(sed -n 's/^tokenized_path: *//p' "$REPO/$CFG")
[ -f "$TOK/dataset_dict.json" ] || { echo "ERROR: tokenized cache missing: $TOK (build it first, never let the pods tokenize)" >&2; exit 1; }
RUN=align_${ARM}; OUT=/storage/openpsi/users/chen/checkpoints/${RUN}
export WORKER_NUM=$((NODES - 1)); export JOB_TAG=""; export JOB_NAME="chenth_${RUN}"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/llamafactory-agent-sft:tf5.12.1"
WANDB_KEY=$(awk '/8.150.1.98/{f=1} f&&/password/{print $2; exit}' /root/.netrc 2>/dev/null)
[ -n "$WANDB_KEY" ] || WANDB_KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key 2>/dev/null)
[ -n "$WANDB_KEY" ] || echo "WARNING: no wandb key found; run will not be logged" >&2
export JOB_COMMAND="\
cd $REPO \
&& N_NODES=$NODES CONFIG=$CFG WANDB_API_KEY=${WANDB_KEY} RUN_NAME=${RUN} \
   EXTRA='output_dir=$OUT overwrite_output_dir=false ${EXTRA_OV:-}' \
   bash coding_sft/launch/ais_train.sh"
echo "JOB_COMMAND: $JOB_COMMAND"
pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ >/dev/null
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python3 /storage/openpsi/users/public/projects/AReaL-miumiu/submit_template.py
