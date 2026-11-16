#!/bin/bash
# AIStudio submission: the SAME TerminalHorizon-3K 256K 2-epoch run as the Slurm job 983941
# (coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml), on 4 x 8 H200.
# Same effective batch as Slurm: 64 trajectories (32 GPUs x ga 2 here, 8 GPUs x ga 8 there) -> 69 steps.
# Separate output_dir / run_name. Data, tokenized cache, model, template, optimizer: identical.
# AIS runs at k8s_priority=low and gets preempted (SIGTERM) every 10-30 min, after which the platform
# restarts the same JOB_COMMAND on fresh pods: overwrite_output_dir=false makes LF auto-resume from the
# newest checkpoint. The yaml saves the full state EVERY step (previous one stripped/deleted by
# keep_optimizer_only_last / keep_weights_every=10), so at most one step is lost per pre-emption and
# a weights-only checkpoint is kept every 10 steps.
export AISTUDIO_LOGIN_NAME="aistudio"
export AISTUDIO_USERNUMBER="492801"
export AISTUDIO_TOKEN="${AISTUDIO_TOKEN:?Set AISTUDIO_TOKEN before submitting}"

export WORKER_NUM="3"                      # 4 nodes = 1 master + 3 workers
export JOB_TAG=""
export JOB_NAME="chenth_th3k_256k_2ep_ais"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/llamafactory-agent-sft:tf5.12.1"

WANDB_KEY=$(awk '/8.150.1.98/{f=1} f&&/password/{print $2; exit}' /root/.netrc 2>/dev/null)
export JOB_COMMAND="\
cd /storage/openpsi/users/chen/llama-factory \
&& N_NODES=4 WANDB_API_KEY=${WANDB_KEY} RUN_NAME=th3k_256k_2ep_bs64_ais4n \
   EXTRA='output_dir=/storage/openpsi/users/chen/checkpoints/th3k_256k_2ep_qwen3.5_4b_ais4n overwrite_output_dir=false gradient_accumulation_steps=2' \
   bash coding_sft/launch/ais_train.sh"

pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python3 /storage/openpsi/users/public/projects/AReaL-miumiu/submit_template.py
