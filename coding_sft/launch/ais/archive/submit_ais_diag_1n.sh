#!/bin/bash
# 1-node diagnostic on AIS: same launcher, 2 steps on the 8 longest dbg samples. Isolates
# image / code-binding / mount problems from multi-node scheduling problems.
export AISTUDIO_LOGIN_NAME="aistudio"; export AISTUDIO_USERNUMBER="492801"; export AISTUDIO_TOKEN="${AISTUDIO_TOKEN:?Set AISTUDIO_TOKEN before submitting}"
export WORKER_NUM="0"; export JOB_TAG=""; export JOB_NAME="chenth_lf_ais_diag_1n"
export KM_IMAGE="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/llamafactory-agent-sft:tf5.12.1"
export JOB_COMMAND="\
echo DIAG_START; hostname; id; which python python3 llamafactory-cli; python --version; ls /storage/openpsi/users/chen/llama-factory/coding_sft | head -3; nvidia-smi -L | head -2; \
cd /storage/openpsi/users/chen/llama-factory \
&& N_NODES=1 RUN_NAME=ais_diag_1n \
   EXTRA='dataset_dir=/storage/openpsi/users/chen/data/terminal_horizon_dbg128 dataset=dbg128_longest8 tokenized_path=null max_steps=2 save_strategy='"no"' output_dir=/storage/openpsi/users/chen/runs/ais/diag_1n overwrite_cache=true' \
   bash coding_sft/launch/ais_train.sh"
pip install "aistudio-common>=0.0.28.75" -i https://pypi.antfin-inc.com/simple/ >/dev/null
pip install aii-pypai -i https://pypi.antfin-inc.com/simple/ >/dev/null
export LAUNCH_CONTAINER_MODE=dev_local
aistudio_user login --name "${AISTUDIO_LOGIN_NAME}" --usernumber "${AISTUDIO_USERNUMBER}" --token "${AISTUDIO_TOKEN}"
python3 /storage/openpsi/users/public/projects/AReaL-miumiu/submit_template.py
