#!/bin/bash
# Run a command inside the agent-sft image on the node of an existing (interactive) Slurm job.
#
#   JOBID=983941 coding_sft/launch/slurm/in_container.sh 'nvidia-smi; python -c "import torch"'
#
# Uses `srun --overlap` so it shares the job's allocation. Same env as run_terminal_agent_sft.slurm
# (PYTHONPATH=repo/src, media placeholders, node-local triton cache). GPUS=0 for CPU-only work.
set -euo pipefail
JOBID=${JOBID:?set JOBID to the running slurm job id}
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
GPUS=${GPUS:-8}
CMD=${1:?command string}

exec srun --jobid="$JOBID" --overlap --mpi=none -N1 -n1 --gres=gpu:${GPUS} ${CPUS:+--cpus-per-task=$CPUS} \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        --env WANDB_API_KEY="${WANDB_API_KEY:-}" "$SIF" bash -c "
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0
export FORCE_TORCHRUN=1 NPROC_PER_NODE=${GPUS} ALLOW_EXTRA_ARGS=1
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface
export TRITON_CACHE_DIR=/tmp/triton_cache_${JOBID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
cd ${REPO}; export PYTHONPATH=${REPO}/src
${CMD}
"
