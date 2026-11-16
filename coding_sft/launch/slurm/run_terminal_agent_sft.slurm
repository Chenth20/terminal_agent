#!/bin/bash
#SBATCH --job-name=chenth_th3k_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/slurm-%x-%j.err
#
# Full SFT of Qwen3.5-4B on TerminalHorizon-3K inside the agent-sft SIF.
#
#   sbatch coding_sft/launch/slurm/run_terminal_agent_sft.slurm                 # full run
#   SMOKE=1 sbatch coding_sft/launch/slurm/run_terminal_agent_sft.slurm         # 20-step smoke test
#   EXTRA="learning_rate=2e-5" sbatch coding_sft/launch/slurm/run_terminal_agent_sft.slurm
#   WANDB_API_KEY=... RUN_NAME=xxx sbatch ...   # log to self-hosted W&B (off when key unset)
#   sbatch --reservation=rsi coding_sft/launch/slurm/run_terminal_agent_sft.slurm   # when no idle nodes
#
# Pre-requisite (once, any node with the data visible):
#   python coding_sft/prep/prepare_terminal_horizon.py --max-tokens 32768   # -> /storage/openpsi/users/chen/data/terminal_horizon
set -euo pipefail

REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
CONFIG=${CONFIG:-coding_sft/configs/archive/qwen3_5_terminal_agent_sft.yaml}
EXTRA=${EXTRA:-}
RUN_NAME=${RUN_NAME:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
WANDB_PROJECT=${WANDB_PROJECT:-chenth_terminal_agent_sft}
WANDB_BASE_URL=${WANDB_BASE_URL:-http://8.150.1.98:8080}
if [ -n "$WANDB_API_KEY" ]; then
    EXTRA="$EXTRA report_to=wandb"
    WANDB_ENV="export WANDB_BASE_URL=${WANDB_BASE_URL} WANDB_PROJECT=${WANDB_PROJECT} WANDB_DIR=/storage/openpsi/users/chen/runs/wandb; mkdir -p /storage/openpsi/users/chen/runs/wandb"
else
    WANDB_ENV="export WANDB_DISABLED=true"
fi
[ -n "$RUN_NAME" ] && EXTRA="$EXTRA run_name=${RUN_NAME}"

if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="$EXTRA max_steps=20 max_samples=256 logging_steps=1 save_steps=20 save_total_limit=1 \
        output_dir=/storage/openpsi/users/chen/checkpoints/smoke_terminal_horizon_3k_qwen3.5_4b overwrite_cache=true"
fi

echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date)  config=${CONFIG}  extra=[${EXTRA}]"

srun --mpi=none --nodes=1 --ntasks=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --nv --pid \
        --bind /storage:/storage \
        --cleanenv \
        --env WANDB_API_KEY="${WANDB_API_KEY}" \
        "${SIF}" \
        bash -c "
            set -euo pipefail
            export NCCL_DEBUG=WARN
            export NCCL_NVLS_ENABLE=0
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            export FORCE_TORCHRUN=1
            export NPROC_PER_NODE=8
            ${WANDB_ENV}
            export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface
            # triton JIT cache must be node-local: on NFS, 8 ranks compiling at once hit 'Stale file handle'
            export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export ALLOW_EXTRA_ARGS=1
            # terminal output contains literal <image>/<video>/<audio> text; keep the VLM
            # template's media placeholders from matching it (see inspect_sft_labels.py)
            export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>'
            export VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>'
            export AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
            cd ${REPO}
            export PYTHONPATH=${REPO}/src   # run THIS checkout, not the snapshot baked into the image
            nvidia-smi --query-gpu=index,name,memory.used --format=csv
            python -c 'import torch, transformers, deepspeed, liger_kernel; from llamafactory.extras.env import VERSION; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__, \"deepspeed\", deepspeed.__version__, \"llamafactory\", VERSION)'
            llamafactory-cli train ${CONFIG} ${EXTRA}
        "

echo "Job ${SLURM_JOB_ID} finished $(date)"
