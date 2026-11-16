#!/bin/bash
#SBATCH --job-name=chenth_demo_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/slurm-%x-%j.err
#
# Small end-to-end image validation on data/alpaca_en_demo.json with W&B logging.
#
#   SIF=/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif \
#   RUN_NAME=demo_tf5.12.1 sbatch coding_sft/launch/slurm/run_demo_test.slurm
#
# W&B: self-hosted server http://8.150.1.98:8080, project ${WANDB_PROJECT:-chenth_lf_image_test}.
# Set WANDB_API_KEY in the environment (it is NOT stored in this file).
set -euo pipefail

REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
CONFIG=${CONFIG:-coding_sft/configs/archive/qwen3_5_demo_test.yaml}
RUN_NAME=${RUN_NAME:-demo_$(basename "$SIF" .sif)}
WANDB_PROJECT=${WANDB_PROJECT:-chenth_lf_image_test}
WANDB_BASE_URL=${WANDB_BASE_URL:-http://8.150.1.98:8080}
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
EXTRA=${EXTRA:-}

echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date)  sif=${SIF}  run=${RUN_NAME}"

srun --mpi=none --nodes=1 --ntasks=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --nv --pid \
        --bind /storage:/storage \
        --cleanenv \
        --env WANDB_API_KEY="${WANDB_API_KEY}" \
        "${SIF}" \
        bash -c "
            set -euo pipefail
            export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
            export FORCE_TORCHRUN=1 NPROC_PER_NODE=8
            export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface
            # triton JIT cache must be node-local: on NFS, 8 ranks compiling at once hit 'Stale file handle'
            export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export ALLOW_EXTRA_ARGS=1
            export WANDB_BASE_URL=${WANDB_BASE_URL} WANDB_PROJECT=${WANDB_PROJECT} WANDB_NAME=${RUN_NAME}
            export WANDB_DIR=/storage/openpsi/users/chen/runs/wandb
            mkdir -p \$WANDB_DIR
            cd ${REPO}
            export PYTHONPATH=${REPO}/src   # run THIS checkout, not the snapshot baked into the image
            python -c 'import torch, transformers, deepspeed, liger_kernel, wandb; from llamafactory.extras.env import VERSION; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__, \"deepspeed\", deepspeed.__version__, \"wandb\", wandb.__version__, \"llamafactory\", VERSION)'
            llamafactory-cli train ${CONFIG} run_name=${RUN_NAME} output_dir=/storage/openpsi/users/chen/checkpoints/${RUN_NAME} ${EXTRA}
        "

echo "Job ${SLURM_JOB_ID} finished $(date)"
