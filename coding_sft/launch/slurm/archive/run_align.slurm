#!/bin/bash
#SBATCH --job-name=chenth_align
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Alignment suite (TH-3K condition-mismatch experiment), always 2 nodes x 8 GPU, one torchrun per node.
# One optimizer step == one 64-trajectory group of group_schedule.json for every arm:
#   ARM=base  sbatch --reservation=rsi --job-name=chenth_align_base  coding_sft/launch/slurm/run_align.slurm
#   ARM=perdh sbatch --reservation=rsi --job-name=chenth_align_perdh coding_sft/launch/slurm/run_align.slurm
# The arm's yaml carries everything (tokenized_path = schedule-ordered cache, ga, disable_shuffling).
# Re-submitting the same ARM resumes from the newest checkpoint (overwrite_output_dir=false).
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
ARM=${ARM:?base|perdh}
NN=${SLURM_JOB_NUM_NODES}; [ "$NN" = 2 ] || { echo "ERROR: the alignment suite is defined for exactly 2 nodes (got $NN)" >&2; exit 1; }
CFG=coding_sft/configs/qwen3_5_align_${ARM}.yaml
TOK=$(sed -n 's/^tokenized_path: *//p' "$REPO/$CFG")
[ -f "$TOK/dataset_dict.json" ] || { echo "ERROR: tokenized cache missing: $TOK" >&2; exit 1; }
KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key)
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
PORT=$((20000 + SLURM_JOB_ID % 10000))
RUN=${RUN_NAME:-align_${ARM}}
OUT=${OUT_DIR:-/storage/openpsi/users/chen/checkpoints/${RUN}}
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
echo "Job ${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} master=${MASTER}:${PORT} start $(date) arm=${ARM} cfg=${CFG} cache=${TOK} extra=[${EXTRA_OV:-}]"

cat > "$INNER" <<EOF
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export NNODES=${NN} MASTER_ADDR=${MASTER} MASTER_PORT=${PORT}
export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 NPROC_PER_NODE=8 ALLOW_EXTRA_ARGS=1 RECORD_VRAM=1
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
export WANDB_BASE_URL=http://8.150.1.98:8080 WANDB_PROJECT=chenth_terminal_agent_sft WANDB_DIR=/storage/openpsi/users/chen/runs/wandb
export WANDB_RUN_ID=${RUN} WANDB_RESUME=allow
cd ${REPO}; export PYTHONPATH=${REPO}/src
echo "[\$(hostname)] node rank \$NODE_RANK of ${NN}, master ${MASTER}:${PORT}"
llamafactory-cli train ${CFG} output_dir=${OUT} overwrite_output_dir=false report_to=wandb run_name=${RUN} ${EXTRA_OV:-}
EOF

srun --mpi=none --ntasks="${NN}" --ntasks-per-node=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        --env WANDB_API_KEY='${KEY}' --env NODE_RANK=\$SLURM_PROCID '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
