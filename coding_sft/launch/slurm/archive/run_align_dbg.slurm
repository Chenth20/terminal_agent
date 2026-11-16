#!/bin/bash
#SBATCH --job-name=chenth_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# DEBUG alignment suite: ONE node x 8 GPU per arm, 8 optimizer steps on the 32 shortest trajectories
# (group_schedule_dbg.json), loss curves compared step by step across base / perdh / tree.
#   ARM=base  sbatch --reservation=rsi --job-name=chenth_dbg_base  coding_sft/launch/slurm/run_align_dbg.slurm
#   ARM=perdh sbatch --reservation=rsi --job-name=chenth_dbg_perdh coding_sft/launch/slurm/run_align_dbg.slurm
#   ARM=tree  sbatch --reservation=rsi --job-name=chenth_dbg_tree  coding_sft/launch/slurm/run_align_dbg.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
ARM=${ARM:?base|perdh|tree}
CFG=coding_sft/configs/qwen3_5_dbg_${ARM}.yaml
TOK=$(sed -n 's/^tokenized_path: *//p' "$REPO/$CFG")
[ -f "$TOK/dataset_dict.json" ] || { echo "ERROR: tokenized cache missing: $TOK" >&2; exit 1; }
KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key)
PORT=$((20000 + SLURM_JOB_ID % 10000))
RUN=${RUN_NAME:-dbg_align_${ARM}}
OUT=${OUT_DIR:-/storage/openpsi/users/chen/checkpoints/${RUN}}
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
echo "Job ${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} start $(date) arm=${ARM} cfg=${CFG} cache=${TOK} extra=[${EXTRA_OV:-}]"

cat > "$INNER" <<EOS
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT}
export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 NPROC_PER_NODE=8 ALLOW_EXTRA_ARGS=1 RECORD_VRAM=1
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
export WANDB_BASE_URL=http://8.150.1.98:8080 WANDB_PROJECT=chenth_terminal_agent_sft WANDB_DIR=/storage/openpsi/users/chen/runs/wandb
export WANDB_RUN_ID=${RUN} WANDB_RESUME=allow
cd ${REPO}; export PYTHONPATH=${REPO}/src
llamafactory-cli train ${CFG} output_dir=${OUT} report_to=wandb run_name=${RUN} ${EXTRA_OV:-}
EOS

srun --mpi=none --ntasks=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        --env WANDB_API_KEY='${KEY}' '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
