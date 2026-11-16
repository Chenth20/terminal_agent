#!/bin/bash
#SBATCH --job-name=chenth_dbg_tree
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Operational A-vs-B parity: arm B (tree) on the dbg cache, same 8 trajectories / 8 steps / ga=8 as
# dbg_align_perdh. Step 1 has lr=0, so its loss must equal dbg_align_perdh's step 1 (1.41041).
#   sbatch --reservation=rsi coding_sft/launch/slurm/run_dbg_tree.slurm
#   EXTRA_OV="max_steps=1" sbatch --reservation=rsi --export=ALL,EXTRA_OV=max_steps=1 .../run_dbg_tree.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
CFG=${CFG:-coding_sft/configs/_dbg_tree.yaml}
RUN=${RUN_NAME:-dbg_align_tree}
OUT=${OUT_DIR:-/storage/openpsi/users/chen/checkpoints/${RUN}}
PORT=$((20000 + SLURM_JOB_ID % 10000))
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh

cat > "$INNER" <<EOS
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
# slurm's gres does not export CUDA_VISIBLE_DEVICES into the (--cleanenv) container here: pick the GPU with the
# least used memory so two 1-GPU jobs on one node (or someone else's job) never share a card.
export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "[gpu] using CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES ( \$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\\n' ';') )"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT}
export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface  TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TREE_KCACHE_BASE=/storage/openpsi/users/chen/.cache/kernels/${RUN} TORCHINDUCTOR_FX_GRAPH_CACHE=1 TREE_ATTN_SPLIT=${TREE_ATTN_SPLIT:-0} TREE_SPLIT_MULT=${TREE_SPLIT_MULT:-32768}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
cd ${REPO}; export PYTHONPATH=${REPO}/src
torchrun --nproc_per_node 1 --master_addr 127.0.0.1 --master_port ${PORT} \\
    coding_sft/patches/train_tree.py ${CFG} output_dir=${OUT} ${EXTRA_OV:-}
EOS

echo "Job ${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} start $(date) cfg=${CFG} out=${OUT} extra=[${EXTRA_OV:-}]"
srun --mpi=none --ntasks=1 --gres=gpu:1 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"