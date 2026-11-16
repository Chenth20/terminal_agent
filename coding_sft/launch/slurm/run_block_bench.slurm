#!/bin/bash
#SBATCH --job-name=chenth_block_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=360G
#SBATCH --time=00:40:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# ONE job / ONE node / 2 GPUs / <= 40 min: TREE_FLEX_BLOCK A/B on real rows (split attention stays on, i.e. the
# production config). GPU0: 0 (autotuned default) vs 64; GPU1: 128. Each arm --repeat 3, 3rd run timed.
# Two lengths only so all six arms fit the budget (the 287K row alone takes ~100 s/iter).
#   sbatch --reservation=game-efficient coding_sft/launch/slurm/run_block_bench.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
TOKENS=${TOKENS:-"120000 200000"}
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
cat > "$INNER" <<EOS
set -uo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
GPUS=(\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -2 | cut -d, -f1))
echo "[gpu] GPU \${GPUS[0]}: BLOCK=0,64   GPU \${GPUS[1]}: BLOCK=128"
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
export TREE_ATTN_SPLIT=1 TREE_SPLIT_MULT=32768
cd ${REPO}; export PYTHONPATH=${REPO}/src
for B in 0 64; do
  CUDA_VISIBLE_DEVICES=\${GPUS[0]} TRITON_CACHE_DIR=/tmp/tc_b\${B}_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/ic_b\${B}_${SLURM_JOB_ID} \\
    TREE_FLEX_BLOCK=\$B python coding_sft/checks/bench_tree.py --tokens ${TOKENS} --repeat 3 \\
    > /storage/openpsi/users/chen/runs/ablation/block_bench_b\${B}_${SLURM_JOB_ID}.log 2>&1 &
  PID=\$!
  if [ "\$B" = 0 ]; then wait \$PID; else wait \$PID; fi
done
for B in 128; do
  CUDA_VISIBLE_DEVICES=\${GPUS[1]} TRITON_CACHE_DIR=/tmp/tc_b\${B}_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/ic_b\${B}_${SLURM_JOB_ID} \\
    TREE_FLEX_BLOCK=\$B python coding_sft/checks/bench_tree.py --tokens ${TOKENS} --repeat 3 \\
    > /storage/openpsi/users/chen/runs/ablation/block_bench_b\${B}_${SLURM_JOB_ID}.log 2>&1
done
for B in 0 64 128; do
  echo "==== TREE_FLEX_BLOCK=\$B"; grep -E "^row |TREE_FLEX_BLOCK|^  T=" /storage/openpsi/users/chen/runs/ablation/block_bench_b\${B}_${SLURM_JOB_ID}.log | cut -c1-190
done
EOS
echo "Job ${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} start $(date)"
srun --mpi=none --ntasks=1 --gres=gpu:2 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
