#!/bin/bash
#SBATCH --job-name=chenth_state_conv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=00:40:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# M3: which recurrent state a branch must start from (fork-state off-by-one).
#   sbatch --reservation=rsi coding_sft/launch/slurm/run_check_state.slurm
#   sbatch --reservation=rsi --export=ALL,SCRIPT=coding_sft/checks/check_tree_synth.py ...   # other check
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
SCRIPT=${SCRIPT:-coding_sft/checks/check_state_chain.py}

INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
cat > "$INNER" <<EOS
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
# slurm's gres does not export CUDA_VISIBLE_DEVICES into the (--cleanenv) container here: pick the GPU with the
# least used memory so two 1-GPU jobs on one node (or someone else's job) never share a card.
export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "[gpu] using CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES ( \$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\\n' ';') )"
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
cd ${REPO}; export PYTHONPATH=${REPO}/src
python ${SCRIPT}
EOS

echo "Job ${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} start $(date) script=${SCRIPT}"
srun --mpi=none --ntasks=1 --gres=gpu:1 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"