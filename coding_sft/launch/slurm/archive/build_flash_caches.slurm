#!/bin/bash
#SBATCH --job-name=chenth_build_flash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Rebuild BOTH flash-full caches from /storage/openpsi/users/chen/code_sft_data (CPU only, one node):
#   exp1  arm B tree   : data/flash_full_tree/tokenized_qwen3_5_text_262144   (--split-tokens SPLIT, bad-pair mask)
#   exp2  whole traj   : data/flash_full_whole/{flash_whole.jsonl, tokenized_qwen3_5_text_262144} (bad-pair mask, overlong drop)
#   WHICH=tree|whole|both (default both), SPLIT (default 262144), SRC jsonl override.
#   sbatch --reservation=rsi coding_sft/launch/slurm/build_flash_caches.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
SRC=${SRC:-/storage/openpsi/users/chen/code_sft_data/TerminalHorizon-dsv41-96k-SFT.jsonl}
WHICH=${WHICH:-both}
SPLIT=${SPLIT:-262144}
DATA=/storage/openpsi/users/chen/data
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
cat > "$INNER" <<EOS
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface ALLOW_EXTRA_ARGS=1
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
cd ${REPO}; export PYTHONPATH=${REPO}/src:${REPO}
if [ "${WHICH}" = tree ] || [ "${WHICH}" = both ]; then
  echo "=== tree cache (split ${SPLIT}) \$(date)"
  python coding_sft/prep/prepare_tree_generic.py --input ${SRC} --name flash_full \\
      --out ${DATA}/flash_full_tree/tokenized_qwen3_5_text_262144 --cutoff 262144 --split-tokens ${SPLIT} \\
      --bad-pair mask --overlong-conv keep --pad-token-id 248044 --workers 40
fi
if [ "${WHICH}" = whole ] || [ "${WHICH}" = both ]; then
  echo "=== whole-trajectory jsonl \$(date)"
  python coding_sft/prep/build_flash_whole.py --input ${SRC} --out-dir ${DATA}/flash_full_whole --name flash_whole \\
      --max-tokens 262144 --bad-pair mask
  echo "=== whole-trajectory tokenized cache \$(date)"
  python coding_sft/prep/build_tokenized.py coding_sft/configs/qwen3_5_flash_whole.yaml --set overwrite_cache=true
fi
echo "=== done \$(date)"
EOS
echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date) which=${WHICH} split=${SPLIT} src=${SRC}"
srun --mpi=none --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} \
    bash -c "exec singularity exec --no-home --writable-tmpfs --bind /storage:/storage --cleanenv '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
