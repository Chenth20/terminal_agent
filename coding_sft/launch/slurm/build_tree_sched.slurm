#!/bin/bash
#SBATCH --job-name=chenth_tree_sched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Build the trajectory-scheduled tree caches (64 trajectories per optimizer step, rank-major, dummy-padded):
#   th3k : rebuild data/th3k_tree with --split-tokens (same 2441 trajectories as th3k_mask) -> _sched16 (2 nodes)
#   flash: data/flash_full_tree (already split) -> _sched32 (4 nodes)
#   WHICH=th3k|flash|flash_solved|both WORLD=16  sbatch coding_sft/launch/slurm/build_tree_sched.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
DATA=/storage/openpsi/users/chen/data
WHICH=${WHICH:-both}; SPLIT=${SPLIT:-294912}; EPOCHS=${EPOCHS:-8}; WORLD=${WORLD:-16}
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
cat > "$INNER" <<EOS
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface ALLOW_EXTRA_ARGS=1
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
cd ${REPO}; export PYTHONPATH=${REPO}/src:${REPO}
if [ "${WHICH}" = th3k ] || [ "${WHICH}" = both ]; then
  echo "=== th3k tree cache (split ${SPLIT}, overlong-conv drop = th3k_mask's 2441 rows) \$(date)"
  python coding_sft/prep/prepare_tree_generic.py \\
      --input /storage/openpsi/users/chen/code_data/terminal_horizon/TerminalHorizon-3K/data/TerminalHorizon-3K-SFT.jsonl \\
      --name th3k --out ${DATA}/th3k_tree/tokenized_qwen3_5_text_262144 --cutoff 262144 --split-tokens ${SPLIT} \\
      --bad-pair mask --overlong-conv drop --pad-token-id 248044 --workers 36
  echo "=== th3k sched (${WORLD} ranks) \$(date)"
  python coding_sft/prep/build_tree_sched.py --cache ${DATA}/th3k_tree/tokenized_qwen3_5_text_262144 \\
      --out ${DATA}/th3k_tree/tokenized_qwen3_5_text_262144_sched${WORLD} --world ${WORLD} --epochs ${EPOCHS}
fi
if [ "${WHICH}" = flash ] || [ "${WHICH}" = both ]; then
  echo "=== flash sched (${WORLD} ranks) \$(date)"
  python coding_sft/prep/build_tree_sched.py --cache ${DATA}/flash_full_tree/tokenized_qwen3_5_text_262144 \\
      --manifest-src /storage/openpsi/users/chen/code_sft_data/TerminalHorizon-dsv41-96k-SFT.manifest.jsonl \\
      --out ${DATA}/flash_full_tree/tokenized_qwen3_5_text_262144_sched${WORLD} --world ${WORLD} --epochs ${EPOCHS}
fi
if [ "${WHICH}" = flash_solved ] || [ "${WHICH}" = both ]; then
  echo "=== flash solved sched (${WORLD} ranks, score==1 rows of the flash tree cache) \$(date)"
  mkdir -p ${DATA}/flash_solved_tree
  python coding_sft/prep/build_tree_sched.py --cache ${DATA}/flash_full_tree/tokenized_qwen3_5_text_262144 \\
      --manifest-src /storage/openpsi/users/chen/code_sft_data/TerminalHorizon-dsv41-96k-SFT.manifest.jsonl --select-score 1 \\
      --out ${DATA}/flash_solved_tree/tokenized_qwen3_5_text_262144_sched${WORLD} --world ${WORLD} --epochs ${EPOCHS}
fi
echo "=== done \$(date)"
EOS
echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date) which=${WHICH}"
srun --mpi=none --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} \
    bash -c "exec singularity exec --no-home --writable-tmpfs --bind /storage:/storage --cleanenv '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
