#!/bin/bash
#SBATCH --job-name=chenth_tok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# CONFIG=path/to/train.yaml SET='dataset=th3k_mask' sbatch <this>
# The variant wrapper passes dataset and cache defaults as positional arguments.
set -euo pipefail
repo=${REPO:-/storage/openpsi/users/chen/llama-factory}
image=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
: "${CONFIG:?set CONFIG to the training yaml}"
echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date) config=$CONFIG"

srun --mpi=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --bind /storage:/storage --cleanenv "$image" \
    bash -s -- "$repo" "$CONFIG" "${SET:-}" "$@" <<'CONTAINER'
set -euo pipefail
cd "$1"
shift
export HOME=/tmp/h_$$
mkdir -p "$HOME"
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface ALLOW_EXTRA_ARGS=1
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>'
export VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>'
export AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
export PYTHONPATH="$PWD/src"

# Preserve shell-style quoting in SET and apply its overrides last.
python - "$@" <<'TOKENIZE'
import os
import shlex
import sys

config, overrides, *defaults = sys.argv[1:]
args = [sys.executable, "coding_sft/prep/build_tokenized.py", config, "--set",
        *defaults, "overwrite_cache=true", *shlex.split(overrides)]
os.execv(sys.executable, args)
TOKENIZE
CONTAINER
echo "Job ${SLURM_JOB_ID} finished $(date)"
