#!/bin/bash
#SBATCH --job-name=chenth_tokenize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# VARIANT=mask_tail sbatch coding_sft/launch/slurm/build_tokenized.slurm
set -euo pipefail
repo=${REPO:-/storage/openpsi/users/chen/llama-factory}
: "${VARIANT:?set VARIANT to an ablation variant}"

selection=$(VARIANT="$VARIANT" bash "$repo/coding_sft/launch/ablation.sh")
read -r dataset_dir dataset overrides <<< "$selection"
cache="$dataset_dir/tokenized_qwen3_5_text_262144"
echo "Job ${SLURM_JOB_ID} on $(hostname) variant=$VARIANT"
CONFIG=coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml SET="$overrides" \
    exec bash "$repo/coding_sft/launch/slurm/build_tokenized_generic.slurm" \
        "dataset_dir=$dataset_dir" "dataset=$dataset" "tokenized_path=$cache"
