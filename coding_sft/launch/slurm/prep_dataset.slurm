#!/bin/bash
#SBATCH --job-name=chenth_prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
# CPU-only data prep for one variant, then its tokenized cache:
#   PREP_ARGS="--bad-pair keep --overlong drop" NAME=th3k_keep sbatch coding_sft/launch/slurm/prep_dataset.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
NAME=${NAME:?}; PREP_ARGS=${PREP_ARGS:?}
OUT=/storage/openpsi/users/chen/data/terminal_horizon_${NAME}
srun --mpi=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --bind /storage:/storage --cleanenv "${SIF}" bash -c "
        set -euo pipefail
        export HOME=/tmp/h_\$\$; mkdir -p \$HOME HF_HOME=/storage/openpsi/users/chen/.cache/huggingface ALLOW_EXTRA_ARGS=1
        export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
        cd ${REPO}; export PYTHONPATH=${REPO}/src
        python coding_sft/prep/prepare_terminal_horizon.py --no-chunk --max-tokens 262144 --workers ${SLURM_CPUS_PER_TASK} ${PREP_ARGS} --out-dir ${OUT} --name ${NAME}
        python coding_sft/prep/build_tokenized.py coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml \
            --set dataset_dir=${OUT} dataset=${NAME} tokenized_path=${OUT}/tokenized_qwen3_5_text_262144 overwrite_cache=true
    "
echo "Job ${SLURM_JOB_ID} finished $(date)"
