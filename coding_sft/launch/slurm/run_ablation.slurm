#!/bin/bash
#SBATCH --job-name=chenth_abl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Batch ablation run (resumable). Usage:
#   VARIANT=mask_cut sbatch --job-name=chenth_abl_mask_cut coding_sft/launch/slurm/run_ablation.slurm
#   VARIANT=mask_cut sbatch --reservation=rsi ... (when no idle node)
# Re-submitting the same VARIANT resumes from the newest complete checkpoint (overwrite_output_dir=false).
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
VARIANT=${VARIANT:?}
read -r DD DS OV < <(VARIANT=$VARIANT bash $REPO/coding_sft/launch/ablation.sh)
KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key)  # batch nodes have no /root/.netrc
echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date) variant=${VARIANT} data=${DD}/${DS} extra=[${OV:-}]"
srun --mpi=none --nodes=1 --ntasks=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv --env WANDB_API_KEY="${KEY}" "${SIF}" bash -c "
        set -euo pipefail
        export HOME=/tmp/h_\$\$; mkdir -p \$HOME
        export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 NPROC_PER_NODE=8 ALLOW_EXTRA_ARGS=1 RECORD_VRAM=1
        export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
        export WANDB_BASE_URL=http://8.150.1.98:8080 WANDB_PROJECT=chenth_terminal_agent_sft WANDB_DIR=/storage/openpsi/users/chen/runs/wandb
        export WANDB_RUN_ID=abl_${VARIANT}_slurm WANDB_RESUME=allow   # re-submits append to the same wandb run
        cd ${REPO}; export PYTHONPATH=${REPO}/src
        llamafactory-cli train coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml dataset_dir=${DD} dataset=${DS} \
            tokenized_path=${DD}/tokenized_qwen3_5_text_262144 output_dir=/storage/openpsi/users/chen/checkpoints/abl_${VARIANT}_slurm \
            overwrite_output_dir=false gradient_accumulation_steps=8 report_to=wandb run_name=abl_${VARIANT}_slurm ${OV:-}
    "
echo "Job ${SLURM_JOB_ID} finished $(date)"
