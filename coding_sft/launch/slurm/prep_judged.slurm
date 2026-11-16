#!/bin/bash
#SBATCH --job-name=chenth_prep_judged
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# One CPU-only job that prepares both runs of the judged dsv41 dataset:
#   1. Qwen3.5-9B text-only export (models/Qwen__Qwen3.5-9B-text; GPU not needed without --verify)
#   2. 4B data:  data/terminal_horizon_dsv41_judged      tokenized_qwen3_5_text_262144 + sched16_gsort
#   3. 9B data:  data/terminal_horizon_dsv41_judged_9b   tokenized_qwen3_5_text_262144 + sched32_gsort
# Every step is skipped when its output already exists (prep_and_sched.py is idempotent).
#
#   sbatch --reservation=rsi coding_sft/launch/slurm/prep_judged.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
SRC=/storage/openpsi/users/chen/code_sft_data/dsv41_judged/dsv41_judged.jsonl
MAN=/storage/openpsi/users/chen/code_sft_data/dsv41_judged/dsv41_judged.manifest.jsonl

srun --mpi=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --bind /storage:/storage --cleanenv "${SIF}" bash -c "
        set -euo pipefail
        export HOME=/tmp/h_\$\$; mkdir -p \$HOME
        export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface
        export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
        cd ${REPO}; export PYTHONPATH=${REPO}/src

        if [ ! -f /storage/openpsi/users/chen/models/Qwen__Qwen3.5-9B-text/config.json ]; then
            echo '=== [1/3] export 9B text-only'
            python coding_sft/prep/export_qwen3_5_text_only.py \
                --src /storage/openpsi/users/chen/models/Qwen__Qwen3.5-9B \
                --dst /storage/openpsi/users/chen/models/Qwen__Qwen3.5-9B-text
        else
            echo '=== [1/3] 9B text-only already exported'
        fi

        echo '=== [2/3] 4B dataset + cache + sched16'
        python coding_sft/prep/prep_and_sched.py --src ${SRC} --manifest ${MAN} \
            --dir /storage/openpsi/users/chen/data/terminal_horizon_dsv41_judged --name dsv41_judged \
            --world 16 --epochs 8 --workers ${SLURM_CPUS_PER_TASK}

        echo '=== [3/3] 9B dataset + cache + sched32'
        python coding_sft/prep/prep_and_sched.py --src ${SRC} --manifest ${MAN} \
            --dir /storage/openpsi/users/chen/data/terminal_horizon_dsv41_judged_9b --name dsv41_judged_9b \
            --world 32 --epochs 8 --workers ${SLURM_CPUS_PER_TASK} \
            --config coding_sft/configs/qwen3_5_9b_terminal_agent_sft_256k.yaml \
            --tokenizer /storage/openpsi/users/chen/models/Qwen__Qwen3.5-9B
    "
echo "Job ${SLURM_JOB_ID} finished $(date)"