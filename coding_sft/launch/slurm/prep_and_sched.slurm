#!/bin/bash
#SBATCH --job-name=chenth_prepsched
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
# CPU-only: raw jsonl -> prepped dataset -> tokenized cache -> trajectory schedule (prep/prep_and_sched.py),
# all inside the training image. One node, no GPU.
#   SRC=/storage/.../X.jsonl MANIFEST=/storage/.../X.manifest.jsonl NAME=dsv41_history \
#       sbatch coding_sft/launch/slurm/prep_and_sched.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
NAME=${NAME:?}; SRC=${SRC:?}; MANIFEST=${MANIFEST:?}; WORLD=${WORLD:-16}; ORDER=${ORDER:-global_sort}; OVERLONG=${OVERLONG:-drop}; THINK_CAP=${THINK_CAP:-0}; EPOCHS=${EPOCHS:-8}
OUT=/storage/openpsi/users/chen/data/terminal_horizon_${NAME}
mkdir -p "$OUT"
[ -f "$OUT/${NAME}.manifest.jsonl" ] || cp "$MANIFEST" "$OUT/${NAME}.manifest.jsonl"
srun --mpi=none --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --bind /storage:/storage --cleanenv "${SIF}" bash -c "
        set -euo pipefail
        export HOME=/tmp/h_\$\$; mkdir -p \$HOME; export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface ALLOW_EXTRA_ARGS=1
        export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
        cd ${REPO}; export PYTHONPATH=${REPO}/src:${REPO}
        python coding_sft/prep/prep_and_sched.py --src ${SRC} --manifest ${OUT}/${NAME}.manifest.jsonl \
            --dir ${OUT} --name ${NAME} --world ${WORLD} --order ${ORDER} --epochs ${EPOCHS} --overlong ${OVERLONG} --think-cap ${THINK_CAP} --workers ${SLURM_CPUS_PER_TASK}
    "
echo "Job ${SLURM_JOB_ID} finished $(date)"
