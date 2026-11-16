#!/bin/bash
#SBATCH --job-name=chenth_abl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Multi-node batch ablation (resumable), one torchrun (8 GPU) per node, ga = 64 / (nodes*8):
#   VARIANT=mask_tail sbatch --nodes=4 --reservation=<res> --job-name=chenth_abl_mask_tail coding_sft/launch/slurm/run_ablation_multi.slurm
# Re-submitting the same VARIANT resumes from the newest complete checkpoint (overwrite_output_dir=false)
# and appends to the same wandb run (fixed WANDB_RUN_ID). The tokenized cache must exist
# (coding_sft/prep/build_tokenized.py) - all ranks load it, nobody builds it.
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
VARIANT=${VARIANT:?}
NN=${SLURM_JOB_NUM_NODES}; GA=$((64 / (NN * 8)))
read -r DD DS OV < <(VARIANT=$VARIANT bash $REPO/coding_sft/launch/ablation.sh)
[ -f "$DD/tokenized_qwen3_5_text_262144/dataset_dict.json" ] || { echo "ERROR: tokenized cache missing in $DD" >&2; exit 1; }
# checkpoint policy (see src/llamafactory/train/callbacks.py): full resume state every step + weights-only
# checkpoint at every epoch boundary (keep_weights_every = steps/epoch), everything else deleted.
# trajectory schedule (prep/prep_and_sched.py, same auto-detect as launch/ais/submit_ablation_ais.sh): prefer the
# global-sort cache (_gsort), else the legacy one; without a schedule fall back to plain per-row batching.
SCHED_TOK=${SCHED_TOK:-}
for cand in "$DD/tokenized_qwen3_5_text_262144_sched$((NN * 8))_gsort" "$DD/tokenized_qwen3_5_text_262144_sched$((NN * 8))"; do
    if [ -z "$SCHED_TOK" ] && [ -f "$cand/manifest.json" ] && [ -f "$DD/$DS.manifest.jsonl" ]; then
        SCHED_TOK=$cand; echo "found trajectory schedule: $SCHED_TOK"
    fi
done
if [ -n "$SCHED_TOK" ]; then
    read -r GA STEPS SPE < <(python3 -c "import json,sys; m=json.load(open(sys.argv[1]+'/manifest.json')); print(m['G'], m['steps'], m['steps_per_epoch'])" "$SCHED_TOK")
    MAXSTEPS="max_steps=$STEPS disable_shuffling=true"
    TP=$SCHED_TOK
    echo "trajectory schedule: $STEPS steps ($SPE/epoch = 64 trajectories each), ga=$GA"
else
    N_ROWS=$(python3 -c "import json;print(json.load(open('$DD/tokenized_qwen3_5_text_262144/train/dataset_info.json'))['splits']['train']['num_examples'])")
    SPE=$(( (N_ROWS + 63) / 64 ))
    MAXSTEPS=""
    TP=$DD/tokenized_qwen3_5_text_262144
fi
KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key)  # batch nodes have no /root/.netrc
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
PORT=$((20000 + SLURM_JOB_ID % 10000))
RUN=${RUN_NAME:-abl_${VARIANT}_slurm${NN}n}
OUT=${OUT_DIR:-/storage/openpsi/users/chen/checkpoints/${RUN}}      # point at a finished run to continue it
OV="${OV:-} ${EXTRA_OV:-}"                                          # EXTRA_OV: extra overrides (e.g. num_train_epochs=5)
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
echo "Job ${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} master=${MASTER}:${PORT} start $(date) variant=${VARIANT} ga=${GA} steps/epoch=${SPE} data=${DD}/${DS} extra=[${OV:-}]"

# per-node script, runs inside the container; NODE_RANK comes in via --env from the srun task id
cat > "$INNER" <<EOF
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export NNODES=${NN} MASTER_ADDR=${MASTER} MASTER_PORT=${PORT}
export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 NPROC_PER_NODE=8 ALLOW_EXTRA_ARGS=1 RECORD_VRAM=1
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
export WANDB_BASE_URL=http://8.150.1.98:8080 WANDB_PROJECT=chenth_terminal_agent_sft WANDB_DIR=/storage/openpsi/users/chen/runs/wandb
export WANDB_RUN_ID=${RUN} WANDB_RESUME=allow
cd ${REPO}; export PYTHONPATH=${REPO}/src
echo "[\$(hostname)] node rank \$NODE_RANK of ${NN}, master ${MASTER}:${PORT}"
llamafactory-cli train coding_sft/configs/qwen3_5_terminal_agent_sft_3k_256k.yaml dataset_dir=${DD} dataset=${DS} \\
    tokenized_path=${TP} output_dir=${OUT} \\
    overwrite_output_dir=false gradient_accumulation_steps=${GA} ${MAXSTEPS} save_steps=1 keep_weights_every=${SPE} save_total_limit=1000 \
    report_to=wandb run_name=${RUN} ${OV:-}
EOF

srun --mpi=none --ntasks="${NN}" --ntasks-per-node=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        --env WANDB_API_KEY='${KEY}' --env NODE_RANK=\$SLURM_PROCID '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
