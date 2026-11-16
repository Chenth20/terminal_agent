#!/bin/bash
#SBATCH --job-name=chenth_tree
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# Arm-B (tree) SFT on Slurm, resumable, effective batch = 64 complete trajectories:
#   NAME=flash_solved sbatch --nodes=2 --reservation=rsi --job-name=chenth_tree_flash_solved coding_sft/launch/slurm/run_tree.slurm
#   NAME=th3k         sbatch --nodes=2 --reservation=rsi --job-name=chenth_tree_th3k         coding_sft/launch/slurm/run_tree.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
NAME=${NAME:?flash|flash_solved|th3k}
NN=${SLURM_JOB_NUM_NODES}; WORLD=$((NN * 8))
case "$NAME" in
  flash)        TOK=/storage/openpsi/users/chen/data/flash_full_tree/tokenized_qwen3_5_text_262144_sched${WORLD};;
  flash_solved) TOK=/storage/openpsi/users/chen/data/flash_solved_tree/tokenized_qwen3_5_text_262144_sched${WORLD};;
  th3k)         TOK=/storage/openpsi/users/chen/data/th3k_tree/tokenized_qwen3_5_text_262144_sched${WORLD};;
  *) echo "unknown NAME $NAME" >&2; exit 1;;
esac
[ -f "$TOK/dataset_dict.json" ] || { echo "ERROR: sched tree cache missing: $TOK (build_tree_sched.py --world $WORLD)" >&2; exit 1; }
# the sched cache fixes everything: step s = 64 trajectories, ga = G from its manifest, one row per micro-batch
read -r GA SPE STEPS < <(python3 -c "import json; m=json.load(open('$TOK/manifest.json')); assert m['world']==$WORLD, m['world']; print(m['G'], m['steps_per_epoch'], m['steps'])")
SAVE=1  # full ckpt every step (prev step deleted by KeepOptimizerOnlyLastCallback), weights-only kept at each epoch (SPE)
ROWS=$(python3 -c "import json; print(json.load(open('$TOK/manifest.json'))['rows'])")
KEY=$(cat /storage/openpsi/users/chen/.secrets/wandb_key)
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
PORT=$((20000 + SLURM_JOB_ID % 10000))
RUN=${RUN_NAME:-tree_${NAME}_slurm${NN}n_v4}
OUT=${OUT_DIR:-/storage/openpsi/users/chen/checkpoints/${RUN}}
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
echo "Job ${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} name=${NAME} ga=${GA} rows=${ROWS} steps=${STEPS} (${SPE}/epoch) cache=${TOK} start $(date)"

cat > "$INNER" <<EOS
set -euo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
export NNODES=${NN} MASTER_ADDR=${MASTER} MASTER_PORT=${PORT} TREE_ATTN_SPLIT=${TREE_ATTN_SPLIT:-1} TREE_SPLIT_MULT=${TREE_SPLIT_MULT:-32768} TREE_T_MULT=${TREE_T_MULT:-8192}
export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ALLOW_EXTRA_ARGS=1 RECORD_VRAM=1
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TREE_KCACHE_BASE=/storage/openpsi/users/chen/.cache/kernels/${RUN} TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID} TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TREE_KCACHE_SEED=/storage/openpsi/users/chen/.cache/kernels/_seed/qwen3_5_4b_tmult${TREE_T_MULT:-8192}
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
export WANDB_BASE_URL=http://8.150.1.98:8080 WANDB_PROJECT=chenth_terminal_agent_sft WANDB_DIR=/storage/openpsi/users/chen/runs/wandb
export WANDB_RUN_ID=${RUN} WANDB_RESUME=allow
cd ${REPO}; export PYTHONPATH=${REPO}/src:${REPO}
torchrun --nnodes ${NN} --nproc_per_node 8 --node_rank \$NODE_RANK --master_addr ${MASTER} --master_port ${PORT} \
  coding_sft/patches/train_tree.py coding_sft/configs/qwen3_5_tree_sft.yaml \
  tokenized_path=${TOK} output_dir=${OUT} gradient_accumulation_steps=${GA} max_steps=${STEPS} \
  save_steps=${SAVE} keep_weights_every=${SPE} save_total_limit=1000 report_to=wandb run_name=${RUN} ${EXTRA_OV:-}
EOS

srun --mpi=none --ntasks="${NN}" --ntasks-per-node=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        --env WANDB_API_KEY='${KEY}' --env NODE_RANK=\$SLURM_PROCID '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
