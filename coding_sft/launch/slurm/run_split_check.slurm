#!/bin/bash
#SBATCH --job-name=chenth_split_check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=360G
#SBATCH --time=00:40:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/ablation/slurm-%x-%j.err
#
# ONE job, ONE node, 2 GPUs, <= 40 min: in parallel
#   GPU a: equivalence  - dbg parity run with TREE_ATTN_SPLIT=1 (8 steps, losses vs dbg_align_tree / perdh)
#   GPU b: speed        - bench_tree.py A/B split=0/1 on real flash rows
#   sbatch --reservation=game-efficient coding_sft/launch/slurm/run_split_check.slurm
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
TOKENS=${TOKENS:-"60000 120000 200000 280000"}
PORT=$((20000 + SLURM_JOB_ID % 10000))
RUN=dbg_align_tree_split
OUT=/storage/openpsi/users/chen/checkpoints/${RUN}
INNER=/storage/openpsi/users/chen/runs/ablation/job-${SLURM_JOB_ID}-inner.sh
cat > "$INNER" <<EOS
set -uo pipefail
export HOME=/tmp/h_\$\$; mkdir -p \$HOME
GPUS=(\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -2 | cut -d, -f1))
echo "[gpu] equivalence on GPU \${GPUS[0]}, bench on GPU \${GPUS[1]}"
export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 ALLOW_EXTRA_ARGS=1
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
export TREE_ATTN_SPLIT=1 TREE_SPLIT_MULT=8192
cd ${REPO}; export PYTHONPATH=${REPO}/src:${REPO}
( CUDA_VISIBLE_DEVICES=\${GPUS[0]} TRITON_CACHE_DIR=/tmp/triton_eq_${SLURM_JOB_ID} TREE_KCACHE_BASE=/storage/openpsi/users/chen/.cache/kernels/${RUN} TORCHINDUCTOR_FX_GRAPH_CACHE=1 \\
  MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT} \\
  torchrun --nproc_per_node 1 --master_addr 127.0.0.1 --master_port ${PORT} \\
    coding_sft/patches/train_tree.py coding_sft/configs/_dbg_tree.yaml output_dir=${OUT} run_name=${RUN} \\
    > /storage/openpsi/users/chen/runs/ablation/split_eq_${SLURM_JOB_ID}.log 2>&1; echo "[eq] exit \$?" ) &
PID_EQ=\$!
( CUDA_VISIBLE_DEVICES=\${GPUS[1]} TRITON_CACHE_DIR=/tmp/triton_bench_${SLURM_JOB_ID} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_bench_${SLURM_JOB_ID} \\
  bash -c 'for S in 0 1; do echo "################ TREE_ATTN_SPLIT=\$S \$(date)"; TREE_ATTN_SPLIT=\$S python coding_sft/checks/bench_tree.py --tokens ${TOKENS} --repeat 3; done' \\
    > /storage/openpsi/users/chen/runs/ablation/split_bench_${SLURM_JOB_ID}.log 2>&1; echo "[bench] exit \$?" ) &
PID_B=\$!
wait \$PID_EQ; wait \$PID_B
echo "=== equivalence losses (split) vs reference"
python - <<'PY'
import json
def losses(p):
    try: return [round(json.loads(l)['loss'],4) for l in open(p) if '"loss"' in l and 'null' not in l.split('"loss"')[1][:8]]
    except FileNotFoundError: return None
sp=losses('${OUT}/trainer_log.jsonl'); tr=losses('/storage/openpsi/users/chen/checkpoints/dbg_align_tree/trainer_log.jsonl'); pd=losses('/storage/openpsi/users/chen/checkpoints/dbg_align_perdh/trainer_log.jsonl')
print('split :',sp); print('tree  :',tr); print('perdh :',pd)
if sp and tr: print('max|split-tree| =',max(abs(a-b) for a,b in zip(sp,tr)))
if sp and pd: print('max|split-perdh|=',max(abs(a-b) for a,b in zip(sp,pd)))
PY
echo "=== bench summary"; grep -E "TREE_ATTN_SPLIT=|^row |^  T=" /storage/openpsi/users/chen/runs/ablation/split_bench_${SLURM_JOB_ID}.log
EOS
echo "Job ${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST} start $(date)"
srun --mpi=none --ntasks=1 --gres=gpu:2 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    bash -c "exec singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv \
        '${SIF}' bash '${INNER}'"
echo "Job ${SLURM_JOB_ID} finished $(date)"
