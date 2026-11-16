#!/bin/bash
#SBATCH --job-name=chenth_th_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=3:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/slurm-%x-%j.err
#
# Throughput / memory sweep for the TerminalHorizon agent-SFT config: runs several
# short (max_steps) variants back-to-back on the same node and prints a table of
# step time, peak GPU memory and loss so the fastest OOM-free setting can be picked.
#
#   sbatch coding_sft/launch/slurm/bench_throughput.slurm
set -euo pipefail

REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
BASE=${BASE:-coding_sft/configs/archive/qwen3_5_terminal_agent_sft_128.yaml}
STEPS=${STEPS:-6}
OUT=/storage/openpsi/users/chen/runs/bench_${SLURM_JOB_ID}
mkdir -p "$OUT"

# name | overrides   (each variant trains STEPS steps on 8 GPUs; batch 1 x 32K per GPU)
VARIANTS=(
  "warmup             | deepspeed=examples/deepspeed/ds_z2_config.json max_steps=2"
  "z2_ga2_nockpt      | deepspeed=examples/deepspeed/ds_z2_config.json disable_gradient_checkpointing=true gradient_checkpointing=false"
  "z2_ga1_nockpt      | deepspeed=examples/deepspeed/ds_z2_config.json disable_gradient_checkpointing=true gradient_checkpointing=false gradient_accumulation_steps=1"
  "z3_ga2_nockpt      | deepspeed=examples/deepspeed/ds_z3_config.json disable_gradient_checkpointing=true gradient_checkpointing=false"
  "z2_ga2_ckpt        | deepspeed=examples/deepspeed/ds_z2_config.json"
)

echo "Job ${SLURM_JOB_ID} on $(hostname) start $(date) sif=${SIF}"
for v in "${VARIANTS[@]}"; do
  name=$(echo "$v" | cut -d'|' -f1 | xargs); ov=$(echo "$v" | cut -d'|' -f2- | xargs)
  echo "================ ${name}: ${ov}"
  srun --mpi=none --nodes=1 --ntasks=1 --gres=gpu:8 ${SLURM_JOB_RESERVATION:+--reservation=$SLURM_JOB_RESERVATION} --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage --cleanenv "${SIF}" \
    bash -c "
      set -uo pipefail
      export NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      export FORCE_TORCHRUN=1 NPROC_PER_NODE=8 WANDB_DISABLED=true ALLOW_EXTRA_ARGS=1
      export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface
      export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
      export RECORD_VRAM=1
      cd ${REPO}
      export PYTHONPATH=${REPO}/src
      llamafactory-cli train ${BASE} logging_steps=1 max_steps=${STEPS} eval_strategy='\"no\"' save_strategy='\"no\"' \
        output_dir=${OUT}/${name} overwrite_cache=false report_to=none ${ov} > ${OUT}/${name}.log 2>&1
      echo exit=\$?
    " || true
  python3 - "$OUT/${name}.log" "$name" <<'PY'
import re, sys, json
log, name = sys.argv[1], sys.argv[2]
txt = open(log, errors="replace").read()
losses = re.findall(r"\{'loss': '([\d.]+)'", txt)
rt = re.search(r"'train_runtime': '([\d.]+)'", txt)
sps = re.search(r"'train_steps_per_second': '([\d.]+)'", txt)
vram = re.findall(r"'vram_reserved': '?([\d.]+)", txt)
oom = "OutOfMemoryError" in txt or "CUDA out of memory" in txt
print(f"RESULT {name:14s} steps={len(losses):2d} runtime={rt.group(1) if rt else 'NA':>7s}s  steps/s={sps.group(1) if sps else 'NA':>6s}  "
      f"s/step={(float(rt.group(1))/max(len(losses),1)) if rt else float('nan'):6.1f}  peak_reserved_GB={max(map(float,vram)) if vram else 'NA'}  OOM={oom}  last_loss={losses[-1] if losses else 'NA'}")
PY
done
echo "Job ${SLURM_JOB_ID} finished $(date)"
