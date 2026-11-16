#!/bin/bash
#SBATCH --job-name=chenth_th_long_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --partition=all
#SBATCH --chdir=/storage/openpsi/users/chen/llama-factory
#SBATCH --output=/storage/openpsi/users/chen/runs/slurm-%x-%j.out
#SBATCH --error=/storage/openpsi/users/chen/runs/slurm-%x-%j.err
#
# Long-context stress sweep on whole TerminalHorizon trajectories (no chunking).
# Each variant: a few steps on the 8 LONGEST samples (sorted-by-length subset) so the
# memory peak is hit immediately. Prints s/step, tok/s, peak VRAM, OOM per variant.
set -euo pipefail
REPO=/storage/openpsi/users/chen/llama-factory
SIF=${SIF:-/storage/openpsi/images/llamafactory-agent-sft-tf5.12.1.sif}
BASE=${BASE:-coding_sft/configs/archive/qwen3_5_terminal_agent_sft_128_long.yaml}
STEPS=${STEPS:-3}
OUT=/storage/openpsi/users/chen/runs/longbench_${SLURM_JOB_ID}; mkdir -p "$OUT"
DD=/storage/openpsi/users/chen/data/terminal_horizon_128_long

VARIANTS=(
  "text_256k          | cutoff_len=262144 model_name_or_path=/storage/openpsi/users/chen/models/Qwen__Qwen3.5-4B-text template=qwen3_5_text"
  "vlm_256k           | cutoff_len=262144"
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
      export FORCE_TORCHRUN=1 NPROC_PER_NODE=8 WANDB_DISABLED=true ALLOW_EXTRA_ARGS=1 RECORD_VRAM=1
      export HF_HOME=/storage/openpsi/users/chen/.cache/huggingface TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
      cd ${REPO}
      export PYTHONPATH=${REPO}/src
      llamafactory-cli train ${BASE} dataset=th128_longest8 logging_steps=1 max_steps=${STEPS} eval_strategy='\"no\"' save_strategy='\"no\"' \
        gradient_accumulation_steps=1 output_dir=${OUT}/${name} overwrite_cache=true report_to=none ${ov} > ${OUT}/${name}.log 2>&1
      echo exit=\$?
    " || true
  python3 - "$OUT/${name}" "$name" <<'PY'
import json, os, sys
d, name = sys.argv[1], sys.argv[2]
log = open(d + ".log", errors="replace").read()
oom = "OutOfMemoryError" in log or "CUDA out of memory" in log
try:
    rows = [json.loads(l) for l in open(d + "/trainer_log.jsonl")]
    rows = [r for r in rows if "current_steps" in r and "loss" in r]
except FileNotFoundError:
    rows = []
def t(r): return sum(a*b for a,b in zip([3600,60,1], map(int, r["elapsed_time"].split(":"))))
if len(rows) >= 2:
    sps = (t(rows[-1]) - t(rows[0])) / (len(rows) - 1); tps = (rows[-1]["total_tokens"] - rows[0]["total_tokens"]) / max(t(rows[-1]) - t(rows[0]), 1)
    print(f"RESULT {name:20s} steps={len(rows)} s/step={sps:6.1f} tok/s={tps:6.0f} tok/step={rows[-1]['total_tokens']//len(rows):7d} peak_alloc={max(r['vram_allocated'] for r in rows):6.1f} peak_resv={max(r['vram_reserved'] for r in rows):6.1f} OOM={oom}")
else:
    print(f"RESULT {name:20s} steps={len(rows)} OOM={oom}")
PY
done
echo "Job ${SLURM_JOB_ID} finished $(date)"
