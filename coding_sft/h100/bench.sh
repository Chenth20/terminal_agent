#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
SQZ_ROOT=$(realpath "${SQZ_ROOT:-$REPO/storage}")
LLAMAFACTORY_ROOT=$(realpath "${LLAMAFACTORY_ROOT:-$REPO}")
export SQZ_ROOT LLAMAFACTORY_ROOT
ST=$SQZ_ROOT
OUT=$(realpath -m "${1:?pass a new run directory on Ceph}")
[[ "$(findmnt -n -o FSTYPE -T "$(dirname "$OUT")")" == ceph ]]
mkdir "$OUT"
exec >"$OUT/driver.log" 2>&1
export CUDA_HOME="$ST/conda/envs/cuda124"
export PATH="$ST/conda/envs/lf/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$LLAMAFACTORY_ROOT/src"
export HF_HOME="$OUT/hf" TRITON_CACHE_DIR="$ST/runs/.triton-h100" HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_PLACEHOLDER='<|lf_image_placeholder_unused|>' VIDEO_PLACEHOLDER='<|lf_video_placeholder_unused|>' AUDIO_PLACEHOLDER='<|lf_audio_placeholder_unused|>'
cd "$LLAMAFACTORY_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/hardware.csv"
python "$HERE/benchmark.py" prepare "$OUT" --profile "${2:-baseline}"
python -m torch.distributed.run --standalone --nproc-per-node=8 "$HERE/benchmark.py" measure "$OUT"
echo 'Benchmark complete'
