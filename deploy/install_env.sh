#!/usr/bin/env bash
# Rebuild the training image's python environment from public sources, package for package.
# Source of truth: deploy/requirements-image.lock (read from the real image's dist-info).
#
#   conda create -n lf python=3.11.11 -y && conda activate lf
#   bash deploy/install_env.sh            # from the repo root
#   python deploy/verify_env.py           # on a GPU node
#
# Env knobs:
#   PIP_INDEX_URL      - mirror for everything except torch (default: PyPI)
#   TORCH_INDEX_URL    - default https://download.pytorch.org/whl/cu124
#   TORCH_CUDA_ARCH_LIST - GPU arch for the causal-conv1d build (default 8.9 = L20/L40/4090;
#                          A100 8.0, H100 9.0, several: "8.0;8.9;9.0")
#   SKIP_FLASHINFER=1  - flashinfer is only used by inference backends, not by training
set -euo pipefail
cd "$(dirname "$0")/.."
LOCK=deploy/requirements-image.lock
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1

py=$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')
[ "$py" = "3.11" ] || { echo "need python 3.11 (have $py): conda create -n lf python=3.11.11"; exit 1; }

echo "== 1/5 torch 2.6.0+cu124 (pins the ABI for every compiled wheel below)"
pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" --index-url "$TORCH_INDEX_URL"

echo "== 2/5 every other pinned package, no dependency resolution (the set is known-consistent)"
# strip comments/blank lines and the torch trio (already installed from the cu124 index, with +cu124 tags pip cannot resolve on PyPI)
grep -vE '^\s*(#|$)' "$LOCK" | grep -vE '^(torch|torchvision|torchaudio)==' > /tmp/lf-lock.txt
pip install --no-deps -r /tmp/lf-lock.txt

echo "== 3/5 this repo (the patched llamafactory 0.9.6.dev0; never the PyPI package)"
pip install --no-deps -e .

echo "== 4/5 flash-attn 2.7.4.post1: the exact prebuilt wheel the image used (cu12 / torch2.6 / cxx11abi=FALSE / cp311)"
pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo "== 5/5 causal-conv1d 1.5.3.post1 (GatedDeltaNet fast path; needs nvcc, ~10 min)"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9} MAX_JOBS=${MAX_JOBS:-32} CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install --no-deps --no-build-isolation "causal-conv1d==1.5.3.post1"

if [ "${SKIP_FLASHINFER:-0}" != "1" ]; then
  echo "== (optional) flashinfer 0.2.2.post1 for the inference backends"
  pip install --no-deps "flashinfer-python==0.2.2.post1" -i https://flashinfer.ai/whl/cu124/torch2.6/ || \
    echo "flashinfer not installed - fine for training"
fi

python - <<'PY'
import torch, transformers, deepspeed, liger_kernel, fla, causal_conv1d, flash_attn
from llamafactory.extras.env import VERSION
from llamafactory.extras.misc import check_dependencies
check_dependencies()
print("ok: llamafactory", VERSION, "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| flash_attn", flash_attn.__version__, "| causal_conv1d", causal_conv1d.__version__)
PY
echo "done. next: python deploy/verify_env.py (on a GPU node)"