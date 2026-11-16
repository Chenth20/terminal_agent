#!/bin/bash
#SBATCH --job-name=lf-agent-sif-build
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/storage/openpsi/users/chen/runs/slurm-%x-%j.log
#
# Builds docker/apptainer/llamafactory-agent-sft.def into a SIF on a compute
# node (login nodes lack the privileges `apptainer build` needs).
#
# Usage (from a login node):
#   REPO=/storage/openpsi/users/chen/llama-factory sbatch coding_sft/env/build_sif.sbatch
#   TRANSFORMERS_VERSION=5.12.1 TAG=tf5.12.1-$(date +%Y%m%d) sbatch coding_sft/env/build_sif.sbatch
#
# REPO must be the /storage copy (visible on compute nodes); /home/admin is not.
# TRANSFORMERS_VERSION="" (default) keeps the base image's transformers.
set -euo pipefail

REPO=${REPO:-/storage/openpsi/users/chen/llama-factory}
OUT_DIR=${OUT_DIR:-/storage/openpsi/images}
TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION:-}
TAG=${TAG:-tf${TRANSFORMERS_VERSION:-5.2.0}-$(date +%Y%m%d)}
NAME="llamafactory-agent-sft-${TAG}.sif"   # rename to llamafactory-agent-sft-tf<ver>.sif once validated
DEF="$REPO/docker/apptainer/llamafactory-agent-sft.def"

mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/build.lock"
flock -n 9 || { echo "another build holds $OUT_DIR/build.lock"; exit 75; }

work=$(mktemp -d /tmp/lf-sif-${SLURM_JOB_ID:-manual}-XXXXXX)
trap 'rm -rf -- "$work"' EXIT
echo "BUILD_START $(date -Is) host=$(hostname) work=$work repo=$REPO"
df -h /tmp | tail -1

# Prefer a newer mksquashfs if one that actually runs on this node exists
# (the system one is 4.2 from 2013 but works); otherwise keep the system one.
NEW_SQ=/storage/openpsi/images/sglang-dev-dsv41-c1633485/squashfs-tools-4.6.1/squashfs-tools
if [ -x "$NEW_SQ/mksquashfs" ] && "$NEW_SQ/mksquashfs" -version >/dev/null 2>&1; then
    export PATH="$NEW_SQ:$PATH"
fi
mksquashfs -version | head -1

export APPTAINER_TMPDIR="$work/tmp" APPTAINER_CACHEDIR="$work/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

cd "$REPO"
apptainer build --disable-cache --build-arg "TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}" "$work/image.sif" "$DEF"

apptainer inspect "$work/image.sif"
# GPU smoke: this job requests one GPU so the CUDA-dependent imports are exercised.
apptainer exec --cleanenv --no-home --nv --writable-tmpfs "$work/image.sif" python - <<'PY'
import torch, transformers, deepspeed, liger_kernel, wandb
import transformers.models.qwen3_5.modeling_qwen3_5 as q
from llamafactory.extras.env import VERSION
assert torch.cuda.is_available(), "no CUDA in container"
assert q.is_fast_path_available, "GatedDeltaNet fast path missing (fla + causal_conv1d)"
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "deepspeed", deepspeed.__version__, "llamafactory", VERSION, "gpu", torch.cuda.get_device_name(0))
PY

cp "$work/image.sif" "$OUT_DIR/$NAME.partial"
mv "$OUT_DIR/$NAME.partial" "$OUT_DIR/$NAME"
(cd "$OUT_DIR" && sha256sum "$NAME" > "$NAME.sha256")
chmod 644 "$OUT_DIR/$NAME" "$OUT_DIR/$NAME.sha256"
echo "BUILD_SUCCESS $(date -Is) -> $OUT_DIR/$NAME"
