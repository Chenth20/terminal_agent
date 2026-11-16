#!/bin/bash
#SBATCH --job-name=lf-official-sif-build
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/storage/openpsi/images/llamafactory-official/slurm-build-%j.log
#
# Converts a `docker save` archive of the official hiyouga/llamafactory image into a SIF,
# same procedure as coding_sft/env/build_sif.sbatch (compute node, local /tmp work dir).
#
#   TAR=/storage/openpsi/users/chen/models/llamafactory-0.9.5-linux-amd64.tar \
#   sbatch coding_sft/env/build_official_sif.sbatch
set -euo pipefail

TAR=${TAR:-/storage/openpsi/users/chen/models/llamafactory-0.9.5-linux-amd64.tar}
OUT_DIR=${OUT_DIR:-/storage/openpsi/images/llamafactory-official}
NAME=${NAME:-llamafactory-official-0.9.5.sif}

mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/build.lock"
flock -n 9 || { echo "another build holds $OUT_DIR/build.lock"; exit 75; }

work=$(mktemp -d /tmp/lf-official-sif-${SLURM_JOB_ID:-manual}-XXXXXX)
trap 'rm -rf -- "$work"' EXIT
echo "BUILD_START $(date -Is) host=$(hostname) work=$work tar=$TAR"
df -h /tmp | tail -1

export APPTAINER_TMPDIR="$work/tmp" APPTAINER_CACHEDIR="$work/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# copy the archive to local disk: the OCI unpack reads it many times
cp "$TAR" "$work/input.tar"
apptainer build --disable-cache "$work/image.sif" "docker-archive:$work/input.tar"

apptainer inspect "$work/image.sif"
apptainer exec --cleanenv --no-home --nv --writable-tmpfs "$work/image.sif" bash -c '
python -c "import torch, transformers; from llamafactory.extras.env import VERSION; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"transformers\", transformers.__version__, \"llamafactory\", VERSION, \"gpu\", torch.cuda.get_device_name(0))"
pip list --format=freeze' > "$OUT_DIR/$NAME.pip-freeze.txt" 2>&1
head -1 "$OUT_DIR/$NAME.pip-freeze.txt"

cp "$work/image.sif" "$OUT_DIR/$NAME.partial"
mv "$OUT_DIR/$NAME.partial" "$OUT_DIR/$NAME"
(cd "$OUT_DIR" && sha256sum "$NAME" > "$NAME.sha256")
chmod 644 "$OUT_DIR/$NAME" "$OUT_DIR/$NAME.sha256" "$OUT_DIR/$NAME.pip-freeze.txt"
echo "BUILD_SUCCESS $(date -Is) -> $OUT_DIR/$NAME"
