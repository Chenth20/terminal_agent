#!/usr/bin/env bash
# TB2 shared_env runner with verifier dependencies baked into the PyTorch env.
# The official verifier remains unchanged; only its dependency source is local.
set -u

ORACLE="${ARENA_ORACLE_PATH:?ARENA_ORACLE_PATH is required}"
RESULT="${ARENA_REWARD_OUTPUT_FILE:?ARENA_REWARD_OUTPUT_FILE is required}"
RDIR="${ARENA_REWARD_DIR:?ARENA_REWARD_DIR is required}"
ART="${ARENA_ARTIFACTS_DIR:-$(dirname "$RESULT")/artifacts}"
mkdir -p "$(dirname "$RESULT")" "$ART"

WORKDIR=/app
if [ -f "$ORACLE/details.json" ]; then
  wd=$(grep -oE '"workdir"[[:space:]]*:[[:space:]]*"[^"]*"' "$ORACLE/details.json" \
        | sed -E 's/.*:[[:space:]]*"([^"]*)".*/\1/')
  [ -n "${wd:-}" ] && WORKDIR="$wd"
fi

TESTS="$ORACLE/tests"
[ -f "$TESTS/test.sh" ] || TESTS="$ORACLE"
rm -rf /tests /logs/verifier 2>/dev/null || true
mkdir -p /logs/verifier
ln -s "$TESTS" /tests 2>/dev/null || cp -a "$TESTS" /tests

export UV_DOWNLOAD_URL="${UV_DOWNLOAD_URL:-file://${RDIR}/uv-dist}"
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://registry.npmmirror.com/-/binary/python-build-standalone}"

if [ "${ARENA_ENV_KEY:-}" = "tb2-pytorch-model-cli" ]; then
  IMAGE_CACHE=/opt/gepa-verifier-cache
  CACHE_RUNTIME=/tmp/gepa-pytorch-verifier-cache
  [ -x "$IMAGE_CACHE/bin/uv" ] || {
    echo "[tb2-reward] missing PyTorch verifier cache in env image" >&2
    exit 96
  }
  rm -rf "$CACHE_RUNTIME"
  mkdir -p "$CACHE_RUNTIME/uv-cache" "$CACHE_RUNTIME/wheelhouse" "$WORKDIR/data/MNIST/raw"
  for wheel in "$IMAGE_CACHE"/wheelhouse/*.whl; do
    ln -s "$wheel" "$CACHE_RUNTIME/wheelhouse/${wheel##*/}"
  done
  ln -s "$IMAGE_CACHE/torch-wheel.payload" \
    "$CACHE_RUNTIME/wheelhouse/torch-2.7.1+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"
  ln -s "$IMAGE_CACHE/torchvision-wheel.payload" \
    "$CACHE_RUNTIME/wheelhouse/torchvision-0.22.1+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"
  cp -a "$IMAGE_CACHE/mnist/." "$WORKDIR/data/MNIST/raw/"
  export UV_CACHE_DIR="$CACHE_RUNTIME/uv-cache"
  export UV_FIND_LINKS="$CACHE_RUNTIME/wheelhouse"
  export UV_NO_INDEX=1
  export UV_OFFLINE=1
  export UV_NO_PROGRESS=1
  uvx() {
    "$IMAGE_CACHE/bin/uv" tool run "$@"
  }
  export -f uvx
  echo "[tb2-reward] using env-baked PyTorch verifier cache"
fi

cd "$WORKDIR" 2>/dev/null || cd /
bash /tests/test.sh > "$ART/tbench_test.log" 2>&1
code=$?

score=0
if [ -f /logs/verifier/reward.txt ]; then
  raw=$(tr -d '[:space:]' < /logs/verifier/reward.txt)
  case "$raw" in
    1|1.0|pass|passed|true|ok|PASS|True|OK) score=1 ;;
    0|""|0.0) score=0 ;;
    *) awk "BEGIN{exit !($raw>0)}" 2>/dev/null && score=1 || score=0 ;;
  esac
elif [ "$code" = "0" ]; then
  score=1
fi

cp -f /logs/verifier/reward.txt "$ART/" 2>/dev/null || true
cp -f /logs/verifier/ctrf.json "$ART/" 2>/dev/null || true
pass=$([ "$score" = 1 ] && echo true || echo false)
excerpt=$(tail -c 3500 "$ART/tbench_test.log" 2>/dev/null | tr -d '\\"' | tr -c ' -~' ' ')
printf '{"status":"OK","score":%s,"pass":%s,"raw":{"exit_code":%s,"mode":"shared_env-envcached-pytorch","reward_txt":"/logs/verifier/reward.txt","log_excerpt":"%s"}}\n' \
  "$score" "$pass" "$code" "$excerpt" > "$RESULT"
echo "reward(shared-envcached): score=$score (test.sh exit=$code)"
