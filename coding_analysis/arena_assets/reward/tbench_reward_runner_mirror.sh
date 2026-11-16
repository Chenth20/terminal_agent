#!/usr/bin/env bash
# cth-tb2-reward-mirror: tb2-reward@4.0.18-gepa-envcache runner with three infra fixes.
# The official verifier (tests/test.sh, test_outputs.py) is executed unchanged; only where its
# dependencies come from and which directory it starts in differ:
#   1. uv is pre-installed from the bundled uv-dist tarball (same 0.9.5 the official reward ships) and
#      $HOME/.local/bin/env is pre-created, so test.sh's `curl https://astral.sh/uv/... | sh` may fail
#      (astral.sh is unreachable from the sandbox) without leaving `uvx: command not found`.
#   2. uv / pip default index -> mirrors.aliyun.com/pypi (pypi.org / files.pythonhosted.org time out
#      from the sandbox); explicit --index-url / UV_NO_INDEX in a test.sh still win.
#   3. WORKDIR falls back to $ARENA_AGENT_WORKDIR before /app (tb2-prove-plus-comm uses /workspace and
#      has no details.json; the official runner ended up in / and test.sh refused to run).
#   4. (1.0.1) uv download timeout 120 -> 600 s, 4 concurrent downloads, 6 HTTP retries: tests that install
#      torch pull 300-400 MB nvidia-* wheels and one of them timed out at 120 s (tb2-mteb-retrieve).
#   5. (1.0.1) HF_HUB_DISABLE_XET=1 + longer HF timeouts: tests that call load_dataset()/hf_hub_download()
#      at verification time (tb2-reshard-c4-data) got 403 from the xet CAS server; the plain HTTP path works.
set -u

ORACLE="${ARENA_ORACLE_PATH:?ARENA_ORACLE_PATH is required}"
RESULT="${ARENA_REWARD_OUTPUT_FILE:?ARENA_REWARD_OUTPUT_FILE is required}"
RDIR="${ARENA_REWARD_DIR:?ARENA_REWARD_DIR is required}"
ART="${ARENA_ARTIFACTS_DIR:-$(dirname "$RESULT")/artifacts}"
mkdir -p "$(dirname "$RESULT")" "$ART"

# --- fix 3: working directory -------------------------------------------------------------
WORKDIR="${ARENA_AGENT_WORKDIR:-/app}"
if [ -f "$ORACLE/details.json" ]; then
  wd=$(grep -oE '"workdir"[[:space:]]*:[[:space:]]*"[^"]*"' "$ORACLE/details.json" \
        | sed -E 's/.*:[[:space:]]*"([^"]*)".*/\1/')
  [ -n "${wd:-}" ] && WORKDIR="$wd"
fi
[ -d "$WORKDIR" ] || WORKDIR=/app

TESTS="$ORACLE/tests"
[ -f "$TESTS/test.sh" ] || TESTS="$ORACLE"
rm -rf /tests /logs/verifier 2>/dev/null || true
mkdir -p /logs/verifier
ln -s "$TESTS" /tests 2>/dev/null || cp -a "$TESTS" /tests

export UV_DOWNLOAD_URL="${UV_DOWNLOAD_URL:-file://${RDIR}/uv-dist}"
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://registry.npmmirror.com/-/binary/python-build-standalone}"

# --- fix 1: pre-install uv from the bundled tarball --------------------------------------
UV_TGZ="$RDIR/uv-dist/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_HOME="${HOME:-/root}/.local/bin"
mkdir -p "$UV_HOME"
if [ -f "$UV_TGZ" ]; then
  tmp=$(mktemp -d)
  if tar -xzf "$UV_TGZ" -C "$tmp" 2>/dev/null; then
    for b in uv uvx; do
      src=$(find "$tmp" -type f -name "$b" | head -1)
      [ -n "$src" ] && [ ! -x "$UV_HOME/$b" ] && install -m 0755 "$src" "$UV_HOME/$b"
    done
  fi
  rm -rf "$tmp"
fi
if [ ! -f "$UV_HOME/env" ]; then
  cat > "$UV_HOME/env" <<'ENVEOF'
#!/bin/sh
# added by cth-tb2-reward-mirror (same shape as the astral installer's env file)
case ":${PATH}:" in
    *:"$HOME/.local/bin":*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
ENVEOF
fi
export PATH="$UV_HOME:$PATH"
echo "[tb2-reward-mirror] uv: $(command -v uv || echo missing) $("$UV_HOME/uv" --version 2>/dev/null || true)"

# --- fix 2: package index mirror ---------------------------------------------------------
MIRROR="${TB2_REWARD_PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$MIRROR}"
export UV_INDEX_URL="${UV_INDEX_URL:-$MIRROR}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-6}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-$MIRROR}"
export PIP_TIMEOUT="${PIP_TIMEOUT:-60}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
# --- fix 5: Hugging Face downloads inside test.sh ---------------------------------------
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

# --- official 4.0.18 pytorch-model-cli branch, unchanged -----------------------------------
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
echo "[tb2-reward-mirror] v1.0.1 workdir=$PWD index=$UV_DEFAULT_INDEX uv_timeout=$UV_HTTP_TIMEOUT hf_xet_disabled=$HF_HUB_DISABLE_XET"
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
printf '{"status":"OK","score":%s,"pass":%s,"raw":{"exit_code":%s,"mode":"shared_env-mirror","workdir":"%s","reward_txt":"/logs/verifier/reward.txt","log_excerpt":"%s"}}\n' \
  "$score" "$pass" "$code" "$WORKDIR" "$excerpt" > "$RESULT"
echo "reward(shared-mirror): score=$score (test.sh exit=$code workdir=$WORKDIR)"
