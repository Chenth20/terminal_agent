#!/bin/bash
# Offline self-test of this bundle: run the harness (bin/runner) and the reward runner once against a
# scripted OpenAI-compatible endpoint, with no real model and no platform.
#
#   sudo bash harness/selftest/run.sh [config-version ...]      # default: 1.6.96 1.7.96 1.6.262
#
# Needs root (chroot + mount namespaces), bash, curl, python3 >= 3.8 for the mock endpoint.
# Everything runs in a private mount namespace on a tmpfs root assembled from read-only bind mounts of the
# host's /usr and /etc, so the harness' /logs and the reward runner's `rm -rf /tests /logs/verifier` never touch
# the real root. Results land in $OUT (default /tmp/arena-selftest).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUNDLE=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-/tmp/arena-selftest}
VERSIONS=${*:-1.6.96 1.7.96 1.6.262}
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
mkdir -p "$OUT"

if [ -z "${SELFTEST_INNER:-}" ]; then
  python3 "$HERE/mock_llm.py" "$OUT/mock.log" "$OUT/mock.port" & MOCK=$!
  trap 'kill $MOCK 2>/dev/null || true' EXIT
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$OUT/mock.port" ] && break; sleep 0.5; done
  [ -s "$OUT/mock.port" ] || { echo "mock endpoint did not start" >&2; exit 1; }
  fail=0
  for v in $VERSIONS; do
    [ -f "$BUNDLE/harness/configs/$v.env" ] || { echo "no config harness/configs/$v.env" >&2; fail=1; continue; }
    SELFTEST_INNER=1 unshare -m bash "$0" "$v" || fail=1
  done
  exit $fail
fi

# ---------------------------------------------------------------- inner: one version, private mount namespace
VER=$1; PORT=$(cat "$OUT/mock.port"); TAG=$(echo "$VER" | tr . _)
ROOT=$OUT/root-$TAG
mount --make-rprivate /
mkdir -p "$ROOT"; mount -t tmpfs -o size=2g tmpfs "$ROOT"
mkdir -p "$ROOT"/{usr,etc,proc,dev,tmp,app,harness,reward,oracle/tests,out,root,run,var/log,env}
ln -s usr/bin "$ROOT/bin"; ln -s usr/lib "$ROOT/lib"; ln -s usr/lib64 "$ROOT/lib64"; ln -s usr/sbin "$ROOT/sbin"
mount --rbind /usr "$ROOT/usr"; mount -o remount,ro,bind "$ROOT/usr" 2>/dev/null || true
mount --rbind /etc "$ROOT/etc"; mount -o remount,ro,bind "$ROOT/etc" 2>/dev/null || true
mount --rbind /dev "$ROOT/dev"; mount -t proc proc "$ROOT/proc"
mount --bind "$BUNDLE/harness" "$ROOT/harness"; mount -o remount,ro,bind "$ROOT/harness"
mount --bind "$BUNDLE/reward" "$ROOT/reward";   mount -o remount,ro,bind "$ROOT/reward"
chmod 1777 "$ROOT/tmp"

echo 'Create a file named hello.txt in the current directory whose first word is "hello".' > "$ROOT/env/instruction.md"
cat > "$ROOT/oracle/tests/test.sh" <<'EOF'
#!/bin/bash
# Same opening as the TB2 test.sh files, with the uv installer pointed at a closed port (unreachable installer).
curl -LsSf --max-time 3 http://127.0.0.1:9/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
echo "uvx: $(uvx --version)"
echo "cwd=$PWD UV_DEFAULT_INDEX=$UV_DEFAULT_INDEX UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
if command -v uvx >/dev/null && [ "$PWD" = /app ] && [ "$(cut -d' ' -f1 hello.txt)" = hello ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
EOF

ENVARGS=()
while IFS= read -r l; do [[ -z $l || $l == \#* ]] || ENVARGS+=("$l"); done < "$BUNDLE/harness/configs/$VER.env"
BASE_ENV=(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root TERM=xterm LANG=C.UTF-8)

set +e
chroot "$ROOT" /usr/bin/env -i "${BASE_ENV[@]}" "${ENVARGS[@]}" \
  ARENA_HARNESS_DIR=/harness ARENA_QUERY_PATH=/env/instruction.md ARENA_OUTPUT_DIR=/out \
  ARENA_AGENT_WORKDIR=/app ARENA_AGENT_TIMEOUT_SEC=300 ARENA_TASK_ID="selftest-$TAG" \
  LLM_API_KEY=sk-selftest LLM_BASE_URL="http://127.0.0.1:$PORT/v1" LLM_MODEL="openai/selftest-$TAG" \
  /harness/bin/runner > "$OUT/harness-$TAG.log" 2>&1
HRC=$?
chroot "$ROOT" /usr/bin/env -i "${BASE_ENV[@]}" \
  ARENA_ORACLE_PATH=/oracle ARENA_REWARD_OUTPUT_FILE=/out/reward.json ARENA_REWARD_DIR=/reward \
  ARENA_ARTIFACTS_DIR=/out/artifacts ARENA_AGENT_WORKDIR=/app \
  /usr/bin/bash /reward/tbench_reward_runner_mirror.sh > "$OUT/reward-$TAG.log" 2>&1
RRC=$?
set -e
rm -rf "$OUT/out-$TAG"; mkdir -p "$OUT/out-$TAG"; cp -a "$ROOT/out/." "$OUT/out-$TAG/"; cp -a "$ROOT/app" "$OUT/out-$TAG/app"
chroot "$ROOT" /usr/bin/env PATH=/harness/bin/fallback-bin:/usr/bin tmux kill-server >/dev/null 2>&1 || true

python3 - "$OUT/out-$TAG" "$VER" "$HRC" "$RRC" "$OUT/mock.log" "selftest-$TAG" <<'EOF'
import json, sys
out, ver, hrc, rrc, mock, model = sys.argv[1:7]
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
h = json.load(open(f"{out}/harness_result.json")); r = json.load(open(f"{out}/reward.json"))
traj = [json.loads(l) for l in open(f"{out}/trajectory.jsonl")]
reqs = [json.loads(l)["body"] for l in open(mock) if json.loads(l)["body"].get("model") == model]
print(f"== {ver}")
check("harness exit 0 and status OK", hrc == "0" and h["status"] == "OK", h["summary"][:60])
check("two task_complete confirmations", h["raw"]["completion_confirmations"] == 2)
check("trajectory = user + 3 agent steps, no parse errors",
      [s["source"] for s in traj] == ["user", "agent", "agent", "agent"] and not any("parsing errors" in json.dumps(s.get("observation")) for s in traj))
check("requests were streamed", reqs and all(b.get("stream") for b in reqs))
check("history carries reasoning_content from call 2 on", len(reqs) >= 2 and all(any("reasoning_content" in m for m in b["messages"]) for b in reqs[1:]))
tool_mode = h["raw"]["tool_call_mode"]
check("tool_call_mode matches config", (tool_mode == "function") == bool(reqs[0].get("tools")), f"mode={tool_mode}")
hello = open(f"{out}/app/hello.txt").read().strip()
check("agent shell saw the sandbox env (PIP_INDEX_URL) when the config exports it",
      ("index=https://" in hello) == ("PIP_INDEX_URL" in h["raw"]["sandbox_env_keys"].split(",")), hello)
check("reward exit 0, score 1 via test.sh (uv from bundled tarball, workdir /app)", rrc == "0" and r["score"] == 1 and r["raw"]["workdir"] == "/app",
      r["raw"]["log_excerpt"][-90:].strip())
print("  result:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
