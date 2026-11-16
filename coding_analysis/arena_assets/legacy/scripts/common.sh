#!/usr/bin/env bash
# Shared helpers for the arena scripts. Source this; do not execute.
#
#   source "$(dirname "$0")/../lib/common.sh"        # from providers/<x>/...
#
# Provides: arena CLI wrapper, control-plane token, logging, gateway data-plane probe,
# and the model register/create-or-update helpers used by every provider.

# arena CLI lives outside the default PATH on this cluster.
[[ -d /storage/openpsi/users/hechuyi.hcy/tools ]] && PATH=/storage/openpsi/users/hechuyi.hcy/tools:$PATH
export PATH

ARENA_TOKEN_FILE=${ARENA_TOKEN_FILE:-$HOME/.envarena/config.toml}
CONTROL_BASE=${CONTROL_BASE:-https://htnn-gateway.alipay.com/envarena/openapi/v1}
# Data plane: what the sandbox's LiteLLM can reach. The office alias (htnn-gateway.alipay.com)
# is NOT reachable from inside sandboxes, so always point models here.
DATA_BASE=${DATA_BASE:-https://envarena-prod.alipay.com/api/v1}
DATA_KEY=${DATA_KEY:-areal}

# arena <subcommand...> — token/cookie env vars are stripped so the CLI uses the profile file.
[[ -f ~/.secrets.env ]] && { set -a; source ~/.secrets.env; set +a; }   # ARENA_OPENAPI_TOKEN / AISTUDIO_TOKEN live here, never in scripts
arena() { env -u ANT_COOKIE arena "$@"; }

log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2; }
die()  { log "error: $*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

arena_token() {
  [[ -n ${ARENA_API_TOKEN:-} ]] && { printf '%s' "$ARENA_API_TOKEN"; return; }
  grep -oP 'token\s*=\s*"\K[^"]+' "${ARENA_TOKEN_FILE/#\~/$HOME}" 2>/dev/null | head -1
}

# http_probe <url> <bearer> [json-body] -> prints "HTTP <code> <seconds>s", returns 0 on 200
http_probe() {
  local url=$1 key=$2 body=${3:-}
  local out
  if [[ -n $body ]]; then
    out=$(curl -sS -m 300 -o /dev/null -w '%{http_code} %{time_total}' "$url" \
            -H "Authorization: Bearer $key" -H 'Content-Type: application/json' -d "$body" 2>&1)
  else
    out=$(curl -sS -m 30 -o /dev/null -w '%{http_code} %{time_total}' "$url" 2>&1)
  fi
  printf 'HTTP %s\n' "$out"
  [[ ${out%% *} == 200 ]]
}

# route_upsert <model_name> <route.json> — create the gateway route, or PATCH it (keeps other endpoints' health).
route_upsert() {
  local name=$1 file=$2
  if arena llm-model get "$name" >/dev/null 2>&1; then
    arena llm-model update "$name" --file "$file" | head -3
  else
    arena llm-model create --file "$file" | head -3
  fi
}

# model_publish <key> <version> <spec.json> — create the entity if absent, then create+publish the version.
# Idempotent: an already-published version is left untouched.
model_publish() {
  local key=$1 ver=$2 spec=$3
  if arena model version-get "$key" "$ver" >/dev/null 2>&1; then log "model $key@$ver already published"; return; fi
  arena model get "$key" >/dev/null 2>&1 \
    || arena model create --key "$key" --name "$key" --tags "chen" --json >/dev/null 2>&1 || true
  arena model version-create "$key" "$ver" --file "$spec" --json >/dev/null || die "version-create $key@$ver failed"
  arena model publish "$key" "$ver" --yes --json >/dev/null || die "publish $key@$ver failed"
  log "registered model $key@$ver"
}

# model_env_json <model_name> <mode: claude|openai|both>
# Claude Code sends an Anthropic `thinking` block; the slurm qwen backends reject it, so the
# claude vars deliberately omit CLAUDE_CODE_EFFORT_LEVEL and set MAX_THINKING_TOKENS=0.
# DashScope-backed routes accept the thinking block, so mode=both is harmless there too.
model_env_json() {
  local m=$1 mode=${2:-both}
  python3 - "$m" "$DATA_BASE" "$DATA_KEY" "$mode" <<'PY'
import json, sys
m, base, key, mode = sys.argv[1:5]
env = {"API_TIMEOUT_MS": "900000", "CLAUDE_CODE_MAX_RETRIES": "2", "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1"}
if mode in ("claude", "both"):
    env |= {"MAX_THINKING_TOKENS": "0",
            "ANTHROPIC_BASE_URL": base.replace("/v1", ""),   # Claude Code appends /v1/messages
            "ANTHROPIC_API_KEY": key, "ANTHROPIC_AUTH_TOKEN": key,
            "ANTHROPIC_MODEL": m, "ANTHROPIC_DEFAULT_MODEL": m,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": m, "ANTHROPIC_DEFAULT_SONNET_MODEL": m,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": m, "AGENT_ANTHROPIC_MODEL": m}
if mode in ("openai", "both"):
    env |= {"OPENAI_BASE_URL": base, "OPENAI_API_KEY": key, "OPENAI_MODEL": m,
            "LLM_BASE_URL": base, "LLM_MODEL": f"openai/{m}", "MODEL_NAME": m}
print(json.dumps(env))
PY
}
