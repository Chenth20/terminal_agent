#!/usr/bin/env bash
# Register @1.4.0 for the 10 envs that failed systematically in cth-thfull14-dsv41-2610270211:
#   - 8 reward-hang envs: oracle re-uploaded from env_assets_v140 (tests/test.sh is a `timeout -k 60 1500`
#     wrapper around the byte-identical original, now tests/test_orig.sh). Root cause: check script waits
#     on a process that never exits; reward exec hit the 7200 s ARCA timeout -> task TIMEOUT.
#   - 2 DNS envs: spec-only change. The task's own bootstrap rewrites /etc/resolv.conf to 127.0.0.1
#     (local dnsmasq/consul), after which the harness cannot resolve envarena-prod.alipay.com and every
#     LLM call fails. The setup hook now pins the gateway in /etc/hosts while DNS still works.
# Idempotent: skips keys whose 1.4.0 is already PUBLISHED.
set -euo pipefail
export PATH=/storage/openpsi/users/hechuyi.hcy/tools:$PATH
[[ -f ~/.secrets.env ]] && { set -a; source ~/.secrets.env; set +a; }
A=(env -u ANT_COOKIE arena)
VER=1.4.0
SPECS=/tmp/opencode/th140/specs           # platform specs of the versions the job ran (arena env version-get); published 1.4.0 specs are kept in work/env-specs/1.4.0/
NEWASSETS=/storage/openpsi/users/chen/code_data/terminal_horizon/env_assets_v140
cd "$(dirname "$0")/../../work"

REWARD_HANG=(
  cth-th-commission-a-calibrated-ros-2-robot-workcell
  cth-th-implement-cancellation-safe-structured-concurrency-in-tokio
  cth-th-make-a-multi-sensor-robot-state-estimate-trustworthy
  cth-th-onboard-a-heterogeneous-vendor-fleet-into-open-rmf
  cth-th-qualify-a-closed-loop-ros-2-mobile-robot-autonomy-stack
  cth-th-qualify-a-multi-sensor-robot-calibration-for-localization
  cth-th-replan-a-cooperative-robot-inspection-mission-through-lost-links-and-low
  cth-th-safely-replace-an-etcd-raft-member-during-live-writes
)
DNS=(
  cth-th-qualify-sip-proxy-paths-for-stateful-transaction-and-dialog-interoperabi
  cth-th-integrate-heterogeneous-workloads-with-consul-health-aware-service-disco
)

HOSTS_PIN='ips=$(getent ahostsv4 envarena-prod.alipay.com 2>/dev/null | awk "{print \$1}" | sort -u | head -3); [ -n "$ips" ] || ips=$(python3 -c "import socket;print(\"\\n\".join(sorted({a[4][0] for a in socket.getaddrinfo(\"envarena-prod.alipay.com\",443,socket.AF_INET)})[:3]))" 2>/dev/null); for ip in $ips; do grep -q "$ip envarena-prod.alipay.com" /etc/hosts 2>/dev/null || echo "$ip envarena-prod.alipay.com" >> /etc/hosts; done; echo "[th-setup] hosts pin: $(grep -c envarena-prod.alipay.com /etc/hosts 2>/dev/null || echo 0) entries"; '

published() {
  "${A[@]}" env version-get "$1" "$VER" --json 2>/dev/null \
    | python3 -c 'import json,sys;sys.exit(0 if json.load(sys.stdin).get("data",{}).get("status")=="PUBLISHED" else 1)'
}

mkdir -p state/$VER /tmp/opencode/th140/out
for key in "${REWARD_HANG[@]}" "${DNS[@]}"; do
  task=${key#cth-th-}
  if published "$key"; then echo "skip $key@$VER (already published)"; continue; fi
  spec=/tmp/opencode/th140/out/$key.json
  if [[ " ${REWARD_HANG[*]} " == *" $key "* ]]; then
    o=$("${A[@]}" upload "$NEWASSETS/$task/oracle" --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"]["uri"])')
    [[ -n $o ]] || { echo "FAILED upload oracle $task"; exit 1; }
    python3 - "$SPECS/$key.json" "$spec" "$o" <<'PY'
import json, sys
src, dst, o = sys.argv[1:4]
s = json.load(open(src))
s["oracle"]["uri"] = o
s.setdefault("metadata", {})["cth_1_4_0"] = "tests/test.sh wraps the byte-identical original (tests/test_orig.sh) in `timeout -k 60 1500`; reward phase hung to the 7200s ARCA exec timeout in cth-thfull14-dsv41-2610270211"
json.dump(s, open(dst, "w"), ensure_ascii=False, indent=1)
PY
  else
    python3 - "$SPECS/$key.json" "$spec" "$HOSTS_PIN" <<'PY'
import json, sys
src, dst, pin = sys.argv[1:4]
s = json.load(open(src))
s["hooks"]["setup_script"] = pin + s["hooks"]["setup_script"]
s.setdefault("metadata", {})["cth_1_4_0"] = "setup hook pins envarena-prod.alipay.com into /etc/hosts: the task bootstrap rewrites /etc/resolv.conf to 127.0.0.1 and the harness lost LLM-gateway DNS in cth-thfull14-dsv41-2610270211"
json.dump(s, open(dst, "w"), ensure_ascii=False, indent=1)
PY
  fi
  "${A[@]}" env version-create "$key" "$VER" --file "$spec" --json \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("ok") else (print(d.get("error"),file=sys.stderr) or 1))' \
    || "${A[@]}" env version-update "$key" "$VER" --file "$spec" --json \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("ok") else (print(d.get("error"),file=sys.stderr) or 1))' \
    || { echo "FAILED version-create/update $key@$VER"; exit 1; }
  "${A[@]}" env publish "$key" "$VER" --yes --json \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("ok") else (print(d.get("error"),file=sys.stderr) or 1))' \
    || { echo "FAILED publish $key@$VER"; exit 1; }
  published "$key" || { echo "FAILED verify $key@$VER"; exit 1; }
  echo "$key@$VER" > "state/$VER/$task.ok"
  echo "published $key@$VER"
done
echo "all done"