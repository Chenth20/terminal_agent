#!/usr/bin/env bash
# Run on an external machine with docker + push access to reg.antgroup-inc.cn.
#   bash build_rt3.sh [images_ca_broken.txt]
# For each "<task> <base-image>" line: pull base -> add ca-certificates (Dockerfile.rt3) -> push as *-rt3.
#
# Why: ARCA bootstrap downloads the egress CA and arca_envd with `curl https://...`; in these 18
# images curl fails with (60) "SSL certificate problem", envd never listens, and every task ends
# TIMEOUT/HARNESS_FAILED. The image cannot be fixed from the cluster (no docker / no registry access).
#
# Verification per image (all must pass, otherwise the script aborts and does not record the tag):
#   1. default user is root (ARCA runs harness/envd/runner as root)
#   2. ENTRYPOINT/CMD byte-identical to the base image
#   3. TLS actually works: `curl -sSf https://mirrors.aliyun.com` inside the new image
# Writes images_rt3.json (task -> rt3 image) for make_assets.py; idempotent, resumable.
set -euo pipefail
cd "$(dirname "$0")"
LIST=${1:-images_ca_broken.txt}
OUT=images_rt3.json
LOG=build_rt3.log
: >> "$LOG"

rt3_tag() {  # base image -> rt3 tag
  case $1 in
    *-rt1) printf '%s-rt3\n' "${1%-rt1}" ;;
    *-rt2) printf '%s-rt3\n' "${1%-rt2}" ;;
    *)     printf '%s-rt3\n' "$1" ;;
  esac
}

declare -A done_map=()
if [[ -f $OUT ]]; then
  while IFS=$'\t' read -r k v; do done_map[$k]=$v; done \
    < <(python3 -c 'import json;[print(k,v,sep="\t") for k,v in json.load(open("'"$OUT"'")).items()]')
fi

while read -r task base; do
  [[ -z ${task:-} ]] && continue
  rt3=$(rt3_tag "$base")
  if [[ -n ${done_map[$task]:-} ]]; then echo "skip $task"; continue; fi
  echo "== $task -> $rt3" | tee -a "$LOG"
  docker pull "$base" >>"$LOG" 2>&1
  docker build --build-arg BASE="$base" -f Dockerfile.rt3 -t "$rt3" . >>"$LOG" 2>&1

  u=$(docker inspect -f '{{.Config.User}}' "$rt3")
  [[ -z $u || $u == root || $u == 0 ]] || { echo "FAILED user=$u $task"; exit 1; }
  [[ $(docker inspect -f '{{json .Config.Entrypoint}}{{json .Config.Cmd}}' "$base") \
     == $(docker inspect -f '{{json .Config.Entrypoint}}{{json .Config.Cmd}}' "$rt3") ]] \
    || { echo "FAILED entrypoint/cmd changed $task"; exit 1; }
  docker run --rm --entrypoint sh "$rt3" -c 'id -u' | grep -qx 0 || { echo "FAILED id -u != 0 $task"; exit 1; }
  docker run --rm --entrypoint sh "$rt3" -c 'curl -sSf -o /dev/null https://mirrors.aliyun.com && echo TLS_OK' \
    | grep -q TLS_OK || { echo "FAILED TLS still broken $task"; exit 1; }

  docker push "$rt3" >>"$LOG" 2>&1
  done_map[$task]=$rt3
  python3 - "$OUT" "$task" "$rt3" <<'PY'
import json, os, sys
p, k, v = sys.argv[1:4]
d = json.load(open(p)) if os.path.exists(p) else {}
d[k] = v
json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
PY
  echo "pushed $rt3"
done < "$LIST"

echo "done: $(python3 -c 'import json;print(len(json.load(open("'"$OUT"'"))))') images in $OUT"
echo "next (on the cluster):"
echo "  IMAGES_FILE=../rt3_ca/images_rt3.json BATCH_TAG=th-ca-rt3 python3 code/reg/make_assets.py \$(cut -d' ' -f1 build/rt3_ca/images_ca_broken.txt)"
echo "  bash code/reg/register_batch.sh 1.4.0 cth-th-ca 1.0.0 <(cut -d' ' -f1 build/rt3_ca/images_ca_broken.txt)"