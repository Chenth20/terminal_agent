#!/bin/sh
# Pin the IPv4 addresses of the given hostnames into /etc/hosts.
#
#   sh dns_hosts_pin.sh <llm-endpoint-host> [<host> ...]
#
# Run it inside the task container before the agent starts, while DNS still works.
#
# Why: two TerminalHorizon tasks (qualify-sip-proxy-paths-..., integrate-heterogeneous-workloads-with-consul-...)
# rewrite /etc/resolv.conf to `nameserver 127.0.0.1` during their own bootstrap (local dnsmasq / consul).
# After that the harness can no longer resolve the LLM endpoint and every model call fails with a
# connection error. Pinning the endpoint's addresses beforehand keeps the harness reachable.
#
# HOSTS_FILE overrides the target file (default /etc/hosts).
HOSTS_FILE=${HOSTS_FILE:-/etc/hosts}
[ $# -gt 0 ] || { echo "usage: $0 <host> [<host> ...]" >&2; exit 2; }
for h in "$@"; do
  ips=$(getent ahostsv4 "$h" 2>/dev/null | awk '{print $1}' | sort -u | head -3)
  [ -n "$ips" ] || ips=$(python3 -c 'import socket,sys;print("\n".join(sorted({a[4][0] for a in socket.getaddrinfo(sys.argv[1],443,socket.AF_INET)})[:3]))' "$h" 2>/dev/null)
  [ -n "$ips" ] || { echo "[hosts-pin] $h: could not resolve, nothing pinned" >&2; continue; }
  for ip in $ips; do
    grep -q "^$ip $h\$" "$HOSTS_FILE" 2>/dev/null || echo "$ip $h" >> "$HOSTS_FILE"
  done
  n=$(grep -c " $h\$" "$HOSTS_FILE" 2>/dev/null) || n=0
  echo "[hosts-pin] $h: $n entries in $HOSTS_FILE"
done
