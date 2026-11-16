#!/bin/sh
# TH flat-image (-flat-*-rt2) setup hook: bring up the sidecar services the image's supervisor would have started.
#
# Why: the platform launches the sandbox with its own wrapper command, NOT the image ENTRYPOINT, so the flattened
# image's supervisor (/opt/flatten/runtime.py -> ready.json) never runs. The original hook waited 10 min for
# ready.json, gave up, and the agent spent 5 h against services that were never started (172/174 flat tasks
# TIMEOUT in cth-thfull14-dsv41; the same in every other job, with every model and harness).
#
# Why not just run runtime.py (diagnosed in cth-th-flat-diag5..23):
#   * network_runtime.prepare() writes /proc/sys/net/ipv4/ip_forward -> read-only under ARCA (OSError 30).
#   * With that patched, the per-service private netns + veth bridge comes up and the service listens inside
#     its netns, but connect() across the bridge fails with EPERM in both directions (ARCA network policy);
#     so the agent could never reach http://api:8080 anyway.
#   * runtime.start() launches the service with unshare --pid --fork and stdout inherited; the hook's stdout
#     pipe then never closes and the platform waits on it forever.
# What works: chroot into the sidecar rootfs and run its argv in the main network namespace (verified: api
# answers 200 on 127.0.0.1:8080). Service hostnames resolve to 127.0.0.1 via /etc/hosts. Ports do not collide
# in the handoff batch (each service has its own port in the original compose).
exec 2>&1
PLAN=/opt/flatten/plan.json
PY=/usr/local/bin/python3; [ -x "$PY" ] || PY=$(command -v python3)
cat > /opt/flatten/th_flat_start.py <<'PYEOF'
import json, os, subprocess, sys, time
plan = json.load(open(sys.argv[1]))
sys.path.insert(0, "/opt/flatten")
import runtime as r  # identity(); start() itself is not used
names = [n for n in plan["order"] if n != "main"]
with open("/etc/hosts", "a") as f:
    f.write("".join(f"127.0.0.1 {n}\n" for n in names + ["main"]))
procs = {}
for name in names:
    svc = plan["services"][name]
    root, cwd = svc["root"], svc.get("cwd") or "/"
    uid, gid, groups, home = r.identity(root, svc.get("user", ""))
    env = dict(svc["env"]); env.setdefault("HOME", home); env.setdefault("HOSTNAME", name)
    env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    def setup(root=root, cwd=cwd, uid=uid, gid=gid, groups=groups):
        if root != "/":
            os.chroot(root)
        os.chdir(cwd)
        os.setgroups(groups); os.setgid(gid); os.setuid(uid)
    log = open(f"/var/log/th-flat-{name}.log", "ab")
    procs[name] = subprocess.Popen(svc["argv"], env=env, preexec_fn=setup, start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
    print("[th-flat] started", name, "pid", procs[name].pid, "argv", svc["argv"], flush=True)
    h = svc.get("healthcheck") or {}
    if h.get("test") or h.get("Test"):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not r.healthy(svc):
            time.sleep(2)
time.sleep(3)
print("[th-flat] exited early:", {n: p.poll() for n, p in procs.items() if p.poll() is not None}, flush=True)
json.dump({"status": "started", "services": names, "started_by": "th-flat-setup-hook", "mode": "chroot-main-netns"},
          open("/opt/flatten/ready.json", "w"))
PYEOF
/usr/bin/timeout -s KILL 600 "$PY" -u /opt/flatten/th_flat_start.py "$PLAN" </dev/null >/var/log/th-flat-start.out 2>&1
echo "[th-flat] starter rc=$? ready=$(cat /opt/flatten/ready.json 2>/dev/null || echo NO)"
cat /var/log/th-flat-start.out
echo "[th-flat] listeners: $(awk 'NR>1 && $4=="0A"{print $2}' /proc/net/tcp | tr '\n' ' ')"
exit 0
