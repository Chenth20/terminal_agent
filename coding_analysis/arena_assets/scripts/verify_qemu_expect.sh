set -u
# Isolate the ONLY textual diff in qemu-startup's test (env -u LD_LIBRARY_PATH before expect) from the VM:
# a fake telnet server on 6665 answers the login prompt, so run.exp succeeds iff `expect`/`telnet` start.
# We run the real test_outputs.py under BOTH test texts, with LD_LIBRARY_PATH set exactly as the reward
# stage sees it in our containers (/.singularity.d/libs) and additionally poisoned with the pyhost lib.
T=/storage/openpsi/users/chen/tb_local
rm -rf /tmp/vpy && mkdir -p /tmp/vpy && tar xzf $T/harness/bin/python/python313.tar.gz -C /tmp/vpy
/tmp/vpy/python313/bin/python3 -m pip -q install --index-url https://mirrors.aliyun.com/pypi/simple/ pytest==8.4.1 2>&1 | tail -1
singularity exec --no-home --writable-tmpfs \
  --bind $T/tasks/qemu-startup/tests:/tests_old:ro --bind $T/tasks/qemu-startup/tests.arena:/tests_new:ro \
  --bind /tmp/vpy:/py:ro $T/sif/tb2-qemu-startup.sif bash -c '
  export PATH=/py/python313/bin:$PATH
  echo "image LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}  expect=$(command -v expect) telnet=$(command -v telnet)"
  # fake "alpine": prints a login prompt, then a shell prompt, answers uname -r
  timeout 120 python3 - <<"PY" &
import socket
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",6665)); s.listen(1)
while True:
    c,_=s.accept()
    c.sendall(b"\r\nlocalhost login: "); c.recv(100)
    c.sendall(b"\r\n# "); c.recv(100)
    c.sendall(b"6.6.4-1-lts\r\n# "); c.close()
PY
  sleep 1; echo "server pid=$!"
  for mode in "as-is" "poisoned"; do
    for which in old new; do
      rm -f /tmp/data.txt; cd /app
      if [ $mode = poisoned ]; then export LD_LIBRARY_PATH=/sandbox-net/py/python313/lib:${LD_LIBRARY_PATH:-}; fi
      r=$(python3 -m pytest /tests_$which/test_outputs.py -q 2>&1 | tail -1)
      printf "  LD_LIBRARY_PATH=%-8s test=%s -> %s | data.txt: %s\n" "$mode" "$which" "$r" "$(tr -d "\r" < /tmp/data.txt 2>/dev/null | tr "\n" " " | cut -c1-40)"
    done
  done
' 2>&1 | grep -v setlocale
