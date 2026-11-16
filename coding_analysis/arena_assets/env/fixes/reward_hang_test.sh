#!/bin/bash
# cth env @1.4.0: run the official verifier (tests/test_orig.sh, byte-identical to upstream
# tests/test.sh) under a 1500 s bound. In job cth-thfull14-dsv41-2610270211 this env's reward
# phase hung until the 7200 s ARCA exec timeout (the check script waits on a process that never
# exits), so the task was recorded TIMEOUT although the agent had finished normally.
mkdir -p /logs/verifier
[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt
timeout -k 60 1500 bash /tests/test_orig.sh
code=$?
if [ "$code" = 124 ] || [ "$code" = 137 ]; then
  echo "[cth-timeout-wrapper] test_orig.sh exceeded 1500 s; reward.txt=$(cat /logs/verifier/reward.txt 2>/dev/null)" >&2
fi
exit $code
