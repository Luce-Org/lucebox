#!/usr/bin/env bash
# Fail fast, with a diagnosis, when ROCm's KFD driver is wedged.
#
# rocminfo on a wedged KFD blocks in uninterruptible sleep (D-state): timeout(1)
# cannot kill it and a foreground wait blocks until the job timeout. Probe in
# the background (output to a file so no pipe keeps the step alive) and enforce
# the deadline in the shell. On a hang, dump the evidence (D-state holders,
# dmesg) and exit 1 within seconds. Each diagnostic has its own deadline, and
# sudo never prompts: reading a wedged process's state can block too.
#
# Usage: kfd_health.sh [rocminfo output file, default /tmp/rocminfo.out]
set -u

out="${1:-/tmp/rocminfo.out}"
/opt/rocm/bin/rocminfo > "$out" 2>&1 &
probe=$!
for _ in $(seq 1 15); do
  kill -0 "$probe" 2>/dev/null || break
  sleep 1
done
if kill -0 "$probe" 2>/dev/null; then
  echo "::error::rocminfo hung (likely D-state) — ROCm/KFD wedged; the box needs a reboot"
  echo "--- probe state:"
  ps -o pid,stat,wchan:32,comm -p "$probe" || true
  echo "--- processes holding /dev/kfd:"
  timeout -k 2 10 sudo -n fuser -v /dev/kfd 2>&1 || true
  echo "--- D-state processes:"
  ps -eo pid,user,stat,wchan:32,comm | awk '$3 ~ /D/' || true
  echo "--- recent amdgpu/kfd dmesg:"
  timeout -k 2 10 sudo -n dmesg 2>/dev/null | grep -iE "amdgpu|kfd" | tail -15 || true
  kill -9 "$probe" 2>/dev/null || true
  disown "$probe" 2>/dev/null || true
  exit 1
fi
if ! wait "$probe"; then
  echo "::error::rocminfo exited non-zero"
  tail -5 "$out"
  exit 1
fi
echo "KFD healthy"
