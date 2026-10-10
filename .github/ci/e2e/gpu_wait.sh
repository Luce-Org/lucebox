#!/usr/bin/env bash
# Wait until no process holds an AMD GPU, so the e2e run does not share the
# machine with someone's manual server or benchmark.
#
# Any KFD process counts: /dev/kfd is shared by every AMD GPU, and a model run
# on the other GPU still competes for host memory and CPU. The holders come
# from the kernel's KFD process list, /sys/class/kfd/kfd/proc/<pid>, which
# lists every user's processes and which any user can read.
#
# The last line printed is state=free|busy. Exits 1 when the process list
# cannot be read, since a busy machine would then look free.
#
# Usage: gpu_wait.sh [max seconds to wait, default 300]
set -u

deadline=$((SECONDS + ${1:-300}))
# Overridable for the tests.
kfd_procs=${KFD_PROC_DIR:-/sys/class/kfd/kfd/proc}

if [ ! -r "$kfd_procs" ] || [ ! -x "$kfd_procs" ]; then
  echo "::error title=GPU check unavailable::Cannot read $kfd_procs to see who holds the GPUs"
  exit 1
fi

while :; do
  # An entry outlives its process for a moment while the driver frees the GPU
  # memory; count it until it is gone.
  pids=$(ls -A "$kfd_procs")
  if [ -z "$pids" ]; then
    echo "No process holds the GPUs; they are free."
    echo "state=free"
    exit 0
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "::warning title=GPU busy::Other processes still hold the GPUs; skipping the model e2e run."
    for pid in $pids; do
      if ! line=$(ps -o pid=,user=,etime=,args= -p "$pid"); then
        line="$pid (exited, but the GPU driver still holds its state)"
      fi
      echo "${line:0:200}"
    done
    echo "state=busy"
    exit 0
  fi
  echo "GPU busy (PIDs: $(echo "$pids" | tr '\n' ' ')); waiting..."
  left=$((deadline - SECONDS))
  sleep $((left < 20 ? left : 20))
done
