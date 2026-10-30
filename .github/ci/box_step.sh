#!/usr/bin/env bash
# The shell of the run steps of lucebox3's jobs:
#
#   defaults: {run: {shell: "bash ${{ github.workspace }}/.github/ci/box_step.sh {0}"}}
#
# Development work on lucebox3 has priority over CI, and a step that runs beside
# it can fail out of GPU memory or RAM (the Strix Halo's memory is the box's RAM)
# for reasons unrelated to the change under test. So a step:
# - starts only when no other process is using a GPU. After waiting for one,
#   and before any rerun, it waits until the GPUs have stayed free for $quiet
#   seconds, so it does not start while a benchmark restarts its server;
# - steps aside when another process starts using a GPU while it runs: it is
#   stopped at once, which frees what it held, and rerun from the start once
#   the GPUs are free again;
# - is rerun once when it fails after the kernel's OOM killer killed a process
#   while it ran, which covers work that started and ended between two looks.
# KFD lists the processes using a GPU, and any user can read the list, so
# developers need to do nothing. A process is this job's when it carries the
# job's RUNNER_TRACKING_ID or descends from this script.
#
# A job waits $BOX_WAIT_MINUTES minutes in all, then fails naming the processes
# that kept the GPUs busy. Each run of a step gets $BOX_STEP_MINUTES minutes,
# and what a step leaves running when its shell exits is stopped with it.
set -uo pipefail

script=$1
wait_limit=$(( ${BOX_WAIT_MINUTES:-120} * 60 ))
step_limit=$(( ${BOX_STEP_MINUTES:-20} * 60 ))
quiet=120
kfd=/sys/class/kfd/kfd/proc
vmstat=/proc/vmstat
job=${RUNNER_TRACKING_ID:-}
waited_file=${RUNNER_TEMP:-/tmp}/box_waited_${job:-$PPID}

[[ -d $kfd ]] || { echo "::error::$kfd is missing; is the amdgpu driver loaded?"; exit 1; }

ours() {
  local pid=$1 environ stat
  if [[ -n $job ]] && environ=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null) &&
     [[ $'\n'$environ$'\n' == *$'\n'"RUNNER_TRACKING_ID=$job"$'\n'* ]]; then
    return 0
  fi
  while (( pid > 1 )); do
    (( pid == $$ )) && return 0
    stat=$(< "/proc/$pid/stat") 2>/dev/null || return 1
    stat=${stat##*) }               # the fields after the command name
    read -r _ pid _ <<< "$stat"     # state, parent
  done
  return 1
}

# rocm-smi and amd-smi open the GPU driver to watch for GPU events, so KFD
# lists them while they run, e.g. `rocm-smi --showevents`, but they use no GPU.
# Nothing else is told apart by what it holds: KFD counts neither managed nor
# pinned host memory, which is what the engine loads a large model into on the
# Strix Halo.
smi='^[^ ]*python[0-9.]* ([^ ]*/)?(rocm-smi|amd-smi|rocm_smi\.py|amdsmi_cli\.py) '
monitor() {
  local args
  args=$(tr '\0' ' ' 2>/dev/null < "/proc/$1/cmdline") || return 1
  [[ $args =~ $smi ]]
}

# The PIDs of the other processes using a GPU.
others() {
  local dir pid
  for dir in "$kfd"/*/; do
    [[ -d $dir ]] || continue
    pid=$(basename "$dir")
    ours "$pid" || monitor "$pid" || echo "$pid"
  done
  return 0
}

describe() {
  local pid
  for pid in $1; do
    ps -o user=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-200 || echo "pid $pid (exited)"
  done
}

waited() { cat "$waited_file" 2>/dev/null || echo 0; }
ooms() { awk '$1 == "oom_kill" { print $2 }' "$vmstat"; }

# Returns at once when no other process uses a GPU, unless the step is being
# rerun ($1, why): it stepped aside or failed for another process, so the GPUs
# must stay free for $quiet seconds even if that process is already gone.
wait_for_gpus() {
  local pids start=$SECONDS free_since='' shown=''
  pids=$(others)
  [[ -z $pids && -z ${1:-} ]] && return 0
  echo "::notice::Waiting until lucebox3's GPUs stay free for $quiet s."
  while :; do
    if (( $(waited) + SECONDS - start >= wait_limit )); then
      echo "::error::lucebox3's GPUs stayed in use; this job waited $(( wait_limit / 60 )) min for them. In use by:"
      describe "$(others)"
      exit 1
    fi
    if [[ -n $pids ]]; then
      free_since=
      [[ $pids == "$shown" ]] || { echo "In use by:"; describe "$pids"; shown=$pids; }
    elif [[ -z $free_since ]]; then
      free_since=$SECONDS shown=''
      echo "The GPUs are free; starting if they stay free for $quiet s."
    elif (( SECONDS - free_since >= quiet )); then
      break
    fi
    sleep 10
    pids=$(others)
  done
  echo $(( $(waited) + SECONDS - start )) > "$waited_file"
  echo "Waited $(( (SECONDS - start) / 60 )) min for the GPUs."
}

# The step runs in its own session, so it stops whole, children included,
# also when its shell has already exited.
step=
stop_step() {
  [[ -n $step ]] || return 0
  kill -TERM -- "-$step" 2>/dev/null
  for _ in $(seq 20); do
    kill -0 -- "-$step" 2>/dev/null || break
    sleep 0.5
  done
  kill -KILL -- "-$step" 2>/dev/null
  wait "$step" 2>/dev/null
  step=
}
trap 'stop_step; exit 143' TERM INT

oom_retried='' outcome=''
while :; do
  wait_for_gpus "$outcome"
  oom_before=$(ooms)
  started=$SECONDS
  # bash -e is the shell GitHub runs a step with when it names none, as these
  # steps did before, so a step's pipelines keep failing as they did: on their
  # last command.
  setsid bash -e "$script" &
  step=$!
  outcome=
  while kill -0 "$step" 2>/dev/null; do
    pids=$(others)
    if [[ -n $pids ]]; then
      echo "::warning::Another process started using a GPU; stopping this step, to rerun it once the GPUs are free. In use by:"
      describe "$pids"
      outcome=yielded
      break
    fi
    if (( SECONDS - started >= step_limit )); then
      echo "::error::The step ran for $(( step_limit / 60 )) min; stopping it."
      outcome=timed_out
      break
    fi
    sleep 1
  done
  if [[ -n $outcome ]]; then
    stop_step
    [[ $outcome == yielded ]] && continue
    exit 124
  fi
  wait "$step"
  status=$?
  # What the step's shell left running when it exited, e.g. a server it started
  # in the background or compilers whose make was killed, stops with it: past
  # here nothing would stop it for its time limit or step aside for others.
  stop_step
  (( status == 0 )) && exit 0
  if [[ -z $oom_retried ]] && (( $(ooms) > oom_before )); then
    echo "::warning::The step failed (exit $status) after the kernel's OOM killer killed a process while it ran; rerunning it once the GPUs are free."
    oom_retried=1 outcome=oom_killed
    continue
  fi
  exit "$status"
done
