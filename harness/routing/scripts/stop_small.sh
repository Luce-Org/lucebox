#!/usr/bin/env bash
# Stop instances started by launch_small.sh, via their pidfiles only
# (never pkill -f: it matches the calling shell).
#   stop_small.sh [name ...]   (default: every pidfile in runs/servers/)
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")/.." && pwd)/runs/servers"
shopt -s nullglob
files=()
if [[ $# -gt 0 ]]; then
  for n in "$@"; do files+=("$RUN_DIR/$n.pid"); done
else
  files=("$RUN_DIR"/*.pid)
fi

for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "no pidfile $f"; continue; }
  name="$(basename "$f" .pid)"; pid="$(cat "$f")"
  # Only signal the pid if it is still a luce_server (pid-reuse guard).
  if kill -0 "$pid" 2>/dev/null && [[ "$(ps -o comm= -p "$pid")" == luce_server* ]]; then
    kill -TERM "$pid"
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then echo "$name: SIGKILL"; kill -KILL "$pid"; fi
    echo "stopped $name ($pid)"
  else
    echo "$name not running"
  fi
  rm -f "$f"
done
