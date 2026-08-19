#!/usr/bin/env bash
# Stop any running dflash_server, relaunch, wait until listening. Prints log path.
cd /home/lucebox5/tbspec
pkill -x dflash_server 2>/dev/null; sleep 2; pkill -9 -x dflash_server 2>/dev/null
for i in $(seq 1 30); do pgrep -x dflash_server >/dev/null || break; sleep 1; done
log=logs/server_$(date +%Y%m%d_%H%M%S).log
setsid nohup ./launch_server.sh > "$log" 2>&1 < /dev/null &
lpid=$!
for i in $(seq 1 240); do
  grep -q "listening on" "$log" && { echo "$log"; exit 0; }
  kill -0 "$lpid" 2>/dev/null || pgrep -x dflash_server >/dev/null || { echo "SERVER_EXITED $log" >&2; tail -20 "$log" >&2; exit 1; }
  sleep 5
done
echo "TIMEOUT $log" >&2; exit 1
