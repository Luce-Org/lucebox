#!/usr/bin/env bash
# Usage: run_tasks.sh <run_tag> <first_arm: control|spec> <task>...
# Runs each task in both arms (alternating which arm goes first per task).
set -uo pipefail
cd ${TBSPEC_DIR:-/home/lucebox5/tbspec}
tag="$1"; first="$2"; shift 2
export PYTHONPATH=${TBSPEC_DIR:-/home/lucebox5/tbspec}
i=0
for task in "$@"; do
  if (( i % 2 == 0 )); then order=("$first" "$([[ $first == control ]] && echo spec || echo control)"); else order=("$([[ $first == control ]] && echo spec || echo control)" "$first"); fi
  for arm in "${order[@]}"; do
    job="${tag}__${task}__${arm}"
    echo "[$(date -u +%FT%TZ)] START $job" | tee -a logs/driver_${tag}.log
    rm -f state/current_container.json
    venv/bin/python prime.py "$task" 2>&1 | tee -a logs/driver_${tag}.log
    if [[ $arm == spec ]] && ! venv/bin/python probe_predictor.py 2>&1 | tee -a logs/driver_${tag}.log | grep -q "probe:.*deferred.*predictor_unavailable.*native_predictor_\(not_initialized\|timeout\)"; then :; elif [[ $arm == spec ]]; then
      echo "[$(date -u +%FT%TZ)] predictor dead before $job -> restarting server" | tee -a logs/driver_${tag}.log
      ./restart_server.sh >> logs/driver_${tag}.log 2>&1 || echo "[$(date -u +%FT%TZ)] SERVER RESTART FAILED" | tee -a logs/driver_${tag}.log
    fi
    venv/bin/harbor run -p datasets/terminal-bench/$task -a tb_agent:ToolSpecAgent \
      --ak arm=$arm --ak run_tag=$tag -n 1 --jobs-dir jobs --job-name "$job" \
      --agent-timeout-multiplier 2 -y -q > logs/harbor_${job}.log 2>&1
    rc=$?
    res=$(python3 - "$job" <<'PY'
import json,sys,glob
job=sys.argv[1]
fs=glob.glob(f"jobs/{job}/*/result.json")
if not fs: print("no-result"); sys.exit()
r=json.load(open(fs[0]))
v=r.get("verifier_result") or {}
ae=r.get("agent_execution") or {}
print(f"reward={v.get('rewards')} agent_exec={ae.get('started_at')}..{ae.get('finished_at')} exc={(r.get('exception_info') or {}).get('exception_type')}")
PY
)
    echo "[$(date -u +%FT%TZ)] END $job rc=$rc $res" | tee -a logs/driver_${tag}.log
    if [[ $arm == spec ]] && grep -q -E "native_predictor_not_initialized|native_predictor_timeout" jobs/$job/*/agent/tbspec_trace.json 2>/dev/null; then
      echo "[$(date -u +%FT%TZ)] predictor died during $job -> restarting server" | tee -a logs/driver_${tag}.log
      ./restart_server.sh >> logs/driver_${tag}.log 2>&1 || echo "[$(date -u +%FT%TZ)] SERVER RESTART FAILED" | tee -a logs/driver_${tag}.log
    fi
  done
  i=$((i+1))
done
echo "[$(date -u +%FT%TZ)] ALL DONE" | tee -a logs/driver_${tag}.log
