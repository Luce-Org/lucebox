#!/usr/bin/env bash
# Multi-client ee7 validation bench — 2026-05-22
# Runs each client × {baseline, ee7} with flock GPU guard
set -euo pipefail

WORKTREE=/home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath
CLIENTS_DIR="$WORKTREE/harness/clients"
RESULTS_DIR="$WORKTREE/dflash/bench/results/2026-05-22_multiclient_ee7"
RUN_DIR="/tmp/lucebox-harness-runs/multiclient-ee7-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

export MODEL_SERVER=lucebox
export LUCEBOX_SERVER_BACKEND=cpp
export DFLASH_SERVER_BIN="$WORKTREE/dflash/build/dflash_server"
export TARGET=/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf
export DRAFT=/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf
export MAX_CTX=98304
export MAX_TOKENS=512
export GGML_CUDA_NO_VMM=1
export DFLASH27B_KV_K=tq3_0
export DFLASH27B_KV_V=tq3_0
export VERIFY_MODE=ddtree
export BUDGET=16
export REPO_DIR="$WORKTREE"
export RUN_DIR
export EXTRA_SERVER_ARGS="--prefill-compression always --prefill-keep-ratio 0.05 --prefill-drafter /home/peppi/models/Qwen3-0.6B-BF16.gguf"

# Client binary overrides (harness defaults to /workspace paths)
export CLAUDE_BIN=/home/peppi/.local/bin/claude
export HERMES_BIN=/home/peppi/.local/bin/hermes
export OPENCODE_BIN=/home/peppi/.nvm/versions/node/v22.17.0/bin/opencode
export PI_BIN=/home/peppi/.asdf/shims/pi
export CODEX_BIN=/home/peppi/.local/bin/codex
export CLIENT_WORK_DIR="$RUN_DIR/client-work"
mkdir -p "$CLIENT_WORK_DIR/clients"

# Prompts
DECODE_CHECK="$CLIENTS_DIR/prompts/decode_check.txt"
REPO_INSPECT="$CLIENTS_DIR/prompts/repo_inspection.txt"

# Results accumulator
JSONL="$RESULTS_DIR/raw_results.jsonl"
> "$JSONL"

log() { echo "[bench] $*"; }

run_client() {
  local client="$1"    # claude_code, opencode, hermes, pi, codex
  local condition="$2" # baseline or ee7
  local prompt_file="$3"

  local stamp
  stamp="mc-ee7-${client}-${condition}-$(date +%Y%m%d-%H%M%S)"
  export STAMP="$stamp"
  export PROMPT_FILE="$prompt_file"
  export MARKER="OK_DONE"

  # Set early-exit env for ee7 condition
  if [[ "$condition" == "ee7" ]]; then
    export PFLASH_DRAFTER_EARLY_EXIT_N=7
    export PFLASH_DRAFTER_SCORE_LAYERS=7
  else
    unset PFLASH_DRAFTER_EARLY_EXIT_N 2>/dev/null || true
    unset PFLASH_DRAFTER_SCORE_LAYERS 2>/dev/null || true
  fi

  local script="$CLIENTS_DIR/run_${client}.sh"
  log "--- client=$client condition=$condition stamp=$stamp ---"

  local wall_start wall_end wall_s
  wall_start=$(date +%s%3N)

  local rc=0
  flock /tmp/dflash_gpu.lock bash "$script" 2>&1 | tee "$RESULTS_DIR/${stamp}.log" || rc=$?

  wall_end=$(date +%s%3N)
  wall_s=$(echo "scale=1; ($wall_end - $wall_start)/1000" | bc)

  # Parse server log
  local server_log="/tmp/lucebox-harness-runs/${stamp}/server.log"
  local client_out="/tmp/lucebox-harness-runs/${stamp}/${client}.out"
  # Alternative names used by some clients
  [[ -f "$client_out" ]] || client_out=$(ls /tmp/lucebox-harness-runs/${stamp}/*.out 2>/dev/null | head -1 || echo "")

  local drafter_fwd="N/A"
  local accept_rate="N/A"
  local ok_done=NO

  if [[ -f "$server_log" ]]; then
    # Extract last drafter forward time
    local fwd_line
    fwd_line=$(grep "forward+score in" "$server_log" | tail -1 || true)
    if [[ -n "$fwd_line" ]]; then
      drafter_fwd=$(echo "$fwd_line" | grep -oP 'forward\+score in \K[0-9.]+')
      drafter_fwd="${drafter_fwd}s"
    fi
    # Extract accept rate from last spec-decode line
    local spec_line
    spec_line=$(grep "spec-decode" "$server_log" | tail -1 || true)
    if [[ -n "$spec_line" ]]; then
      accept_rate=$(echo "$spec_line" | grep -oP 'accepted=\d+/\d+ \(\K[0-9.]+(?=%)' || true)
      if [[ -n "$accept_rate" ]]; then
        accept_rate="${accept_rate}%"
      fi
    fi
  fi

  if [[ -f "$client_out" ]]; then
    if grep -q "OK_DONE" "$client_out" 2>/dev/null; then
      ok_done=YES
    fi
  fi

  log "RESULT client=$client condition=$condition wall=${wall_s}s drafter_fwd=$drafter_fwd accept=$accept_rate ok_done=$ok_done rc=$rc"
  echo "{\"client\":\"$client\",\"condition\":\"$condition\",\"wall_s\":\"$wall_s\",\"drafter_fwd\":\"$drafter_fwd\",\"accept_rate\":\"$accept_rate\",\"ok_done\":\"$ok_done\",\"rc\":$rc,\"stamp\":\"$stamp\"}" >> "$JSONL"

  # Kill any lingering server on non-zero rc to avoid GPU contention.
  # Match on the specific binary path to avoid killing unrelated processes.
  if [[ $rc -ne 0 ]]; then
    pkill -f "$DFLASH_SERVER_BIN" 2>/dev/null || true
    sleep 2
  fi
}

# Matrix: client × prompt × {baseline, ee7}
# claude_code: decode_check (matches b04b678 validation)
# hermes, opencode, pi, codex: repo_inspection (exercises tool use)

for condition in baseline ee7; do
  run_client claude_code "$condition" "$DECODE_CHECK"
done

for condition in baseline ee7; do
  run_client hermes "$condition" "$REPO_INSPECT"
done

for condition in baseline ee7; do
  run_client opencode "$condition" "$REPO_INSPECT"
done

for condition in baseline ee7; do
  run_client pi "$condition" "$REPO_INSPECT"
done

for condition in baseline ee7; do
  run_client codex "$condition" "$REPO_INSPECT"
done

log "All runs complete. Results in $JSONL"
