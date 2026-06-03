#!/usr/bin/env bash
# Run luce-bench as a harness client against a freshly-started Lucebox server.
#
# Slots into the same start-server → run-client → save-logs → stop-server
# pattern as the other harness/clients/run_*.sh wrappers (run_codex.sh,
# run_claude_code.sh, etc.). The "client" here is luce-bench (the standalone
# HTTP capability bench, now an in-tree workspace member at luce-bench/).
#
# Why this exists: luce-bench is just another HTTP client of /v1/chat/completions.
# Wrapping it in the harness pattern gives operators a uniform way to invoke
# it ("did this server change break luce-bench?") alongside real-client smoke
# tests, and lets the harness sweep matrix surface luce-bench regressions the
# same way it surfaces an OpenCode or Hermes regression.
#
# Knobs (env var or default):
#   LUCEBENCH_AREA      area(s) to run; pass the comma list (or `all`) to
#                       luce-bench directly.
#                       (default: empty → the level1 set
#                       `smoke,code,gsm8k,agent,longctx` — matches
#                       `luce-bench/src/lucebench/levels.py:LEVELS["level1"]`.
#                       Use `LUCEBENCH_AREA=all` for the full stdlib sweep;
#                       `LUCEBENCH_AREA=forge` requires the [forge] extra.)
#   LUCEBENCH_THINK     1 → --think, 0 → --no-think, empty → per-area
#                       defaults from luce-bench's area cards (recommended).
#                       Default empty so we don't override card-defined
#                       defaults; set `LUCEBENCH_THINK=0` for the
#                       ~4× faster nothink mode on gemma-4-26b (see
#                       2026-05-26 think/nothink comparison) when running
#                       A/B sweeps.
#   LUCEBENCH_MAX_TOKENS overrides per-request decode cap when set
#   LUCEBENCH_TIMEOUT   per-request wall timeout in seconds (default 300)
#   LUCEBENCH_PARALLEL  in-flight concurrency (default 1 — single-GPU)
#
# All harness/common.sh knobs apply: MODEL_SERVER, LUCEBOX_SERVER_BACKEND
# (use `cpp` to drive the native dflash_server), MAX_CTX, BUDGET, MODEL_ID,
# EXTRA_SERVER_ARGS, PORT, etc.
#
# Output:
#   $LOG_DIR/lucebench-{area,sweep}.{json,md}  — bench results (per-case rows
#                                                + markdown summary)
#   $LOG_DIR/lucebench.out                     — stdout/stderr from the run
#   $LOG_DIR/server.log                        — server stdout/stderr
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MAX_CTX:=32768}"
: "${BUDGET:=22}"
: "${VERIFY_MODE:=ddtree}"
: "${EXTRA_SERVER_ARGS:=--lazy-draft}"
: "${LUCEBENCH_AREA:=}"
: "${LUCEBENCH_THINK:=}"
: "${LUCEBENCH_MAX_TOKENS:=}"
: "${LUCEBENCH_TIMEOUT:=300}"
: "${LUCEBENCH_PARALLEL:=1}"
source "$SCRIPT_DIR/common.sh"

CLIENT_OUT="$LOG_DIR/lucebench.out"

# Build the luce-bench argv. With no LUCEBENCH_AREA, we run the level1 set
# (smoke + code + gsm8k + agent + longctx — the standard capability gate
# documented in luce-bench/src/lucebench/levels.py), and write per-area
# JSONs + `_summary.{json,md}` under $LOG_DIR/lucebench-sweep/.
# With LUCEBENCH_AREA=X (single area), we write a single JSON to
# $LOG_DIR/lucebench-X.json so the file name carries the area.
# With LUCEBENCH_AREA=<comma list> or `all`, we sweep into lucebench-sweep/.
# `--areas` is the canonical flag since luce-bench v0.2.5; the older
# `--sweep` is still accepted but emits a deprecation note.
lucebench_args=(--base-url "$BASE_URL" --model "$MODEL_ID" \
                --timeout "$LUCEBENCH_TIMEOUT" --parallel "$LUCEBENCH_PARALLEL")

# Default area set when LUCEBENCH_AREA is unset/empty: the level1 capability
# gate (mirrors luce-bench's `--level level1`). Picking `all` here was too
# broad — it tripped slow areas (ds4-eval, forge, agent_recorded) on every
# default run.
: "${LUCEBENCH_AREA_DEFAULT:=smoke,code,gsm8k,agent,longctx}"
effective_area="${LUCEBENCH_AREA:-$LUCEBENCH_AREA_DEFAULT}"

if [[ "$effective_area" == *","* || "$effective_area" == "all" ]]; then
  # Multi-area or `all`: sweep, write per-area JSONs + a roll-up.
  lucebench_args+=(--areas "$effective_area" --out-dir "$LOG_DIR" --name lucebench-sweep)
else
  # Single area: one JSON named after the area for convenient diffing.
  lucebench_args+=(--areas "$effective_area" \
                   --json-out "$LOG_DIR/lucebench-$effective_area.json")
fi

# --think / --no-think only applies when explicitly set. Leaving the flag
# off lets the server's card-defined defaults govern (recommended for
# capability gates; explicit modes are for A/B sweeps).
if [[ "$LUCEBENCH_THINK" == "1" ]]; then
  lucebench_args+=(--think)
elif [[ "$LUCEBENCH_THINK" == "0" ]]; then
  lucebench_args+=(--no-think)
fi

if [[ -n "$LUCEBENCH_MAX_TOKENS" ]]; then
  lucebench_args+=(--max-tokens "$LUCEBENCH_MAX_TOKENS")
fi

start_lucebox_server
trap stop_lucebox_server EXIT
wait_lucebox_server

set +e
cd "$REPO_DIR"
# Delegate to harness.bench (the Python entry point) so this wrapper, the
# `lucebox profile` framework, and ad-hoc operators all go through the
# same argv-building source of truth.
uv run python -m harness.bench "${lucebench_args[@]}" \
  > "$CLIENT_OUT" 2>&1
RC=$?
set -e

finish_report "$CLIENT_OUT" "$RC"
exit "$RC"
