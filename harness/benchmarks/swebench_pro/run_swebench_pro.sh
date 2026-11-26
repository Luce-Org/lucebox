#!/usr/bin/env bash
set -euo pipefail

# Run SWE-bench Pro via Harbor against a running luce_server endpoint.
#
# Defaults target the curated 24-task long-horizon slice on Harbor's built-in
# `scale-ai/swe-bench-pro` dataset using `terminus-2` (runs on the host, so it
# reaches 127.0.0.1 directly with zero container networking setup).
#
# For in-container agents (`opencode`, `claude-code`, `codex`, `pi`, `hermes`,
# `mini-swe-agent`), start `luce_server` with `--host 0.0.0.0`; this launcher
# automatically adds `docker-compose.host-gateway.yaml` and routes requests
# through `host.docker.internal` on Linux Docker hosts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

: "${HARBOR_BIN:=$(command -v harbor || echo "$HOME/.local/bin/harbor")}"
: "${DATASET:=scale-ai/swe-bench-pro}"
: "${TASKS_PATH:=}"
: "${SLICE_FILE:=$SCRIPT_DIR/long_horizon_24.txt}"
: "${AGENT:=terminus-2}"
: "${MODEL:=Qwen3.8-27B}"
: "${SERVER_HOST:=127.0.0.1}"
: "${SERVER_PORT:=8000}"
: "${BASE_URL:=}"
: "${API_KEY:=lucebox}"
: "${API_FORMAT:=}"
: "${CONCURRENCY:=1}"
: "${LIMIT:=}"
: "${JOBS_DIR:=$REPO_ROOT/.harness-work/harbor-jobs}"
: "${JOB_NAME:=swebench-pro-${AGENT}-$(date +%Y%m%d-%H%M%S)}"
: "${DRY_RUN:=0}"

if [[ ! -x "$HARBOR_BIN" ]]; then
  echo "error: harbor binary not found at $HARBOR_BIN (install via: uv tool install harbor)" >&2
  exit 2
fi

is_in_container_agent() {
  case "$1" in
    terminus|terminus-1|terminus-2|oracle|nop|*:*) return 1 ;;
    *) return 0 ;;
  esac
}

default_api_format() {
  case "$1" in
    claude-code) echo "anthropic_messages" ;;
    codex) echo "openai_responses" ;;
    *) echo "openai_chat_completions" ;;
  esac
}

default_provider_prefix() {
  case "$1" in
    anthropic_messages) echo "anthropic" ;;
    *) echo "openai" ;;
  esac
}

cmd=("$HARBOR_BIN" run)

if [[ -n "$TASKS_PATH" ]]; then
  cmd+=(-p "$TASKS_PATH")
else
  cmd+=(-d "$DATASET")
fi

# Apply slice filter unless SLICE_FILE=all or empty.
if [[ -n "$SLICE_FILE" && "$SLICE_FILE" != "all" ]]; then
  if [[ ! -f "$SLICE_FILE" ]]; then
    echo "error: slice file not found: $SLICE_FILE" >&2
    exit 2
  fi
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    if [[ -n "$TASKS_PATH" ]]; then
      cmd+=(-i "$line")
    else
      # Harbor package registry task names carry the `<org>/` prefix (e.g. `scale-ai/<id>`).
      if [[ "$line" == */* ]]; then
        cmd+=(-i "$line")
      else
        cmd+=(-i "*${line}")
      fi
    fi
  done < "$SLICE_FILE"
fi

if [[ -n "$LIMIT" ]]; then
  cmd+=(-l "$LIMIT")
fi

cmd+=(-a "$AGENT" -n "$CONCURRENCY" -o "$JOBS_DIR" --job-name "$JOB_NAME")

if [[ "$AGENT" != "oracle" && "$AGENT" != "nop" ]]; then
  fmt="${API_FORMAT:-$(default_api_format "$AGENT")}"
  provider="$(default_provider_prefix "$fmt")"

  model_arg="$MODEL"
  if [[ "$model_arg" != */* ]]; then
    model_arg="${provider}/${model_arg}"
  fi

  resolved_host="$SERVER_HOST"
  if is_in_container_agent "$AGENT"; then
    if [[ "$resolved_host" == "127.0.0.1" || "$resolved_host" == "localhost" ]]; then
      resolved_host="host.docker.internal"
    fi
    if [[ "$resolved_host" == "host.docker.internal" ]]; then
      cmd+=(--extra-docker-compose "$SCRIPT_DIR/docker-compose.host-gateway.yaml")
    fi
    if [[ -n "$TASKS_PATH" ]]; then
      # SWE-bench Pro V2 sets `[agent] network_mode = "no-network"`; allowlist the local server host.
      cmd+=(--allow-agent-host "$resolved_host")
    fi
  fi

  if [[ -n "$BASE_URL" ]]; then
    url="$BASE_URL"
  elif [[ "$fmt" == "anthropic_messages" ]]; then
    url="http://${resolved_host}:${SERVER_PORT}"
  else
    url="http://${resolved_host}:${SERVER_PORT}/v1"
  fi

  cmd+=(-m "$model_arg" --base-url "$url" --api-format "$fmt")

  if [[ "$fmt" == "anthropic_messages" ]]; then
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$API_KEY}"
  else
    export OPENAI_API_KEY="${OPENAI_API_KEY:-$API_KEY}"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if (($# > 0)); then
  cmd+=("$@")
fi

exec "${cmd[@]}"
