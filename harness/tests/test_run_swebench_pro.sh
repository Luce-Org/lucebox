#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBP_DIR="$REPO_ROOT/harness/benchmarks/swebench_pro"
LAUNCHER="$SBP_DIR/run_swebench_pro.sh"
SLICE="$SBP_DIR/long_horizon_24.txt"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

FAKE_HARBOR="$TMP_DIR/harbor"
cat >"$FAKE_HARBOR" <<'EOF'
#!/usr/bin/env bash
printf 'OPENAI_API_KEY=%s\n' "${OPENAI_API_KEY:-}"
printf 'ANTHROPIC_API_KEY=%s\n' "${ANTHROPIC_API_KEY:-}"
printf 'ARGS:%s\n' "$*"
EOF
chmod +x "$FAKE_HARBOR"

# 1. Verify slice has exactly 24 unique valid instance IDs.
mapfile -t ids < <(grep -vE '^[[:space:]]*(#|$)' "$SLICE")
if [[ "${#ids[@]}" -ne 24 ]]; then
  echo "expected 24 tasks in slice, got ${#ids[@]}" >&2
  exit 1
fi
unique_count="$(printf '%s\n' "${ids[@]}" | sort -u | wc -l | tr -d ' ')"
if [[ "$unique_count" -ne 24 ]]; then
  echo "expected 24 unique tasks in slice, got $unique_count" >&2
  exit 1
fi
for id in "${ids[@]}"; do
  if [[ "$id" != instance_* ]]; then
    echo "unexpected task ID format: $id" >&2
    exit 1
  fi
done

# 2. Default terminus-2 host-side invocation uses 127.0.0.1 and no host-gateway overlay.
out="$(HARBOR_BIN="$FAKE_HARBOR" "$LAUNCHER")"
grep -q -- '-d scale-ai/swe-bench-pro' <<<"$out"
grep -q -- '-a terminus-2' <<<"$out"
grep -q -- '-m openai/Qwen3.8-27B' <<<"$out"
grep -q -- '--base-url http://127.0.0.1:8000/v1' <<<"$out"
grep -q -- '--api-format openai_chat_completions' <<<"$out"
grep -q -- 'OPENAI_API_KEY=lucebox' <<<"$out"
if grep -q -- 'docker-compose.host-gateway.yaml' <<<"$out"; then
  echo "terminus-2 should not require host-gateway compose overlay" >&2
  exit 1
fi

# 3. In-container opencode maps 127.0.0.1 -> host.docker.internal and attaches compose overlay.
out="$(HARBOR_BIN="$FAKE_HARBOR" AGENT=opencode SERVER_PORT=18080 CONCURRENCY=4 "$LAUNCHER")"
grep -q -- '-a opencode' <<<"$out"
grep -q -- '-n 4' <<<"$out"
grep -q -- '--base-url http://host.docker.internal:18080/v1' <<<"$out"
grep -q -- '--extra-docker-compose .*docker-compose.host-gateway.yaml' <<<"$out"

# 4. In-container claude-code against local V2 tasks uses anthropic_messages and allowlists host.
out="$(HARBOR_BIN="$FAKE_HARBOR" AGENT=claude-code TASKS_PATH=/tmp/v2/tasks DRY_RUN=1 "$LAUNCHER")"
grep -q -- '-p /tmp/v2/tasks' <<<"$out"
grep -q -- '-a claude-code' <<<"$out"
grep -q -- '-m anthropic/Qwen3.8-27B' <<<"$out"
grep -q -- '--base-url http://host.docker.internal:8000' <<<"$out"
grep -q -- '--api-format anthropic_messages' <<<"$out"
grep -q -- '--allow-agent-host host.docker.internal' <<<"$out"
grep -q -- 'ANTHROPIC_API_KEY=lucebox' <<<"$out"
grep -q -- '--dry-run' <<<"$out"

# 5. Oracle skips model/auth flags.
out="$(HARBOR_BIN="$FAKE_HARBOR" AGENT=oracle LIMIT=1 "$LAUNCHER")"
grep -q -- '-a oracle' <<<"$out"
grep -q -- '-l 1' <<<"$out"
if grep -q -- '--base-url' <<<"$out"; then
  echo "oracle should not pass --base-url" >&2
  exit 1
fi

echo "test_run_swebench_pro: OK"
