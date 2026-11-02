#!/usr/bin/env bash
# Serve a model the way lucebox3 serves it and check what it answers: one
# request whose drafts the drafter must get accepted, then, when the server
# takes several at once, that many together. Each must count to 30. Run from
# the repository root, where a profile finds its server/share files. With
# SERVE_SMOKE_PROFILE set, the server must have applied that launch profile.
#
#   serve_smoke.sh <model id> <requests at once> <luce_server> <model.gguf> [luce_server flags...]
set -euo pipefail

name=$1 at_once=$2 server=$3 model=$4
shift 4
port=${SERVE_SMOKE_PORT:-8098}
log=${RUNNER_TEMP:-/tmp}/serve_smoke_$name.log
url=http://127.0.0.1:$port

[[ -r $model ]] || { echo "::error::$model is missing or not readable by $(id -un)"; exit 1; }

"$server" "$model" "$@" --host 127.0.0.1 --port "$port" --model-name "$name" \
  --kv-cache-dir off > "$log" 2>&1 &
pid=$!
# Best effort under set -e: kill fails when the server already exited (the case the
# log explains) and wait returns the signal's status, so either would end the trap
# before the log.
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
      echo "--- luce_server log (tail)"; tail -n 40 "$log"' EXIT

# A model that streams experts from the SSD (DeepSeek V4.1) loads its dense
# weights and hot experts first: about 80 s with the file in the page cache,
# longer cold.
for _ in $(seq 600); do
  curl -sf -m 5 "$url/v1/models" > /dev/null && break
  kill -0 "$pid" 2>/dev/null || { echo "::error::luce_server exited while loading"; exit 1; }
  sleep 1
done
curl -sf -m 5 "$url/v1/models" > /dev/null || { echo "::error::luce_server was not ready in 10 minutes"; exit 1; }
if [[ -n ${SERVE_SMOKE_PROFILE:-} ]] && ! grep -q "^\[server\] profile $SERVE_SMOKE_PROFILE: flags" "$log"; then
  echo "::error::luce_server did not apply the $SERVE_SMOKE_PROFILE profile"; exit 1
fi

ask() {
  curl -sf -m 300 "$url/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "'"$name"'", "max_tokens": 128, "temperature": 0, "reasoning_effort": "none",
    "messages": [{"role": "user", "content": "Count from 1 to 30, separated by commas. Reply with the numbers only."}]}'
}

replies=${RUNNER_TEMP:-/tmp}/serve_smoke_$name.replies
rm -rf "$replies" && mkdir -p "$replies"
ask > "$replies/alone.json" || { echo "::error::the first request failed"; exit 1; }
status=$(curl -sf -m 5 "$url/status/json")
if (( at_once > 1 )); then
  # Only the requests: a bare wait would also wait for luce_server. One that fails
  # leaves its reply empty, which the check below reports.
  asking=()
  for i in $(seq "$at_once"); do ask > "$replies/together-$i.json" & asking+=($!); done
  for request in "${asking[@]}"; do wait "$request" || true; done
fi

python3 - "$replies" "$status" "$at_once" <<'EOF'
import json
import re
import sys
from pathlib import Path

replies, status, at_once = Path(sys.argv[1]), json.loads(sys.argv[2]), int(sys.argv[3])


def answer(path):
    text = path.read_text()
    return (json.loads(text)["choices"][0]["message"].get("content") or "") if text else ""


def counts_to_30(text):
    # Every number from 1 to 30 in order and nothing else, whatever the separators.
    return re.findall(r"\d+", text) == [str(n) for n in range(1, 31)]


problems = []
alone = answer(replies / "alone.json")
served = status["perf_history"][-1]
print(f"alone: {alone!r}, decode {served['decode_tok_s']:.1f} tok/s, "
      f"drafter {'on' if served.get('spec_decode') else 'off'}, accepted {served.get('accept_rate', 0):.0%}")
if not counts_to_30(alone):
    problems.append("the answer does not count to 30")
if not served.get("spec_decode"):
    problems.append("the drafter did not decode the answer")
elif not served.get("accept_rate"):
    problems.append("the drafter drafted, but no draft was accepted")
if at_once > 1:
    together = [answer(path) for path in sorted(replies.glob("together-*.json"))]
    counted = sum(counts_to_30(text) for text in together)
    print(f"{at_once} at once: {counted} of {at_once} counted to 30")
    if counted != at_once:
        problems.append(f"{at_once - counted} of {at_once} requests at once did not count to 30")
for problem in problems:
    print(f"::error::{problem}")
sys.exit(1 if problems else 0)
EOF
