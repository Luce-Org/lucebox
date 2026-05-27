#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${RUN_DIR:-/workspace/lucebox-client-harness-runs/claude-llamacpp-matrix}}"
TARGET="${TARGET:-$REPO_DIR/server/models/Qwen3.6-27B-Q4_K_M.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/workspace/llama-cpp-server-build/bin/llama-server}"
PROMPT="${PROMPT:-Write exactly 120 words about why a repo-level agent benchmark should run the client harness instead of a handcrafted HTTP request. End with OK_DONE.}"
MARKER="${MARKER:-OK_DONE}"

mkdir -p "$RUN_ROOT"

run_one() {
  local name="$1"
  local ctx="$2"
  local cache_k="$3"
  local cache_v="$4"
  local max_tokens="$5"
  local extra="${6:-}"

  echo "== $name =="
  REPO_DIR="$REPO_DIR" \
  RUN_DIR="$RUN_ROOT" \
  STAMP="$name" \
  TARGET="$TARGET" \
  MODEL_SERVER=llamacpp \
  LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" \
  LLAMA_COMPAT_PROXY=anthropic \
  LLAMA_COMPAT_MAX_TOKENS="$max_tokens" \
  LLAMA_CACHE_TYPE_K="$cache_k" \
  LLAMA_CACHE_TYPE_V="$cache_v" \
  LLAMA_FLASH_ATTN=1 \
  LLAMA_N_GPU_LAYERS=999 \
  LLAMA_PARALLEL=1 \
  LLAMA_CACHE_RAM=0 \
  LLAMA_EXTRA_ARGS="--no-warmup $extra" \
  MAX_CTX="$ctx" \
  CLAUDE_TIMEOUT=900 \
  PROMPT="$PROMPT" \
  MARKER="$MARKER" \
  "$SCRIPT_DIR/run_claude_code.sh" > "$RUN_ROOT/$name.wrapper.log" 2>&1 || true
}

run_one tq3_32k_cap192 32768 tq3_0 tq3_0 192
run_one q8_32k_cap192 32768 q8_0 q8_0 192
run_one tq3_48k_cap192 49152 tq3_0 tq3_0 192
run_one q8_48k_cap192 49152 q8_0 q8_0 192
run_one f16_32k_cap192 32768 f16 f16 192

python3 - "$RUN_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
prompt_re = re.compile(r"prompt eval time\s*=.*?/\s*(\d+)\s+tokens.*?,\s*([0-9.]+)\s+tokens per second", re.I)
decode_re = re.compile(r"^\s*eval time\s*=.*?/\s*(\d+)\s+tokens.*?,\s*([0-9.]+)\s+tokens per second", re.I | re.M)
rows = []

for run in sorted(p for p in root.iterdir() if p.is_dir()):
    server = (run / "server.log").read_text(errors="replace") if (run / "server.log").exists() else ""
    out = (run / "claude-code.out").read_text(errors="replace") if (run / "claude-code.out").exists() else ""
    wrapper = (root / f"{run.name}.wrapper.log").read_text(errors="replace") if (root / f"{run.name}.wrapper.log").exists() else ""
    prompts = prompt_re.findall(server)
    decodes = decode_re.findall(server)
    usage = {}
    try:
        usage = json.loads(out.splitlines()[0]).get("usage", {})
    except Exception:
        pass
    rows.append({
        "name": run.name,
        "rc": (re.search(r"^rc=(\d+)$", wrapper, re.M) or ["", ""])[1],
        "marker": "OK_DONE" in out,
        "prompt_tokens": int(prompts[-1][0]) if prompts else usage.get("input_tokens"),
        "prompt_tok_s": float(prompts[-1][1]) if prompts else None,
        "output_tokens": int(decodes[-1][0]) if decodes else usage.get("output_tokens"),
        "decode_tok_s": float(decodes[-1][1]) if decodes else None,
        "preview": " ".join(out.split())[:180],
    })

print("| profile | rc | marker | prompt tok | prompt tok/s | output tok | decode tok/s | preview |")
print("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |")
for r in rows:
    preview = r["preview"].replace("|", "\\|")
    print(f"| {r['name']} | {r['rc']} | {'yes' if r['marker'] else 'no'} | {r['prompt_tokens'] or ''} | {r['prompt_tok_s'] or ''} | {r['output_tokens'] or ''} | {r['decode_tok_s'] or ''} | {preview} |")

(root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY
