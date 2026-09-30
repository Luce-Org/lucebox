#!/usr/bin/env bash
# Download the routing prototype GGUFs into $MODELS_DIR (default ~/models/routing).
# Uses `uvx --from huggingface_hub hf` (no global pip installs). Idempotent:
# hf download skips files already present. Downloads run in parallel (the
# link here is per-connection limited).
#
# Env: MODELS_DIR, SKIP_EMBED=1 to skip the (optional) embedding model.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models/routing}"
mkdir -p "$MODELS_DIR"
HF=(uvx --from huggingface_hub hf)

# repo  file  [fallback-file]
get() {
  local repo="$1" file="$2" fallback="${3:-}"
  if "${HF[@]}" download "$repo" "$file" --local-dir "$MODELS_DIR" >/dev/null 2>&1; then
    echo "ok   $MODELS_DIR/$file"
  elif [[ -n "$fallback" ]] &&
       "${HF[@]}" download "$repo" "$fallback" --local-dir "$MODELS_DIR" >/dev/null 2>&1; then
    echo "ok   $MODELS_DIR/$fallback (fallback for $file)"
  else
    echo "FAIL $repo/$file" >&2
    return 1
  fi
}

# All qwen35-arch (Qwen3.5 hybrid DeltaNet), standard llama.cpp GGUF layout,
# tied embeddings (no output.weight). Q8_0 preferred; Q4_K_M fallback.
pids=()
get unsloth/Qwen3.5-0.8B-GGUF               Qwen3.5-0.8B-Q8_0.gguf  Qwen3.5-0.8B-Q4_K_M.gguf & pids+=($!)
get unsloth/Qwen3.5-2B-GGUF                 Qwen3.5-2B-Q8_0.gguf    Qwen3.5-2B-Q4_K_M.gguf   & pids+=($!)
get regolo/brick-complexity-2-max-Q8_0-GGUF brick-complexity-2-max-Q8_0.gguf & pids+=($!)
get regolo/brick-complexity-2-eco-Q8_0-GGUF brick-complexity-2-eco-Q8_0.gguf & pids+=($!)
if [[ "${SKIP_EMBED:-0}" != 1 ]]; then
  # qwen3 arch (not qwen35); optional, for a kNN router. Not servable by
  # luce_server today (no embedding endpoint).
  get Qwen/Qwen3-Embedding-0.6B-GGUF        Qwen3-Embedding-0.6B-Q8_0.gguf & pids+=($!)
fi
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

ls -la "$MODELS_DIR"/*.gguf
exit $rc
