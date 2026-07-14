#!/usr/bin/env bash
# Phase 1 RED test: Gemma4 MTP assistant converter
# Should FAIL until convert_dflash_to_gguf.py learns --mtp-assistant mode.
# When this passes, our converter produces a GGUF byte-for-byte equivalent in
# tensor list and metadata to AtomicChat's pre-converted reference.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAFETENSORS_DIR="${MTP_SAFETENSORS_DIR:-$ROOT/server/models/gemma-4-31B-it-assistant}"
GOLD_GGUF="${MTP_GOLD_GGUF:-$ROOT/server/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf}"
OUT_GGUF="${MTP_OUT_GGUF:-/tmp/mtp_phase1_ours.gguf}"
CONVERT="$ROOT/server/scripts/convert_dflash_to_gguf.py"
GGUF_DUMP=(python3 "$ROOT/server/scripts/gguf_dump.py")

# Sanity: gold reference must exist (downloaded in Phase 0)
if [ ! -f "$GOLD_GGUF" ]; then
    echo "SKIP: gold reference missing at $GOLD_GGUF"
    echo "       (Phase 0 download not yet complete — re-run after spike lands)"
    exit 77   # autotools "skip" exit code
fi

# Sanity: source safetensors must exist
if [ ! -d "$SAFETENSORS_DIR" ]; then
    echo "SKIP: source safetensors dir missing at $SAFETENSORS_DIR"
    echo "       (run: hf download google/gemma-4-31B-it-assistant --local-dir $SAFETENSORS_DIR)"
    exit 77
fi

echo "[red] running our converter — expected to FAIL until --mtp-assistant lands"
python3 "$CONVERT" --mtp-assistant "$SAFETENSORS_DIR" "$OUT_GGUF"

# --- Assertion 1: tensor list parity ---
echo "[assert 1] tensor list matches gold"
"${GGUF_DUMP[@]}" \
    --no-tensors "$GOLD_GGUF" 2>&1 | grep -E '^\s+[0-9]+:\s+' | awk '{print $NF}' | sort > /tmp/gold_tensors.txt
"${GGUF_DUMP[@]}" \
    --no-tensors "$OUT_GGUF" 2>&1 | grep -E '^\s+[0-9]+:\s+' | awk '{print $NF}' | sort > /tmp/ours_tensors.txt
diff /tmp/gold_tensors.txt /tmp/ours_tensors.txt
echo "[assert 1] PASS"

# --- Assertion 2: required tensors present (the v1 set) ---
echo "[assert 2] required MTP tensors present"
for t in \
    mtp.pre_projection.weight \
    mtp.post_projection.weight \
    output_norm.weight \
    mtp.blk.0.attn_norm.weight \
    mtp.blk.0.wq.weight \
    mtp.blk.0.wo.weight \
    mtp.blk.0.ffn_up.weight \
    mtp.blk.0.ffn_gate.weight \
    mtp.blk.0.ffn_down.weight \
    mtp.blk.3.attn_norm.weight \
    mtp.blk.3.wq.weight ; do
    grep -q "$t" /tmp/ours_tensors.txt || { echo "MISSING: $t"; exit 1; }
done
echo "[assert 2] PASS"

# --- Assertion 3: required metadata keys present (vLLM #41789 guard) ---
echo "[assert 3] required GGUF metadata keys"
GOLD_META=$("${GGUF_DUMP[@]}" "$GOLD_GGUF" 2>&1)
OURS_META=$("${GGUF_DUMP[@]}" "$OUT_GGUF" 2>&1)
for k in \
    gemma4_assistant.n_embd_backbone \
    gemma4_assistant.requires_target_arch \
    gemma4_assistant.attention_k_eq_v ; do
    echo "$OURS_META" | grep -q "$k" || { echo "MISSING META: $k"; exit 1; }
done
# n_embd_backbone must equal target n_embd (5376 for Dense 31B)
N_EMBD=$(echo "$OURS_META" | grep -oE 'gemma4_assistant.n_embd_backbone[^=]*= *[0-9]+' | grep -oE '[0-9]+$' || echo "0")
if [ "$N_EMBD" != "5376" ]; then
    echo "FAIL: n_embd_backbone=$N_EMBD, expected 5376 (Dense 31B target hidden)"
    exit 1
fi
echo "[assert 3] PASS"

# --- Assertion 4: requires_target_arch == "gemma4" (vLLM #41789 guard) ---
echo "[assert 4] requires_target_arch == gemma4"
arch_val=$(echo "$OURS_META" | grep "gemma4_assistant.requires_target_arch" | awk -F'"' '{print $2}')
[ "$arch_val" = "gemma4" ] \
    || { echo "FAIL: target arch mismatch (got '$arch_val', want 'gemma4')"; exit 1; }
echo "[assert 4] PASS"

echo "[red→green] all 4 assertions PASS — Phase 1 GREEN"
