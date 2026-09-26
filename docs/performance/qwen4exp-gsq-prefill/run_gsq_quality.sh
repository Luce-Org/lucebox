#!/usr/bin/env bash
set -euo pipefail

repo=/home/duster/lucebox-qwen4exp
model=/home/duster/models/qwen4exp/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
port=18750
out=/tmp/q4exp-gsq-prefill
mkdir -p "$out"

env HIP_VISIBLE_DEVICES=1 DFLASH_HIP_NO_AUTO_UMA=1 GGML_CUDA_MMB=1 \
    QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5 DFLASH_MMB_SHADOW=1 \
    LLAMA_MMB_HC16=2 QWEN4EXP_LAST_TOKEN_FFN=1 QWEN4EXP_DENSE_TABLE=1 \
    QWEN4EXP_HC_TILE16=1 \
    "$repo/server/build-hip/dflash_server" --model "$model" \
    --host 127.0.0.1 --port "$port" --target-device hip:0 \
    --max-ctx 40000 --chunk 16384 >"$out/t13-gsq-quality.server.log" 2>&1 &
pid=$!
cleanup() {
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 240); do
    if curl -sf --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        tail -80 "$out/t13-gsq-quality.server.log"
        exit 1
    fi
    sleep 2
done
test "$ready" = 1

python3 - "$pid" "$out/t13-gsq-quality.environ.json" <<'PY'
import json, pathlib, sys
pid, output = sys.argv[1:]
keys = [
    "HIP_VISIBLE_DEVICES", "DFLASH_HIP_NO_AUTO_UMA", "GGML_CUDA_MMB",
    "QWEN4EXP_QSA", "QWEN4EXP_MMB_CUBLAS", "DFLASH_MMB_SHADOW",
    "LLAMA_MMB_HC16", "QWEN4EXP_LAST_TOKEN_FFN", "QWEN4EXP_DENSE_TABLE",
    "QWEN4EXP_HC_TILE16",
]
env = dict(x.decode().split("=", 1) for x in pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in x)
got = {key: env.get(key) for key in keys}
pathlib.Path(output).write_text(json.dumps(got, indent=2) + "\n")
expected = {key: "1" for key in keys}
expected["QWEN4EXP_MMB_CUBLAS"] = "5"
expected["LLAMA_MMB_HC16"] = "2"
assert got == expected, got
PY

cd "$repo"
python3 /tmp/gsq_sanity.py "$port" "$out/t13-gsq-sanity.json" \
    2>&1 | tee "$out/t13-gsq-sanity.log"
python3 harness/client_test_runner.py bench \
    --url "http://127.0.0.1:$port" --suite he,gsm,math,recall --model dflash \
    --json-out "$out/t13-gsq-quality.json" \
    2>&1 | tee "$out/t13-gsq-quality.log"
