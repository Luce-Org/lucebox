#!/usr/bin/env bash
# DS4V-3 vision GGUF quant recipe for Strix Halo plus 7900 XT.
# Follows docs/ds4v-uncensored-vision-plan.md section DS4V-3.
# See docs/ds4v-quant.md for the rung table and the budget math.
#
# Usage:
#   scripts/ds4v-quant.sh plan               print the exact quantize commands for the chosen rung
#   scripts/ds4v-quant.sh run                execute the recipe (needs weights, imatrix and a pinned llama.cpp)
#   scripts/ds4v-quant.sh verify SRC MMPROJ  assert bias_vl on 43 layers and that the mmproj loads
#   scripts/ds4v-quant.sh help               this text, exit 0
#
# Env knobs.
#   MODEL_DIR     source model directory      (default models/DeepSeek-V4-Flash-Vision-Uncensored)
#   IMATRIX       importance matrix gguf      (default $OUT_DIR/imatrix.gguf)
#   RUNG          IQ1_M, IQ2_XXS, IQ2_S, IQ3_XXS  (default IQ2_XXS)
#   OUT_DIR       output directory            (default models/ds4v-vision-quant)
#   LLAMA_CPP_DIR llama.cpp checkout          (default llama.cpp)

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/DeepSeek-V4-Flash-Vision-Uncensored}"
OUT_DIR="${OUT_DIR:-models/ds4v-vision-quant}"
RUNG="${RUNG:-IQ2_XXS}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-llama.cpp}"
IMATRIX="${IMATRIX:-$OUT_DIR/imatrix.gguf}"

# llama.cpp master pin. The quant needs both vision PRs from 9400c894 or later.
# Release b10763 predates them and is refused.
PIN_COMMIT="9400c894"
REFUSED_BUILD="b10763"

runge_ftype() {
    case "$RUNG" in
        IQ1_M)   echo "iq1_m" ;;
        IQ2_XXS) echo "iq2_xxs" ;;
        IQ2_S)   echo "iq2_s" ;;
        IQ3_XXS) echo "iq3_xxs" ;;
        *) echo "unknown rung $RUNG. Use IQ1_M, IQ2_XXS, IQ2_S or IQ3_XXS" >&2; return 1 ;;
    esac
}

usage() {
    cat <<'EOF'
DS4V-3 vision GGUF quant recipe.

Usage:
  scripts/ds4v-quant.sh plan               print the exact quantize commands for the chosen rung
  scripts/ds4v-quant.sh run                execute the recipe (needs weights, imatrix and a pinned llama.cpp)
  scripts/ds4v-quant.sh verify SRC MMPROJ  assert bias_vl on 43 layers and that the mmproj loads
  scripts/ds4v-quant.sh help               this text, exit 0

Env knobs.
  MODEL_DIR     source model directory      (default models/DeepSeek-V4-Flash-Vision-Uncensored)
  IMATRIX       importance matrix gguf      (default $OUT_DIR/imatrix.gguf)
  RUNG          IQ1_M, IQ2_XXS, IQ2_S, IQ3_XXS  (default IQ2_XXS)
  OUT_DIR       output directory            (default models/ds4v-vision-quant)
  LLAMA_CPP_DIR llama.cpp checkout          (default llama.cpp)

The recipe pins llama.cpp master 9400c894 or later with both vision PRs.
Release b10763 is refused.
EOF
}

die() {
    echo "error. $1" >&2
    exit 1
}

quantize_bin() {
    for c in "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$LLAMA_CPP_DIR/llama-quantize"; do
        if [ -x "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    if command -v llama-quantize >/dev/null 2>&1; then
        command -v llama-quantize
        return 0
    fi
    die "llama-quantize not found under LLAMA_CPP_DIR=$LLAMA_CPP_DIR"
}

check_pin() {
    local bin ver hash
    bin="$(quantize_bin)"
    ver="$("$bin" --version 2>&1 | head -n 1)" || die "cannot read version from $bin"
    echo "$ver"
    if printf '%s' "$ver" | grep -q "$REFUSED_BUILD"; then
        die "refusing llama.cpp release $REFUSED_BUILD. It lacks the vision PRs. Use master at or after $PIN_COMMIT"
    fi
    if printf '%s' "$ver" | grep -q "$PIN_COMMIT"; then
        return 0
    fi
    hash="$(printf '%s' "$ver" | sed -n 's/.*(\([0-9a-f]\{7,\}\)).*/\1/p')"
    if [ -n "$hash" ] && [ -d "$LLAMA_CPP_DIR/.git" ]; then
        if git -C "$LLAMA_CPP_DIR" merge-base --is-ancestor "$PIN_COMMIT" HEAD 2>/dev/null; then
            return 0
        fi
    fi
    die "llama.cpp build $ver is not master at or after $PIN_COMMIT with the vision PRs"
}

# One llama-quantize pass. Routed experts take the rung type as the main type.
# Attention plus shared experts get Q6_K. token_embd plus output get Q8_0.
# Routers and bias_vl stay at BF16. The mmproj ships at F16 and is never requantized.
quantize_command() {
    local src="$1" out="$2" ftype
    ftype="$(runge_ftype)"
    echo "$LLAMA_CPP_DIR/build/bin/llama-quantize \\"
    echo "  --imatrix \"$IMATRIX\" \\"
    echo "  --token-embedding-type q8_0 \\"
    echo "  --output-tensor-type q8_0 \\"
    echo "  --tensor-type 'blk\\.[0-9]+\\.attn_.*=q6_k' \\"
    echo "  --tensor-type 'blk\\.[0-9]+\\.ffn_(gate|up|down)_shexp=q6_k' \\"
    echo "  --tensor-type 'blk\\.[0-9]+\\.ffn_gate_inp=bf16' \\"
    echo "  --tensor-type 'blk\\.[0-9]+\\.exp_probs_b=bf16' \\"
    echo "  --tensor-type 'blk\\.[0-9]+\\.bias_vl=bf16' \\"
    echo "  \"$src\" \"$out\" $ftype"
}

cmd_plan() {
    local ftype src out
    ftype="$(runge_ftype)"
    src="$MODEL_DIR/DeepSeek-V4-Flash-Vision-Uncensored-BF16.gguf"
    mkdir -p "$OUT_DIR"
    out="$OUT_DIR/DeepSeek-V4-Flash-Vision-Uncensored-$RUNG.gguf"
    echo "# llama.cpp pin $PIN_COMMIT or later with both vision PRs. $REFUSED_BUILD is refused."
    echo "# rung $RUNG, expert ftype $ftype"
    echo "# step 1, imatrix from the abliterated source"
    echo "$LLAMA_CPP_DIR/build/bin/llama-imatrix -m \"$src\" --mmproj \"$MODEL_DIR/mmproj-DeepSeek-V4-Flash-Vision-Uncensored-f16.gguf\" -f \"$MODEL_DIR/calibration.txt\" -o \"$IMATRIX\""
    echo "# step 2, quantize. bias_vl stays on all 43 layers. mmproj ships at F16."
    quantize_command "$src" "$out"
    echo "# source weights are never deleted. FP8 shards and BF16 gguf stay in $MODEL_DIR"
}

cmd_run() {
    local bin src out mmproj
    mkdir -p "$OUT_DIR"
    bin="$(quantize_bin)"
    [ -d "$MODEL_DIR" ] || die "MODEL_DIR=$MODEL_DIR does not exist. Run scripts/ds4v-fetch-source.sh first"
    src="$MODEL_DIR/DeepSeek-V4-Flash-Vision-Uncensored-BF16.gguf"
    [ -f "$src" ] || die "missing $src. Dequantize the FP8 shards to BF16 and convert with convert_hf_to_gguf.py first. See docs/ds4v-quant.md"
    [ -f "$IMATRIX" ] || die "missing imatrix $IMATRIX. Run the llama-imatrix step printed by plan first"
    check_pin
    out="$OUT_DIR/DeepSeek-V4-Flash-Vision-Uncensored-$RUNG.gguf"
    echo "run. quantize $src to rung $RUNG into $out"
    bash -c "$(quantize_command "$src" "$out")"
    mmproj="$MODEL_DIR/mmproj-DeepSeek-V4-Flash-Vision-Uncensored-f16.gguf"
    [ -f "$mmproj" ] || echo "warning. mmproj $mmproj not found. Vision needs it at F16"
    echo "done. source weights in $MODEL_DIR were not touched"
    echo "verify with. scripts/ds4v-quant.sh verify \"$out\" \"$mmproj\""
}

gguf_dump_cmd() {
    local f="$1"
    if [ -n "${LLAMA_CPP_DIR:-}" ] && [ -f "$LLAMA_CPP_DIR/gguf-py/gguf/scripts/gguf_dump.py" ]; then
        echo "python3 $LLAMA_CPP_DIR/gguf-py/gguf/scripts/gguf_dump.py"
        return 0
    fi
    echo "python3 -m gguf.scripts.gguf_dump"
}

cmd_verify() {
    local src="$1" mmproj="$2" dump n
    [ -f "$src" ] || die "missing gguf $src"
    dump="$(gguf_dump_cmd "$src")"
    n="$($dump "$src" 2>/dev/null | grep -o 'bias_vl' | wc -l | tr -d ' ')"
    if [ "$n" -eq 43 ]; then
        echo "pass. bias_vl present on 43 layers"
    else
        die "bias_vl count is $n, expected 43. The vision PRs or the quant type map dropped it"
    fi
    [ -f "$mmproj" ] || die "missing mmproj $mmproj"
    echo "mmproj load check. one token through the mtmd path with the mmproj attached"
    "$LLAMA_CPP_DIR/build/bin/llama-mtmd-cli" \
        --model "$src" \
        --mmproj "$mmproj" \
        -p "load check" \
        -n 1 \
        --temp 0
    echo "pass. mmproj loaded and generated one token"
}

case "${1:-help}" in
    plan)
        cmd_plan
        ;;
    run)
        cmd_run
        ;;
    verify)
        if [ "${2:-}" = "--help" ] || [ "${2:-}" = "-h" ] || [ $# -lt 3 ]; then
            usage
            echo ""
            echo "verify needs the quantized gguf and the mmproj path. Both checks need the weights. Run it on the operator machine."
            if [ "${2:-}" = "--help" ] || [ "${2:-}" = "-h" ]; then
                exit 0
            fi
            exit 1
        fi
        cmd_verify "$2" "$3"
        ;;
    help|-h|--help)
        usage
        exit 0
        ;;
    *)
        usage
        echo ""
        echo "unknown subcommand $1" >&2
        exit 1
        ;;
esac
