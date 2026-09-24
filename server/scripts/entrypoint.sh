#!/usr/bin/env bash
# In-container ENTRYPOINT for lucebox-hub (CUDA and ROCm images).
#
#   docker run IMAGE [serve] [luce_server flags...]   start the server (default)
#   docker run IMAGE devices                          list GPUs and the auto choice
#   docker run IMAGE shell                            bash inside the container
#   docker run IMAGE <command> [args...]              run any other command
#
# `serve` finds the model files under models/, then execs luce_server with the
# flags below. Any luce_server flag can follow `serve`, or come first, and it
# replaces the value the entrypoint would pass, e.g.
#
#   docker run ... IMAGE --profile ds4-strix
#   docker run ... IMAGE serve --target-device hip:1 --max-ctx 65536
#
# Environment (all optional):
#   LUCE_TARGET               target GGUF (default: the one >5 GB file in models/)
#   LUCE_DRAFT                draft file or directory (default: models/draft;
#                             "none" runs without a draft)
#   LUCE_TARGET_DEVICE        backend:gpu or auto (default: auto, which picks
#                             a GPU the model fits on; see `devices`)
#   LUCE_PROFILE              luce_server --profile name
#   LUCE_ARGS                 extra luce_server flags, split on whitespace
#   LUCE_MAX_CTX              context length (default: from the GPU's memory)
#   LUCE_HOST, LUCE_PORT      listen address (default: 0.0.0.0:8080)
#   LUCE_BUDGET               DDTree budget (default: 22)
#   LUCE_LAZY                 1 = request-scoped draft (needs LUCE_PREFILL_DRAFTER)
#   LUCE_CACHE_TYPE_K/V       KV cache types
#   LUCE_PREFILL_MODE         off|auto|always PFlash compression (default: off)
#   LUCE_PREFILL_KEEP, LUCE_PREFILL_THRESHOLD, LUCE_PREFILL_DRAFTER
#   LUCE_PREFIX_CACHE_SLOTS, LUCE_PREFILL_CACHE_SLOTS
#   LUCE_DEFAULT_MAX_TOKENS, LUCE_MODEL_NAME, LUCE_THINK_MAX, LUCE_FA_WINDOW,
#   LUCE_MMPROJ
# Other LUCE_* variables reach luce_server unchanged.

set -euo pipefail

# Honor a pre-set LUCE_DIR (used by the entrypoint tests to drive a synthetic
# models/draft layout). In the shipped image this var is unset.
LUCE_DIR="${LUCE_DIR:-/opt/lucebox-hub/server}"
: "${LUCE_SERVER_BIN:=$LUCE_DIR/build/luce_server}"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*" >&2; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── target ─────────────────────────────────────────────────────────────────
# Targets are 10-100 GB; drafts and projectors are 1-4 GB and live under
# models/draft/. When several targets are present we refuse to guess: picking
# the wrong one silently has produced wrong benchmark numbers before.
target_candidates() {
    [ -d "$LUCE_DIR/models" ] || return 0
    find -L "$LUCE_DIR/models" -maxdepth 4 -type f -name '*.gguf' -size +5G \
        -not -path '*/draft/*' -not -iname '*mmproj*' -printf '%p\n' 2>/dev/null | sort
}
resolve_target() {
    : "${LUCE_TARGET:=}"
    if [ -z "$LUCE_TARGET" ]; then
        local candidates=()
        mapfile -t candidates < <(target_candidates)
        case "${#candidates[@]}" in
            0) ;;
            1) LUCE_TARGET="${candidates[0]}"
               info "Auto-detected target: $(basename "$LUCE_TARGET")" ;;
            *) warn "Multiple candidate target GGUFs in $LUCE_DIR/models:"
               local c
               for c in "${candidates[@]}"; do warn "    $c"; done
               die "Ambiguous target: set LUCE_TARGET=<path> to one of the candidates above." ;;
        esac
    fi
    if [ -z "$LUCE_TARGET" ] || [ ! -f "$LUCE_TARGET" ]; then
        die "No target GGUF found. Mount a model dir: -v /host/models:/opt/lucebox-hub/server/models, or set LUCE_TARGET=<path-inside-container>."
    fi
}

# ── device probe ───────────────────────────────────────────────────────────
# luce_server reports the GPUs of this image's runtime (CUDA or ROCm), the
# model architecture, and the device `--target-device auto` would pick. Its
# lines look like:
#   device hip:0 arch=gfx1201 type=discrete total_mib=32624 free_mib=... name=...
#   model arch=deepseek4 size_mib=89740 kv_mib=0 path=...
#   auto hip:1 fits=no total_mib=98304 reason=...
PROBE=""
probe_field() {  # probe_field <line-kind> <key>  (first matching line)
    awk -v kind="$1" -v key="$2" '
        $1 == kind { for (i = 2; i <= NF; i++) if (index($i, key "=") == 1) {
            print substr($i, length(key) + 2); exit } }' <<<"$PROBE"
}
device_total_mib() {  # device_total_mib <backend:gpu>
    awk -v dev="$1" '$1 == "device" && $2 == dev {
        for (i = 3; i <= NF; i++) if (index($i, "total_mib=") == 1) {
            print substr($i, 11); exit } }' <<<"$PROBE"
}

# Last value of a flag in an argv list, e.g. the --target-device a user passed.
arg_value() {  # arg_value <flag> <args...>
    local flag="$1" value="" prev="" a
    shift
    for a in "$@"; do
        [ "$prev" = "$flag" ] && value="$a"
        prev="$a"
    done
    printf '%s' "$value"
}
has_arg() {  # has_arg <flag> <args...>
    local flag="$1" a
    shift
    for a in "$@"; do [ "$a" = "$flag" ] && return 0; done
    return 1
}

# ── draft ──────────────────────────────────────────────────────────────────
# Drafts are architecture-specific (a Qwen3.6 DFlash draft crashes a Gemma
# target and vice versa), so a draft directory is searched with the target's
# family patterns first. DeepSeek V4 only takes its DSpark drafter.
resolve_draft() {  # sets DRAFT_ARG
    local arch="$1"
    DRAFT_ARG=""
    : "${LUCE_DRAFT:=$LUCE_DIR/models/draft}"
    case "$LUCE_DRAFT" in none|off|"") return ;; esac

    # Common host layouts link ~/models/qwen3.6-27b-dflash instead of draft/.
    if [ "$LUCE_DRAFT" = "$LUCE_DIR/models/draft" ] && [ ! -e "$LUCE_DRAFT" ]; then
        local cand
        for cand in "$LUCE_DIR/models/qwen3.6-27b-dflash" \
                    "$LUCE_DIR/models/Qwen3.6-27B-DFlash" \
                    "$LUCE_DIR/models/dflash"; do
            if [ -e "$cand" ]; then LUCE_DRAFT="$cand"; break; fi
        done
    fi
    if [ -f "$LUCE_DRAFT" ]; then
        DRAFT_ARG="$LUCE_DRAFT"
        return
    fi
    if [ ! -d "$LUCE_DRAFT" ]; then
        [ "$LUCE_DRAFT" = "$LUCE_DIR/models/draft" ] ||
            warn "Draft path $LUCE_DRAFT not found — running without draft"
        return
    fi

    local target_name family=() generic=()
    target_name="$(basename "$LUCE_TARGET" .gguf | tr 'A-Z' 'a-z')"
    if [ "$arch" = deepseek4 ]; then
        family=('*dspark*.gguf')
    else
        case "$target_name" in
            *gemma-4-26b*|*gemma4-26b*) family=('*gemma*4*26b*dflash*.gguf' '*dflash*gemma*4*26b*.gguf') ;;
            *gemma-4-31b*|*gemma4-31b*) family=('*gemma*4*31b*dflash*.gguf' '*dflash*gemma*4*31b*.gguf') ;;
            *gemma-4*|*gemma4*)         family=('*gemma*4*dflash*.gguf' '*dflash*gemma*4*.gguf') ;;
            *qwen3.6*|*qwen36*)         family=('dflash-draft-3.6-*.gguf' '*qwen*3.6*dflash*.gguf') ;;
        esac
        generic=('dflash-draft-*.gguf' '*dflash*.gguf' '*.gguf' 'model.safetensors' '*.safetensors')
    fi

    # Projectors are never drafts, and DSpark drafters only serve DeepSeek V4.
    local exclude=(-not -iname '*mmproj*')
    [ "$arch" = deepseek4 ] || exclude+=(-not -iname '*dspark*')
    local pattern file
    for pattern in "${family[@]}" "${generic[@]}"; do
        # Sorted so the pick does not depend on filesystem order.
        file="$(find -L "$LUCE_DRAFT" -maxdepth 4 -type f -iname "$pattern" \
                    "${exclude[@]}" -print 2>/dev/null | sort | head -n 1)"
        if [ -n "$file" ]; then
            DRAFT_ARG="$file"
            info "Resolved draft dir $LUCE_DRAFT → $DRAFT_ARG (pattern: $pattern)"
            return
        fi
    done
    warn "No draft for $(basename "$LUCE_TARGET") in $LUCE_DRAFT — running without draft"
}

# ── dispatch ───────────────────────────────────────────────────────────────
SUBCMD="${1:-serve}"
case "$SUBCMD" in
    serve) [ $# -eq 0 ] || shift ;;
    -*)    ;;  # bare luce_server flags: serve with them
    devices)
        shift
        target=("${LUCE_TARGET:-}")
        if [ -z "${target[0]}" ]; then
            mapfile -t target < <(target_candidates)
            if [ "${#target[@]}" -gt 1 ]; then
                warn "Several target GGUFs in $LUCE_DIR/models; set LUCE_TARGET to include one in the listing."
                target=()
            fi
        fi
        exec "$LUCE_SERVER_BIN" --list-devices "${target[@]}" "$@"
        ;;
    shell)
        shift
        exec /bin/bash "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
USER_ARGS=("$@")
EXTRA_ARGS=()
if [ -n "${LUCE_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS <<<"$LUCE_ARGS"
fi
ALL_ARGS=("${EXTRA_ARGS[@]}" "${USER_ARGS[@]}")

# These mapped to flags luce_server no longer accepts; forwarding them made
# the server exit with "unknown option".
for retired in LUCE_THINK_SOFT_CLOSE_MIN_RATIO LUCE_DEBUG_THINKING_LOGITS; do
    [ -n "${!retired:-}" ] && warn "$retired is no longer supported by luce_server and is ignored"
done

[ -x "$LUCE_SERVER_BIN" ] || die "luce_server binary missing at $LUCE_SERVER_BIN (image build failed?)"
resolve_target

# The server reads LUCE_TARGET_DEVICE whenever no device flag or profile names
# one, so the default never overrides an explicit placement.
export LUCE_TARGET_DEVICE="${LUCE_TARGET_DEVICE:-auto}"

PROBE="$("$LUCE_SERVER_BIN" --list-devices "$LUCE_TARGET" 2>/dev/null || true)"
MODEL_ARCH="$(probe_field model arch)"
GPU_COUNT="$(grep -c '^device ' <<<"$PROBE" || true)"
if [ -z "$PROBE" ] || [ "$GPU_COUNT" = 0 ]; then
    warn "No GPU visible to luce_server. CUDA: --gpus all. ROCm: --device /dev/kfd --device /dev/dri --group-add video --group-add render."
fi

# A --profile in the arguments replaces LUCE_PROFILE; luce_server takes one.
PROFILE_ARG="$(arg_value --profile "${ALL_ARGS[@]}")"
PROFILE="${PROFILE_ARG:-${LUCE_PROFILE:-}}"

# The device the model will run on, for sizing: an explicit single device, else
# the auto choice. A profile may name its own device, so with a profile only an
# explicit one is known here. Multi-device placements (--target-devices) keep
# the server's own defaults and are reported as given.
TARGET_DEVICES="$(arg_value --target-devices "${ALL_ARGS[@]}")"
USED_DEVICE=""
if [ -z "$TARGET_DEVICES" ]; then
    USED_DEVICE="$(arg_value --target-device "${ALL_ARGS[@]}")"
    if [ -z "$USED_DEVICE" ] && [ "$LUCE_TARGET_DEVICE" != auto ]; then
        USED_DEVICE="$LUCE_TARGET_DEVICE"
    fi
    if [ -z "$PROFILE" ] && { [ -z "$USED_DEVICE" ] || [ "$USED_DEVICE" = auto ]; }; then
        USED_DEVICE="$(awk '$1 == "auto" { print $2; exit }' <<<"$PROBE")"
    fi
    [ "$USED_DEVICE" = auto ] && USED_DEVICE=""
fi
TARGET_DESC="${TARGET_DEVICES:-${USED_DEVICE:-chosen by luce_server}}"
GPU_MIB=0
if [ -n "$USED_DEVICE" ]; then
    GPU_MIB="$(device_total_mib "$USED_DEVICE")"
    GPU_MIB="${GPU_MIB:-0}"
fi
[ "$GPU_COUNT" -gt 1 ] 2>/dev/null &&
    info "$GPU_COUNT GPUs visible; target $TARGET_DESC (LUCE_TARGET_DEVICE=$LUCE_TARGET_DEVICE; list with: docker run ... devices)"

# DeepSeek V4 serves concurrent requests (paged attention) autoregressively,
# so it takes no drafter there.
PAGED=0
CONCURRENCY="$(arg_value --max-concurrency "${ALL_ARGS[@]}")"
if has_arg --paged-attention "${ALL_ARGS[@]}" || [ "${CONCURRENCY:-1}" -gt 1 ] 2>/dev/null; then
    PAGED=1
fi

# ── context size from GPU memory ───────────────────────────────────────────
# Only when the operator set neither LUCE_MAX_CTX nor a profile: a profile's
# --max-ctx is part of its qualified configuration. An explicit --max-ctx flag
# still gets the memory-based draft settings, but its context is the one used.
MAX_CTX_ARG="$(arg_value --max-ctx "${ALL_ARGS[@]}")"
LAZY_EXPLICIT="${LUCE_LAZY:-}"
GPU_GB=$((GPU_MIB / 1024))
if [ -z "${LUCE_MAX_CTX:-}" ] && [ -z "$PROFILE" ] && [ "$GPU_GB" -gt 0 ]; then
    IS_WSL=0
    if grep -qi microsoft /proc/version 2>/dev/null || [ -e /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        IS_WSL=1
    fi
    if [ "$GPU_GB" -lt 12 ]; then
        LUCE_MAX_CTX=4096; : "${LUCE_LAZY:=1}"
        warn "GPU memory ${GPU_GB} GB < 12 GB — a 27B target is unlikely to fit"
    elif [ "$GPU_GB" -lt 22 ]; then
        LUCE_MAX_CTX=32768; : "${LUCE_LAZY:=1}"
    elif [ "$GPU_GB" -lt 32 ]; then
        : "${LUCE_LAZY:=1}"
        if [ "$IS_WSL" = 1 ]; then
            LUCE_MAX_CTX=65536; : "${LUCE_BUDGET:=16}"
        else
            LUCE_MAX_CTX=98304
        fi
    else
        LUCE_MAX_CTX=131072
    fi
fi
[ -n "$PROFILE" ] || : "${LUCE_MAX_CTX:=16384}"

# Qwen3.6 DFlash drafters use sliding-window attention; older GGUFs lack the
# metadata, so keep the documented default for them.
case "$(basename "$LUCE_TARGET")" in
    *Qwen3.6*|*qwen3.6*)
        if [ -z "${LUCE_DRAFT_SWA:-}" ]; then
            export LUCE_DRAFT_SWA=2048
            info "LUCE_DRAFT_SWA=2048 (Qwen3.6 draft SWA)"
        fi ;;
esac

if [ "$MODEL_ARCH" = deepseek4 ] && [ "$PAGED" = 1 ]; then
    DRAFT_ARG=""
    info "DeepSeek V4 concurrent serving decodes autoregressively; not loading a DSpark drafter"
else
    resolve_draft "$MODEL_ARCH"
fi

if [ "$MODEL_ARCH" = deepseek4 ] && [ -z "$PROFILE" ] && [ "$PAGED" = 0 ]; then
    case "$(awk -v dev="$USED_DEVICE" '$1 == "device" && $2 == dev { print $3 }' <<<"$PROBE")" in
        arch=gfx1151) info "DeepSeek V4 on Strix Halo: add --profile ds4-strix for the qualified serving profile" ;;
    esac
fi

# ── build + exec ───────────────────────────────────────────────────────────
: "${LUCE_HOST:=0.0.0.0}"
: "${LUCE_PORT:=8080}"
: "${LUCE_BUDGET:=22}"
: "${LUCE_THINK_MAX:=15488}"   # ds4_eval.c: max_tokens(16000) - reply budget(512)
: "${LUCE_PREFILL_MODE:=off}"

CMD=("$LUCE_SERVER_BIN" "$LUCE_TARGET"
     --host "$LUCE_HOST"
     --port "$LUCE_PORT"
     --think-max-tokens "$LUCE_THINK_MAX")
[ -n "$PROFILE" ] && [ -z "$PROFILE_ARG" ] && CMD+=(--profile "$PROFILE")
[ -n "${LUCE_MAX_CTX:-}" ] && [ -z "$MAX_CTX_ARG" ] && CMD+=(--max-ctx "$LUCE_MAX_CTX")

# Cache defaults belong to luce_server: omitting the variable keeps the native
# default, and explicit values (including 0) are forwarded.
[ -n "${LUCE_PREFIX_CACHE_SLOTS:-}" ] && CMD+=(--prefix-cache-slots "$LUCE_PREFIX_CACHE_SLOTS")
[ -n "${LUCE_PREFILL_CACHE_SLOTS:-}" ] && CMD+=(--prefill-cache-slots "$LUCE_PREFILL_CACHE_SLOTS")

if [ -n "$DRAFT_ARG" ]; then
    CMD+=(--draft "$DRAFT_ARG")
    # DeepSeek V4 verifies DSpark proposals itself; DDTree is a DFlash mode.
    [ "$MODEL_ARCH" = deepseek4 ] || CMD+=(--ddtree --ddtree-budget "$LUCE_BUDGET")
fi
[ -n "${LUCE_DEFAULT_MAX_TOKENS:-}" ] && CMD+=(--default-max-tokens "$LUCE_DEFAULT_MAX_TOKENS")
[ -n "${LUCE_MODEL_NAME:-}" ]         && CMD+=(--model-name "$LUCE_MODEL_NAME")
[ -n "${LUCE_MMPROJ:-}" ]             && CMD+=(--mmproj "$LUCE_MMPROJ")
[ -n "${LUCE_CACHE_TYPE_K:-}" ]       && CMD+=(--cache-type-k "$LUCE_CACHE_TYPE_K")
[ -n "${LUCE_CACHE_TYPE_V:-}" ]       && CMD+=(--cache-type-v "$LUCE_CACHE_TYPE_V")
[ "${LUCE_FA_WINDOW:-0}" -gt 0 ] 2>/dev/null && CMD+=(--fa-window "$LUCE_FA_WINDOW")

# --lazy-draft parks a decode draft only while PFlash compresses a prompt.
if [ "${LUCE_LAZY:-0}" = 1 ]; then
    if [ -n "$DRAFT_ARG" ] && [ -n "${LUCE_PREFILL_DRAFTER:-}" ]; then
        CMD+=(--lazy-draft)
    elif [ "$LAZY_EXPLICIT" = 1 ]; then
        warn "LUCE_LAZY=1 ignored: requires a draft and LUCE_PREFILL_DRAFTER"
    fi
fi

if [ "$LUCE_PREFILL_MODE" != off ]; then
    [ -n "${LUCE_PREFILL_DRAFTER:-}" ] || die "LUCE_PREFILL_MODE=$LUCE_PREFILL_MODE requires LUCE_PREFILL_DRAFTER"
    [ -f "$LUCE_PREFILL_DRAFTER" ] || die "Prefill drafter not found at $LUCE_PREFILL_DRAFTER"
    CMD+=(--prefill-compression "$LUCE_PREFILL_MODE"
          --prefill-keep-ratio "${LUCE_PREFILL_KEEP:-0.05}"
          --prefill-threshold "${LUCE_PREFILL_THRESHOLD:-32000}"
          --prefill-drafter "$LUCE_PREFILL_DRAFTER")
fi

# Operator flags go last: luce_server keeps the last value of a repeated flag.
CMD+=("${ALL_ARGS[@]}")

info "lucebox-hub starting: target=$(basename "$LUCE_TARGET") arch=${MODEL_ARCH:-unknown} device=$TARGET_DESC max_ctx=${MAX_CTX_ARG:-${LUCE_MAX_CTX:-profile}}${PROFILE:+ profile=$PROFILE}"

cd "$LUCE_DIR"
exec "${CMD[@]}"
