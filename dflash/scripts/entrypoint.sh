#!/usr/bin/env bash
# In-container ENTRYPOINT for lucebox-hub.
#
# Normal path: the host-side `lucebox` CLI has already populated every
# DFLASH_* env var from its detection / benchmark, so this script just
# resolves paths and execs the native dflash_server binary.
#
# Fallback path: a user runs the image directly (`docker run --gpus all
# ghcr.io/luce-org/lucebox-hub:cuda12`) with no env-var prep. We then do a
# minimal VRAM-tiered autotune — same tiers as `lucebox configure`, kept in
# sync by hand. Anything more elaborate (driver-version probes, AMD paths,
# lspci fallbacks) belongs in the host CLI, not here.

set -euo pipefail

DFLASH_DIR="/opt/lucebox-hub/dflash"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── arg dispatch ───────────────────────────────────────────────────────────
# `serve` (default) — start the OpenAI-compatible server.
# `benchmark`        — run the in-container sweep (lucebox_bench.py).
# `shell`            — drop into bash inside the container (debug).
# `lucebox`          — dispatch to the Python CLI. Any subcommand
#                      `lucebox.sh` doesn't handle on the host arrives here
#                      (check, configure, pull, print-run, smoke, …).
# `python` or anything else
#                    — pass through to exec, so `docker run … python -m foo`
#                      still works for dev.
SUBCMD="${1:-serve}"
[ $# -gt 0 ] && shift || true

LUCEBOX_PKG="/opt/lucebox-hub"

case "$SUBCMD" in
    lucebox)
        exec uv run --directory "$LUCEBOX_PKG" python -m lucebox "$@"
        ;;
    benchmark)
        export DFLASH_DIR
        cd "$DFLASH_DIR"
        exec uv run --directory "$DFLASH_DIR" python scripts/lucebox_bench.py "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    serve|"")
        : # fall through to server startup below
        ;;
    *)
        exec "$SUBCMD" "$@"
        ;;
esac

# ── detect ─────────────────────────────────────────────────────────────────
# nvidia-smi is always present here (--gpus all wires the driver in).
GPU_VRAM_GB=0
if command -v nvidia-smi &>/dev/null; then
    if mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
                  | head -1) && [ -n "$mem_mib" ]; then
        GPU_VRAM_GB=$((mem_mib / 1024))
    fi
fi
GPU_COUNT=0
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | awk '/^GPU /{n++} END{print n+0}') || GPU_COUNT=0
fi

# ── fallback autotune (only fills unset env) ───────────────────────────────
# Keep these tiers in lockstep with lucebox::autotune_env on the host. The
# divergence we accept is the lower-VRAM error tier — the host CLI refuses
# to start there with a clear message; here we just warn and let the server
# decide whether it can load.

if [ "$GPU_VRAM_GB" -gt 0 ]; then
    IS_WSL=0
    if grep -qi microsoft /proc/version 2>/dev/null || [ -e /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        IS_WSL=1
    fi
    if [ "$GPU_VRAM_GB" -lt 12 ]; then
        : "${DFLASH_LAZY:=1}"
        : "${DFLASH_MAX_CTX:=4096}"
        warn "VRAM ${GPU_VRAM_GB} GB < 12 GB — 27B target unlikely to fit"
    elif [ "$GPU_VRAM_GB" -lt 22 ]; then
        : "${DFLASH_LAZY:=1}"
        : "${DFLASH_MAX_CTX:=32768}"
    elif [ "$GPU_VRAM_GB" -lt 32 ]; then
        : "${DFLASH_LAZY:=1}"
        if [ "$IS_WSL" = "1" ]; then
            : "${DFLASH_BUDGET:=16}"
            : "${DFLASH_MAX_CTX:=65536}"
        else
            : "${DFLASH_MAX_CTX:=98304}"
        fi
    elif [ "$GPU_VRAM_GB" -lt 48 ]; then
        : "${DFLASH_MAX_CTX:=131072}"
    else
        : "${DFLASH_PREFIX_CACHE_SLOTS:=0}"
        : "${DFLASH_MAX_CTX:=131072}"
    fi
fi

: "${DFLASH_BIN:=$DFLASH_DIR/build/test_dflash}"
: "${DFLASH_SERVER_BIN:=$DFLASH_DIR/build/dflash_server}"
: "${DFLASH_HOST:=0.0.0.0}"
: "${DFLASH_PORT:=8080}"
: "${DFLASH_BUDGET:=22}"
: "${DFLASH_MAX_CTX:=16384}"
: "${DFLASH_LAZY:=0}"
: "${DFLASH_PREFIX_CACHE_SLOTS:=0}"
: "${DFLASH_PREFILL_CACHE_SLOTS:=0}"
: "${DFLASH_CACHE_TYPE_K:=}"
: "${DFLASH_CACHE_TYPE_V:=}"
: "${DFLASH_VERBOSE:=0}"
: "${DFLASH_TARGET:=}"
: "${DFLASH_DRAFT:=$DFLASH_DIR/models/draft}"
: "${DFLASH_PREFILL_MODE:=off}"
: "${DFLASH_PREFILL_KEEP:=0.05}"
: "${DFLASH_PREFILL_THRESHOLD:=32000}"
: "${DFLASH_PREFILL_DRAFTER:=}"
# Phase-1 (thinking) cap when a request opts into thinking. Default mirrors
# antirez/ds4 ds4_eval.c: think_max_tokens = max_tokens(16000) - hard_limit
# reply budget(512) = 15488. The server's own hardcoded default is 10000;
# overriding here aligns ds4-eval and similar reasoning benches with upstream.
: "${DFLASH_THINK_MAX:=15488}"

# ── auto-detect target ─────────────────────────────────────────────────────
# Largest .gguf wins (target ~16 GB vs ~1 GB drafter). Follow symlinks so a
# symlinked 27B target is ranked by target size instead of being skipped.
if [ -z "$DFLASH_TARGET" ] && [ -d "$DFLASH_DIR/models" ]; then
    # Prefer the canonical Qwen3.6 Q4_K_M target when several same-sized
    # variants are present; otherwise fall back to largest GGUF.
    DFLASH_TARGET=$(find -L "$DFLASH_DIR/models" -maxdepth 4 -type f \
                      -name '*Qwen3.6-27B-Q4_K_M.gguf' \
                      -printf '%s %p\n' 2>/dev/null \
                      | sort -nr | head -1 | awk '{ $1=""; sub(/^ /,""); print }')
    if [ -z "$DFLASH_TARGET" ]; then
        DFLASH_TARGET=$(find -L "$DFLASH_DIR/models" -maxdepth 4 -type f \
                          -name '*Qwen3.6*Q4_K_M*.gguf' \
                          -printf '%s %p\n' 2>/dev/null \
                          | sort -nr | head -1 | awk '{ $1=""; sub(/^ /,""); print }')
    fi
    if [ -z "$DFLASH_TARGET" ]; then
        DFLASH_TARGET=$(find -L "$DFLASH_DIR/models" -maxdepth 4 -type f -name '*.gguf' \
                          -printf '%s %p\n' 2>/dev/null \
                          | sort -nr | head -1 | awk '{ $1=""; sub(/^ /,""); print }')
    fi
fi

if [ -z "$DFLASH_TARGET" ] || [ ! -f "$DFLASH_TARGET" ]; then
    die "No target GGUF found. Mount a model dir: -v /host/models:/opt/lucebox-hub/dflash/models"
fi
[ -x "$DFLASH_SERVER_BIN" ] || die "dflash_server binary missing at $DFLASH_SERVER_BIN (image build failed?)"

# Qwen3.6 DFlash drafters use sliding-window attention in the draft. Some GGUFs
# carry this metadata directly; keep the documented env override as the startup
# default so older drafts behave like the benchmark path.
case "$(basename "$DFLASH_TARGET")" in
    *Qwen3.6*|*qwen3.6*)
        if [ -z "${DFLASH27B_DRAFT_SWA:-}" ]; then
            export DFLASH27B_DRAFT_SWA=2048
            info "Autotune: DFLASH27B_DRAFT_SWA=2048 (Qwen3.6 draft SWA)"
        fi
        ;;
esac

# Common host layouts use ~/models/qwen3.6-27b-dflash as an absolute symlink
# rather than a literal models/draft directory. If the default is absent, find
# that draft before deciding to run without DFlash.
if [ "$DFLASH_DRAFT" = "$DFLASH_DIR/models/draft" ] && [ ! -e "$DFLASH_DRAFT" ]; then
    for cand in "$DFLASH_DIR/models/qwen3.6-27b-dflash" \
                "$DFLASH_DIR/models/Qwen3.6-27B-DFlash" \
                "$DFLASH_DIR/models/dflash"; do
        if [ -e "$cand" ]; then
            DFLASH_DRAFT="$cand"
            break
        fi
    done
fi

# Draft: directory holding GGUF/safetensors, or a direct draft file.
# The native dflash_server expects --draft to be a FILE path (not a dir).
# If DFLASH_DRAFT points at a directory, resolve it to a draft GGUF inside.
#
# Draft files are arch-specific: a draft trained for qwen3.6 has a fc
# weight shape that only divides evenly into the qwen3.6 target's hidden
# size, and crashes hard at spec-decode time when fed gemma4 (or vice
# versa) — see Gemma4Backend draft-incompatibility check. So when the
# draft dir contains multiple drafts (e.g. a host with both qwen3.6 and
# gemma4 drafts pre-downloaded), pick the one whose filename matches the
# target's family. Falls back to the generic dflash-draft-*.gguf pattern
# (legacy qwen3.6-only behavior) when the target family is unknown.
DRAFT_ARG="$DFLASH_DRAFT"
if [ -d "$DFLASH_DRAFT" ]; then
    # Derive a target-family hint from the target filename. Matching the
    # GGUF arch metadata would be cleaner but requires parsing the header
    # in shell; the filename convention is enforced upstream by the
    # publish-side dflash quantize scripts.
    TARGET_BASENAME="$(basename "$DFLASH_TARGET" .gguf 2>/dev/null)"
    case "$(echo "$TARGET_BASENAME" | tr 'A-Z' 'a-z')" in
        *gemma-4-26b*|*gemma4-26b*)   DRAFT_FAMILY_GLOB='dflash-gemma-4-26b-*.gguf' ;;
        *gemma-4-31b*|*gemma4-31b*)   DRAFT_FAMILY_GLOB='dflash-gemma-4-31b-*.gguf' ;;
        *gemma-4*|*gemma4*)           DRAFT_FAMILY_GLOB='dflash-gemma-*.gguf'       ;;
        *qwen3.6*|*qwen36*)           DRAFT_FAMILY_GLOB='dflash-draft-3.6-*.gguf'   ;;
        *)                            DRAFT_FAMILY_GLOB=''                          ;;
    esac

    DRAFT_FILE=""
    # Priority-ordered patterns. The family-specific pattern (if any)
    # goes first; the generic dflash-draft-*.gguf legacy pattern comes
    # after so older single-draft setups still resolve. Generic *.gguf
    # / safetensors come last as a last-resort fallback.
    PATTERNS=""
    [ -n "$DRAFT_FAMILY_GLOB" ] && PATTERNS="$DRAFT_FAMILY_GLOB"
    PATTERNS="$PATTERNS dflash-draft-*.gguf *.gguf model.safetensors *.safetensors"
    for pattern in $PATTERNS; do
        DRAFT_FILE="$(find -L "$DFLASH_DRAFT" -maxdepth 4 -type f -name "$pattern" -print -quit 2>/dev/null)"
        [ -n "$DRAFT_FILE" ] && break
    done
    if [ -n "$DRAFT_FILE" ] && [ -f "$DRAFT_FILE" ]; then
        DRAFT_ARG="$DRAFT_FILE"
        if [ -n "$DRAFT_FAMILY_GLOB" ]; then
            info "Resolved draft dir $DFLASH_DRAFT → $DRAFT_ARG (target family: $DRAFT_FAMILY_GLOB)"
        else
            info "Resolved draft dir $DFLASH_DRAFT → $DRAFT_ARG"
        fi
    else
        warn "No DFlash draft GGUF/safetensors in draft dir $DFLASH_DRAFT — running without draft"
        DRAFT_ARG=""
    fi
elif [ -n "$DFLASH_DRAFT" ] && [ ! -f "$DFLASH_DRAFT" ]; then
    warn "Draft path $DFLASH_DRAFT not found — running without draft"
    DRAFT_ARG=""
fi

[ "$GPU_COUNT" -gt 1 ] && warn "${GPU_COUNT} GPUs detected — native server layer sharding is not auto-enabled"

# ── build + exec native server ────────────────────────────────────────────
CMD=("$DFLASH_SERVER_BIN" "$DFLASH_TARGET"
     --host "$DFLASH_HOST"
     --port "$DFLASH_PORT"
     --max-ctx "$DFLASH_MAX_CTX"
     --prefix-cache-slots "$DFLASH_PREFIX_CACHE_SLOTS"
     --think-max-tokens "$DFLASH_THINK_MAX")

[ -n "$DRAFT_ARG" ]                && CMD+=(--draft "$DRAFT_ARG")
[ -n "$DRAFT_ARG" ]                && CMD+=(--ddtree --ddtree-budget "$DFLASH_BUDGET")
[ "$DFLASH_LAZY" = "1" ]           && CMD+=(--lazy-draft)
[ -n "$DFLASH_CACHE_TYPE_K" ]      && CMD+=(--cache-type-k "$DFLASH_CACHE_TYPE_K")
[ -n "$DFLASH_CACHE_TYPE_V" ]      && CMD+=(--cache-type-v "$DFLASH_CACHE_TYPE_V")

if [ "$DFLASH_PREFILL_MODE" != "off" ]; then
    [ -n "$DFLASH_PREFILL_DRAFTER" ] || die "DFLASH_PREFILL_MODE=$DFLASH_PREFILL_MODE requires DFLASH_PREFILL_DRAFTER"
    [ -f "$DFLASH_PREFILL_DRAFTER" ] || die "Prefill drafter not found at $DFLASH_PREFILL_DRAFTER"
    CMD+=(--prefill-compression "$DFLASH_PREFILL_MODE"
          --prefill-keep-ratio "$DFLASH_PREFILL_KEEP"
          --prefill-threshold "$DFLASH_PREFILL_THRESHOLD"
          --prefill-drafter "$DFLASH_PREFILL_DRAFTER")
fi

info "lucebox-hub container starting (target=$(basename "$DFLASH_TARGET"), max_ctx=$DFLASH_MAX_CTX, budget=$DFLASH_BUDGET, lazy=$DFLASH_LAZY)"

cd "$DFLASH_DIR"
exec "${CMD[@]}"
