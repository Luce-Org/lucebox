#!/usr/bin/env bash
set -euo pipefail

# ── lucebox-hub server startup ──────────────────────────────────────────────
# Detects local hardware and starts the dflash OpenAI-compatible HTTP server
# with sensible defaults for this machine.
#
# Override any path or flag via environment variables:
#   DFLASH_TARGET    path to target GGUF (default: dflash/models/*.gguf)
#   DFLASH_DRAFT     path to draft dir, GGUF, or safetensors (default: dflash/models/draft/)
#   DFLASH_BIN       path to test_dflash binary (default: dflash/build/test_dflash)
#   DFLASH_PORT      server port (default: 8080)
#   DFLASH_HOST      server host (default: 0.0.0.0)
#   DFLASH_MAX_CTX   max context length (autotuned by VRAM; static default 16384)
#   DFLASH_BUDGET    DDTree budget (default: 22)
#   DFLASH_LAZY      set to 1 to park draft when idle (default: 0)
#   DFLASH_PREFIX_CACHE_SLOTS   (default: 1)
#   DFLASH_PREFILL_CACHE_SLOTS  (default: 0)
#   DFLASH_VERBOSE   set to 1 for verbose daemon logs (default: 0)
#
# pFlash speculative prefill (requires additional drafter GGUF):
#   DFLASH_PREFILL_MODE   off|auto|always (default: off)
#   DFLASH_PREFILL_KEEP   keep ratio (default: 0.05)
#   DFLASH_PREFILL_THRESHOLD  token threshold for auto (default: 32000)
#   DFLASH_PREFILL_DRAFTER   path to Qwen3-0.6B BF16 GGUF
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DFLASH_DIR="$PROJECT_ROOT/dflash"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── hardware detection ───────────────────────────────────────────────────────
# Populates: NPROC, TOTAL_RAM_GB, GPU_VENDOR (nvidia|amd|none),
# GPU_NAME, GPU_COUNT, GPU_MEM_GB (0 = unknown), GPU_ARCH (AMD gfx* if known).
# Falls back from nvidia-smi → nvidia-smi -L → lspci, so a broken NVML driver
# (mismatch after a kernel update) still yields a usable name and vendor.

NPROC=$(nproc 2>/dev/null || echo 4)
TOTAL_RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 16)
GPU_VENDOR="none"
GPU_NAME=""
GPU_COUNT=0
GPU_MEM_GB=0       # 0 means "unknown" — autotune treats this as low-confidence
GPU_ARCH=""        # AMD only

# NVIDIA path. nvidia-smi writes NVML errors to stdout (not stderr) but signals
# them via non-zero exit — we use the exit code, not output emptiness, to decide.
NVSMI_BROKEN=0
if command -v nvidia-smi &>/dev/null; then
    if NVSMI_OUT=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null) \
       && [ -n "$NVSMI_OUT" ]; then
        GPU_VENDOR="nvidia"
        GPU_NAME=$(printf '%s\n' "$NVSMI_OUT" | head -1 | awk -F', ' '{print $1}')
        GPU_MEM_MIB=$(printf '%s\n' "$NVSMI_OUT" | head -1 | awk -F', ' '{print $2}')
        GPU_MEM_GB=$((GPU_MEM_MIB / 1024))
        GPU_COUNT=$(printf '%s\n' "$NVSMI_OUT" | wc -l)
    elif NVSMI_L=$(nvidia-smi -L 2>/dev/null) && [ -n "$NVSMI_L" ]; then
        # NVML query failed but -L worked (rare). Use it for name only.
        GPU_VENDOR="nvidia"
        GPU_NAME=$(printf '%s\n' "$NVSMI_L" | head -1 | sed -E 's/^GPU [0-9]+: //; s/ \(UUID:.*$//')
        GPU_COUNT=$(printf '%s\n' "$NVSMI_L" | wc -l)
        NVSMI_BROKEN=1
        warn "nvidia-smi NVML query failed; using nvidia-smi -L for name only (VRAM unknown — autotune disabled for VRAM-dependent flags)"
    else
        # Both NVML calls failed — typical after a driver upgrade pending reboot.
        # Don't set GPU_VENDOR yet; let the lspci fallback identify the card.
        NVSMI_BROKEN=1
        warn "nvidia-smi present but NVML calls fail (likely driver/library version mismatch — reboot may be required). Falling back to lspci for card identification; VRAM-dependent autotune disabled."
    fi
fi

# AMD path (only if NVIDIA wasn't found)
if [ "$GPU_VENDOR" = "none" ] && command -v rocm-smi &>/dev/null; then
    if ROCM_NAME=$(rocm-smi --showproductname 2>/dev/null | awk -F': ' '/Card series|GPU\[0\].*Card model/{print $2; exit}') \
       && [ -n "$ROCM_NAME" ]; then
        GPU_VENDOR="amd"
        GPU_NAME="$ROCM_NAME"
        GPU_COUNT=$(rocm-smi --showproductname 2>/dev/null | grep -c '^GPU\[' || echo 1)
        # VRAM in bytes
        if ROCM_MEM=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Total Memory/{print $NF; exit}') && [ -n "$ROCM_MEM" ]; then
            GPU_MEM_GB=$((ROCM_MEM / 1024 / 1024 / 1024))
        fi
        if command -v rocminfo &>/dev/null; then
            GPU_ARCH=$(rocminfo 2>/dev/null | awk '/Name:.*gfx/{print $2; exit}')
        fi
    fi
fi

# lspci fallback — last resort for name + vendor when smi tools are missing/broken.
# Vendor matching uses lspci's canonical proper-case strings (NVIDIA / AMD /
# Advanced Micro Devices / ATI Technologies) — case-insensitive "ati" would
# substring-match "VGA compATIble controller" and pick up the integrated Intel GPU.
# Single awk avoids the pipefail+SIGPIPE trap of `grep | grep | head`.
if [ "$GPU_VENDOR" = "none" ] || [ -z "$GPU_NAME" ]; then
    if command -v lspci &>/dev/null; then
        LSPCI_LINE=$(lspci 2>/dev/null | awk '
            tolower($0) ~ /vga|3d|display/ &&
            (/NVIDIA/ || /AMD/ || /Advanced Micro/ || /ATI Technologies/) { print; exit }
        ') || LSPCI_LINE=""
        if [ -n "$LSPCI_LINE" ]; then
            case "$LSPCI_LINE" in
                *NVIDIA*) [ "$GPU_VENDOR" = "none" ] && GPU_VENDOR="nvidia" ;;
                *AMD*|*"ATI Technologies"*|*"Advanced Micro"*)
                    [ "$GPU_VENDOR" = "none" ] && GPU_VENDOR="amd" ;;
            esac
            if [ -z "$GPU_NAME" ]; then
                # Strip "BB:DD.F class: vendor " prefix and trailing "(rev xx)".
                GPU_NAME=$(printf '%s' "$LSPCI_LINE" | sed -E 's/^[0-9a-f:.]+ [^:]+: //; s/ \(rev [0-9a-f]+\)$//')
            fi
            [ "$GPU_COUNT" -eq 0 ] && GPU_COUNT=1
        fi
    fi
fi

GPU_DISPLAY=""
if [ "$GPU_VENDOR" != "none" ]; then
    GPU_DISPLAY="${GPU_NAME:-unknown}"
    if [ "$GPU_MEM_GB" -gt 0 ]; then
        GPU_DISPLAY="${GPU_DISPLAY}, ${GPU_MEM_GB} GB"
    else
        GPU_DISPLAY="${GPU_DISPLAY}, VRAM unknown"
    fi
    [ "$GPU_COUNT" -gt 1 ] && GPU_DISPLAY="${GPU_COUNT}× ${GPU_DISPLAY}"
    [ -n "$GPU_ARCH" ] && GPU_DISPLAY="${GPU_DISPLAY} (${GPU_ARCH})"
fi

info "Hardware: ${NPROC} CPUs · ${TOTAL_RAM_GB} GB RAM${GPU_DISPLAY:+ · GPU: ${GPU_DISPLAY}}"

# ── venv ─────────────────────────────────────────────────────────────────────

VENV_DIR="$PROJECT_ROOT/.venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    ok "venv activated ($VENV_DIR)"
else
    warn "no .venv found at $VENV_DIR — attempting system python"
fi

# ── hardware-driven autotune (conservative) ──────────────────────────────────
# Only set defaults that are well-documented or high-confidence given detection.
# Anything still unset after this block falls through to the static defaults
# below. User env vars always win because every assignment here uses `:=`.

AUTOTUNED=()

# AMD: gfx1100/1101/1102 want --ddtree-budget=8 (~+53% on HE per docs/HIP_PERF_PLAN.md).
# gfx1151 (Strix Halo) and gfx1201 keep budget=22.
case "$GPU_ARCH" in
    gfx1100|gfx1101|gfx1102)
        if [ -z "${DFLASH_BUDGET:-}" ]; then
            DFLASH_BUDGET=8
            AUTOTUNED+=("budget=8 (RDNA3 ${GPU_ARCH} prefers MMVQ path)")
        fi
        ;;
esac

# VRAM-tiered tuning — only when we actually know VRAM (GPU_MEM_GB > 0).
# Tiers target Qwen3.x-27B Q4_K_M (~16 GB) + DFlash draft (~3.3 GB) + KV cache.
# server.py auto-enables TQ3_0 KV (3.5 bpv) when max_ctx > 6144, which is what
# lets a 24 GB-class consumer card carry ~112K ctx. RTX 5090 Laptop reports
# 23 GiB via nvidia-smi (24 564 MiB integer-divided by 1024), so the 24 GB tier
# starts at 22 to cover it; desktop 3090/4090/5090 land cleanly in the same tier.
if [ "$GPU_MEM_GB" -gt 0 ]; then
    if [ "$GPU_MEM_GB" -lt 12 ]; then
        # Tiny VRAM (<12 GB) — 27B Q4_K_M won't fit; this is a safety floor.
        if [ -z "${DFLASH_LAZY:-}" ]; then
            DFLASH_LAZY=1
            AUTOTUNED+=("lazy-draft=1 (VRAM ${GPU_MEM_GB} GB < 12 GB)")
        fi
        if [ -z "${DFLASH_MAX_CTX:-}" ]; then
            DFLASH_MAX_CTX=4096
            AUTOTUNED+=("max_ctx=4096 (tiny VRAM)")
        fi
    elif [ "$GPU_MEM_GB" -lt 22 ]; then
        # 12-21 GB (RTX 4070/4080/3080) — model fits but headroom is tight.
        if [ -z "${DFLASH_LAZY:-}" ]; then
            DFLASH_LAZY=1
            AUTOTUNED+=("lazy-draft=1 (VRAM ${GPU_MEM_GB} GB < 22 GB)")
        fi
        if [ -z "${DFLASH_MAX_CTX:-}" ]; then
            DFLASH_MAX_CTX=32768
            AUTOTUNED+=("max_ctx=32768 (12-21 GB tier)")
        fi
    elif [ "$GPU_MEM_GB" -lt 32 ]; then
        # 22-31 GB — 24 GB-class consumer flagships (RTX 3090/4090/5090,
        # RTX 5090 Laptop). Q4_K_M target + draft + TQ3_0 KV fits ~112K ctx;
        # full 128K leaves no margin for verify/rollback buffers (#114-style OOM).
        if [ -z "${DFLASH_LAZY:-}" ]; then
            DFLASH_LAZY=1
            AUTOTUNED+=("lazy-draft=1 (24 GB-class consumer GPU)")
        fi
        if [ -z "${DFLASH_MAX_CTX:-}" ]; then
            DFLASH_MAX_CTX=114688
            AUTOTUNED+=("max_ctx=114688 (24 GB-class consumer GPU; TQ3_0 KV auto-enabled by server.py)")
        fi
    elif [ "$GPU_MEM_GB" -lt 48 ]; then
        # 32-47 GB (RTX 6000 Ada, A100 40 GB) — full 128K fits comfortably.
        if [ -z "${DFLASH_MAX_CTX:-}" ]; then
            DFLASH_MAX_CTX=131072
            AUTOTUNED+=("max_ctx=131072 (32-47 GB tier)")
        fi
    else
        # ≥48 GB (A100 80 GB, H100, RTX 6000 Pro). Plenty of headroom for
        # extra prefix-cache snapshots.
        if [ -z "${DFLASH_PREFIX_CACHE_SLOTS:-}" ]; then
            DFLASH_PREFIX_CACHE_SLOTS=4
            AUTOTUNED+=("prefix_cache_slots=4 (VRAM ${GPU_MEM_GB} GB ≥ 48 GB)")
        fi
        if [ -z "${DFLASH_MAX_CTX:-}" ]; then
            DFLASH_MAX_CTX=131072
            AUTOTUNED+=("max_ctx=131072 (ample VRAM)")
        fi
    fi
fi

# Suggestions (printed, not applied) — multi-GPU sharding is workload-dependent
# and currently has no env var hook in this script, so we surface it instead.
if [ "$GPU_COUNT" -gt 1 ]; then
    warn "Detected ${GPU_COUNT} GPUs — consider --target-gpus / --draft-gpu for sharding (not auto-enabled). See server.py --help."
fi

if [ ${#AUTOTUNED[@]} -gt 0 ]; then
    info "Autotune applied:"
    for item in "${AUTOTUNED[@]}"; do
        printf '       - %s\n' "$item"
    done
fi

# ── resolve paths ────────────────────────────────────────────────────────────

: "${DFLASH_BIN:="$DFLASH_DIR/build/test_dflash"}"
: "${DFLASH_TARGET:=""}"
: "${DFLASH_DRAFT:="$DFLASH_DIR/models/draft"}"
: "${DFLASH_PORT:=8080}"
: "${DFLASH_HOST:=0.0.0.0}"
: "${DFLASH_MAX_CTX:=16384}"
: "${DFLASH_BUDGET:=22}"
: "${DFLASH_LAZY:=0}"
: "${DFLASH_PREFIX_CACHE_SLOTS:=1}"
: "${DFLASH_PREFILL_CACHE_SLOTS:=0}"
: "${DFLASH_VERBOSE:=0}"

# pFlash defaults
: "${DFLASH_PREFILL_MODE:=off}"
: "${DFLASH_PREFILL_KEEP:=0.05}"
: "${DFLASH_PREFILL_THRESHOLD:=32000}"
: "${DFLASH_PREFILL_DRAFTER:=""}"

# Auto-detect target if not set.
# Search recursively (HF downloads sometimes land in subdirs of --local-dir).
# If multiple .gguf files are present (e.g. 27B target + 0.6B pFlash drafter),
# pick the largest — the target is ~16-19 GB vs ~1.2 GB for the drafter.
if [ -z "$DFLASH_TARGET" ]; then
    if [ -d "$DFLASH_DIR/models" ]; then
        # `find -printf '%s %p\n' | sort -nr` ranks by size, biggest first.
        DFLASH_TARGET=$(find "$DFLASH_DIR/models" -maxdepth 4 -type f -name "*.gguf" \
                        -printf '%s %p\n' 2>/dev/null | sort -nr | head -1 | awk '{ $1=""; sub(/^ /,""); print }')
        CANDIDATE_COUNT=$(find "$DFLASH_DIR/models" -maxdepth 4 -type f -name "*.gguf" 2>/dev/null | wc -l)
        if [ "$CANDIDATE_COUNT" -gt 1 ]; then
            warn "Multiple GGUF models in $DFLASH_DIR/models — selected largest: ${DFLASH_TARGET}"
        fi
    fi
    if [ -z "$DFLASH_TARGET" ]; then
        printf '\033[1;31m[ERROR]\033[0m No GGUF target found under %s/\n' "$DFLASH_DIR/models"
        printf '\nDownload one of the supported targets, e.g.:\n'
        printf '  mkdir -p %s/models %s/models/draft\n' "$DFLASH_DIR" "$DFLASH_DIR"
        printf '  huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir %s/models\n' "$DFLASH_DIR"
        printf '  huggingface-cli download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q8_0.gguf --local-dir %s/models/draft\n' "$DFLASH_DIR"
        printf '\nOr point DFLASH_TARGET at an existing .gguf file.\n' >&2
        exit 1
    fi
fi

# Qwen3.6 DFlash drafters use sliding-window attention in the draft. Some GGUFs
# carry this metadata directly; keep the documented env override as the Docker
# default so the image behaves like the benchmark path even for older drafts.
case "$(basename "$DFLASH_TARGET")" in
    *Qwen3.6*|*qwen3.6*)
        if [ -z "${DFLASH27B_DRAFT_SWA:-}" ]; then
            export DFLASH27B_DRAFT_SWA=2048
            info "Autotune: DFLASH27B_DRAFT_SWA=2048 (Qwen3.6 draft SWA)"
        fi
        ;;
esac

# ── validation ───────────────────────────────────────────────────────────────

[ -f "$DFLASH_BIN" ]      || die "Binary not found at $DFLASH_BIN (build with: cmake --build dflash/build --target test_dflash -j)"
[ -f "$DFLASH_TARGET" ]   || die "Target GGUF not found at $DFLASH_TARGET"

# Draft resolution: accept dir (server.py resolves GGUF/safetensors) or direct file
DRAFT_ARG="$DFLASH_DRAFT"
if [ -d "$DFLASH_DRAFT" ]; then
    if ! find "$DFLASH_DRAFT" -maxdepth 4 -type f \( \
            -name 'dflash-draft-*.gguf' -o \
            -name '*.gguf' -o \
            -name 'model.safetensors' -o \
            -name '*.safetensors' \
        \) -print -quit | grep -q .; then
        warn "No DFlash draft GGUF/safetensors in draft dir $DFLASH_DRAFT — starting without draft"
        DRAFT_ARG=""
    fi
elif [ -n "$DFLASH_DRAFT" ] && [ ! -f "$DFLASH_DRAFT" ]; then
    warn "Draft path $DFLASH_DRAFT not found — starting without draft"
    DRAFT_ARG=""
fi

# ── build command ────────────────────────────────────────────────────────────

CMD=(uv run --directory "$DFLASH_DIR" python scripts/server.py)
CMD+=(--host "$DFLASH_HOST")
CMD+=(--port "$DFLASH_PORT")
CMD+=(--target "$DFLASH_TARGET")
CMD+=(--bin "$DFLASH_BIN")
CMD+=(--budget "$DFLASH_BUDGET")
CMD+=(--max-ctx "$DFLASH_MAX_CTX")
CMD+=(--prefix-cache-slots "$DFLASH_PREFIX_CACHE_SLOTS")
CMD+=(--prefill-cache-slots "$DFLASH_PREFILL_CACHE_SLOTS")

if [ -n "$DRAFT_ARG" ]; then
    CMD+=(--draft "$DRAFT_ARG")
fi

if [ "$DFLASH_LAZY" = "1" ]; then
    CMD+=(--lazy-draft)
fi

if [ "$DFLASH_VERBOSE" = "1" ]; then
    CMD+=(--verbose-daemon)
fi

# pFlash
if [ "$DFLASH_PREFILL_MODE" != "off" ]; then
    if [ -z "$DFLASH_PREFILL_DRAFTER" ]; then
        die "DFLASH_PREFILL_MODE=$DFLASH_PREFILL_MODE requires DFLASH_PREFILL_DRAFTER (path to Qwen3-0.6B BF16 GGUF)"
    fi
    [ -f "$DFLASH_PREFILL_DRAFTER" ] || die "Prefill drafter not found at $DFLASH_PREFILL_DRAFTER"
    CMD+=(--prefill-compression "$DFLASH_PREFILL_MODE")
    CMD+=(--prefill-keep-ratio "$DFLASH_PREFILL_KEEP")
    CMD+=(--prefill-threshold "$DFLASH_PREFILL_THRESHOLD")
    CMD+=(--prefill-drafter "$DFLASH_PREFILL_DRAFTER")
fi

# ── display configuration ────────────────────────────────────────────────────

printf '\n'
info "Starting server with:"
printf '       host     = %s\n' "$DFLASH_HOST"
printf '       port     = %s\n' "$DFLASH_PORT"
printf '       target   = %s\n' "$DFLASH_TARGET"
printf '       draft    = %s\n' "${DRAFT_ARG:-(none)}"
printf '       bin      = %s\n' "$DFLASH_BIN"
printf '       budget   = %s\n' "$DFLASH_BUDGET"
printf '       max_ctx  = %s\n' "$DFLASH_MAX_CTX"
printf '       prefill  = %s\n' "$DFLASH_PREFILL_MODE"
printf '       cpus     = %s\n' "$NPROC"
printf '       ram      = %s GB\n' "$TOTAL_RAM_GB"
printf '\n'

# ── start server ─────────────────────────────────────────────────────────────
# Exec so the script's PID becomes the server's PID (clean signal handling)
exec "${CMD[@]}"
