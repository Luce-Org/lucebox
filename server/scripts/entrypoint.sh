#!/usr/bin/env bash
# In-container ENTRYPOINT for lucebox-hub.
#
# Normal path: the host-side `lucebox` CLI has already populated every
# DFLASH_* env var from its detection / autotune sweep, so this script
# just resolves paths and execs the native dflash_server binary.
#
# Fallback path: a user runs the image directly (`docker run --gpus all
# ghcr.io/luce-org/lucebox-hub:cuda12`) with no env-var prep. We then do a
# minimal VRAM-tiered autotune — same tiers as `lucebox autotune`, kept in
# sync by hand. Anything more elaborate (driver-version probes, AMD paths,
# lspci fallbacks) belongs in the host CLI, not here.

set -euo pipefail

# Honor a pre-set DFLASH_DIR (used by the host-side smoke tests to drive
# the entrypoint with a synthetic models/draft layout). In the shipped
# image this var is unset, so the fallback is the normal install prefix.
DFLASH_DIR="${DFLASH_DIR:-/opt/lucebox-hub/server}"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── arg dispatch ───────────────────────────────────────────────────────────
# `serve` (default) — start the OpenAI-compatible server.
# `shell`            — drop into bash inside the container (debug).
# `lucebox`          — dispatch to the Python CLI. Any subcommand
#                      `lucebox.sh` doesn't handle on the host arrives here
#                      (check, config, pull, print-run, smoke, …).
# `python` or anything else
#                    — pass through to exec, so `docker run … python -m foo`
#                      still works for dev.
SUBCMD="${1:-serve}"
[ $# -gt 0 ] && shift || true

LUCEBOX_PKG="/opt/lucebox-hub"

case "$SUBCMD" in
    lucebox)
        # --no-sync: the venv was fully populated at image build time
        # (`uv sync --no-editable` in the Dockerfile). Skipping the env
        # consistency check here prevents hatch-vcs from firing its
        # `_version.py` write hook against the read-only workspace source
        # dirs and crashing the entire subcommand.
        exec uv run --no-sync --directory "$LUCEBOX_PKG" python -m lucebox "$@"
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

# ── HOST_INFO (host-identity capture) ──────────────────────────────────────
# Write /opt/lucebox-hub/HOST_INFO as JSON before exec'ing dflash_server.
# The C++ server reads this file at startup and surfaces the parsed JSON
# under /props.host. Mirrors the IMAGE_INFO pattern (server_main.cpp
# read_image_info) but in JSON instead of KEY=VALUE — host facts have
# nested structure (gpu array, multi-field per GPU) that doesn't fit a
# flat KEY=VALUE layout.
#
# Inputs: the LUCEBOX_HOST_* env vars set by the host wrapper's
# probe_host(). When none are set (e.g. someone ran `docker run` directly,
# bypassing lucebox.sh), we still write a stub `{"source":"unknown", ...}`
# so the C++ side doesn't have to special-case missing-vs-blank.
#
# Failure is never fatal — the host_info file is informational. A
# write-failure (read-only FS, etc.) gets a warning and we continue.
write_host_info() {
    local target="/opt/lucebox-hub/HOST_INFO"
    local tmp="${target}.tmp.$$"
    local collected_at
    collected_at=$(date -u +%FT%TZ 2>/dev/null || echo "")
    # If any LUCEBOX_HOST_* var was supplied, the source is "lucebox.sh"
    # (the host wrapper probed and forwarded these via -e). Otherwise the
    # container was launched outside the wrapper — we still emit a stub
    # so the C++ side can read /props.host without special-casing missing.
    local source_tag="unknown"
    local collector_tag="entrypoint.sh"
    if [ -n "${LUCEBOX_HOST_OS_PRETTY:-}" ] \
       || [ -n "${LUCEBOX_HOST_KERNEL:-}" ] \
       || [ -n "${LUCEBOX_HOST_GPU_LIST_CSV:-}" ] \
       || [ -n "${LUCEBOX_HOST_CPU_MODEL:-}" ]; then
        source_tag="lucebox.sh"
        collector_tag="lucebox.sh"
    fi

    if ! _build_host_info_json "$source_tag" "$collector_tag" "$collected_at" > "$tmp" 2>/dev/null; then
        warn "Failed to build HOST_INFO JSON — skipping"
        rm -f "$tmp" 2>/dev/null || true
        return 0
    fi
    if ! mv -f "$tmp" "$target" 2>/dev/null; then
        warn "Failed to write $target (continuing without it)"
        rm -f "$tmp" 2>/dev/null || true
        return 0
    fi
    info "Wrote $target (source=$source_tag)"
}

# Build the HOST_INFO JSON on stdout. Real JSON escape via python3 (always
# present in the runtime image — uv pulls it in for the venv stage) with
# a bash fallback for parsing emergencies (broken venv, debug invocations
# from a minimal base). The bash fallback covers the realistic char set
# that leaks from lscpu / /etc/os-release / nvidia-smi (backslash, quote,
# newline, tab, CR); the python path covers every JSON-illegal char
# including the full U+0000-U+001F control range, so a misbehaved upstream
# can't silently invalidate the entire HOST_INFO and turn /props.host
# into null on the C++ side (which silently drops a parse-failed block).
_json_escape() {
    # Read from $1, emit on stdout. No quotes around the result — the
    # caller wraps with `"..."`.
    if command -v python3 >/dev/null 2>&1; then
        # json.dumps emits `"…escaped…"`; strip the surrounding quotes
        # so callers can keep their existing `"..."` wrap convention.
        python3 -c '
import json, sys
out = json.dumps(sys.argv[1])
sys.stdout.write(out[1:-1])
' "$1"
        return
    fi
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

# Emit a JSON value for a string field. Empty input → JSON null. Caller
# embeds the result directly (no leading/trailing whitespace).
_json_str_or_null() {
    if [ -z "${1:-}" ]; then
        printf 'null'
    else
        printf '"%s"' "$(_json_escape "$1")"
    fi
}

# Emit a JSON value for an integer field. Empty / non-numeric → null.
_json_int_or_null() {
    local v="${1:-}"
    if [ -z "$v" ] || ! [[ "$v" =~ ^-?[0-9]+$ ]]; then
        printf 'null'
    else
        printf '%s' "$v"
    fi
}

# Parse the LUCEBOX_HOST_GPU_LIST_CSV (whatever
# `nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,power.limit
#               --format=csv,noheader` produced on the host) into a JSON
# array. Empty CSV → "[]". Each row becomes one object.
_emit_gpu_array() {
    local csv="${LUCEBOX_HOST_GPU_LIST_CSV:-}"
    if [ -z "$csv" ]; then
        printf '[]'
        return
    fi
    local out="[" first=1
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        # Trim surrounding whitespace from each field. nvidia-smi prints
        # `0, GPU-abc..., 00000000:01:00.0, NVIDIA RTX 5090, 12.0, 24576 MiB, 175.00 W`.
        local idx uuid pci name cc mem plimit
        idx=$(printf '%s' "$line" | awk -F', ' '{print $1}' | sed 's/^ *//; s/ *$//')
        uuid=$(printf '%s' "$line" | awk -F', ' '{print $2}' | sed 's/^ *//; s/ *$//')
        pci=$(printf '%s' "$line" | awk -F', ' '{print $3}' | sed 's/^ *//; s/ *$//')
        name=$(printf '%s' "$line" | awk -F', ' '{print $4}' | sed 's/^ *//; s/ *$//')
        cc=$(printf '%s' "$line" | awk -F', ' '{print $5}' | sed 's/^ *//; s/ *$//')
        mem=$(printf '%s' "$line" | awk -F', ' '{print $6}' | sed 's/^ *//; s/ *$//')
        plimit=$(printf '%s' "$line" | awk -F', ' '{print $7}' | sed 's/^ *//; s/ *$//')
        # Strip units. "24576 MiB" → 24576; "175.00 W" → 175 (truncate).
        local mem_mib vram_gb power_w
        mem_mib=$(printf '%s' "$mem" | awk '{print $1+0}')
        vram_gb=""
        if [ -n "$mem_mib" ] && [ "$mem_mib" -gt 0 ] 2>/dev/null; then
            vram_gb=$((mem_mib / 1024))
        fi
        power_w=$(printf '%s' "$plimit" | awk '{printf "%d", $1+0.5}')
        if [ "$first" = "1" ]; then
            first=0
        else
            out+=","
        fi
        out+="{\"index\":$(_json_int_or_null "$idx"),\"uuid\":$(_json_str_or_null "$uuid"),"
        out+="\"pci_bus_id\":$(_json_str_or_null "$pci"),\"name\":$(_json_str_or_null "$name"),"
        out+="\"sm\":$(_json_str_or_null "$cc"),\"vram_gb\":$(_json_int_or_null "$vram_gb"),"
        out+="\"power_limit_w\":$(_json_int_or_null "$power_w")}"
    done <<<"$csv"
    out+="]"
    printf '%s' "$out"
}

_build_host_info_json() {
    local source_tag="$1" collector_tag="$2" collected_at="$3"
    printf '{'
    printf '"os_pretty":%s,'        "$(_json_str_or_null "${LUCEBOX_HOST_OS_PRETTY:-}")"
    printf '"kernel":%s,'           "$(_json_str_or_null "${LUCEBOX_HOST_KERNEL:-}")"
    printf '"wsl_version":%s,'      "$(_json_str_or_null "${LUCEBOX_HOST_WSL_VERSION:-}")"
    printf '"docker_version":%s,'   "$(_json_str_or_null "${LUCEBOX_HOST_DOCKER_VERSION:-}")"
    printf '"nvidia_driver":%s,'    "$(_json_str_or_null "${LUCEBOX_HOST_DRIVER_VERSION:-}")"
    printf '"nvidia_ctk_version":%s,' "$(_json_str_or_null "${LUCEBOX_HOST_NVIDIA_CTK_VERSION:-}")"
    printf '"cpu_model":%s,'        "$(_json_str_or_null "${LUCEBOX_HOST_CPU_MODEL:-}")"
    printf '"nproc":%s,'            "$(_json_int_or_null "${LUCEBOX_HOST_NPROC:-}")"
    printf '"ram_gb":%s,'           "$(_json_int_or_null "${LUCEBOX_HOST_RAM_GB:-}")"
    printf '"gpus":%s,'             "$(_emit_gpu_array)"
    printf '"cuda_visible_devices":%s,' "$(_json_str_or_null "${LUCEBOX_HOST_CUDA_VISIBLE_DEVICES:-}")"
    printf '"source":%s,'           "$(_json_str_or_null "$source_tag")"
    printf '"collector":%s,'        "$(_json_str_or_null "$collector_tag")"
    printf '"collected_at":%s'      "$(_json_str_or_null "$collected_at")"
    printf '}\n'
}

write_host_info

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
# Target .gguf is typically 10-30 GB (Q4_K_M). Drafts are 1-2 GB (Q8_0 / Q4)
# and live under models/draft/ or have a dflash- prefix. The 5 GB threshold
# excludes drafts cleanly without needing to parse GGUF arch metadata.
#
# CRITICAL UX RULE: when multiple candidate targets exist, we DO NOT silently
# pick one based on filename pattern. That hid a real bug for the matrix
# bench — a hardcoded Qwen3.6 preference made the container run the wrong
# model when both gemma4 and qwen3.6 GGUFs were present, and the operator
# only noticed when the bench numbers came out wrong. Either set
# DFLASH_TARGET=... explicitly, or have exactly one .gguf in models/.
if [ -z "$DFLASH_TARGET" ] && [ -d "$DFLASH_DIR/models" ]; then
    # Collect candidates: .gguf files ≥5 GB (target-sized), excluding
    # anything under models/draft/. Sort alphabetically for determinism.
    mapfile -t TARGET_CANDIDATES < <(
        find -L "$DFLASH_DIR/models" -maxdepth 4 -type f -name '*.gguf' \
            -size +5G \
            -not -path '*/draft/*' \
            -printf '%p\n' 2>/dev/null \
          | sort
    )
    case "${#TARGET_CANDIDATES[@]}" in
        0)
            : # fall through to the missing-target die below
            ;;
        1)
            DFLASH_TARGET="${TARGET_CANDIDATES[0]}"
            info "Auto-detected target: $(basename "$DFLASH_TARGET")"
            ;;
        *)
            warn "Multiple candidate targets in $DFLASH_DIR/models — running the FIRST alphabetically."
            warn "To pick a specific one, set DFLASH_TARGET=<path>. Candidates:"
            for c in "${TARGET_CANDIDATES[@]}"; do
                marker=" "
                [ "$c" = "${TARGET_CANDIDATES[0]}" ] && marker="*"
                warn "  ${marker} $c"
            done
            DFLASH_TARGET="${TARGET_CANDIDATES[0]}"
            ;;
    esac
fi

if [ -z "$DFLASH_TARGET" ] || [ ! -f "$DFLASH_TARGET" ]; then
    die "No target GGUF found. Mount a model dir: -v /host/models:/opt/lucebox-hub/server/models, or set DFLASH_TARGET=<path-inside-container>."
fi
[ -x "$DFLASH_SERVER_BIN" ] || die "dflash_server binary missing at $DFLASH_SERVER_BIN (image build failed?)"

# Qwen3.6 DFlash drafters use sliding-window attention in the draft. Some GGUFs
# carry this metadata directly; keep the documented env override as the startup
# default so older drafts behave like the autotune-sweep path.
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
    # Use -iname (case-insensitive) throughout so both naming conventions
    # work: legacy "dflash-gemma-4-31b-*.gguf" and the Lucebox HF repo's
    # "gemma-4-31B-it-DFlash-q8_0.gguf". Glob list is family-specific first,
    # then generic dflash-draft-*.gguf legacy, then last-resort *.gguf.
    # The 31B match in the Lucebox repo uses capital B in the filename —
    # -iname handles that without needing to enumerate every case form.
    case "$(echo "$TARGET_BASENAME" | tr 'A-Z' 'a-z')" in
        *gemma-4-26b*|*gemma4-26b*)
            FAMILY_GLOBS=('*gemma*4*26b*dflash*.gguf' '*dflash*gemma*4*26b*.gguf') ;;
        *gemma-4-31b*|*gemma4-31b*)
            FAMILY_GLOBS=('*gemma*4*31b*dflash*.gguf' '*dflash*gemma*4*31b*.gguf') ;;
        *gemma-4*|*gemma4*)
            FAMILY_GLOBS=('*gemma*4*dflash*.gguf' '*dflash*gemma*4*.gguf') ;;
        *qwen3.6*|*qwen36*)
            FAMILY_GLOBS=('dflash-draft-3.6-*.gguf' '*qwen*3.6*dflash*.gguf') ;;
        *)
            FAMILY_GLOBS=() ;;
    esac

    DRAFT_FILE=""
    # Track which glob actually matched so the info() log can show whether
    # we picked via the family-specific pattern or fell back to a generic
    # one. Initialize empty up front — `set -u` will fire if we read the
    # var without an assignment having run, and the for-loop below may
    # exit on the very first iteration without entering the body.
    DRAFT_FAMILY_GLOB=""
    # Family-specific globs first (most specific). Then the legacy
    # `dflash-draft-*.gguf` for single-draft setups. Then the generic
    # `*.gguf` / safetensors fallbacks.
    GENERIC_GLOBS=('dflash-draft-*.gguf' '*dflash*.gguf' '*.gguf' 'model.safetensors' '*.safetensors')
    family_count="${#FAMILY_GLOBS[@]}"
    i=0
    for pattern in "${FAMILY_GLOBS[@]}" "${GENERIC_GLOBS[@]}"; do
        DRAFT_FILE="$(find -L "$DFLASH_DRAFT" -maxdepth 4 -type f -iname "$pattern" -print -quit 2>/dev/null)"
        if [ -n "$DRAFT_FILE" ]; then
            # Mark the family-specific match so the log line below can
            # distinguish "matched on family hint" from "generic fallback".
            if [ "$i" -lt "$family_count" ]; then
                DRAFT_FAMILY_GLOB="$pattern"
            fi
            break
        fi
        i=$((i + 1))
    done
    # Defensive: every read of DRAFT_FAMILY_GLOB below must survive `set -u`
    # even if the init on line ~257 was somehow skipped (e.g. a future refactor
    # that moves the init out of this block, or a partial-rewrite during a
    # rebase that drops it). Coalesce-to-empty inline so a regression can't
    # re-trip the unbound-variable crash that fired on the sindri sweep with
    # multiple target GGUFs in models/ (commit a87bb93 was a partial fix —
    # the recurrence proved that "initialize once at the top of the block"
    # is too easy to undo). Cost: zero bytes at runtime.
    DRAFT_FAMILY_GLOB="${DRAFT_FAMILY_GLOB:-}"
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
# `--lazy-draft` is silently dropped by the C++ server unless both
# `--prefill-drafter` and `--draft` are present (look for the runtime
# warning `--lazy-draft ignored: requires both --prefill-drafter and
# --draft`). Warn loudly here when the operator's config asked for lazy
# but we're about to drop it — sweeping past the silent no-op was the
# fingerprint left in every sindri decode-tuning docker.stderr.
if [ "$DFLASH_LAZY" = "1" ]; then
    if [ -z "$DRAFT_ARG" ] || [ -z "$DFLASH_PREFILL_DRAFTER" ]; then
        warn "DFLASH_LAZY=1 ignored: requires both DFLASH_DRAFT and DFLASH_PREFILL_DRAFTER (see entrypoint.sh comment). Continuing without --lazy-draft."
    else
        CMD+=(--lazy-draft)
    fi
fi
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
