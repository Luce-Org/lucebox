#!/usr/bin/env bash
# lucebox.sh — host-side wrapper for the lucebox-hub container.
#
# Two jobs:
#
#   1) Probe the host (NVIDIA/AMD GPU, driver/runtime, docker, VRAM, RAM,
#      systemd), select the CUDA 12 or ROCm image, and
#      dispatch into the in-container Python CLI via `docker run`. The
#      Python CLI lives at /opt/lucebox-hub/lucebox/ inside the image and is
#      the single source of truth for orchestration logic — TOML config,
#      optimization planning, calibration probes, and model downloads.
#
#   2) Manage a user-level systemd unit (~/.config/systemd/user/lucebox.service)
#      so the server can run as a long-lived service without keeping a shell
#      open. install/uninstall/start/stop/enable/disable/status/logs all
#      delegate to systemctl --user / journalctl --user.
#
# Install:
#   curl -fsSL https://raw.githubusercontent.com/Luce-Org/lucebox/main/install.sh | bash
#
# The installer bakes the source URL into the installed copy as
# `LUCEBOX_INSTALLED_FROM=`, so `lucebox update` later re-pulls from the
# same channel (canonical, dev fork, branch — whatever you originally
# installed from).
#
# Then: lucebox check && lucebox install && lucebox start
#
# The runtime works whether the file is installed as `lucebox` (preferred)
# or `lucebox.sh` — all self-referencing hints use the actual basename.
#
# No root is ever taken automatically. Anything that needs sudo (package
# install, loginctl enable-linger) is printed for the user to run.

set -euo pipefail

# GPU SDK packages on minimal Ubuntu images do not always update a login
# shell's PATH.  Keep probing and native builds self-contained; this changes
# PATH only for the wrapper process and its children.
for lucebox_toolchain_bin in /opt/rocm/bin /usr/local/cuda/bin; do
    [ -d "$lucebox_toolchain_bin" ] || continue
    case ":${PATH:-}:" in
        *:"$lucebox_toolchain_bin":*) ;;
        # Respect explicit caller paths (including test shims and operator
        # toolchain pins); SDK defaults are fallbacks, not overrides.
        *) export PATH="${PATH:-/usr/bin:/bin}:$lucebox_toolchain_bin" ;;
    esac
done
unset lucebox_toolchain_bin

VERSION="0.2.0"
SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"

# ── tunables / env overrides ───────────────────────────────────────────────
# Host-side scalars (image registry+variant, port, container name, models
# dir). Resolution order, applied uniformly via _lucebox_resolve below:
#   1. $LUCEBOX_<NAME>            per-invocation env override
#   2. config.toml <section>.<key>  persisted user choice (system of record)
#   3. derived / canonical default
# This keeps the wrapper and the in-container Python CLI agreeing on
# effective values — config.toml is the single source of truth, both
# sides read it.
UNIT_NAME="lucebox.service"
UNIT_PATH="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT_NAME"

# CUDA driver floor for the prebuilt CUDA 12 image.
# shellcheck disable=SC2034
MIN_DRIVER_CUDA12=525
MIN_DRIVER_CUDA128=570
MIN_DRIVER_CUDA13=580

# Canonical source of `lucebox.sh`. The bootstrap installer (`install.sh`)
# rewrites this line at install time to record which URL the user actually
# installed from — `lucebox update` then re-pulls from the same channel
# without losing track of forks. Falls back to the Luce-Org main branch
# when nothing was baked in (e.g. someone curl'd the script directly).
LUCEBOX_INSTALLED_FROM="${LUCEBOX_INSTALLED_FROM:-https://raw.githubusercontent.com/Luce-Org/lucebox/main/lucebox.sh}"

# Path to the persisted config.toml. Mirrors
# lucebox.config.default_config_path: $LUCEBOX_HOME/config.toml if set,
# else $HOME/.lucebox/config.toml. Read-only from this wrapper — the
# Python CLI is the writer.
_lucebox_config_path() {
    if [ -n "${LUCEBOX_HOME:-}" ]; then
        printf '%s/config.toml' "$LUCEBOX_HOME"
        return
    fi
    printf '%s/.lucebox/config.toml' "$HOME"
}

# Read a `<section>.<key>` value from config.toml. Returns empty if the
# file is missing, the section/key is absent, or the value is empty.
# Handles the subset of TOML that lucebox writes:
#   [section]
#   key = "string"      # surrounding double-quotes are stripped
#   key = 8080          # bare scalars passed through verbatim
#   key = true          # same
#   key = [             # simple multi-line arrays written by tomli_w
#       "hip:0",
#       "hip:1",
#   ]
# Inline `# comment` is honored. Inline tables and multi-line strings are not
# part of the host wrapper's config contract.
_lucebox_config_get() {
    local dotted="$1" cfg
    cfg="$(_lucebox_config_path)"
    [ -f "$cfg" ] || return 0
    local section="${dotted%.*}"
    local key="${dotted##*.}"
    [ "$section" = "$dotted" ] && section=""
    awk -v want_section="$section" -v want_key="$key" '
        function strip_comment(text,    i, ch, in_quote, escaped) {
            in_quote = 0
            escaped = 0
            for (i = 1; i <= length(text); i++) {
                ch = substr(text, i, 1)
                if (escaped) {
                    escaped = 0
                    continue
                }
                if (in_quote && ch == "\\") {
                    escaped = 1
                    continue
                }
                if (ch == "\"") {
                    in_quote = !in_quote
                    continue
                }
                if (ch == "#" && !in_quote)
                    return substr(text, 1, i - 1)
            }
            return text
        }
        BEGIN { current = "" }
        /^[[:space:]]*\[/ {
            t = $0
            sub(/^[[:space:]]*\[[[:space:]]*/, "", t)
            sub(/[[:space:]]*\][[:space:]]*$/, "", t)
            current = t
            next
        }
        /^[[:space:]]*#/ { next }
        /=/ {
            if (current != want_section) next
            line = strip_comment($0)
            eq = index(line, "=")
            if (eq == 0) next
            k = substr(line, 1, eq - 1)
            v = substr(line, eq + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            if (k != want_key) next
            if (substr(v, 1, 1) == "[" && index(v, "]") == 0) {
                while ((getline continuation) > 0) {
                    continuation = strip_comment(continuation)
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", continuation)
                    v = v continuation
                    if (index(continuation, "]") > 0) break
                }
            }
            if (length(v) >= 2 && substr(v, 1, 1) == "\"" && substr(v, length(v), 1) == "\"")
                v = substr(v, 2, length(v) - 2)
            print v
            exit
        }
    ' "$cfg"
}

# Resolve a scalar through the precedence ladder. env_value comes from
# the caller (typically `"${LUCEBOX_FOO:-}"` — the `:-` matters under
# `set -u`).
_lucebox_resolve() {
    local env_value="$1" toml_key="$2" default="$3" v
    if [ -n "$env_value" ]; then
        printf '%s' "$env_value"
        return
    fi
    v="$(_lucebox_config_get "$toml_key")"
    if [ -n "$v" ]; then
        printf '%s' "$v"
        return
    fi
    printf '%s' "$default"
}

# Derive the default image URL from the install source so a fork install
# (e.g. easel/lucebox-hub) gets the fork's GHCR image automatically when
# config.toml hasn't pinned one yet. Pattern:
#   https://raw.githubusercontent.com/<org>/<repo>/<ref>/lucebox.sh
#   → ghcr.io/<org-lowercase>/<repo>
# GHCR rejects mixed-case org paths so the org segment is lowercased; the
# repo name is preserved as-is. Falls back to the canonical Luce-Org image
# when the URL doesn't match the raw.githubusercontent.com pattern.
_lucebox_derive_image() {
    # The ref segment can contain slashes (e.g. `feat/lucebox-docker`), so
    # the middle `.+` greedily eats everything up to the trailing
    # `/lucebox.sh`. The first two `[^/]+` capture org + repo, which are
    # never slash-containing on GitHub.
    local url="$1" org repo
    if [[ "$url" =~ ^https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/.+/lucebox\.sh$ ]]; then
        org=$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')
        repo="${BASH_REMATCH[2]}"
        # The source repository was renamed from lucebox-hub to lucebox, while
        # the published runtime image intentionally keeps its established
        # ghcr.io/luce-org/lucebox-hub name.
        if [ "$org" = "luce-org" ] && [ "$repo" = "lucebox" ]; then
            repo="lucebox-hub"
        fi
        printf 'ghcr.io/%s/%s' "$org" "$repo"
        return
    fi
    printf 'ghcr.io/luce-org/lucebox-hub'
}

# Effective scalars, env > config.toml > default.
CONTAINER_NAME=$(_lucebox_resolve "${LUCEBOX_CONTAINER:-}" runtime.container_name "lucebox")
DEFAULT_PORT=$(_lucebox_resolve "${LUCEBOX_PORT:-}" runtime.port "8080")
DEFAULT_MODELS_DIR=$(_lucebox_resolve "${LUCEBOX_MODELS:-}" paths.models "${XDG_DATA_HOME:-$HOME/.local/share}/lucebox/models")
IMAGE_BASE=$(_lucebox_resolve "${LUCEBOX_IMAGE:-}" image.registry "$(_lucebox_derive_image "$LUCEBOX_INSTALLED_FROM")")
CONFIG_HOME="${LUCEBOX_HOME:-$HOME/.lucebox}"
CONNECTOR_STATE_DIR="$CONFIG_HOME/connectors"
CONNECTOR_SELECTION_FILE="$CONNECTOR_STATE_DIR/selected"

# ── LUCEBOX_HOST_* safe defaults (belt-and-suspenders) ────────────────────
# `set -u` makes any unbound LUCEBOX_HOST_* read fatal. Historically this has
# been the #1 source of regressions in this wrapper: someone adds a code path
# that touches a LUCEBOX_HOST_* var before probe_host has run, the call sites
# that DO pre-probe still work, and the bug ships. To make the bug literally
# unrepresentable we seed every LUCEBOX_HOST_* with an explicit safe default
# at script-load time (these mirror probe_host's "nothing detected" state).
# probe_host then overwrites them with real values. Any future read — pre- or
# post-probe — is now well-defined.
: "${LUCEBOX_HOST_NPROC:=1}"
: "${LUCEBOX_HOST_RAM_GB:=0}"
: "${LUCEBOX_HOST_GPU_VENDOR:=none}"
: "${LUCEBOX_HOST_HAS_NVIDIA_GPU:=0}"
: "${LUCEBOX_HOST_HAS_AMD_GPU:=0}"
: "${LUCEBOX_HOST_GPU_NAME:=}"
: "${LUCEBOX_HOST_GPU_COUNT:=0}"
: "${LUCEBOX_HOST_VRAM_GB:=0}"
: "${LUCEBOX_HOST_GPU_SM:=}"
: "${LUCEBOX_HOST_DRIVER_VERSION:=}"
: "${LUCEBOX_HOST_DRIVER_MAJOR:=0}"
: "${LUCEBOX_HOST_NVIDIA_GPU_NAME:=}"
: "${LUCEBOX_HOST_NVIDIA_GPU_COUNT:=0}"
: "${LUCEBOX_HOST_NVIDIA_VRAM_GB:=0}"
: "${LUCEBOX_HOST_NVIDIA_GPU_ARCH:=}"
: "${LUCEBOX_HOST_NVIDIA_GPU_LIST_CSV:=}"
: "${LUCEBOX_HOST_NVIDIA_UNIFIED_MEMORY:=0}"
: "${LUCEBOX_HOST_ROCM_VERSION:=}"
: "${LUCEBOX_HOST_HAS_KFD:=0}"
: "${LUCEBOX_HOST_HAS_DRI:=0}"
: "${LUCEBOX_HOST_AMD_GPU_NAME:=}"
: "${LUCEBOX_HOST_AMD_GPU_COUNT:=0}"
: "${LUCEBOX_HOST_AMD_VRAM_GB:=0}"
: "${LUCEBOX_HOST_AMD_GPU_ARCH:=}"
: "${LUCEBOX_HOST_AMD_GPU_LIST_CSV:=}"
: "${LUCEBOX_HOST_HAS_SYSTEMD:=0}"
: "${LUCEBOX_HOST_IS_WSL:=0}"
: "${LUCEBOX_HOST_HAS_DOCKER:=0}"
: "${LUCEBOX_HOST_DOCKER_VERSION:=}"
: "${LUCEBOX_HOST_HAS_CTK:=none}"
# Host-identity facts (item 1 — host-identity capture). These ride along
# the existing LUCEBOX_HOST_* convoy into the container so /opt/lucebox-hub/
# HOST_INFO can be written without re-probing inside the container (where
# /proc and nvidia-smi see the container's view, not the rig's).
: "${LUCEBOX_HOST_OS_PRETTY:=}"
: "${LUCEBOX_HOST_KERNEL:=}"
: "${LUCEBOX_HOST_WSL_VERSION:=}"
: "${LUCEBOX_HOST_NVIDIA_CTK_VERSION:=}"
: "${LUCEBOX_HOST_CPU_MODEL:=}"
: "${LUCEBOX_HOST_GPU_LIST_CSV:=}"
: "${LUCEBOX_HOST_CUDA_VISIBLE_DEVICES:=}"
: "${LUCEBOX_HOST_HIP_VISIBLE_DEVICES:=}"
: "${LUCEBOX_HOST_ROCR_VISIBLE_DEVICES:=}"
: "${LUCEBOX_HOST_HAS_HYBRID_RUNTIME:=0}"
: "${LUCEBOX_HOST_HYBRID_SERVER_BIN:=}"
: "${LUCEBOX_HOST_HYBRID_IPC_BIN:=}"
: "${LUCEBOX_HOST_HYBRID_DFLASH_DIR:=}"
: "${LUCEBOX_HOST_HYBRID_ENTRYPOINT:=}"
# Tracks whether probe_host has actually run; pieces of the code that need
# fresh host facts (e.g. cmd_check, cmd_serve) gate on this. Default 0.
: "${_LUCEBOX_HOST_PROBED:=0}"

# ── output helpers ────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_INFO='\033[1;34m'; C_OK='\033[1;32m'; C_WARN='\033[1;33m'
    C_ERR='\033[1;31m'; C_DIM='\033[2m'; C_BRAND='\033[38;2;245;200;66m'; C_RST='\033[0m'
else
    C_INFO=''; C_OK=''; C_WARN=''; C_ERR=''; C_DIM=''; C_RST=''
    C_BRAND=''
fi

info()  { printf '%b[INFO]%b  %s\n' "$C_INFO" "$C_RST" "$*"; }
ok()    { printf '%b[OK]%b    %s\n' "$C_OK"   "$C_RST" "$*"; }
warn()  { printf '%b[WARN]%b  %s\n' "$C_WARN" "$C_RST" "$*"; }
err()   { printf '%b[ERROR]%b %s\n' "$C_ERR"  "$C_RST" "$*" >&2; }
hint()  { printf '       %b%s%b\n'  "$C_DIM"  "$*"     "$C_RST"; }
die()   { err "$*"; exit 1; }

print_logo() {
    printf '%b' "$C_BRAND"
    cat <<'EOF'
 ░██                                          ░██
 ░██         ░██    ░██  ░███████   ░███████  ░████████   ░███████  ░██    ░██
 ░██         ░██    ░██ ░██    ░██ ░██    ░██ ░██    ░██ ░██    ░██  ░██  ░██
 ░██         ░██    ░██ ░██        ░█████████ ░██    ░██ ░██    ░██   ░█████
 ░██         ░██   ░███ ░██    ░██ ░██        ░███   ░██ ░██    ░██  ░██  ░██
 ░██████████  ░█████░██  ░███████   ░███████  ░██░█████   ░███████  ░██    ░██
EOF
    printf '%b   computers for agents%b\n\n' "$C_DIM" "$C_RST"
}

# Find a source checkout for contributor-only actions. An explicit path wins;
# otherwise inspect the current directory and the wrapper's own directory,
# walking upward until the repository markers are found. Buyer installs simply
# return no path and never see build/harness actions.
_find_repo_root() {
    local candidate="${LUCEBOX_REPO:-}" dir
    if [ -n "$candidate" ]; then
        if [ -f "$candidate/server/CMakeLists.txt" ] && [ -d "$candidate/harness" ]; then
            (cd "$candidate" && pwd)
            return 0
        fi
        return 1
    fi

    for candidate in "$PWD" "$(dirname "$SCRIPT_PATH")"; do
        dir="$candidate"
        while [ "$dir" != "/" ] && [ -n "$dir" ]; do
            if [ -f "$dir/server/CMakeLists.txt" ] && [ -d "$dir/harness" ]; then
                (cd "$dir" && pwd)
                return 0
            fi
            dir="$(dirname "$dir")"
        done
    done
    return 1
}

_native_binary_ready() {
    local binary="$1" dependencies
    [ -x "$binary" ] || return 1
    command -v ldd >/dev/null 2>&1 || return 0
    dependencies=$(ldd "$binary" 2>&1 || true)
    [[ "$dependencies" != *"not found"* ]]
}

_confirm() {
    # usage: _confirm "question" [default_yes]
    local question="$1" default_yes="${2:-1}" answer prompt
    if [ "$default_yes" = "1" ]; then prompt="Y/n"; else prompt="y/N"; fi
    printf '%s [%s] ' "$question" "$prompt"
    IFS= read -r answer || return 1
    case "$answer" in
        y|Y|yes|YES|Yes) return 0 ;;
        n|N|no|NO|No)   return 1 ;;
        "")             [ "$default_yes" = "1" ] ;;
        *)               return 1 ;;
    esac
}

sha256_file() {
    local sum
    if command -v sha256sum >/dev/null 2>&1; then
        sum=$(sha256sum "$1")
    elif command -v shasum >/dev/null 2>&1; then
        sum=$(shasum -a 256 "$1")
    else
        die "checksum requested, but neither sha256sum nor shasum is installed"
    fi
    sum="${sum%% *}"
    printf '%s' "$sum" | tr '[:upper:]' '[:lower:]'
}

# ── host probing ──────────────────────────────────────────────────────────
# Sets the LUCEBOX_HOST_* variables consumed by the in-container Python CLI
# (passed through with -e). The Python side trusts these and doesn't reprobe
# — it can't see the host's /proc anyway, only the container's.

# Normalize ``amd-smi static --asic --vram --csv`` into the compact internal
# form ``index|name|gfx_arch|vram_mib|rocr_selector``. Discrete cards use
# their stable GPU UUID instead of assuming amd-smi and ROCr enumerate devices
# in the same order; devices without a usable ASIC serial fall back to index.
# Header lookup keeps this resilient to columns being added or reordered.
_parse_amd_smi_csv() {
    awk -F',' '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                key = tolower($i)
                gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", key)
                col[key] = i
            }
            next
        }
        {
            idx = $(col["gpu"])
            name = $(col["market_name"])
            arch = $(col["target_graphics_version"])
            mem = $(col["size"])
            serial = $(col["asic_serial"])
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", idx)
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", name)
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", arch)
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", mem)
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", serial)
            selector = idx
            serial = tolower(serial)
            sub(/^0x/, "", serial)
            if (serial ~ /^[0-9a-f]+$/ && serial !~ /^0+$/)
                selector = "GPU-" serial
            if (idx ~ /^[0-9]+$/ && arch ~ /^gfx[0-9a-z]+$/ && mem ~ /^[0-9]+([.][0-9]+)?$/)
                printf "%s|%s|%s|%d|%s\n", idx, name, arch, mem, selector
        }
    '
}

# Older ROCm installs ship rocm-smi but not amd-smi. Normalize its CSV to the
# same internal form; rocm-smi reports VRAM in bytes rather than MiB.
_parse_rocm_smi_csv() {
    awk -F',' '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                key = tolower($i)
                gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", key)
                col[key] = i
            }
            next
        }
        {
            dev = $(col["device"])
            bytes = $(col["vram total memory (b)"])
            name = $(col["card series"])
            arch = $(col["gfx version"])
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", dev)
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", bytes)
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", name)
            gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", arch)
            idx = dev
            sub(/^card/, "", idx)
            if (idx ~ /^[0-9]+$/ && arch ~ /^gfx[0-9a-z]+$/ && bytes ~ /^[0-9]+$/)
                printf "%s|%s|%s|%d|%s\n", idx, name, arch, bytes / 1048576, idx
        }
    '
}

probe_host() {
    LUCEBOX_HOST_NPROC=$(nproc 2>/dev/null || echo 1)
    # RAM: try Linux /proc/meminfo first, then macOS/BSD sysctl, else 0.
    LUCEBOX_HOST_RAM_GB=0
    if [ -r /proc/meminfo ]; then
        LUCEBOX_HOST_RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    elif command -v sysctl &>/dev/null; then
        mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        LUCEBOX_HOST_RAM_GB=$(( mem_bytes / 1024 / 1024 / 1024 ))
    fi
    LUCEBOX_HOST_GPU_VENDOR="none"
    LUCEBOX_HOST_HAS_NVIDIA_GPU=0
    LUCEBOX_HOST_HAS_AMD_GPU=0
    LUCEBOX_HOST_GPU_NAME=""
    LUCEBOX_HOST_GPU_COUNT=0
    LUCEBOX_HOST_VRAM_GB=0
    LUCEBOX_HOST_GPU_SM=""
    LUCEBOX_HOST_DRIVER_VERSION=""
    LUCEBOX_HOST_DRIVER_MAJOR=0
    LUCEBOX_HOST_GPU_LIST_CSV=""
    LUCEBOX_HOST_NVIDIA_GPU_NAME=""
    LUCEBOX_HOST_NVIDIA_GPU_COUNT=0
    LUCEBOX_HOST_NVIDIA_VRAM_GB=0
    LUCEBOX_HOST_NVIDIA_GPU_ARCH=""
    LUCEBOX_HOST_NVIDIA_GPU_LIST_CSV=""
    LUCEBOX_HOST_NVIDIA_UNIFIED_MEMORY=0
    LUCEBOX_HOST_ROCM_VERSION=""
    LUCEBOX_HOST_HAS_KFD=0
    LUCEBOX_HOST_HAS_DRI=0
    LUCEBOX_HOST_AMD_GPU_NAME=""
    LUCEBOX_HOST_AMD_GPU_COUNT=0
    LUCEBOX_HOST_AMD_VRAM_GB=0
    LUCEBOX_HOST_AMD_GPU_ARCH=""
    LUCEBOX_HOST_AMD_GPU_LIST_CSV=""
    LUCEBOX_HOST_HAS_HYBRID_RUNTIME=0
    LUCEBOX_HOST_HYBRID_SERVER_BIN=""
    LUCEBOX_HOST_HYBRID_IPC_BIN=""
    LUCEBOX_HOST_HYBRID_DFLASH_DIR=""
    LUCEBOX_HOST_HYBRID_ENTRYPOINT=""

    if command -v nvidia-smi &>/dev/null; then
        local q
        if q=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
                          --format=csv,noheader,nounits 2>/dev/null) && [ -n "$q" ]; then
            LUCEBOX_HOST_GPU_VENDOR="nvidia"
            LUCEBOX_HOST_HAS_NVIDIA_GPU=1
            LUCEBOX_HOST_GPU_NAME=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $1}')
            LUCEBOX_HOST_DRIVER_VERSION=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $3}')
            LUCEBOX_HOST_DRIVER_MAJOR=${LUCEBOX_HOST_DRIVER_VERSION%%.*}
            local cc mem_mib
            cc=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $4}')
            LUCEBOX_HOST_GPU_SM="${cc//./}"
            mem_mib=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $2}')
            if [[ "$mem_mib" =~ ^[0-9]+$ ]]; then
                LUCEBOX_HOST_VRAM_GB=$((mem_mib / 1024))
            elif [ "$LUCEBOX_HOST_GPU_SM" = "121" ] \
                 && [[ "$LUCEBOX_HOST_GPU_NAME" == *GB10* ]] \
                 && [ "$LUCEBOX_HOST_RAM_GB" -gt 16 ]; then
                # GB10 exposes one coherent CPU/GPU memory pool. NVML reports
                # memory.total as [N/A], while the CUDA runtime sees nearly all
                # system RAM. Reserve 16 GB for the OS and CPU-side buffers,
                # matching the Strix UMA planner policy.
                LUCEBOX_HOST_VRAM_GB=$((LUCEBOX_HOST_RAM_GB - 16))
                LUCEBOX_HOST_NVIDIA_UNIFIED_MEMORY=1
            else
                # Unknown/non-numeric NVML memory must never enter arithmetic
                # or be guessed as system RAM on a discrete card.
                LUCEBOX_HOST_VRAM_GB=0
            fi
            LUCEBOX_HOST_GPU_COUNT=$(printf '%s\n' "$q" | wc -l)
            LUCEBOX_HOST_NVIDIA_GPU_NAME="$LUCEBOX_HOST_GPU_NAME"
            LUCEBOX_HOST_NVIDIA_GPU_COUNT="$LUCEBOX_HOST_GPU_COUNT"
            LUCEBOX_HOST_NVIDIA_VRAM_GB="$LUCEBOX_HOST_VRAM_GB"
            LUCEBOX_HOST_NVIDIA_GPU_ARCH="$LUCEBOX_HOST_GPU_SM"
        fi
        # Multi-GPU enumeration for /props.host. The single-GPU vars
        # above (GPU_NAME / GPU_SM / VRAM_GB / DRIVER_VERSION) keep
        # describing GPU 0 for back-compat with cmd_check + autotune;
        # the full per-GPU CSV rides along separately so HOST_INFO can
        # emit the whole array.
        LUCEBOX_HOST_GPU_LIST_CSV=$(nvidia-smi \
            --query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,power.limit \
            --format=csv,noheader 2>/dev/null || echo "")
        LUCEBOX_HOST_NVIDIA_GPU_LIST_CSV="$LUCEBOX_HOST_GPU_LIST_CSV"
    fi

    # Probe AMD independently even on mixed NVIDIA + Strix systems. A working
    # NVIDIA GPU remains the default backend (RTX 3090 + Strix → cuda12), but
    # recording the AMD companion prevents the APU from confusing readiness
    # reporting and lets an explicit rocm variant remain possible.
    local amd_csv="" amd_rows="" amd_primary_selector=""
    if command -v amd-smi &>/dev/null; then
        amd_csv=$(amd-smi static --asic --vram --csv 2>/dev/null || echo "")
        if [ -n "$amd_csv" ]; then
            amd_rows=$(printf '%s\n' "$amd_csv" | _parse_amd_smi_csv)
        fi
    fi
    if [ -z "$amd_rows" ] && command -v rocm-smi &>/dev/null; then
        amd_csv=$(rocm-smi --showproductname --showmeminfo vram --csv 2>/dev/null || echo "")
        if [ -n "$amd_csv" ]; then
            amd_rows=$(printf '%s\n' "$amd_csv" | _parse_rocm_smi_csv)
        fi
    fi
    # Minimal fallback for ROCm installations without either SMI frontend.
    if [ -z "$amd_rows" ] && [ -e /dev/kfd ] && command -v rocminfo &>/dev/null; then
        local roc_arches
        roc_arches=$(rocminfo 2>/dev/null \
            | awk '/^[[:space:]]*Name:[[:space:]]+gfx[0-9a-z]+/{print $2}' \
            | awk '!seen[$0]++' || echo "")
        if [ -n "$roc_arches" ]; then
            local amd_idx=0 arch
            while IFS= read -r arch; do
                amd_rows+="${amd_idx}|AMD GPU|${arch}|0|${amd_idx}"$'\n'
                amd_idx=$((amd_idx + 1))
            done <<<"$roc_arches"
            amd_rows=${amd_rows%$'\n'}
        fi
    fi

    if [ -n "$amd_rows" ]; then
        LUCEBOX_HOST_HAS_AMD_GPU=1
        LUCEBOX_HOST_AMD_GPU_COUNT=$(printf '%s\n' "$amd_rows" | awk 'NF{n++} END{print n+0}')
        local amd_primary
        # Prefer a discrete accelerator over Strix Halo UMA even when a newer
        # firmware reports the APU's large shared-memory aperture as VRAM.
        # Within the same memory class, use the device with most physical
        # memory. This keeps R9700 + Strix builds on the faster R9700 while a
        # Strix-only machine still selects its integrated GPU.
        amd_primary=$(printf '%s\n' "$amd_rows" \
            | awk -F'|' '$3 != "gfx1151"' \
            | sort -t'|' -k4,4nr \
            | head -1)
        if [ -z "$amd_primary" ]; then
            amd_primary=$(printf '%s\n' "$amd_rows" \
                | sort -t'|' -k4,4nr \
                | head -1)
        fi
        local amd_idx amd_name amd_arch amd_mem_mib amd_selector
        IFS='|' read -r amd_idx amd_name amd_arch amd_mem_mib amd_selector <<<"$amd_primary"
        amd_primary_selector="${amd_selector:-$amd_idx}"
        LUCEBOX_HOST_AMD_GPU_NAME="$amd_name"
        LUCEBOX_HOST_AMD_GPU_ARCH="$amd_arch"
        LUCEBOX_HOST_AMD_VRAM_GB=$((amd_mem_mib / 1024))
        LUCEBOX_HOST_AMD_GPU_LIST_CSV=$(printf '%s\n' "$amd_rows" \
            | awk -F'|' '{printf "%s, , , %s, %s, %s MiB,\n", $1, $2, $3, $4}')
        # Strix Halo exposes most memory as unified system RAM, while SMI may
        # report only a 512 MiB carve-out. Use host RAM as the effective model
        # capacity on a Strix-only build; a discrete R9700 remains primary on
        # the R9700 + Strix build by the policy above.
        if [ "$amd_arch" = "gfx1151" ] \
           && [ "$LUCEBOX_HOST_AMD_VRAM_GB" -lt 12 ] \
           && [ "$LUCEBOX_HOST_RAM_GB" -ge 32 ]; then
            LUCEBOX_HOST_AMD_VRAM_GB=$LUCEBOX_HOST_RAM_GB
        fi

        if command -v amd-smi &>/dev/null; then
            LUCEBOX_HOST_ROCM_VERSION=$(amd-smi version 2>/dev/null \
                | sed -n 's/.*ROCm version: \([^ |]*\).*/\1/p' \
                | head -1 || echo "")
        fi
        if [ -z "$LUCEBOX_HOST_ROCM_VERSION" ] && command -v hipconfig &>/dev/null; then
            LUCEBOX_HOST_ROCM_VERSION=$(hipconfig --version 2>/dev/null \
                | sed 's/-.*//' | head -1 || echo "")
        fi

        if [ "$LUCEBOX_HOST_GPU_VENDOR" = "none" ]; then
            LUCEBOX_HOST_GPU_VENDOR="amd"
            LUCEBOX_HOST_GPU_NAME="$LUCEBOX_HOST_AMD_GPU_NAME"
            LUCEBOX_HOST_GPU_COUNT=$LUCEBOX_HOST_AMD_GPU_COUNT
            LUCEBOX_HOST_VRAM_GB=$LUCEBOX_HOST_AMD_VRAM_GB
            LUCEBOX_HOST_GPU_SM="$LUCEBOX_HOST_AMD_GPU_ARCH"
            LUCEBOX_HOST_GPU_LIST_CSV="$LUCEBOX_HOST_AMD_GPU_LIST_CSV"
        fi
    fi

    if [ -r /dev/kfd ] && [ -w /dev/kfd ]; then
        LUCEBOX_HOST_HAS_KFD=1
    fi
    local render_node
    for render_node in /dev/dri/renderD*; do
        [ -e "$render_node" ] || continue
        if [ -r "$render_node" ] && [ -w "$render_node" ]; then
            LUCEBOX_HOST_HAS_DRI=1
            break
        fi
    done
    # CUDA_VISIBLE_DEVICES from the caller's env (empty default = "all GPUs").
    LUCEBOX_HOST_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
    # Legacy profiles use one isolated primary device. A resolved placement
    # later clears this mask so the engine can address every named device. On
    # R9700 + Strix systems, the legacy pin remains aligned with the discrete
    # primary even when ROCm enumerates the integrated GPU first.
    LUCEBOX_HOST_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"
    LUCEBOX_HOST_ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"
    if [ -z "$LUCEBOX_HOST_HIP_VISIBLE_DEVICES" ] \
       && [ -z "$LUCEBOX_HOST_ROCR_VISIBLE_DEVICES" ]; then
        # AMD recommends ROCR_VISIBLE_DEVICES for Linux. Prefer the physical
        # GPU UUID derived from amd-smi's ASIC serial; it remains stable even
        # if SMI and ROCr enumeration orders differ. Use one isolation layer,
        # not both, so an index fallback is never interpreted twice.
        LUCEBOX_HOST_ROCR_VISIBLE_DEVICES="$amd_primary_selector"
    fi

    # A mixed CUDA/HIP process cannot be assembled from a single-backend
    # Docker image. Buyers may receive the paired native runtime under
    # /opt/lucebox/runtime; contributors get the same contract after building
    # both backends in their checkout. Record it only when both executables
    # exist so Automatic never emits an unlaunchable cross-vendor plan.
    if [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ] \
       && [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ]; then
        local hybrid_repo="" hybrid_server="" hybrid_ipc=""
        local hybrid_dir="" hybrid_entrypoint=""
        hybrid_server="${LUCEBOX_HYBRID_SERVER_BIN:-}"
        hybrid_ipc="${LUCEBOX_HYBRID_IPC_BIN:-}"
        hybrid_dir="${LUCEBOX_HYBRID_DFLASH_DIR:-}"
        hybrid_entrypoint="${LUCEBOX_HYBRID_ENTRYPOINT:-}"
        if [ -z "$hybrid_entrypoint" ] && [ -n "$hybrid_dir" ]; then
            hybrid_entrypoint="$hybrid_dir/scripts/entrypoint.sh"
        fi
        if ! _native_binary_ready "$hybrid_server" \
           || ! _native_binary_ready "$hybrid_ipc" \
           || [ ! -f "$hybrid_entrypoint" ]; then
            hybrid_repo=$(_find_repo_root 2>/dev/null || true)
            if [ -n "$hybrid_repo" ]; then
                hybrid_server="$hybrid_repo/server/build-cuda/dflash_server"
                hybrid_ipc="$hybrid_repo/server/build-hip/backend_ipc_daemon"
                hybrid_dir="$hybrid_repo/server"
                hybrid_entrypoint="$hybrid_dir/scripts/entrypoint.sh"
            fi
        fi
        if ! _native_binary_ready "$hybrid_server" \
           || ! _native_binary_ready "$hybrid_ipc" \
           || [ ! -f "$hybrid_entrypoint" ]; then
            hybrid_server="/opt/lucebox/runtime/cuda/dflash_server"
            hybrid_ipc="/opt/lucebox/runtime/hip/backend_ipc_daemon"
            hybrid_dir="/opt/lucebox/runtime/server"
            hybrid_entrypoint="$hybrid_dir/scripts/entrypoint.sh"
        fi
        if _native_binary_ready "$hybrid_server" \
           && _native_binary_ready "$hybrid_ipc" \
           && [ -f "$hybrid_entrypoint" ]; then
            LUCEBOX_HOST_HAS_HYBRID_RUNTIME=1
            LUCEBOX_HOST_HYBRID_SERVER_BIN="$hybrid_server"
            LUCEBOX_HOST_HYBRID_IPC_BIN="$hybrid_ipc"
            LUCEBOX_HOST_HYBRID_DFLASH_DIR="$hybrid_dir"
            LUCEBOX_HOST_HYBRID_ENTRYPOINT="$hybrid_entrypoint"
        fi
    fi

    # OS / kernel identity. /etc/os-release is the freedesktop spec for
    # "what distro is this?" and we keep PRETTY_NAME verbatim (it already
    # includes the version, e.g. "Ubuntu 22.04.3 LTS").
    LUCEBOX_HOST_OS_PRETTY=""
    if [ -r /etc/os-release ]; then
        # shellcheck source=/dev/null
        LUCEBOX_HOST_OS_PRETTY=$(. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-}")
    fi
    LUCEBOX_HOST_KERNEL=$(uname -r 2>/dev/null || echo "")

    # WSL version detection. "wsl2" matches the kernel-side string the
    # MS-shipped WSL2 kernel embeds; "wsl1" is what the legacy translation
    # layer writes. Anything else stays empty (= not WSL).
    LUCEBOX_HOST_WSL_VERSION=""
    if [ -r /proc/version ]; then
        if grep -q "microsoft-standard-WSL2" /proc/version 2>/dev/null; then
            LUCEBOX_HOST_WSL_VERSION="wsl2"
        elif grep -qi "Microsoft" /proc/version 2>/dev/null; then
            LUCEBOX_HOST_WSL_VERSION="wsl1"
        fi
    fi

    # CPU model — first "model name" hit in /proc/cpuinfo. Cheaper than
    # lscpu and keeps the bash side dep-free.
    LUCEBOX_HOST_CPU_MODEL=""
    if [ -r /proc/cpuinfo ]; then
        LUCEBOX_HOST_CPU_MODEL=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "")
    fi

    LUCEBOX_HOST_HAS_SYSTEMD=0
    if command -v systemctl &>/dev/null && systemctl --user show-environment &>/dev/null; then
        LUCEBOX_HOST_HAS_SYSTEMD=1
    fi

    LUCEBOX_HOST_IS_WSL=0
    if grep -qi microsoft /proc/version 2>/dev/null \
       || [ -e /proc/sys/fs/binfmt_misc/WSLInterop ]; then
        LUCEBOX_HOST_IS_WSL=1
    fi

    LUCEBOX_HOST_HAS_DOCKER=0
    LUCEBOX_HOST_DOCKER_VERSION=""
    if command -v docker &>/dev/null && docker ps &>/dev/null; then
        LUCEBOX_HOST_HAS_DOCKER=1
        LUCEBOX_HOST_DOCKER_VERSION=$(timeout 5 docker version --format '{{.Server.Version}}' 2>/dev/null || echo "")
    fi

    LUCEBOX_HOST_HAS_CTK="none"
    if [ "$LUCEBOX_HOST_HAS_DOCKER" = "1" ]; then
        if command -v nvidia-container-runtime &>/dev/null; then
            LUCEBOX_HOST_HAS_CTK="runtime"
        elif command -v nvidia-ctk &>/dev/null \
             && nvidia-ctk cdi list 2>/dev/null | grep -q 'nvidia.com/gpu'; then
            LUCEBOX_HOST_HAS_CTK="cdi"
        elif command -v nvidia-ctk &>/dev/null; then
            LUCEBOX_HOST_HAS_CTK="installed-unwired"
        fi
    fi

    # NVIDIA Container Toolkit version (best-effort; empty when nvidia-ctk
    # is not installed). nvidia-ctk --version prints "NVIDIA Container
    # Toolkit CLI version 1.16.2" on a single line — extract the trailing
    # token so the host-info JSON carries just the version, not the banner.
    LUCEBOX_HOST_NVIDIA_CTK_VERSION=""
    if command -v nvidia-ctk &>/dev/null; then
        LUCEBOX_HOST_NVIDIA_CTK_VERSION=$(nvidia-ctk --version 2>/dev/null \
            | awk '/version/{print $NF; exit}' \
            || echo "")
    fi

    export LUCEBOX_HOST_NPROC LUCEBOX_HOST_RAM_GB LUCEBOX_HOST_GPU_VENDOR
    export LUCEBOX_HOST_HAS_NVIDIA_GPU LUCEBOX_HOST_HAS_AMD_GPU
    export LUCEBOX_HOST_GPU_NAME LUCEBOX_HOST_GPU_COUNT LUCEBOX_HOST_VRAM_GB
    export LUCEBOX_HOST_GPU_SM LUCEBOX_HOST_DRIVER_VERSION LUCEBOX_HOST_DRIVER_MAJOR
    export LUCEBOX_HOST_HAS_SYSTEMD LUCEBOX_HOST_IS_WSL
    export LUCEBOX_HOST_HAS_DOCKER LUCEBOX_HOST_DOCKER_VERSION
    export LUCEBOX_HOST_HAS_CTK
    export LUCEBOX_HOST_ROCM_VERSION LUCEBOX_HOST_HAS_KFD LUCEBOX_HOST_HAS_DRI
    export LUCEBOX_HOST_AMD_GPU_NAME LUCEBOX_HOST_AMD_GPU_COUNT
    export LUCEBOX_HOST_AMD_VRAM_GB LUCEBOX_HOST_AMD_GPU_ARCH
    export LUCEBOX_HOST_AMD_GPU_LIST_CSV
    export LUCEBOX_HOST_OS_PRETTY LUCEBOX_HOST_KERNEL LUCEBOX_HOST_WSL_VERSION
    export LUCEBOX_HOST_NVIDIA_CTK_VERSION LUCEBOX_HOST_CPU_MODEL
    export LUCEBOX_HOST_GPU_LIST_CSV LUCEBOX_HOST_CUDA_VISIBLE_DEVICES
    export LUCEBOX_HOST_HIP_VISIBLE_DEVICES LUCEBOX_HOST_ROCR_VISIBLE_DEVICES
    export LUCEBOX_HOST_NVIDIA_GPU_NAME LUCEBOX_HOST_NVIDIA_GPU_COUNT
    export LUCEBOX_HOST_NVIDIA_VRAM_GB LUCEBOX_HOST_NVIDIA_GPU_ARCH
    export LUCEBOX_HOST_NVIDIA_GPU_LIST_CSV
    export LUCEBOX_HOST_NVIDIA_UNIFIED_MEMORY
    export LUCEBOX_HOST_HAS_HYBRID_RUNTIME
    export LUCEBOX_HOST_HYBRID_SERVER_BIN LUCEBOX_HOST_HYBRID_IPC_BIN
    export LUCEBOX_HOST_HYBRID_DFLASH_DIR LUCEBOX_HOST_HYBRID_ENTRYPOINT
    _LUCEBOX_HOST_PROBED=1
}

# Cheap idempotency wrapper. Anything that needs real host facts (vs the safe
# defaults seeded at script-load) calls this. Subcommands that go straight to
# `systemctl`/`journalctl` no longer need to remember to call probe_host.
ensure_probed() {
    [ "$_LUCEBOX_HOST_PROBED" = "1" ] || probe_host
}

pick_variant() {
    # Explicit env/config always wins. On a fresh install choose the backend
    # from hardware: a working NVIDIA GPU takes priority on RTX + Strix builds;
    # otherwise an AMD GPU selects ROCm (R9700 + Strix and Strix-only builds).
    local configured
    if [ -n "${LUCEBOX_VARIANT:-}" ]; then
        printf '%s' "$LUCEBOX_VARIANT"
        return
    fi
    configured=$(_lucebox_config_get image.variant)
    if [ -n "$configured" ]; then
        # cuda12 used to contain sm_120. New releases keep that newer toolkit
        # in cuda128 so RTX 20/30/40 users retain the r525 driver floor. Migrate
        # the old moving tag in memory; explicit LUCEBOX_VARIANT still wins.
        if [ "$configured" = "cuda12" ]; then
            ensure_probed
            if [ "$LUCEBOX_HOST_GPU_SM" = "120" ]; then
                printf 'cuda128'
                return
            fi
            if [ "$LUCEBOX_HOST_GPU_SM" = "121" ]; then
                case "$(uname -m 2>/dev/null || echo unknown)" in
                    aarch64|arm64) printf 'cuda13'; return ;;
                esac
            fi
        fi
        printf '%s' "$configured"
        return
    fi
    ensure_probed
    if [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ]; then
        _default_cuda_variant
    elif [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ]; then
        printf 'rocm'
    else
        # Keep the historical default so `lucebox check` can still tell a
        # GPU-less host what image it would otherwise use.
        printf 'cuda12'
    fi
}

_default_cuda_variant() {
    # Each Blackwell target needs a newer compiler/runtime than the broad
    # CUDA-12 image. Keep that newer driver floor isolated from Turing through
    # Hopper hosts, which remain on cuda12.
    local machine
    machine=$(uname -m 2>/dev/null || echo unknown)
    if [ "$LUCEBOX_HOST_GPU_SM" = "121" ]; then
        case "$machine" in
            aarch64|arm64) printf 'cuda13'; return ;;
        esac
    fi
    if [ "$LUCEBOX_HOST_GPU_SM" = "120" ]; then
        printf 'cuda128'
        return
    fi
    printf 'cuda12'
}

_variant_is_rocm() {
    # Variant names are not limited to the moving `rocm` tag. Releases and
    # CI produce tags such as `0.3.0-rocm` and `pr-335-rocm`; treat any tag
    # containing the backend marker as ROCm. Spell out case-insensitivity so
    # host-only commands such as `check` reach no Bash-4-only expansion. Full
    # container dispatch still requires Bash 4.3+ for the argv namerefs below.
    case "$1" in
        *[Rr][Oo][Cc][Mm]*) return 0 ;;
        *)                  return 1 ;;
    esac
}

_variant_is_cuda13() {
    case "$1" in
        *[Cc][Uu][Dd][Aa]13*) return 0 ;;
        *)                    return 1 ;;
    esac
}

_variant_is_cuda128() {
    case "$1" in
        *[Cc][Uu][Dd][Aa]128*) return 0 ;;
        *)                     return 1 ;;
    esac
}

# ── prereq checks (host-only) ─────────────────────────────────────────────
# Print-and-exit on anything that needs root to install. The Python CLI does
# the richer reporting; this is the bare minimum to make `docker run` viable.

require_host_prereqs() {
    ensure_probed
    local variant="${1:-}"
    [ -n "$variant" ] || variant=$(pick_variant)
    local missing=0
    if ! command -v docker &>/dev/null; then
        err "docker is not installed"
        hint "Install: https://docs.docker.com/engine/install/"
        missing=1
    elif ! docker ps &>/dev/null; then
        err "docker daemon not reachable"
        hint "sudo systemctl start docker   (or: add your user to the 'docker' group, then re-login)"
        missing=1
    fi

    if _variant_is_rocm "$variant"; then
        if [ "$LUCEBOX_HOST_HAS_AMD_GPU" != "1" ]; then
            err "ROCm image selected but no working AMD GPU was detected"
            hint "Install ROCm/amd-smi, or choose LUCEBOX_VARIANT=cuda12 on an NVIDIA build."
            missing=1
        fi
        if [ "$LUCEBOX_HOST_HAS_KFD" != "1" ]; then
            err "/dev/kfd is missing or not accessible"
            hint "Add the user to the render group, then re-login: sudo usermod -aG render \"$USER\""
            missing=1
        fi
        if [ "$LUCEBOX_HOST_HAS_DRI" != "1" ]; then
            err "no accessible /dev/dri/renderD* device was found"
            hint "Add the user to the render and video groups, then re-login."
            missing=1
        fi
    else
        if [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" != "1" ]; then
            err "CUDA image selected but no working NVIDIA GPU was detected"
            hint "Install the NVIDIA driver, or choose LUCEBOX_VARIANT=rocm on an AMD build."
            missing=1
        fi
    fi

    [ "$missing" = "0" ] || exit 1
}

require_ctk() {
    local variant="${1:-$(pick_variant)}"
    _variant_is_rocm "$variant" && return 0
    case "$LUCEBOX_HOST_HAS_CTK" in
        runtime|cdi) return 0 ;;
        installed-unwired)
            err "NVIDIA Container Toolkit installed but not wired into docker"
            hint "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            hint "  or generate a CDI spec: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
            exit 1 ;;
        none|*)
            err "NVIDIA Container Toolkit not installed"
            hint "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
            hint "Then register with docker:"
            hint "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            exit 1 ;;
    esac
}

require_systemd() {
    # Earlier versions of this wrapper had `start`/`stop`/`logs`/etc. drop
    # straight into cmd_systemctl_passthrough without probing first, which
    # tripped `set -u` on the reference below. Two layers of defence now:
    #   1) top-of-script seeds LUCEBOX_HOST_HAS_SYSTEMD=0 unconditionally, so
    #      no read can be unbound even if probe_host is bypassed entirely.
    #   2) ensure_probed runs probe_host on first call so we still get the
    #      real answer for the require_systemd error path.
    ensure_probed
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" != "1" ]; then
        err "user systemd is not available — required for $1"
        hint "On WSL: set 'systemd=true' under [boot] in /etc/wsl.conf, then 'wsl --shutdown'."
        hint "Otherwise: install systemd, or run '$SCRIPT_NAME serve' to run in the foreground without systemd."
        exit 1
    fi
}

# ── docker run construction ───────────────────────────────────────────────
# All the Python-CLI subcommands share the same docker run incantation:
# mount the host docker socket (so the in-container CLI can spawn server /
# bench containers on the host daemon), mount only Lucebox's config/models
# state, and pass host facts via env. The selected image gets its native
# accelerator contract: --gpus all for
# CUDA; /dev/kfd + /dev/dri and the render/video groups for ROCm.

DOCKER_SOCK_PATH="${DOCKER_HOST:-/var/run/docker.sock}"
DOCKER_SOCK_PATH="${DOCKER_SOCK_PATH#unix://}"

# Append `-e LUCEBOX_HOST_<x>=<val>` for every exported host fact onto the
# named docker-argv array (bash 4.3+ nameref). The Python side reads these
# instead of reprobing — see build_orchestrator_argv / cmd_exec_in_container.
_append_host_env() {
    # shellcheck disable=SC2178  # nameref to a caller's array, not a string
    local -n _arr="$1"
    local var
    for var in $(compgen -e | grep '^LUCEBOX_HOST_' || true); do
        _arr+=(-e "$var=${!var}")
    done
}

# Override the generic primary-GPU facts when the user deliberately selects
# the AMD backend on a mixed NVIDIA + AMD machine. probe_host keeps NVIDIA as
# the mixed-build default, but Automatic must tune the accelerator that will
# actually run the model.
_append_selected_backend_facts() {  # usage: arrayname variant
    # shellcheck disable=SC2178
    local -n _facts_arr="$1"
    local variant="$2"
    if _variant_is_rocm "$variant" && [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ]; then
        _facts_arr+=(
            -e "LUCEBOX_HOST_GPU_VENDOR=amd"
            -e "LUCEBOX_HOST_GPU_NAME=$LUCEBOX_HOST_AMD_GPU_NAME"
            -e "LUCEBOX_HOST_GPU_COUNT=$LUCEBOX_HOST_AMD_GPU_COUNT"
            -e "LUCEBOX_HOST_VRAM_GB=$LUCEBOX_HOST_AMD_VRAM_GB"
            -e "LUCEBOX_HOST_GPU_SM=$LUCEBOX_HOST_AMD_GPU_ARCH"
            -e "LUCEBOX_HOST_GPU_LIST_CSV=$LUCEBOX_HOST_AMD_GPU_LIST_CSV"
        )
    fi
}

# Append the LUCEBOX_* scalar overrides (image/variant/port/container/models)
# plus the optional HF_TOKEN guard onto the named docker-argv array. Shared
# by the docker-run (build_orchestrator_argv) and docker-exec
# (cmd_exec_in_container) paths so both forward an identical env subset.
_append_scalar_env() {
    # shellcheck disable=SC2178  # nameref to a caller's array, not a string
    local -n _arr="$1"
    local variant="$2"
    _arr+=(-e "LUCEBOX_IMAGE=$IMAGE_BASE")
    _arr+=(-e "LUCEBOX_VARIANT=$variant")
    _arr+=(-e "LUCEBOX_PORT=$DEFAULT_PORT")
    _arr+=(-e "LUCEBOX_CONTAINER=$CONTAINER_NAME")
    _arr+=(-e "LUCEBOX_MODELS=$DEFAULT_MODELS_DIR")
    _arr+=(-e "LUCEBOX_HOME=$CONFIG_HOME")
    [ -n "${HF_TOKEN:-}" ] && _arr+=(-e "HF_TOKEN=$HF_TOKEN")
    return 0
}

# Append the Docker accelerator contract for the chosen image variant.
# CUDA and ROCm use fundamentally different runtime flags; keeping this in
# one helper prevents the orchestrator, canonical server argv, and fallback
# server path from drifting apart.
_append_gpu_args() {  # usage: _append_gpu_args arrayname variant
    # shellcheck disable=SC2178
    local -n _gpu_arr="$1"
    local variant="$2"
    if _variant_is_rocm "$variant"; then
        _gpu_arr+=(
            --device /dev/kfd
            --device /dev/dri
            --group-add video
            --group-add render
            --security-opt seccomp=unconfined
        )
        if [ -n "${LUCEBOX_HOST_ROCR_VISIBLE_DEVICES:-}" ]; then
            _gpu_arr+=(-e "ROCR_VISIBLE_DEVICES=$LUCEBOX_HOST_ROCR_VISIBLE_DEVICES")
        elif [ -n "${LUCEBOX_HOST_HIP_VISIBLE_DEVICES:-}" ]; then
            _gpu_arr+=(
                -e "HIP_VISIBLE_DEVICES=$LUCEBOX_HOST_HIP_VISIBLE_DEVICES"
            )
        fi
    else
        _gpu_arr+=(--gpus all)
        if [ -n "${LUCEBOX_HOST_CUDA_VISIBLE_DEVICES:-}" ]; then
            _gpu_arr+=(-e "CUDA_VISIBLE_DEVICES=$LUCEBOX_HOST_CUDA_VISIBLE_DEVICES")
        fi
    fi
}

# Pick docker's interactive flags: -it on a real tty, -i otherwise.
# Writes into a caller-supplied array via nameref. This MUST run in the
# caller's scope (not a subshell or `< <(...)` process substitution): the
# `[ -t 1 ]` test inspects fd 1, and inside a process substitution fd 1 is
# the pipe to the consumer, not the terminal — which would force -i even on
# a real tty and break the interactive client TUIs (lucebox claude, etc.).
_set_tty_flags() {  # usage: _set_tty_flags arrayname
    # shellcheck disable=SC2178
    local -n _a="$1"
    if [ -t 0 ] && [ -t 1 ]; then
        _a=(-it)
    else
        _a=(-i)
    fi
}

build_orchestrator_argv() {
    local variant="$1" caller_has_tty="$2"; shift 2
    local cli_group="${1:-}" cli_action="${2:-}"
    local tty=(-i)
    [ "$caller_has_tty" = "1" ] && tty=(-it)
    local argv=(docker run --rm "${tty[@]}")
    _append_gpu_args argv "$variant"
    argv+=(--name "${CONTAINER_NAME}-cli-$$")
    argv+=(--user "$(id -u):$(id -g)")
    # Native/hybrid servers listen on the host loopback rather than in a
    # Lucebox container. The internal calibration probe needs that namespace
    # when it cannot use docker exec against a running inference container.
    if [ "$cli_group" = "_calibration" ] && [ "$cli_action" = "probe" ]; then
        argv+=(--network host)
    fi
    # Only bind-mount the docker socket when DOCKER_HOST actually points
    # at a unix socket on this host. With DOCKER_HOST=tcp://… or ssh://…
    # the path we'd construct is `tcp` or empty, and `docker run -v` would
    # bark with an "invalid mount" error before the orchestrator even
    # starts. The orchestrator-in-container relies on docker access only
    # when actually needed; pulling that mount when the host talks to
    # docker over TCP/SSH is fine.
    if [ -S "$DOCKER_SOCK_PATH" ]; then
        local socket_gid
        socket_gid=$(stat -c '%g' "$DOCKER_SOCK_PATH" 2>/dev/null \
            || stat -f '%g' "$DOCKER_SOCK_PATH" 2>/dev/null \
            || echo "")
        [ -n "$socket_gid" ] && argv+=(--group-add "$socket_gid")
        argv+=(-v "$DOCKER_SOCK_PATH:/var/run/docker.sock")
    fi
    # Bind only Lucebox-owned state. The orchestrator used to receive all of
    # $HOME read-write to support model symlinks; the Python launch builder
    # now handles selected symlinks with narrow read-only mounts instead.
    # Keeping credentials and unrelated user files outside the container is
    # the safer default for both buyer appliances and contributor machines.
    mkdir -p "$DEFAULT_MODELS_DIR"
    argv+=(-v "$DEFAULT_MODELS_DIR:$DEFAULT_MODELS_DIR")
    # A custom LUCEBOX_HOME may sit anywhere, so mount it explicitly and use
    # it as the ephemeral CLI's HOME (its caches then remain app-scoped too).
    # Use an image-owned working directory: callers may invoke lucebox from
    # /tmp or another path that is not bind-mounted into the container.
    mkdir -p "$CONFIG_HOME"
    argv+=(-v "$CONFIG_HOME:$CONFIG_HOME")
    argv+=(-w /opt/lucebox-hub)
    argv+=(-e "HOME=$CONFIG_HOME")
    # Host facts — Python side reads these instead of reprobing.
    _append_host_env argv
    _append_selected_backend_facts argv "$variant"
    # User overrides for image/port/container/models scalars + HF_TOKEN.
    # Always exports the resolved models dir so the in-container CLI sees
    # the same path the wrapper mounts (the XDG default flows through too).
    _append_scalar_env argv "$variant"

    argv+=("${IMAGE_BASE}:${variant}")
    # `lucebox` is the entrypoint subcommand handled by server/scripts/entrypoint.sh
    # — it execs `python -m lucebox` with whatever args we pass on.
    argv+=(lucebox "$@")
    printf '%s\n' "${argv[@]}"
}

# ── subcommand implementations ────────────────────────────────────────────

_config_requires_hybrid_runtime() {
    local value target target_devices remote target_backend remote_backend
    for value in \
        "$(_lucebox_config_get placement.remote_draft)" \
        "$(_lucebox_config_get placement.remote_target_shard)"; do
        case "$value" in true|1|yes|on) return 0 ;; esac
    done

    remote=$(_lucebox_config_get placement.remote_expert_device)
    [ -n "$remote" ] || return 1
    target=$(_lucebox_config_get placement.target_device)
    if [ -z "$target" ]; then
        target_devices=$(_toml_array_to_csv \
            "$(_lucebox_config_get placement.target_devices)")
        target="${target_devices%%,*}"
    fi
    [ -n "$target" ] || return 1
    target_backend="${target%%:*}"
    remote_backend="${remote%%:*}"
    [ "$target_backend" != "$remote_backend" ]
}

_validate_hybrid_profile() {
    local target target_devices draft remote_expert value device
    local remote_draft=0 remote_target=0 seen_remote=0
    local hybrid_targets=()
    target=$(_lucebox_config_get placement.target_device)
    target_devices=$(_toml_array_to_csv \
        "$(_lucebox_config_get placement.target_devices)")
    if [ -z "$target" ]; then
        target="${target_devices%%,*}"
    fi
    [[ "$target" =~ ^cuda:[0-9]+$ ]] \
        || die "the installed hybrid runtime requires a CUDA target server (got '${target:-none}')"

    value=$(_lucebox_config_get placement.remote_draft)
    case "$value" in true|1|yes|on) remote_draft=1 ;; esac
    value=$(_lucebox_config_get placement.remote_target_shard)
    case "$value" in true|1|yes|on) remote_target=1 ;; esac

    if [ "$remote_draft" = "1" ]; then
        draft=$(_lucebox_config_get placement.draft_device)
        [[ "$draft" =~ ^hip:[0-9]+$ ]] \
            || die "the installed hybrid runtime requires the remote draft on HIP (got '${draft:-none}')"
    fi

    if [ "$remote_target" = "1" ]; then
        [ -n "$target_devices" ] \
            || die "remote target sharding requires placement.target_devices"
        IFS=',' read -r -a hybrid_targets <<<"$target_devices"
        for device in "${hybrid_targets[@]}"; do
            if [[ "$device" =~ ^cuda:[0-9]+$ ]]; then
                [ "$seen_remote" = "0" ] \
                    || die "hybrid target devices must list CUDA shards before HIP shards"
            elif [[ "$device" =~ ^hip:[0-9]+$ ]]; then
                seen_remote=1
            else
                die "bad hybrid target device '$device'"
            fi
        done
        [ "$seen_remote" = "1" ] \
            || die "remote target sharding requires at least one HIP shard"
    fi

    remote_expert=$(_lucebox_config_get placement.remote_expert_device)
    if [ -n "$remote_expert" ] \
       && [ "${remote_expert%%:*}" != "${target%%:*}" ]; then
        [[ "$remote_expert" =~ ^hip:[0-9]+$ ]] \
            || die "the installed hybrid runtime requires remote Spark experts on HIP (got '$remote_expert')"
    fi
}

cmd_hybrid_serve() {
    ensure_probed
    [ "$LUCEBOX_HOST_HAS_HYBRID_RUNTIME" = "1" ] \
        || die "this profile needs the paired CUDA + HIP runtime, but it is not installed"
    [ -x "$LUCEBOX_HOST_HYBRID_SERVER_BIN" ] \
        || die "hybrid server is missing: $LUCEBOX_HOST_HYBRID_SERVER_BIN"
    [ -x "$LUCEBOX_HOST_HYBRID_IPC_BIN" ] \
        || die "hybrid backend daemon is missing: $LUCEBOX_HOST_HYBRID_IPC_BIN"
    [ -f "$LUCEBOX_HOST_HYBRID_ENTRYPOINT" ] \
        || die "hybrid runtime entrypoint is missing: $LUCEBOX_HOST_HYBRID_ENTRYPOINT"
    _validate_hybrid_profile

    if [ -z "${INVOCATION_ID:-}" ] \
       && systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
        die "$UNIT_NAME is already running; use '$SCRIPT_NAME restart' or '$SCRIPT_NAME logs'"
    fi
    if _lucebox_container_running; then
        die "container '$CONTAINER_NAME' is already running; stop it before starting the hybrid runtime"
    fi

    local selected=() target draft model_id
    mapfile -t selected < <(_selected_model_paths)
    target="${selected[0]:-}"
    draft="${selected[1]:-none}"
    model_id="${selected[2]:-lucebox}"
    _model_artifact_ready "$target" \
        || die "selected target is not installed: $target — run '$SCRIPT_NAME models select'"
    if [ "$draft" != "none" ] && ! _model_artifact_ready "$draft"; then
        die "selected draft is not installed: $draft — run '$SCRIPT_NAME models select'"
    fi

    _export_native_config
    export DFLASH_DIR="$LUCEBOX_HOST_HYBRID_DFLASH_DIR"
    export DFLASH_SERVER_BIN="$LUCEBOX_HOST_HYBRID_SERVER_BIN"
    export DFLASH_BACKEND_IPC_BIN="$LUCEBOX_HOST_HYBRID_IPC_BIN"
    export DFLASH_TARGET="$target"
    _export_selected_decode_companion "$model_id" "$draft"
    export DFLASH_HOST="${LUCEBOX_NATIVE_HOST:-127.0.0.1}"
    export DFLASH_PORT="$DEFAULT_PORT"
    export DFLASH_MODEL_NAME="$model_id"
    export LUCEBOX_NATIVE=1
    info "Starting topology-aware CUDA + HIP engine at http://$DFLASH_HOST:$DFLASH_PORT"
    exec bash "$LUCEBOX_HOST_HYBRID_ENTRYPOINT" serve
}

cmd_serve() {
    # Long-running foreground server. Also what systemd's ExecStart= calls.
    #
    # Two-stage so config.toml takes effect:
    #   1. Run an ephemeral orchestrator container that emits the canonical
    #      server docker-run argv from .lucebox/config.toml (one arg per
    #      line on stdout).
    #   2. Exec that argv.
    #
    # If stage 1 fails (image not pulled yet, no config), fall back to a
    # conservative docker run — the container's own VRAM-tiered autotune
    # picks reasonable defaults from there.
    ensure_probed
    local variant
    variant=$(pick_variant)
    if _config_requires_hybrid_runtime; then
        cmd_hybrid_serve
        return $?
    fi
    require_host_prereqs "$variant"
    require_ctk "$variant"

    # Pre-flight: refuse to stomp on something that's already serving this
    # slot. Three states to distinguish, because silently `docker rm -f`-ing
    # whatever is there hides real bugs (e.g. the user forgot they had a
    # systemd unit up, and we'd happily race two servers on the same port):
    #
    #   1. systemd unit active           → refuse, redirect to `logs`/`stop`
    #   2. container running (no systemd)→ refuse, redirect to `docker logs`
    #   3. container present but stopped → orphan from a SIGKILLed previous
    #      run (docker run --rm only cleans up on clean exit). Remove it,
    #      but TELL the user — they need to know their last run died dirty.
    # CRITICAL: when systemd invokes US as the unit's ExecStart, is-active
    # returns true *because of us* — refusing here would deadlock the unit
    # in a restart loop (and historically did — commit a30dbe5 shipped this
    # bug). systemd sets $INVOCATION_ID in every service exec, so its
    # presence is the unambiguous "I am running as the systemd ExecStart"
    # signal. Skip the unit-active check in that case; the container-state
    # check below still catches a stale container holding the slot.
    if [ -z "${INVOCATION_ID:-}" ] \
       && systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
        err "${UNIT_NAME} is already running under systemd."
        hint "  $SCRIPT_NAME logs            # follow the journal"
        hint "  $SCRIPT_NAME restart         # bounce the service"
        hint "  $SCRIPT_NAME stop            # stop the service"
        exit 1
    fi
    local container_state
    container_state=$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo absent)
    case "$container_state" in
        absent)
            ;;
        running|restarting)
            err "Container '$CONTAINER_NAME' is already running (outside systemd)."
            hint "  docker logs -f $CONTAINER_NAME    # follow output"
            hint "  $SCRIPT_NAME stop                # stop it"
            exit 1
            ;;
        exited|created|paused|dead)
            info "Removing stale '$CONTAINER_NAME' container (state=$container_state, likely from a previous unclean exit)"
            docker rm -f "$CONTAINER_NAME" >/dev/null
            ;;
        *)
            warn "Container '$CONTAINER_NAME' is in unexpected state '$container_state' — removing"
            docker rm -f "$CONTAINER_NAME" >/dev/null
            ;;
    esac

    local orch_argv server_argv server_output orch_error_file orch_rc=0
    mapfile -t orch_argv < <(build_orchestrator_argv "$variant" 0 print-serve-argv)

    orch_error_file=$(mktemp -t lucebox-orchestrator.XXXXXX) \
        || die "couldn't create temporary orchestrator log"
    if server_output=$("${orch_argv[@]}" 2>"$orch_error_file"); then
        if [ -n "$server_output" ]; then
            mapfile -t server_argv <<<"$server_output"
            if [ "${#server_argv[@]}" -gt 0 ] \
               && [ "${server_argv[0]}" = "docker" ]; then
                rm -f "$orch_error_file"
                info "Starting lucebox server (variant=$variant, from config.toml)"
                _serve_and_track "${server_argv[@]}"
                return $?
            fi
        fi
    else
        orch_rc=$?
    fi

    # Configuration/path errors must never turn into a successful launch with
    # defaults: that could start the wrong model or ignore explicit tuning.
    # Docker infrastructure failures (for example an image not pulled yet)
    # retain the conservative fallback below.
    if [ "$orch_rc" -eq 2 ] \
       || grep -qE 'Invalid configuration:|Cannot build server command:' "$orch_error_file"; then
        [ ! -s "$orch_error_file" ] || cat "$orch_error_file" >&2
        rm -f "$orch_error_file"
        die "refusing to ignore invalid Lucebox configuration"
    fi
    rm -f "$orch_error_file"

    warn "Couldn't fetch server argv from container (image not pulled?) — using fallback"
    info "Starting lucebox server (variant=$variant, port=$DEFAULT_PORT, defaults only)"
    local fallback_models="$DEFAULT_MODELS_DIR"
    mkdir -p "$fallback_models"
    # Forward host facts even on the fallback path so the in-container
    # entrypoint can still write /opt/lucebox-hub/HOST_INFO from the host's
    # view of the rig. Matches the orchestrator path (see
    # build_orchestrator_argv) — without it, HOST_INFO would be written
    # with "source: unknown" any time print-serve-argv fails.
    local fallback_argv=(docker run --rm
        --name "$CONTAINER_NAME"
        -p "$DEFAULT_PORT:8080"
        -v "$CONFIG_HOME:$CONFIG_HOME"
        -v "$fallback_models:/opt/lucebox-hub/server/models"
        -e "HOME=$CONFIG_HOME")
    _append_gpu_args fallback_argv "$variant"
    _append_host_env fallback_argv
    _append_selected_backend_facts fallback_argv "$variant"
    _append_scalar_env fallback_argv "$variant"
    fallback_argv+=("${IMAGE_BASE}:${variant}")
    _serve_and_track "${fallback_argv[@]}"
}

# Foreground server runner with controlling-process lifetime semantics:
# the docker daemon owns containers independently of the CLI, so a bare
# `exec docker run` leaves the container alive after the wrapper's parent
# (a terminal, a systemd unit, anything) goes away. `docker run --rm` only
# cleans up on the container's own clean exit, not on our death.
#
# Fix: run docker as a child, install signal traps that issue `docker stop`
# before exiting. Now `lucebox serve` behaves like a normal foreground
# program — close the terminal, kill the wrapper, send SIGTERM from
# systemd, the container goes down with it.
#
# Stops also from EXIT so even a `set -e` propagation cleans up.
_serve_and_track() {
    "$@" &
    local docker_pid=$!
    # shellcheck disable=SC2317  # called via trap, not "unreachable"
    _serve_stop() {
        trap - HUP INT TERM EXIT
        # Best-effort: container may already be exiting / never started.
        # `docker stop` blocks up to -t seconds for graceful shutdown
        # (server handles SIGTERM), then SIGKILLs. 10s is enough for the
        # in-flight request to finish on a typical decode.
        docker stop -t 10 "$CONTAINER_NAME" >/dev/null 2>&1 || true
        # docker_pid may be out of scope when this fires as the EXIT trap
        # after _serve_and_track has unwound (e.g. the server fast-failed);
        # guard so `set -u` doesn't turn cleanup into its own error.
        [ -n "${docker_pid:-}" ] && wait "$docker_pid" 2>/dev/null || true
    }
    trap _serve_stop HUP INT TERM EXIT
    wait "$docker_pid"
    local rc=$?
    trap - HUP INT TERM EXIT
    return $rc
}

cmd_systemd_install() {
    ensure_probed
    local variant unit_after unit_wants unit_pre="" unit_stop="" unit_hybrid_env=""
    variant=$(pick_variant)
    require_systemd "service install"
    unit_after="network-online.target"
    unit_wants="network-online.target"
    if _config_requires_hybrid_runtime; then
        [ "$LUCEBOX_HOST_HAS_HYBRID_RUNTIME" = "1" ] \
            || die "install the paired CUDA + HIP runtime before enabling this profile"
        _validate_hybrid_profile
        # A contributor build may live in a checkout that systemd cannot
        # rediscover from its working directory. Persist the already-validated
        # executable contract in the unit; factory installs use the same lines
        # with their stable /opt/lucebox/runtime paths.
        unit_hybrid_env=$(printf '%s\n%s\n%s\n%s' \
            "Environment=LUCEBOX_HYBRID_SERVER_BIN=$LUCEBOX_HOST_HYBRID_SERVER_BIN" \
            "Environment=LUCEBOX_HYBRID_IPC_BIN=$LUCEBOX_HOST_HYBRID_IPC_BIN" \
            "Environment=LUCEBOX_HYBRID_DFLASH_DIR=$LUCEBOX_HOST_HYBRID_DFLASH_DIR" \
            "Environment=LUCEBOX_HYBRID_ENTRYPOINT=$LUCEBOX_HOST_HYBRID_ENTRYPOINT")
    else
        require_host_prereqs "$variant"
        local docker_bin
        docker_bin=$(command -v docker)
        unit_after+=" docker.service"
        unit_wants+=" docker.service"
        unit_pre="ExecStartPre=-$docker_bin rm -f $CONTAINER_NAME"
        unit_stop="ExecStop=$docker_bin stop -t 30 $CONTAINER_NAME"
    fi

    mkdir -p "$(dirname "$UNIT_PATH")"
    # Capture the user's resolved env at install time so the unit launches
    # with the same image/variant/port/models the user expected when they
    # ran `lucebox install`. Systemd's user-session env is sparse — without
    # this block, the wrapper inside the unit would fall back to the
    # in-script defaults and silently pick a different image or models
    # directory than the user's interactive session uses.
    #
    # Docker profiles add an ExecStartPre cleanup for an orphaned container
    # name. Native hybrid profiles intentionally leave both Docker directives
    # empty because their server process is owned directly by systemd.
    cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Lucebox hub LLM inference server
Documentation=https://github.com/Luce-Org/lucebox-hub
After=$unit_after
Wants=$unit_wants

[Service]
Type=exec
Restart=on-failure
RestartSec=10
# A normal SIGTERM may surface as 143 after transport-specific cleanup. Treat
# that as a clean stop; genuine crashes still use other codes and trip Restart.
SuccessExitStatus=143 SIGTERM
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LUCEBOX_IMAGE=$IMAGE_BASE
Environment=LUCEBOX_VARIANT=$variant
Environment=LUCEBOX_PORT=$DEFAULT_PORT
Environment=LUCEBOX_CONTAINER=$CONTAINER_NAME
Environment=LUCEBOX_MODELS=$DEFAULT_MODELS_DIR
Environment=LUCEBOX_HOME=$CONFIG_HOME
$unit_hybrid_env
$unit_pre
ExecStart=$SCRIPT_PATH serve
$unit_stop
TimeoutStopSec=45

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    ok "Installed $UNIT_PATH"

    # Check for linger — without it, the unit dies when the user logs out.
    local linger
    linger=$(loginctl show-user "$USER" 2>/dev/null | awk -F= '/^Linger=/{print $2}')
    if [ "$linger" != "yes" ]; then
        warn "Linger is off for $USER — the service will stop when you log out"
        hint "To enable (requires sudo): sudo loginctl enable-linger \"$USER\""
    fi

    printf '\nNext:\n'
    hint "  $SCRIPT_NAME start            # start now"
    hint "  $SCRIPT_NAME enable           # start at every login"
    hint "  $SCRIPT_NAME logs             # follow the journal"
}

cmd_systemd_uninstall() {
    require_systemd "service uninstall"
    if systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
        info "Stopping $UNIT_NAME"
        systemctl --user stop "$UNIT_NAME" || true
    fi
    if systemctl --user is-enabled --quiet "$UNIT_NAME" 2>/dev/null; then
        info "Disabling $UNIT_NAME"
        systemctl --user disable "$UNIT_NAME" || true
    fi
    if [ -f "$UNIT_PATH" ]; then
        rm -f "$UNIT_PATH"
        ok "Removed $UNIT_PATH"
    else
        info "No unit at $UNIT_PATH — nothing to remove"
    fi
    systemctl --user daemon-reload
    hint "Config and models are left in place. Remove them by hand if you want."
}

cmd_systemctl_passthrough() {
    local action="$1"
    require_systemd "$action"
    if [ ! -f "$UNIT_PATH" ]; then
        err "$UNIT_NAME is not installed — run '$SCRIPT_NAME install' first"
        exit 1
    fi
    case "$action" in
        start|restart)
            # `systemctl start` is fire-and-forget for Type=exec: it returns
            # success as soon as execve() completes, even if the wrapper
            # exits 1 a millisecond later. That gave us the worst possible
            # UX — `lucebox start` reports no error but no container ever
            # binds port 8080. Poll is-active for a few seconds and dump
            # status + recent journal lines so the user sees the real cause.
            local current
            current=$(systemctl --user is-active "$UNIT_NAME" 2>/dev/null || true)
            # `start` against an already-active unit: systemctl returns 0
            # silently. That's polite for scripts but confusing for humans
            # — say so explicitly. For `restart` always run through.
            if [ "$action" = "start" ] && [ "$current" = "active" ]; then
                ok "$UNIT_NAME is already active"
                hint "logs:    $SCRIPT_NAME logs"
                hint "smoke:   curl -s http://localhost:$DEFAULT_PORT/v1/models"
                hint "(use \`$SCRIPT_NAME restart\` to bounce, \`$SCRIPT_NAME stop\` to halt)"
                return 0
            fi
            # `start` against a unit stuck in restart-loop ("activating") is
            # the symptom of a broken ExecStart — calling start would just
            # block waiting for active that never comes. Surface this
            # specifically so the user goes to `lucebox logs` to find the
            # exit reason rather than waiting for the poll to give up.
            if [ "$action" = "start" ] && [ "$current" = "activating" ]; then
                err "$UNIT_NAME is in restart-loop (state=activating)"
                hint "the unit is failing and being auto-restarted by systemd"
                hint "  $SCRIPT_NAME stop          # halt the loop first"
                hint "  $SCRIPT_NAME logs          # find the exit reason"
                exit 1
            fi
            info "$action $UNIT_NAME"
            if ! systemctl --user "$action" "$UNIT_NAME"; then
                err "systemctl --user $action $UNIT_NAME failed"
                systemctl --user status "$UNIT_NAME" --no-pager -n 30 || true
                exit 1
            fi
            local i state
            for i in 1 2 3 4 5 6 7 8 9 10; do
                state=$(systemctl --user is-active "$UNIT_NAME" 2>/dev/null || true)
                case "$state" in
                    active) break ;;     # already up — no need to keep polling
                    activating) ;;       # still booting; keep waiting
                    *) break ;;          # failed / inactive — fall through to error path
                esac
                sleep 1
            done
            state=$(systemctl --user is-active "$UNIT_NAME" 2>/dev/null || true)
            if [ "$state" != "active" ]; then
                err "$UNIT_NAME did not reach active state (current: ${state:-unknown})"
                if [ "$state" = "activating" ]; then
                    hint "the unit is in a restart loop — \`$SCRIPT_NAME stop\` to halt it"
                fi
                hint "status:"
                systemctl --user status "$UNIT_NAME" --no-pager -n 30 || true
                hint "recent journal:"
                journalctl --user -u "$UNIT_NAME" -n 30 --no-pager || true
                exit 1
            fi
            ok "$UNIT_NAME is active"
            hint "logs:    $SCRIPT_NAME logs"
            hint "smoke:   curl -s http://localhost:$DEFAULT_PORT/v1/models"
            ;;
        stop|enable|disable)
            exec systemctl --user "$action" "$UNIT_NAME" ;;
        status)
            exec systemctl --user status "$UNIT_NAME" --no-pager ;;
        *)
            die "unknown systemctl passthrough: $action" ;;
    esac
}

# One-time machine calibration. The Python package owns candidate generation,
# workload measurement, quality-equivalence checks, winner selection, and the
# result cache. This host layer owns only the lifecycle it alone can control:
# restart the user service for each startup-scoped DDTree budget and restore
# both config and service state on every failure or interrupt.
_calibration_remove_run_dir() {
    if [ -n "${LUCEBOX_CALIBRATION_RUN_DIR:-}" ] \
       && [[ "$LUCEBOX_CALIBRATION_RUN_DIR" == "$CONFIG_HOME"/.calibration-run.* ]]; then
        rm -rf -- "$LUCEBOX_CALIBRATION_RUN_DIR"
    fi
}

_calibration_on_exit() {
    local rc=$?
    trap - EXIT HUP INT TERM
    if [ "${LUCEBOX_CALIBRATION_COMMITTED:-0}" != "1" ] \
       && [ -f "${LUCEBOX_CALIBRATION_BACKUP:-}" ]; then
        cp "$LUCEBOX_CALIBRATION_BACKUP" "$(_lucebox_config_path)" || true
        if [ "${LUCEBOX_CALIBRATION_HAD_RECORD:-0}" = "1" ] \
           && [ -f "${LUCEBOX_CALIBRATION_RECORD_BACKUP:-}" ]; then
            cp "$LUCEBOX_CALIBRATION_RECORD_BACKUP" \
                "$CONFIG_HOME/calibration.json" || true
        else
            rm -f "$CONFIG_HOME/calibration.json"
        fi
        if [ "${LUCEBOX_CALIBRATION_WAS_ACTIVE:-0}" = "1" ]; then
            systemctl --user restart "$UNIT_NAME" >/dev/null 2>&1 || true
        else
            systemctl --user stop "$UNIT_NAME" >/dev/null 2>&1 || true
        fi
        warn "Calibration did not complete; the original profile and service state were restored."
    fi
    _calibration_remove_run_dir
    exit "$rc"
}

cmd_calibrate() {
    local force=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --force) force=1 ;;
            --help|-h)
                cat <<EOF
Usage: $SCRIPT_NAME calibrate [--force]

Measure prefill, decode, and warm-prefix performance on the selected model.
For DDTree-capable models, test at most three budgets and keep a new value only
when its output/cache behavior matches and it is at least 5% faster. Results
are cached for the exact model, runtime profile, driver, and GPU.
EOF
                return 0
                ;;
            *) die "unknown calibrate option: $1" ;;
        esac
        shift
    done

    ensure_probed
    local config_path model
    config_path=$(_lucebox_config_path)
    [ -f "$config_path" ] || die "no config.toml — run '$SCRIPT_NAME setup' first"
    model=$(_lucebox_config_get model.preset)
    [ -n "$model" ] || die "no model selected — run '$SCRIPT_NAME models select' first"
    _ensure_configured_image || return $?
    if [ "$force" = "0" ] \
       && bash "$SCRIPT_PATH" _calibration status; then
        hint "Use '$SCRIPT_NAME calibrate --force' to measure again."
        return 0
    fi

    require_systemd "safe calibration restarts"
    [ -f "$UNIT_PATH" ] \
        || die "$UNIT_NAME is not installed — run '$SCRIPT_NAME install' first"
    command -v flock >/dev/null 2>&1 \
        || die "calibration requires flock (provided by util-linux)"

    mkdir -p "$CONFIG_HOME"
    exec 9>"$CONFIG_HOME/calibration.lock"
    flock -n 9 || die "another Lucebox calibration is already running"

    LUCEBOX_CALIBRATION_RUN_DIR=$(mktemp -d "$CONFIG_HOME/.calibration-run.XXXXXX") \
        || die "could not create calibration workspace"
    LUCEBOX_CALIBRATION_BACKUP="$LUCEBOX_CALIBRATION_RUN_DIR/config.toml.backup"
    cp "$config_path" "$LUCEBOX_CALIBRATION_BACKUP"
    LUCEBOX_CALIBRATION_RECORD_BACKUP="$LUCEBOX_CALIBRATION_RUN_DIR/calibration.json.backup"
    LUCEBOX_CALIBRATION_HAD_RECORD=0
    if [ -f "$CONFIG_HOME/calibration.json" ]; then
        cp "$CONFIG_HOME/calibration.json" "$LUCEBOX_CALIBRATION_RECORD_BACKUP"
        LUCEBOX_CALIBRATION_HAD_RECORD=1
    fi
    LUCEBOX_CALIBRATION_WAS_ACTIVE=0
    systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null \
        && LUCEBOX_CALIBRATION_WAS_ACTIVE=1
    LUCEBOX_CALIBRATION_COMMITTED=0
    export LUCEBOX_CALIBRATION_RUN_DIR LUCEBOX_CALIBRATION_BACKUP
    export LUCEBOX_CALIBRATION_RECORD_BACKUP LUCEBOX_CALIBRATION_HAD_RECORD
    export LUCEBOX_CALIBRATION_WAS_ACTIVE LUCEBOX_CALIBRATION_COMMITTED
    trap _calibration_on_exit EXIT
    trap 'exit 130' HUP INT TERM

    local budgets=() baseline budget result_path successful=0
    mapfile -t budgets < <(bash "$SCRIPT_PATH" _calibration budgets)
    [ "${#budgets[@]}" -gt 0 ] || die "calibration planner returned no budget"
    [ "${#budgets[@]}" -le 3 ] || die "calibration planner exceeded its three-cell bound"
    baseline="${budgets[0]}"
    info "Calibrating $model on this machine (${#budgets[@]} server start(s))."
    hint "The engine may take several minutes to load each cell. Ctrl-C safely restores the original profile."

    for budget in "${budgets[@]}"; do
        [[ "$budget" =~ ^[0-9]+$ ]] || die "invalid calibration budget: $budget"
        info "Calibration cell: DDTree budget $budget"
        bash "$SCRIPT_PATH" _calibration apply "$budget" \
            || die "could not apply calibration budget $budget"
        if ! systemctl --user restart "$UNIT_NAME"; then
            if [ "$budget" = "$baseline" ]; then
                die "the baseline server failed to start"
            fi
            warn "Skipping budget $budget because the server failed to start."
            continue
        fi
        result_path="$LUCEBOX_CALIBRATION_RUN_DIR/budget-$budget.json"
        if bash "$SCRIPT_PATH" _calibration probe "$budget" "$result_path"; then
            successful=$((successful + 1))
        elif [ "$budget" = "$baseline" ]; then
            die "the baseline performance probe failed"
        else
            warn "Skipping budget $budget because its performance probe failed."
        fi
    done
    [ "$successful" -gt 0 ] || die "calibration produced no measurements"

    bash "$SCRIPT_PATH" _calibration finish \
        "$LUCEBOX_CALIBRATION_RUN_DIR" --baseline "$baseline" \
        || die "calibration could not select a safe result"
    local winner
    winner=$(<"$LUCEBOX_CALIBRATION_RUN_DIR/winner")
    [[ "$winner" =~ ^[0-9]+$ ]] || die "calibration returned an invalid winner"

    if [ "$LUCEBOX_CALIBRATION_WAS_ACTIVE" = "1" ]; then
        systemctl --user restart "$UNIT_NAME" \
            || die "the calibrated server failed to restart"
        ok "Calibrated profile is active (DDTree budget $winner)."
    else
        systemctl --user stop "$UNIT_NAME" \
            || die "could not restore the engine's stopped state"
        ok "Calibrated profile saved (DDTree budget $winner); engine remains stopped."
    fi

    LUCEBOX_CALIBRATION_COMMITTED=1
    export LUCEBOX_CALIBRATION_COMMITTED
    trap - EXIT HUP INT TERM
    _calibration_remove_run_dir
    flock -u 9
}

cmd_logs() {
    ensure_probed
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ] && [ -f "$UNIT_PATH" ]; then
        # Pure passthrough: any flags the user wants (-f, -n, --since, ...)
        # go straight to journalctl. Default is follow.
        if [ $# -eq 0 ]; then
            exec journalctl --user -u "$UNIT_NAME" -f
        fi
        exec journalctl --user -u "$UNIT_NAME" "$@"
    fi
    if _lucebox_container_running; then
        if [ $# -eq 0 ]; then
            exec docker logs -f "$CONTAINER_NAME"
        fi
        exec docker logs "$@" "$CONTAINER_NAME"
    fi
    die "the inference engine is not running — use '$SCRIPT_NAME start' or '$SCRIPT_NAME serve'"
}

cmd_status() {
    ensure_probed
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ] && [ -f "$UNIT_PATH" ]; then
        exec systemctl --user status "$UNIT_NAME" --no-pager
    fi
    if _lucebox_container_running; then
        exec docker ps --filter "name=^${CONTAINER_NAME}\$" \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    fi
    info "Lucebox inference engine is stopped"
    hint "Run '$SCRIPT_NAME setup' for first-time setup, or '$SCRIPT_NAME serve' in the foreground."
}

cmd_stop() {
    ensure_probed
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ] && [ -f "$UNIT_PATH" ]; then
        exec systemctl --user stop "$UNIT_NAME"
    fi
    if _lucebox_container_running; then
        exec docker stop "$CONTAINER_NAME"
    fi
    ok "Lucebox inference engine is already stopped"
}

cmd_pull() {
    # Pull has to run on the host. Delegating this into the container creates a
    # stale-image trap: docker may start an old local tag before the fresh tag
    # has been pulled.
    ensure_probed
    local variant
    variant=$(pick_variant)
    require_host_prereqs "$variant"
    info "Pulling ${IMAGE_BASE}:${variant}"
    exec docker pull "${IMAGE_BASE}:${variant}"
}

# ── contributor-native workflow ───────────────────────────────────────────

_native_backend() {
    local requested="${1:-}" variant
    case "$requested" in
        cuda|cuda12|cuda128|cuda13|nvidia) printf 'cuda'; return ;;
        rocm|hip|amd)       printf 'rocm'; return ;;
        "")
            variant=$(pick_variant)
            if _variant_is_rocm "$variant"; then printf 'rocm'; else printf 'cuda'; fi
            return
            ;;
        *) die "unknown backend '$requested' — choose cuda or rocm" ;;
    esac
}

_native_build_dir() {
    local repo="$1" backend="$2"
    if [ -n "${LUCEBOX_BUILD_DIR:-}" ]; then
        printf '%s' "$LUCEBOX_BUILD_DIR"
    elif [ "$backend" = "rocm" ]; then
        printf '%s/server/build-hip' "$repo"
    else
        printf '%s/server/build-cuda' "$repo"
    fi
}

_nvidia_build_arches() {
    printf '%s\n' "${LUCEBOX_HOST_NVIDIA_GPU_LIST_CSV:-}" | awk -F',' '
        {
            arch = $5
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", arch)
            gsub(/[.]/, "", arch)
            if (arch ~ /^[0-9]+$/ && !seen[arch]++) {
                out = out (out ? ";" : "") arch
            }
        }
        END { print out }
    '
}

_amd_build_arches() {
    printf '%s\n' "${LUCEBOX_HOST_AMD_GPU_LIST_CSV:-}" | awk -F',' '
        {
            arch = $5
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", arch)
            if (arch ~ /^gfx[0-9a-z]+$/ && !seen[arch]++) {
                out = out (out ? ";" : "") arch
            }
        }
        END { print out }
    '
}

_safe_model_relative_path() {
    local value="$1" part parts=()
    [ -n "$value" ] || return 1
    [[ "$value" != /* ]] || return 1
    IFS='/' read -r -a parts <<<"$value"
    for part in "${parts[@]}"; do
        [ "$part" != ".." ] || return 1
    done
    return 0
}

_model_artifact_ready() {
    local path="$1"
    if [ -f "$path" ]; then
        [ -s "$path" ]
        return
    fi
    if [ -d "$path" ]; then
        [ -n "$(find -L "$path" -maxdepth 4 -type f \
            \( -name '*.gguf' -o -name '*.safetensors' \) \
            -size +0c -print -quit 2>/dev/null)" ]
        return
    fi
    return 1
}

_selected_model_paths() {
    # Emit target path, draft path (or "none"), and model id on separate lines.
    # `models select` persists the filenames, so preset mapping is only a
    # compatibility fallback for hand-written config files.
    local preset target_file draft_file target_path draft_path="none" speculative_decode
    preset=$(_lucebox_config_get model.preset)
    target_file=$(_lucebox_config_get model.target_file)
    draft_file=$(_lucebox_config_get model.draft_file)
    if [ -z "$target_file" ]; then
        case "$preset" in
            qwen3.6-27b)  target_file="Qwen3.6-27B-Q4_K_M.gguf" ;;
            gemma-4-26b)  target_file="google_gemma-4-26B-A4B-it-Q4_K_M.gguf" ;;
            gemma-4-31b)  target_file="google_gemma-4-31B-it-Q4_K_M.gguf" ;;
            laguna-xs.2)  target_file="laguna-xs2-Q4_K_M.gguf" ;;
            qwen3.6-moe)  target_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" ;;
            deepseek-v4-flash) target_file="DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf" ;;
        esac
    fi
    if [ -z "$draft_file" ]; then
        case "$preset" in
            qwen3.6-27b) draft_file="dflash-draft-3.6-q4_k_m.gguf" ;;
            gemma-4-26b) draft_file="gemma-4-26B-A4B-it-DFlash-q8_0.gguf" ;;
            gemma-4-31b) draft_file="gemma-4-31B-it-DFlash-q8_0.gguf" ;;
            deepseek-v4-flash) draft_file="DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf" ;;
        esac
    fi
    speculative_decode=$(_lucebox_config_get dflash.speculative_decode)
    case "$speculative_decode" in
        false|0|no|off) draft_file="" ;;
    esac
    _safe_model_relative_path "$target_file" \
        || die "no valid model is selected — run '$SCRIPT_NAME models select' first"
    target_path="$DEFAULT_MODELS_DIR/$target_file"
    if [ -n "$draft_file" ]; then
        _safe_model_relative_path "$draft_file" \
            || die "invalid model.draft_file in $(_lucebox_config_path)"
        draft_path="$DEFAULT_MODELS_DIR/draft/$draft_file"
    elif [ "$preset" = "laguna-xs.2" ] \
         && [ "$speculative_decode" != "false" ] \
         && [ "$speculative_decode" != "0" ] \
         && [ "$speculative_decode" != "no" ] \
         && [ "$speculative_decode" != "off" ] \
         && [ -s "$DEFAULT_MODELS_DIR/draft/laguna-xs2-speculator/model.safetensors" ] \
         && [ -s "$DEFAULT_MODELS_DIR/draft/laguna-xs2-speculator/config.json" ]; then
        draft_path="$DEFAULT_MODELS_DIR/draft/laguna-xs2-speculator"
    fi
    printf '%s\n%s\n%s\n' "$target_path" "$draft_path" "${preset:-lucebox}"
}

_export_selected_decode_companion() {
    # Generic DFlash consumes --draft. DeepSeek's architecture-specific
    # DSpark backend consumes a separate switch/path and deliberately ignores
    # --draft, so native and packaged-hybrid launches must mirror the Python
    # container planner instead of relying on generic draft discovery.
    local model_id="$1" draft="$2"
    unset DFLASH_DS4_SPEC DFLASH_DS4_DRAFT
    if [ "$model_id" = "deepseek-v4-flash" ]; then
        export DFLASH_DRAFT="$DEFAULT_MODELS_DIR/.lucebox-no-draft"
        if [ "$draft" != "none" ]; then
            export DFLASH_DS4_SPEC=1
            export DFLASH_DS4_DRAFT="$draft"
        fi
    elif [ "$draft" = "none" ]; then
        export DFLASH_DRAFT="$DEFAULT_MODELS_DIR/.lucebox-no-draft"
    else
        export DFLASH_DRAFT="$draft"
    fi
}

# ── engine client connectors ─────────────────────────────────────────────
#
# A connector uses a client the user already installed and points only that
# invocation (or a dedicated Lucebox profile) at the local inference API. It
# never installs a client and never replaces the client's normal cloud config.

_connector_normalize() {
    case "${1:-}" in
        1|claude|claude-code|claude_code) printf 'claude_code' ;;
        2|codex)                          printf 'codex' ;;
        3|opencode|open-code)             printf 'opencode' ;;
        4|hermes)                         printf 'hermes' ;;
        5|pi)                             printf 'pi' ;;
        6|openclaw|open-claw)             printf 'openclaw' ;;
        7|openwebui|open-webui|webui)     printf 'openwebui' ;;
        *) return 1 ;;
    esac
}

_connector_label() {
    case "$1" in
        claude_code) printf 'Claude Code' ;;
        codex)       printf 'Codex' ;;
        opencode)    printf 'OpenCode' ;;
        hermes)      printf 'Hermes' ;;
        pi)          printf 'Pi' ;;
        openclaw)    printf 'OpenClaw' ;;
        openwebui)   printf 'Open WebUI' ;;
        *)           printf '%s' "$1" ;;
    esac
}

_connector_binary() {
    local client="$1" override="" command_name=""
    case "$client" in
        claude_code) override="${LUCEBOX_CLAUDE_BIN:-}";  command_name=claude ;;
        codex)       override="${LUCEBOX_CODEX_BIN:-}";   command_name=codex ;;
        opencode)    override="${LUCEBOX_OPENCODE_BIN:-}"; command_name=opencode ;;
        hermes)      override="${LUCEBOX_HERMES_BIN:-}";  command_name=hermes ;;
        pi)          override="${LUCEBOX_PI_BIN:-}";      command_name=pi ;;
        openclaw)    override="${LUCEBOX_OPENCLAW_BIN:-}"; command_name=openclaw ;;
        openwebui)   override="${LUCEBOX_OPENWEBUI_BIN:-}"; command_name=open-webui ;;
        *) return 1 ;;
    esac
    if [ -n "$override" ]; then
        if [[ "$override" == */* ]]; then
            [ -f "$override" ] && [ -x "$override" ] || return 1
            printf '%s' "$override"
            return 0
        fi
        command -v "$override" 2>/dev/null
        return
    fi
    command -v "$command_name" 2>/dev/null
}

_connector_selected() {
    local selected
    [ -f "$CONNECTOR_SELECTION_FILE" ] || return 0
    IFS= read -r selected < "$CONNECTOR_SELECTION_FILE" || return 0
    _connector_normalize "$selected" 2>/dev/null || true
}

_connector_write_file() {
    # Atomically replace a Lucebox-owned connector file with private mode.
    # Content is read from stdin so callers can use a quoted or expanded
    # heredoc without lossy shell escaping.
    local target="$1" mode="${2:-600}" parent tmp
    parent=$(dirname "$target")
    mkdir -p "$parent"
    tmp=$(mktemp "${target}.tmp.XXXXXX") \
        || die "couldn't create a temporary connector file next to $target"
    if ! cat > "$tmp"; then
        rm -f "$tmp"
        die "couldn't write connector file: $target"
    fi
    chmod "$mode" "$tmp"
    mv "$tmp" "$target"
}

_connector_private_dir() {
    local path="$1"
    mkdir -p "$path"
    chmod 700 "$path"
}

_connector_remember() {
    local client="$1"
    _connector_private_dir "$CONNECTOR_STATE_DIR"
    _connector_write_file "$CONNECTOR_SELECTION_FILE" 600 <<EOF
$client
EOF
}

_connector_model_id() {
    local model
    model=$(_lucebox_config_get model.preset)
    [ -n "$model" ] \
        || die "no model is selected — run '$SCRIPT_NAME models select' first"
    # The value is embedded in small TOML/JSON/YAML connector profiles. Keep
    # hand-edited config from turning that serialization into code or syntax.
    [[ "$model" =~ ^[A-Za-z0-9._:/-]+$ ]] \
        || die "model.preset contains unsupported characters: $model"
    printf '%s' "$model"
}

_connector_context_size() {
    local value
    value=$(_lucebox_config_get dflash.max_ctx)
    [ -n "$value" ] || value=65536
    [[ "$value" =~ ^[0-9]+$ ]] && [ "$value" -gt 0 ] \
        || die "dflash.max_ctx must be a positive integer before connecting a harness"
    printf '%s' "$value"
}

_connector_output_size() {
    # Never advertise an output allowance larger than the active context.
    # Small-memory automatic profiles can use a 4K context, while the server
    # and larger profiles support up to a 32K response budget.
    local max_ctx="$1" value=32768
    if [ "$max_ctx" -lt 65536 ]; then
        value=$((max_ctx / 2))
        [ "$value" -gt 0 ] || value=1
    fi
    printf '%s' "$value"
}

_connector_api_root() {
    [[ "$DEFAULT_PORT" =~ ^[0-9]+$ ]] \
        && [ "$DEFAULT_PORT" -ge 1 ] && [ "$DEFAULT_PORT" -le 65535 ] \
        || die "runtime.port must be an integer between 1 and 65535"
    printf 'http://127.0.0.1:%s' "$DEFAULT_PORT"
}

_connector_api_ready() {
    local api_root="$1" expected_model="${2:-}" response compact
    command -v curl >/dev/null 2>&1 || return 1
    response=$(curl -fsS --connect-timeout 2 --max-time 4 \
        "$api_root/v1/models" 2>/dev/null) || return 1
    # A generic 200 response from another process on the configured port is
    # not sufficient. Require an OpenAI-compatible model catalog shape.
    [[ "$response" == *'"data"'* || "$response" == *'"models"'* ]] || return 1
    [ -n "$expected_model" ] || return 0
    compact=$(printf '%s' "$response" | tr -d '[:space:]')
    [[ "$compact" == *"\"id\":\"$expected_model\""* ]]
}

_connector_ensure_api() {
    local api_root="$1" expected_model="${2:-}" i state
    if ! command -v curl >/dev/null 2>&1; then
        err "curl is required to verify the local Lucebox API"
        return 1
    fi
    if _connector_api_ready "$api_root" "$expected_model"; then
        return 0
    fi
    ensure_probed
    state=$(_engine_state)
    if [ "$state" = "stopped" ] && [ -t 0 ] \
       && [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ] \
       && [ -f "$UNIT_PATH" ] \
       && _confirm "The inference engine is stopped. Start it now?" 1; then
        bash "$SCRIPT_PATH" start || return $?
        state=running
        for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
            _connector_api_ready "$api_root" "$expected_model" && return 0
            sleep 1
        done
    fi
    if [ "$state" = "running" ]; then
        err "the inference engine is running, but its API is not ready at $api_root"
        hint "Wait for model loading to finish, or inspect: $SCRIPT_NAME logs"
    else
        err "the Lucebox API is not reachable at $api_root"
        hint "Start it from the menu or run: $SCRIPT_NAME start"
    fi
    return 1
}

_connector_print_argv() {
    local arg
    printf '  '
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '\n'
}

_connector_choose() {
    local client label selected status number selected_number=""
    selected=$(_connector_selected)
    printf 'Choose an installed harness to connect:\n'
    number=0
    for client in claude_code codex opencode hermes pi openclaw openwebui; do
        number=$((number + 1))
        label=$(_connector_label "$client")
        if _connector_binary "$client" >/dev/null 2>&1; then
            status="installed"
            [ "$client" = "$selected" ] && selected_number="$number"
        else
            status="not found"
        fi
        [ "$client" = "$selected" ] && status="$status, selected"
        printf '  %s  %-12s %b(%s)%b\n' "$number" "$label" "$C_DIM" "$status" "$C_RST"
    done
    printf '  b  Back\n\n'
    if [ -n "$selected_number" ]; then
        printf 'Harness number [%s]: ' "$selected_number"
    else
        printf 'Harness number: '
    fi
    IFS= read -r client || return 1
    case "$client" in b|B|q|Q) return 1 ;; esac
    [ -n "$client" ] || client="$selected_number"
    [ -n "$client" ] || return 1
    CONNECTOR_CHOICE=$(_connector_normalize "$client") \
        || { warn "Choose 1–7 or b"; return 1; }
}

# The prepare helpers generate only Lucebox-owned files or additive named
# profiles. They populate CONNECTOR_SETUP_ARGV (optional) and
# CONNECTOR_LAUNCH_ARGV; the caller applies setup before remembering the choice.
_connector_prepare_claude() {
    local binary="$1" api_root="$2" model="$3"
    CONNECTOR_LAUNCH_ARGV=(
        env
        -u CLAUDE_CODE_USE_BEDROCK
        -u CLAUDE_CODE_USE_VERTEX
        -u CLAUDE_CODE_USE_FOUNDRY
        -u ANTHROPIC_API_KEY
        -u CLAUDE_CODE_OAUTH_TOKEN
        "ANTHROPIC_BASE_URL=$api_root"
        "ANTHROPIC_AUTH_TOKEN=lucebox-local"
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
        "$binary" --model "$model"
    )
    CONNECTOR_NOTE="Only this Claude Code session uses Lucebox; your normal Claude configuration is unchanged."
}

_connector_prepare_codex() {
    local binary="$1" api_root="$2" model="$3" codex_home profile
    codex_home="${CODEX_HOME:-$HOME/.codex}"
    profile="$codex_home/lucebox-local.config.toml"
    mkdir -p "$codex_home"
    if [ -e "$profile" ] \
       && ! grep -Fqx '# Generated by Lucebox. Normal Codex configuration is not modified.' "$profile"; then
        die "refusing to replace an unowned Codex profile: $profile"
    fi
    _connector_write_file "$profile" 600 <<EOF
# Generated by Lucebox. Normal Codex configuration is not modified.
model = "$model"
model_provider = "lucebox"

[model_providers.lucebox]
name = "Lucebox"
base_url = "$api_root/v1"
wire_api = "responses"
EOF
    CONNECTOR_LAUNCH_ARGV=("$binary" --profile lucebox-local --model "$model")
    CONNECTOR_NOTE="Codex uses the additive 'lucebox-local' profile; your default profile and login are unchanged."
}

_connector_prepare_opencode() {
    local binary="$1" api_root="$2" model="$3" max_ctx="$4" output_tokens="$5"
    local state config version major=2
    state="$CONNECTOR_STATE_DIR/opencode"
    config="$state/opencode.json"
    _connector_private_dir "$state"
    version=$("$binary" --version 2>/dev/null | head -n 1 || true)
    if [[ "$version" =~ (^|[^0-9])([0-9]+)\.[0-9]+ ]]; then
        major="${BASH_REMATCH[2]}"
    fi
    if [ "$major" -ge 2 ]; then
        _connector_write_file "$config" 600 <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "lucebox-local/$model",
  "providers": {
    "lucebox-local": {
      "name": "Lucebox",
      "package": "@opencode-ai/ai/providers/openai-compatible",
      "settings": {"baseURL": "$api_root/v1"},
      "models": {
        "$model": {
          "name": "Lucebox local model",
          "capabilities": {"tools": true, "input": ["text"], "output": ["text"]},
          "limit": {"context": $max_ctx, "output": $output_tokens}
        }
      }
    }
  }
}
EOF
    else
        _connector_write_file "$config" 600 <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "lucebox-local/$model",
  "small_model": "lucebox-local/$model",
  "provider": {
    "lucebox-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Lucebox",
      "options": {"baseURL": "$api_root/v1", "apiKey": "lucebox-local"},
      "models": {
        "$model": {
          "name": "Lucebox local model",
          "limit": {"context": $max_ctx, "output": $output_tokens}
        }
      }
    }
  }
}
EOF
    fi
    CONNECTOR_LAUNCH_ARGV=(
        env "OPENCODE_CONFIG=$config" "OPENAI_API_KEY=lucebox-local"
        "$binary" --model "lucebox-local/$model"
    )
    CONNECTOR_NOTE="OpenCode receives a Lucebox config overlay; its global and project files are not replaced."
}

_connector_prepare_hermes() {
    local binary="$1" api_root="$2" model="$3" max_ctx="$4" output_tokens="$5" state
    state="$CONNECTOR_STATE_DIR/hermes"
    _connector_private_dir "$state"
    _connector_write_file "$state/config.yaml" 600 <<EOF
model:
  default: "$model"
  provider: "custom"
  base_url: "$api_root/v1"
  api_key: "lucebox-local"
  api_mode: "chat_completions"
  context_length: $max_ctx
  max_tokens: $output_tokens
EOF
    _connector_write_file "$state/.env" 600 <<EOF
OPENAI_API_KEY=lucebox-local
OPENAI_BASE_URL=$api_root/v1
HERMES_INFERENCE_PROVIDER=custom
HERMES_INFERENCE_MODEL=$model
EOF
    CONNECTOR_LAUNCH_ARGV=(
        env "HERMES_HOME=$state" "OPENAI_API_KEY=lucebox-local"
        "OPENAI_BASE_URL=$api_root/v1" "HERMES_INFERENCE_PROVIDER=custom"
        "HERMES_INFERENCE_MODEL=$model" "$binary" chat --model "$model"
    )
    CONNECTOR_NOTE="Hermes uses a Lucebox-only provider profile; your normal Hermes home is unchanged."
}

_connector_prepare_pi() {
    local binary="$1" api_root="$2" model="$3" max_ctx="$4" output_tokens="$5" state
    state="$CONNECTOR_STATE_DIR/pi"
    _connector_private_dir "$state"
    _connector_private_dir "$state/sessions"
    _connector_write_file "$state/models.json" 600 <<EOF
{
  "providers": {
    "lucebox": {
      "baseUrl": "$api_root/v1",
      "api": "openai-completions",
      "apiKey": "lucebox-local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "$model",
          "name": "Lucebox local model",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": $max_ctx,
          "maxTokens": $output_tokens,
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        }
      ]
    }
  }
}
EOF
    CONNECTOR_LAUNCH_ARGV=(
        env "PI_CODING_AGENT_DIR=$state" "PI_CODING_AGENT_SESSION_DIR=$state/sessions"
        "PI_OFFLINE=1" "$binary" --provider lucebox --model "$model"
        --api-key lucebox-local
    )
    CONNECTOR_NOTE="Pi uses a Lucebox-only model directory; your normal Pi models and sessions are unchanged."
}

_connector_prepare_openclaw() {
    local binary="$1" api_root="$2" model="$3" max_ctx="$4" output_tokens="$5"
    local state runtime config patch
    state="$CONNECTOR_STATE_DIR/openclaw"
    runtime="$state/state"
    config="$runtime/openclaw.json"
    patch="$state/lucebox.patch.json"
    _connector_private_dir "$state"
    _connector_private_dir "$runtime"
    _connector_write_file "$patch" 600 <<EOF
{
  "models": {
    "mode": "merge",
    "providers": {
      "lucebox": {
        "baseUrl": "$api_root/v1",
        "apiKey": "\${OPENAI_API_KEY}",
        "auth": "api-key",
        "api": "openai-completions",
        "models": [
          {
            "id": "$model",
            "name": "Lucebox local model",
            "api": "openai-completions",
            "contextWindow": $max_ctx,
            "maxTokens": $output_tokens,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {"primary": "lucebox/$model"},
      "models": {"lucebox/$model": {"alias": "Lucebox local model"}},
      "skipBootstrap": true
    }
  }
}
EOF
    CONNECTOR_SETUP_ARGV=(
        env "OPENCLAW_STATE_DIR=$runtime" "OPENCLAW_CONFIG_PATH=$config"
        "OPENAI_API_KEY=lucebox-local"
        "$binary" config patch --file "$patch"
    )
    CONNECTOR_LAUNCH_ARGV=(
        env "OPENCLAW_STATE_DIR=$runtime" "OPENCLAW_CONFIG_PATH=$config"
        "OPENCLAW_WORKSPACE_DIR=$PWD"
        "OPENAI_API_KEY=lucebox-local" "$binary" chat
    )
    CONNECTOR_NOTE="OpenClaw uses Lucebox-only state; your default OpenClaw profile is unchanged."
}

_connector_prepare_openwebui() {
    local binary="$1" api_root="$2" model="$3" state webui_port
    state="$CONNECTOR_STATE_DIR/openwebui"
    webui_port="${LUCEBOX_WEBUI_PORT:-3000}"
    [[ "$webui_port" =~ ^[0-9]+$ ]] \
        && [ "$webui_port" -ge 1 ] && [ "$webui_port" -le 65535 ] \
        || die "LUCEBOX_WEBUI_PORT must be an integer between 1 and 65535"
    [ "$webui_port" != "$DEFAULT_PORT" ] \
        || die "Open WebUI and the Lucebox API cannot use the same port ($webui_port)"
    _connector_private_dir "$state"
    _connector_private_dir "$state/data"
    CONNECTOR_LAUNCH_ARGV=(
        env "DATA_DIR=$state/data" "WEBUI_AUTH=True"
        "DEFAULT_MODELS=$model" "OPENAI_API_BASE_URL=$api_root/v1"
        "OPENAI_API_KEY=lucebox-local" "$binary" serve
        --host 127.0.0.1 --port "$webui_port"
    )
    CONNECTOR_NOTE="Open WebUI will use a Lucebox-only data directory at http://127.0.0.1:$webui_port."
}

cmd_connect() {
    local requested="" launch=1 client label binary api_root model max_ctx output_tokens
    while [ $# -gt 0 ]; do
        case "$1" in
            --no-launch) launch=0 ;;
            --help|-h)
                cat <<EOF
$SCRIPT_NAME connect [harness] [--no-launch]

Connect an already-installed Claude Code, Codex, OpenCode, Hermes, Pi,
OpenClaw, or Open WebUI client to the local Lucebox API. Lucebox never
installs the client and never replaces its normal cloud configuration.

  --no-launch  create/update the Lucebox profile but do not open the client
EOF
                return 0
                ;;
            -*) die "unknown connect option: $1" ;;
            *)
                [ -z "$requested" ] || die "connect accepts one harness name"
                requested="$1"
                ;;
        esac
        shift
    done

    if [ -z "$requested" ]; then
        [ -t 0 ] && [ -t 1 ] \
            || die "choose a harness name in non-interactive mode (for example: $SCRIPT_NAME connect codex)"
        CONNECTOR_CHOICE=""
        _connector_choose || return 0
        client="$CONNECTOR_CHOICE"
    else
        client=$(_connector_normalize "$requested") \
            || die "unknown harness '$requested' — choose claude, codex, opencode, hermes, pi, openclaw, or openwebui"
    fi

    label=$(_connector_label "$client")
    binary=$(_connector_binary "$client") || {
        err "$label is not installed or not on PATH"
        hint "Lucebox does not install harnesses. Install $label yourself, then run this command again."
        return 1
    }
    api_root=$(_connector_api_root)
    model=$(_connector_model_id)
    max_ctx=$(_connector_context_size)
    output_tokens=$(_connector_output_size "$max_ctx")
    if [ "$client" = "hermes" ] && [ "$max_ctx" -lt 65536 ]; then
        die "Hermes requires at least a 65536-token context; choose a larger optimization profile first"
    fi

    CONNECTOR_SETUP_ARGV=()
    CONNECTOR_LAUNCH_ARGV=()
    CONNECTOR_NOTE=""
    case "$client" in
        claude_code) _connector_prepare_claude "$binary" "$api_root" "$model" ;;
        codex)       _connector_prepare_codex "$binary" "$api_root" "$model" ;;
        opencode)    _connector_prepare_opencode "$binary" "$api_root" "$model" "$max_ctx" "$output_tokens" ;;
        hermes)      _connector_prepare_hermes "$binary" "$api_root" "$model" "$max_ctx" "$output_tokens" ;;
        pi)          _connector_prepare_pi "$binary" "$api_root" "$model" "$max_ctx" "$output_tokens" ;;
        openclaw)    _connector_prepare_openclaw "$binary" "$api_root" "$model" "$max_ctx" "$output_tokens" ;;
        openwebui)   _connector_prepare_openwebui "$binary" "$api_root" "$model" ;;
    esac

    if [ "${#CONNECTOR_SETUP_ARGV[@]}" -gt 0 ]; then
        "${CONNECTOR_SETUP_ARGV[@]}" \
            || die "$label rejected the generated Lucebox profile"
    fi
    _connector_remember "$client"

    ok "$label is linked to Lucebox"
    hint "Endpoint: $api_root/v1"
    hint "Model:    $model"
    [ -z "$CONNECTOR_NOTE" ] || hint "$CONNECTOR_NOTE"

    if [ "$launch" = "0" ]; then
        printf '\nLaunch command:\n'
        _connector_print_argv "${CONNECTOR_LAUNCH_ARGV[@]}"
        return 0
    fi

    _connector_ensure_api "$api_root" "$model" || return $?
    info "Opening $label with the Lucebox model"
    exec "${CONNECTOR_LAUNCH_ARGV[@]}"
}

_toml_array_to_csv() {
    # Config arrays written by tomli_w contain only validated device names or
    # finite positive numbers. Strip TOML punctuation for the engine's comma-
    # separated CLI flags without evaluating the value as shell code.
    printf '%s' "$1" | tr -d '[]"[:space:]' | sed 's/,$//'
}

_export_native_config() {
    local key env_name value target_device target_devices
    target_device=$(_lucebox_config_get placement.target_device)
    target_devices=$(_lucebox_config_get placement.target_devices)
    # A resolved placement uses physical device indexes and therefore needs the
    # complete backend inventory. Legacy configs without [placement] retain the
    # wrapper's historical primary-device isolation.
    if [ -n "$target_device" ] || [ -n "$target_devices" ]; then
        unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    else
        if [ -n "${LUCEBOX_HOST_ROCR_VISIBLE_DEVICES:-}" ]; then
            export ROCR_VISIBLE_DEVICES="$LUCEBOX_HOST_ROCR_VISIBLE_DEVICES"
            unset HIP_VISIBLE_DEVICES
        elif [ -n "${LUCEBOX_HOST_HIP_VISIBLE_DEVICES:-}" ]; then
            export HIP_VISIBLE_DEVICES="$LUCEBOX_HOST_HIP_VISIBLE_DEVICES"
            unset ROCR_VISIBLE_DEVICES
        fi
        if [ -n "${LUCEBOX_HOST_CUDA_VISIBLE_DEVICES:-}" ]; then
            export CUDA_VISIBLE_DEVICES="$LUCEBOX_HOST_CUDA_VISIBLE_DEVICES"
        fi
    fi
    while IFS='|' read -r key env_name; do
        [ -n "$key" ] || continue
        value=$(_lucebox_config_get "$key")
        [ -n "$value" ] || continue
        case "$key" in
            dflash.speculative_decode|dflash.lazy|dflash.spark|\
            dflash.debug_thinking_logits|placement.remote_draft|\
            placement.remote_target_shard|placement.peer_access)
                case "$value" in
                    true|1|yes|on) value=1 ;;
                    *)             value=0 ;;
                esac
                ;;
            placement.target_devices|placement.target_layer_split)
                value=$(_toml_array_to_csv "$value")
                ;;
            dflash.prefill_drafter)
                case "$value" in
                    /opt/lucebox-hub/server/models/*)
                        value="$DEFAULT_MODELS_DIR/${value#/opt/lucebox-hub/server/models/}"
                        ;;
                esac
                ;;
        esac
        export "$env_name=$value"
    done <<'EOF'
dflash.budget|DFLASH_BUDGET
dflash.max_ctx|DFLASH_MAX_CTX
dflash.speculative_decode|DFLASH_SPECULATIVE_DECODE
dflash.lazy|DFLASH_LAZY
dflash.prefix_cache_slots|DFLASH_PREFIX_CACHE_SLOTS
dflash.prefill_cache_slots|DFLASH_PREFILL_CACHE_SLOTS
dflash.cache_type_k|DFLASH_CACHE_TYPE_K
dflash.cache_type_v|DFLASH_CACHE_TYPE_V
dflash.prefill_mode|DFLASH_PREFILL_MODE
dflash.prefill_keep_ratio|DFLASH_PREFILL_KEEP
dflash.prefill_threshold|DFLASH_PREFILL_THRESHOLD
dflash.prefill_drafter|DFLASH_PREFILL_DRAFTER
dflash.kvflash|DFLASH_KVFLASH
dflash.kvflash_policy|DFLASH_KVFLASH_POLICY
dflash.kvflash_tau|DFLASH_KVFLASH_TAU
dflash.spark|DFLASH_SPARK
dflash.spark_vram_gb|DFLASH_SPARK_VRAM_GB
dflash.ds4_prefill|DFLASH_DS4_PREFILL
dflash.think_max|DFLASH_THINK_MAX
dflash.fa_window|DFLASH_FA_WINDOW
dflash.think_soft_close_min_ratio|DFLASH_THINK_SOFT_CLOSE_MIN_RATIO
dflash.debug_thinking_logits|DFLASH_DEBUG_THINKING_LOGITS
placement.mode|DFLASH_PLACEMENT_MODE
placement.target_device|DFLASH_TARGET_DEVICE
placement.target_devices|DFLASH_TARGET_DEVICES
placement.target_layer_split|DFLASH_TARGET_LAYER_SPLIT
placement.draft_device|DFLASH_DRAFT_DEVICE
placement.remote_draft|DFLASH_REMOTE_DRAFT
placement.remote_target_shard|DFLASH_REMOTE_TARGET_SHARD
placement.peer_access|DFLASH_PEER_ACCESS
placement.remote_expert_device|DFLASH_REMOTE_EXPERT_DEVICE
EOF
}

cmd_native_build() {
    local requested="${1:-}" repo backend build_dir jobs hip_wmma=OFF build_arches
    if [ "$requested" = "hybrid" ]; then
        [ -z "${LUCEBOX_BUILD_DIR:-}" ] \
            || die "LUCEBOX_BUILD_DIR cannot be shared by a hybrid build"
        ensure_probed
        [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ] \
            && [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ] \
            || die "hybrid build requires both NVIDIA and AMD GPUs"
        cmd_native_build cuda
        cmd_native_build rocm
        _LUCEBOX_HOST_PROBED=0
        ensure_probed
        [ "$LUCEBOX_HOST_HAS_HYBRID_RUNTIME" = "1" ] \
            || die "both builds completed, but the paired runtime contract is incomplete"
        ok "Hybrid CUDA + HIP runtime ready"
        return 0
    fi
    repo=$(_find_repo_root) \
        || die "native build requires a lucebox repository checkout (cd into it or set LUCEBOX_REPO)"
    ensure_probed
    backend=$(_native_backend "$requested")
    build_dir=$(_native_build_dir "$repo" "$backend")
    command -v cmake >/dev/null 2>&1 || die "cmake is required to build the inference engine"
    [ -f "$repo/server/deps/llama.cpp/ggml/CMakeLists.txt" ] \
        || die "git submodules are missing — run: git -C '$repo' submodule update --init --recursive"

    local configure=(cmake)
    # Prefer Ninja for a fresh build when it is available. Minimal buyer
    # images often ship Ninja with the ROCm SDK but omit GNU make; relying on
    # CMake's Unix Makefiles default makes an otherwise complete toolchain
    # fail before compiler detection. Preserve an existing build's generator.
    if [ ! -f "$build_dir/CMakeCache.txt" ] && command -v ninja >/dev/null 2>&1; then
        configure+=(-G Ninja)
    fi
    configure+=(
        -S "$repo/server" -B "$build_dir"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    )
    if [ "$backend" = "rocm" ]; then
        [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ] \
            || die "ROCm native build selected but no AMD GPU was detected"
        if [ -f /opt/rocm/include/rocwmma/rocwmma.hpp ] \
           || [ -f /usr/include/rocwmma/rocwmma.hpp ]; then
            hip_wmma=ON
        fi
        build_arches=$(_amd_build_arches)
        build_arches="${build_arches:-${LUCEBOX_HOST_AMD_GPU_ARCH:-gfx1151}}"
        configure+=(
            -DDFLASH27B_GPU_BACKEND=hip
            "-DDFLASH27B_HIP_ARCHITECTURES=$build_arches"
            "-DDFLASH27B_HIP_SM80_EQUIV=$hip_wmma"
        )
    else
        [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ] \
            || die "CUDA native build selected but no NVIDIA GPU was detected"
        configure+=(-DDFLASH27B_GPU_BACKEND=cuda)
        build_arches=$(_nvidia_build_arches)
        build_arches="${build_arches:-${LUCEBOX_HOST_NVIDIA_GPU_ARCH:-$LUCEBOX_HOST_GPU_SM}}"
        if [[ "$build_arches" =~ ^[0-9]+([;][0-9]+)*$ ]]; then
            configure+=("-DCMAKE_CUDA_ARCHITECTURES=$build_arches")
        fi
    fi
    info "Configuring native $backend build in $build_dir"
    "${configure[@]}"
    jobs="${LUCEBOX_HOST_NPROC:-1}"
    [ "$jobs" -gt 0 ] 2>/dev/null || jobs=1
    info "Building dflash_server + backend_ipc_daemon ($jobs jobs)"
    cmake --build "$build_dir" \
        --target dflash_server backend_ipc_daemon -j "$jobs"
    ok "Native engine ready: $build_dir/dflash_server"
    ok "Backend companion ready: $build_dir/backend_ipc_daemon"
}

cmd_package_runtime() {
    local repo destination stage runtime_dir backend build_dir package_backend
    repo=$(_find_repo_root) \
        || die "runtime packaging requires a lucebox repository checkout"
    destination="${1:-$repo/dist/lucebox-runtime}"
    case "$destination" in
        ""|/|"$repo") die "refusing unsafe runtime package destination: $destination" ;;
    esac
    [ ! -e "$destination" ] \
        || die "runtime package destination already exists: $destination"

    for backend in cuda rocm; do
        build_dir=$(_native_build_dir "$repo" "$backend")
        [ -x "$build_dir/dflash_server" ] \
            || die "missing $backend server — run '$SCRIPT_NAME build hybrid' first"
        [ -x "$build_dir/backend_ipc_daemon" ] \
            || die "missing $backend companion — run '$SCRIPT_NAME build hybrid' first"
    done

    stage=$(mktemp -d -t lucebox-runtime.XXXXXX) \
        || die "could not create runtime staging directory"
    trap 'if [ -n "${stage:-}" ] && [ -d "$stage" ]; then rm -rf -- "$stage"; fi' EXIT
    runtime_dir="$stage/lucebox-runtime"
    mkdir -p "$runtime_dir/cuda" "$runtime_dir/hip" \
        "$runtime_dir/server/scripts" "$runtime_dir/server/share"

    for backend in cuda rocm; do
        build_dir=$(_native_build_dir "$repo" "$backend")
        package_backend="$backend"
        [ "$backend" = "rocm" ] && package_backend=hip
        cp "$build_dir/dflash_server" "$runtime_dir/$package_backend/"
        cp "$build_dir/backend_ipc_daemon" "$runtime_dir/$package_backend/"
        if [ -d "$build_dir/deps" ]; then
            cp -R "$build_dir/deps" "$runtime_dir/$package_backend/deps"
            find "$runtime_dir/$package_backend/deps" -type f \
                ! -name 'lib*.so*' -delete
            find "$runtime_dir/$package_backend/deps" -depth -type d \
                -empty -delete
        fi
    done
    cp "$repo/server/scripts/entrypoint.sh" "$runtime_dir/server/scripts/"
    cp -R "$repo/server/share/." "$runtime_dir/server/share/"
    printf 'source_commit=%s\ncreated_at=%s\n' \
        "$(git -C "$repo" rev-parse HEAD 2>/dev/null || printf unknown)" \
        "$(date -u +%FT%TZ 2>/dev/null || printf unknown)" \
        > "$runtime_dir/MANIFEST"

    mkdir -p "$(dirname "$destination")"
    mv "$runtime_dir" "$destination"
    rmdir "$stage" 2>/dev/null || true
    stage=""
    trap - EXIT
    ok "Paired native runtime packaged: $destination"
    hint "Factory install location: /opt/lucebox/runtime"
}

cmd_native_serve() {
    local repo backend build_dir binary selected=() target draft model_id
    ensure_probed
    if _config_requires_hybrid_runtime; then
        cmd_hybrid_serve
        return $?
    fi
    repo=$(_find_repo_root) \
        || die "native run requires a lucebox repository checkout (cd into it or set LUCEBOX_REPO)"
    backend=$(_native_backend "${1:-}")
    build_dir=$(_native_build_dir "$repo" "$backend")
    binary="$build_dir/dflash_server"
    [ -x "$binary" ] \
        || die "native engine is not built — run '$SCRIPT_NAME build $backend' first"
    mapfile -t selected < <(_selected_model_paths)
    target="${selected[0]:-}"
    draft="${selected[1]:-none}"
    model_id="${selected[2]:-lucebox}"
    _model_artifact_ready "$target" \
        || die "selected target is not installed: $target — run '$SCRIPT_NAME models select'"
    if [ "$draft" != "none" ] && ! _model_artifact_ready "$draft"; then
        die "selected draft is not installed: $draft — run '$SCRIPT_NAME models select'"
    fi

    _export_native_config
    export DFLASH_DIR="$repo/server"
    export DFLASH_SERVER_BIN="$binary"
    export DFLASH_BACKEND_IPC_BIN="$build_dir/backend_ipc_daemon"
    export DFLASH_TARGET="$target"
    _export_selected_decode_companion "$model_id" "$draft"
    export DFLASH_HOST="${LUCEBOX_NATIVE_HOST:-127.0.0.1}"
    export DFLASH_PORT="$DEFAULT_PORT"
    export DFLASH_MODEL_NAME="$model_id"
    export LUCEBOX_NATIVE=1
    info "Starting native $backend engine at http://$DFLASH_HOST:$DFLASH_PORT"
    exec "$repo/server/scripts/entrypoint.sh" serve
}

HARNESS_ENGINE_PID=""
HARNESS_ENGINE_LOG=""

_stop_harness_engine() {
    if [ -n "$HARNESS_ENGINE_PID" ] \
       && kill -0 "$HARNESS_ENGINE_PID" 2>/dev/null; then
        kill "$HARNESS_ENGINE_PID" 2>/dev/null || true
        wait "$HARNESS_ENGINE_PID" 2>/dev/null || true
    fi
    HARNESS_ENGINE_PID=""
}

_wait_for_harness_engine() {
    local api_root="$1" model="$2" attempts=0
    while [ "$attempts" -lt 300 ]; do
        attempts=$((attempts + 1))
        _connector_api_ready "$api_root" "$model" && return 0
        if [ -n "$HARNESS_ENGINE_PID" ] \
           && ! kill -0 "$HARNESS_ENGINE_PID" 2>/dev/null; then
            err "the inference engine exited while loading"
            [ ! -f "$HARNESS_ENGINE_LOG" ] \
                || tail -n 120 "$HARNESS_ENGINE_LOG" >&2
            return 1
        fi
        sleep 1
    done
    err "the inference engine did not become ready within 5 minutes"
    [ ! -f "$HARNESS_ENGINE_LOG" ] || tail -n 120 "$HARNESS_ENGINE_LOG" >&2
    return 1
}

cmd_harness() {
    local repo name="${1:-}" backend script api_root model max_ctx log_dir rc=0
    repo=$(_find_repo_root) \
        || die "harness launchers are contributor tools; run this command inside the lucebox repository"
    if [ -z "$name" ]; then
        cat <<'EOF'
Choose a harness:
  1  Claude Code
  2  Codex
  3  OpenCode
  4  Hermes
  5  Pi
  6  OpenClaw
  7  Open WebUI
EOF
        printf 'Harness number: '
        IFS= read -r name || return 1
    fi
    case "$name" in
        1|claude|claude-code) name=claude_code ;;
        2|codex)              name=codex ;;
        3|opencode)           name=opencode ;;
        4|hermes)             name=hermes ;;
        5|pi)                 name=pi ;;
        6|openclaw)           name=openclaw ;;
        7|openwebui|webui)    name=openwebui ;;
        *) die "unknown harness '$name'" ;;
    esac
    script="$repo/harness/clients/run_${name}.sh"
    [ -x "$script" ] || die "harness launcher is missing: $script"
    ensure_probed
    backend=$(_native_backend "${LUCEBOX_HARNESS_BACKEND:-}")
    api_root=$(_connector_api_root)
    model=$(_connector_model_id)
    max_ctx=$(_connector_context_size)
    log_dir="$CONFIG_HOME/logs"
    _connector_private_dir "$log_dir"
    HARNESS_ENGINE_LOG="$log_dir/harness-engine.log"

    if _connector_api_ready "$api_root"; then
        _connector_api_ready "$api_root" "$model" \
            || die "the API at $api_root is running a different model; stop it or select its advertised model"
        # We do not own the reused process and therefore cannot promise a log
        # path to the harness.  Avoid exposing a stale log from an earlier
        # CLI-owned launch if the client later reports an API error.
        HARNESS_ENGINE_LOG=/dev/null
        info "Reusing the running Lucebox API at $api_root"
    else
        # The canonical native path owns model resolution, placement, and all
        # optimization flags. Harness scripts are protocol/client adapters;
        # they must not reconstruct a second, divergent engine command line.
        info "Starting the optimized Lucebox engine for $name"
        bash "$SCRIPT_PATH" native "$backend" >"$HARNESS_ENGINE_LOG" 2>&1 &
        HARNESS_ENGINE_PID=$!
        trap _stop_harness_engine EXIT
        _wait_for_harness_engine "$api_root" "$model" || return $?
    fi

    export REPO_DIR="$repo"
    export MODEL_SERVER=external
    export HOST=127.0.0.1
    export PORT="$DEFAULT_PORT"
    export MODEL_ID="$model"
    export MAX_CTX="$max_ctx"
    export SERVER_LOG="$HARNESS_ENGINE_LOG"
    info "Launching $name against the selected Lucebox model"
    "$script" || rc=$?
    _stop_harness_engine
    trap - EXIT
    return "$rc"
}

cmd_update() {
    # Download the wrapper itself as data; never execute a second remote
    # installer. Override the persisted channel with LUCEBOX_INSTALL_URL.
    local source_url target wrapper_tmp expected_sha actual_sha escaped_url
    source_url="${LUCEBOX_INSTALL_URL:-$LUCEBOX_INSTALLED_FROM}"
    if [[ "$source_url" != */lucebox.sh ]]; then
        die "LUCEBOX_INSTALLED_FROM doesn't end in /lucebox.sh: $source_url"
    fi
    case "$source_url" in
        *['"$`\']*|*$'\n'*|*$'\r'*)
            die "update URL contains unsafe characters: $source_url" ;;
    esac
    target=$(realpath "$SCRIPT_PATH")

    info "Updating lucebox"
    info "  source: $source_url"
    info "  target: $target"

    # Create the temporary file next to the destination so the final rename
    # is atomic even when /tmp and the install directory are different mounts.
    wrapper_tmp=$(mktemp "${target}.update.XXXXXX") \
        || die "couldn't create temporary update file next to $target"
    trap 'if [ -n "${wrapper_tmp:-}" ]; then rm -f "$wrapper_tmp" "$wrapper_tmp.baked"; fi' EXIT
    curl --connect-timeout 10 --max-time 120 --retry 2 --retry-delay 1 \
        -fsSL "$source_url" -o "$wrapper_tmp" \
        || die "failed to download wrapper from $source_url"

    [ "$(head -1 "$wrapper_tmp")" = '#!/usr/bin/env bash' ] \
        || die "downloaded wrapper has an unexpected shebang"
    grep -Fqx 'set -euo pipefail' "$wrapper_tmp" \
        || die "downloaded wrapper is missing strict shell mode"
    grep -q '^VERSION=' "$wrapper_tmp" \
        || die "downloaded file is missing the Lucebox version marker"
    grep -q '^LUCEBOX_INSTALLED_FROM=' "$wrapper_tmp" \
        || die "downloaded file is missing the Lucebox update-channel marker"
    bash -n "$wrapper_tmp" \
        || die "downloaded wrapper does not parse as valid Bash"

    expected_sha="${LUCEBOX_WRAPPER_SHA256:-}"
    if [ -n "$expected_sha" ]; then
        [[ "$expected_sha" =~ ^[0-9a-fA-F]{64}$ ]] \
            || die "LUCEBOX_WRAPPER_SHA256 must be exactly 64 hexadecimal characters"
        actual_sha=$(sha256_file "$wrapper_tmp")
        expected_sha=$(printf '%s' "$expected_sha" | tr '[:upper:]' '[:lower:]')
        [ "$actual_sha" = "$expected_sha" ] \
            || die "wrapper checksum mismatch (expected $expected_sha, got $actual_sha)"
        ok "wrapper sha256 verified"
    fi

    # Preserve the chosen branch/fork in the new copy. The URL was validated
    # above for safe embedding in a Bash double-quoted string.
    escaped_url=$(printf '%s' "$source_url" | sed 's/[&|]/\\&/g')
    sed "s|^LUCEBOX_INSTALLED_FROM=.*|LUCEBOX_INSTALLED_FROM=\"$escaped_url\"|" \
        "$wrapper_tmp" > "$wrapper_tmp.baked"
    mv "$wrapper_tmp.baked" "$wrapper_tmp"
    grep -Fqx "LUCEBOX_INSTALLED_FROM=\"$source_url\"" "$wrapper_tmp" \
        || die "failed to preserve update channel in downloaded wrapper"
    bash -n "$wrapper_tmp" || die "updated wrapper failed validation after channel rewrite"

    chmod +x "$wrapper_tmp"
    mv "$wrapper_tmp" "$target"
    trap - EXIT
    ok "updated lucebox → $target"
}

cmd_completion() {
    # Print shell completion script for bash / zsh / fish. Usage:
    #
    #   # bash  (in ~/.bashrc):
    #   source <(lucebox completion bash)
    #
    #   # zsh  (in ~/.zshrc, before `compinit`):
    #   source <(lucebox completion zsh)
    #
    #   # fish:
    #   lucebox completion fish | source
    #
    # Keep this in sync with the dispatch table in main() and the sub-app
    # verbs (config get/set/unset, models list/download/select). Adding a new
    # top-level command means adding it here too.
    local shell="${1:-}"
    case "$shell" in
        bash)
            cat <<'BASH'
# lucebox bash completion. Source from ~/.bashrc:
#   source <(lucebox completion bash)
_lucebox_complete() {
    local cur prev cmds config_verbs models_verbs connector_names completion_shells
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmds="menu setup calibrate connect install uninstall start stop restart enable disable status logs \
          serve build package-runtime native harness pull update check completion config models \
          optimize print-run help version"
    config_verbs="get set unset"
    models_verbs="list download select"
    connector_names="claude codex opencode hermes pi openclaw openwebui"
    completion_shells="bash zsh fish"

    # Sub-app verbs / shell args.
    case "$prev" in
        config)     COMPREPLY=( $(compgen -W "$config_verbs"     -- "$cur") ); return ;;
        models)     COMPREPLY=( $(compgen -W "$models_verbs"     -- "$cur") ); return ;;
        connect)    COMPREPLY=( $(compgen -W "$connector_names"  -- "$cur") ); return ;;
        completion) COMPREPLY=( $(compgen -W "$completion_shells" -- "$cur") ); return ;;
    esac

    # Top-level command.
    if [ "$COMP_CWORD" = 1 ]; then
        COMPREPLY=( $(compgen -W "$cmds" -- "$cur") )
        return
    fi
}
complete -F _lucebox_complete lucebox lucebox.sh
BASH
            ;;
        zsh)
            # Bash-compat shim: zsh sources our bash completion through
            # bashcompinit. Users who prefer native zsh _arguments-style
            # completion can write their own; this gets `<TAB>` working
            # in two lines for free.
            cat <<'ZSH'
# lucebox zsh completion. Source from ~/.zshrc (after compinit):
#   source <(lucebox completion zsh)
autoload -Uz compinit bashcompinit
compinit
bashcompinit
ZSH
            cmd_completion bash
            ;;
        fish)
            cat <<'FISH'
# lucebox fish completion. Source from ~/.config/fish/config.fish:
#   lucebox completion fish | source
complete -c lucebox -f
set -l __lucebox_cmds menu setup calibrate connect install uninstall start stop restart enable disable \
    status logs serve build package-runtime native harness pull update check completion config models \
    optimize print-run help version
for cmd in $__lucebox_cmds
    complete -c lucebox -n "not __fish_seen_subcommand_from $__lucebox_cmds" -a $cmd
end
complete -c lucebox -n "__fish_seen_subcommand_from config" -a "get set unset"
complete -c lucebox -n "__fish_seen_subcommand_from models" -a "list download select"
complete -c lucebox -n "__fish_seen_subcommand_from connect" -a "claude codex opencode hermes pi openclaw openwebui"
complete -c lucebox -n "__fish_seen_subcommand_from completion" -a "bash zsh fish"
FISH
            ;;
        ""|--help|-h)
            cat <<EOF
$SCRIPT_NAME completion {bash|zsh|fish}

Emits a shell completion script. Source it from your shell's rc file:

  bash:  source <($SCRIPT_NAME completion bash)
  zsh:   source <($SCRIPT_NAME completion zsh)
  fish:  $SCRIPT_NAME completion fish | source
EOF
            ;;
        *)
            die "unknown shell: $shell — want bash, zsh, or fish" ;;
    esac
}

cmd_check() {
    # Host-only readiness report. Pure shell — never enters the container,
    # since the point is to verify the host can run the container in the
    # first place. Reuses probe_host (LUCEBOX_HOST_* env vars) for the
    # actual detection so the formatting is the only thing here.
    ensure_probed

    local variant
    variant=$(pick_variant)

    # Two-column grid: "  name        ✓  detail" — matches the visual
    # style of the lucebench preflight output.
    local mark
    _row() {
        # Brace every var ref so multi-byte glyphs (✓ ✗) don't get parsed
        # as part of the identifier — some bash builds with permissive
        # locales count them as identifier characters and `set -u` then
        # errors out on the resulting "C_OK✓" / "C_ERR✗" names.
        if [ "$1" = "1" ]; then mark="${C_OK}✓${C_RST}"
        elif [ "$1" = "warn" ]; then mark="${C_WARN}!${C_RST}"
        else mark="${C_ERR}✗${C_RST}"; fi
        printf '  %-22s %b  %s\n' "$2" "$mark" "$3"
    }

    echo "[lucebox] host readiness report"

    # docker
    if [ "$LUCEBOX_HOST_HAS_DOCKER" = "1" ]; then
        _row 1 "docker daemon" "reachable (server ${LUCEBOX_HOST_DOCKER_VERSION:-?})"
    elif command -v docker &>/dev/null; then
        _row 0 "docker daemon" "installed but unreachable — start the daemon or add user to 'docker' group"
    else
        _row 0 "docker daemon" "not installed — https://docs.docker.com/engine/install/"
    fi

    if [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ]; then
        # nvidia container toolkit
        case "$LUCEBOX_HOST_HAS_CTK" in
            runtime)            _row 1    "nvidia ctk" "wired into docker (runtime)" ;;
            cdi)                _row 1    "nvidia ctk" "wired via CDI (nvidia.com/gpu)" ;;
            installed-unwired)  _row warn "nvidia ctk" "installed but not registered with docker — sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" ;;
            none|*)             _row 0    "nvidia ctk" "not installed — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" ;;
        esac

        local required_driver cuda_label memory_label
        if _variant_is_cuda13 "$variant"; then
            required_driver=$MIN_DRIVER_CUDA13
            cuda_label=cuda13
        elif _variant_is_cuda128 "$variant"; then
            required_driver=$MIN_DRIVER_CUDA128
            cuda_label=cuda128
        else
            required_driver=$MIN_DRIVER_CUDA12
            cuda_label=cuda12
        fi
        if [ "$LUCEBOX_HOST_DRIVER_MAJOR" -ge "$required_driver" ]; then
            _row 1 "nvidia driver" "$LUCEBOX_HOST_DRIVER_VERSION (≥ $required_driver required for $cuda_label)"
        else
            _row 0 "nvidia driver" "$LUCEBOX_HOST_DRIVER_VERSION (< $required_driver — $cuda_label image will fail)"
        fi
        memory_label="${LUCEBOX_HOST_VRAM_GB} GB VRAM"
        if [ "$LUCEBOX_HOST_NVIDIA_UNIFIED_MEMORY" = "1" ]; then
            memory_label="${LUCEBOX_HOST_VRAM_GB} GB usable shared memory"
        fi
        _row 1 "nvidia gpu" "$LUCEBOX_HOST_GPU_NAME × $LUCEBOX_HOST_GPU_COUNT (sm_$LUCEBOX_HOST_GPU_SM, $memory_label)"
        if _variant_is_cuda13 "$variant"; then
            case "$LUCEBOX_HOST_GPU_SM" in
                121) _row 1    "cuda13 arch" "sm_121 covered by the arm64 image" ;;
                "")  _row warn "cuda13 arch" "compute capability not detected" ;;
                *)   _row warn "cuda13 arch" "sm_$LUCEBOX_HOST_GPU_SM is not covered by the arm64 image (121)" ;;
            esac
        elif _variant_is_cuda128 "$variant"; then
            case "$LUCEBOX_HOST_GPU_SM" in
                120) _row 1    "cuda128 arch" "sm_120 covered by the CUDA 12.8 image" ;;
                "")  _row warn "cuda128 arch" "compute capability not detected" ;;
                *)   _row warn "cuda128 arch" "sm_$LUCEBOX_HOST_GPU_SM is not covered by the CUDA 12.8 image (120)" ;;
            esac
        else
            case "$LUCEBOX_HOST_GPU_SM" in
                75|80|86|89|90) _row 1    "cuda12 arch" "sm_$LUCEBOX_HOST_GPU_SM covered by image" ;;
                "")             _row warn "cuda12 arch" "compute capability not detected" ;;
                *)              _row warn "cuda12 arch" "sm_$LUCEBOX_HOST_GPU_SM not in image arch list (75;80;86;89;90)" ;;
            esac
        fi
    elif command -v nvidia-smi &>/dev/null; then
        _row 0 "nvidia driver" "nvidia-smi present but NVML calls fail — driver/library mismatch, try reboot"
    fi

    if [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ]; then
        if [ -n "$LUCEBOX_HOST_ROCM_VERSION" ]; then
            _row 1 "rocm" "$LUCEBOX_HOST_ROCM_VERSION"
        else
            _row warn "rocm" "GPU detected, userspace version unavailable"
        fi
        _row 1 "amd gpu" "${LUCEBOX_HOST_AMD_GPU_COUNT} device(s); primary ${LUCEBOX_HOST_AMD_GPU_NAME} (${LUCEBOX_HOST_AMD_GPU_ARCH}, ${LUCEBOX_HOST_AMD_VRAM_GB} GB effective)"
        if [ "$LUCEBOX_HOST_AMD_GPU_COUNT" -gt 1 ]; then
            local selected_device
            selected_device="${LUCEBOX_HOST_ROCR_VISIBLE_DEVICES:-${LUCEBOX_HOST_HIP_VISIBLE_DEVICES:-0}}"
            _row 1 "gpu placement" "Automatic resolves single- or multi-GPU execution after model selection (primary ${selected_device})"
        fi
        local amd_line amd_device_idx amd_device_name amd_device_arch amd_device_mem
        while IFS= read -r amd_line; do
            [ -n "$amd_line" ] || continue
            amd_device_idx=$(printf '%s' "$amd_line" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}')
            amd_device_name=$(printf '%s' "$amd_line" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}')
            amd_device_arch=$(printf '%s' "$amd_line" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}')
            amd_device_mem=$(printf '%s' "$amd_line" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6); print $6}')
            _row 1 "  amd[$amd_device_idx]" "$amd_device_name ($amd_device_arch, $amd_device_mem physical)"
        done <<<"$LUCEBOX_HOST_AMD_GPU_LIST_CSV"
        case "$LUCEBOX_HOST_AMD_GPU_ARCH" in
            gfx1100|gfx1151|gfx1200|gfx1201) _row 1    "rocm arch" "$LUCEBOX_HOST_AMD_GPU_ARCH covered by published image" ;;
            "")                            _row warn "rocm arch" "gfx architecture not detected" ;;
            *)                             _row warn "rocm arch" "$LUCEBOX_HOST_AMD_GPU_ARCH not in published image arch list" ;;
        esac
        if [ "$LUCEBOX_HOST_HAS_KFD" = "1" ]; then
            _row 1 "amd /dev/kfd" "accessible"
        else
            _row 0 "amd /dev/kfd" "missing or inaccessible — user needs the render group"
        fi
        if [ "$LUCEBOX_HOST_HAS_DRI" = "1" ]; then
            _row 1 "amd /dev/dri" "render node accessible"
        else
            _row 0 "amd /dev/dri" "no accessible renderD* node — user needs render/video groups"
        fi
    fi

    if [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" != "1" ] \
       && [ "$LUCEBOX_HOST_HAS_AMD_GPU" != "1" ]; then
        _row 0 "gpu" "no supported NVIDIA or AMD GPU detected"
    fi

    # systemd
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ]; then
        _row 1 "user systemd" "available (needed for '$SCRIPT_NAME install')"
    elif [ "$LUCEBOX_HOST_IS_WSL" = "1" ]; then
        _row warn "user systemd" "WSL detected — set 'systemd=true' under [boot] in /etc/wsl.conf, then 'wsl --shutdown'"
    else
        _row warn "user systemd" "not available — '$SCRIPT_NAME install' (service unit) won't work; '$SCRIPT_NAME serve' (foreground) will"
    fi

    # Selected backend and image. On heterogeneous builds the unselected APU
    # stays visible in the inventory above but does not change this decision.
    if _variant_is_rocm "$variant"; then
        if [ "$LUCEBOX_HOST_HAS_AMD_GPU" != "1" ]; then
            _row 0 "image" "${IMAGE_BASE}:${variant} — requires an AMD GPU"
        elif [ "$LUCEBOX_HOST_HAS_KFD" != "1" ] || [ "$LUCEBOX_HOST_HAS_DRI" != "1" ]; then
            _row 0 "image" "${IMAGE_BASE}:${variant} — needs accessible /dev/kfd and /dev/dri"
        else
            _row 1 "image" "${IMAGE_BASE}:${variant} (AMD selected)"
        fi
    else
        if [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" != "1" ]; then
            _row 0 "image" "${IMAGE_BASE}:${variant} — requires an NVIDIA GPU"
        elif [ "$LUCEBOX_HOST_HAS_CTK" = "none" ] || [ "$LUCEBOX_HOST_HAS_CTK" = "installed-unwired" ]; then
            _row 0 "image" "${IMAGE_BASE}:${variant} — needs NVIDIA Container Toolkit wired into docker"
        else
            _row 1 "image" "${IMAGE_BASE}:${variant} (NVIDIA selected)"
        fi
    fi
    # RAM / cores (informational)
    _row 1 "host" "${LUCEBOX_HOST_NPROC} cpus, ${LUCEBOX_HOST_RAM_GB} GB RAM"
}

cmd_in_container() {
    # Generic dispatcher: anything that isn't a systemd action goes here.
    # Runs the in-container Python CLI with the supplied argv.
    ensure_probed
    # CTK isn't strictly required for every subcommand (e.g. `config get`
    # or `autotune` only touch local files), but the server-spawning
    # subcommands need it.
    # Letting docker error its own way is fine for the no-CTK case.
    local variant
    variant=$(pick_variant)
    require_host_prereqs "$variant"
    local caller_has_tty=0 argv
    if [ -t 0 ] && [ -t 1 ]; then
        caller_has_tty=1
    fi
    mapfile -t argv < <(build_orchestrator_argv "$variant" "$caller_has_tty" "$@")
    exec "${argv[@]}"
}

# Is the long-running lucebox container currently up? Used by the dispatcher
# to decide between `docker exec` into it (cheap, shares the running server's
# network namespace so localhost:8080 reaches the server) vs. `docker run`
# (cold start, isolated network — can't reach the live server).
#
# `docker ps -q -f name=^<CONTAINER>$` prints the container id when running,
# empty otherwise. The anchored regex avoids matching `lucebox-cli-12345`
# style ephemeral siblings.
_lucebox_container_running() {
    # No docker on PATH → definitely not running. Don't even probe.
    command -v docker >/dev/null 2>&1 || return 1
    local id
    id=$(docker ps -q -f "name=^${CONTAINER_NAME}\$" 2>/dev/null || true)
    [ -n "$id" ]
}

# `docker exec` variant of cmd_in_container. Same calling convention, but:
#   - shares the running container's network namespace (localhost:8080 → the
#     server), filesystem, and mounts — no bind mounts needed.
#   - skips the ~1-3s cold-start cost of a fresh `docker run --rm`.
#   - only safe for steady-state / read-only / config-only subcommands. Any
#     command that restarts the lucebox service (calibrate, serve)
#     would kill the very container the exec is in — caller must route those
#     to cmd_in_container instead.
#
# Pass through the same env-var subset the run path uses so the in-container
# CLI sees consistent overrides whichever route it took: HOME, every
# LUCEBOX_HOST_*, the image/port/container/models scalars, and HF_TOKEN.
cmd_exec_in_container() {
    ensure_probed
    local variant
    variant=$(pick_variant)
    require_host_prereqs "$variant"
    local tty=()
    _set_tty_flags tty
    local argv=(docker exec "${tty[@]}")
    argv+=(--user "$(id -u):$(id -g)")
    argv+=(-w /opt/lucebox-hub)
    argv+=(-e "HOME=$CONFIG_HOME")
    _append_host_env argv
    _append_selected_backend_facts argv "$variant"
    _append_scalar_env argv "$variant"
    # The image has no top-level `lucebox` binary on PATH — that name only
    # works as the first arg to /opt/lucebox-hub/server/scripts/entrypoint.sh,
    # which then `exec uv run ... python -m lucebox`s. docker exec bypasses
    # the image's ENTRYPOINT, so we invoke the entrypoint shim explicitly
    # with `lucebox` as its SUBCMD and the user's argv tail. Keeps the
    # exec path bit-for-bit equivalent to what docker run does on the
    # SUBCMD=lucebox branch.
    argv+=("$CONTAINER_NAME" /opt/lucebox-hub/server/scripts/entrypoint.sh lucebox "$@")
    exec "${argv[@]}"
}

# Decide whether a given (subcommand, argv) pair is safe to run via
# `docker exec` into the live container. Returns 0 (yes, prefer exec) or 1
# (no, must use docker run / host-side).
#
# The safe-to-exec set is exactly the steady-state / read-only / hits-the-
# running-server subcommands. Anything that restarts the service, mutates
# images, or is itself the long-running service must stay on cmd_in_container.
#
_lucebox_prefer_exec() {
    local cmd="$1"; shift
    case "$cmd" in
        config|models|optimize|check|print-run|print-serve-argv)
            return 0
            ;;
        _calibration)
            case "${1:-}" in
                apply|budgets|finish|probe|status) return 0 ;;
            esac
            return 1
            ;;
        *)
            return 1
            ;;
    esac
}

# Top-level routing for the in-container Python CLI. Picks between exec
# (cheap, shares the live server's namespace) and run (cold start, isolated).
#
# Decision tree:
#   1. LUCEBOX_NO_EXEC=1 / --no-exec was set → always run, never exec.
#      Useful for debugging the wrapper or when the in-container Python is
#      stale relative to the image.
#   2. cmd is not in the prefer-exec list → run (service mutators).
#   3. container is running → exec (the fast path, hits the live server).
#   4. container is not running → run (fall back so first-run / pre-install
#      flows still work without a live service).
cmd_route_to_container() {
    local cmd="$1"; shift
    if [ "${LUCEBOX_NO_EXEC:-0}" = "1" ]; then
        cmd_in_container "$cmd" "$@"
        return
    fi
    if _lucebox_prefer_exec "$cmd" "$@" && _lucebox_container_running; then
        cmd_exec_in_container "$cmd" "$@"
        return
    fi
    cmd_in_container "$cmd" "$@"
}

# ── interactive product surface ───────────────────────────────────────────

_engine_state() {
    if command -v systemctl >/dev/null 2>&1 \
       && systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null; then
        printf 'running'
    elif _lucebox_container_running; then
        printf 'running'
    else
        printf 'stopped'
    fi
}

_menu_clear() {
    if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ] \
       && [ "${LUCEBOX_NO_CLEAR:-0}" != "1" ]; then
        printf '\033[2J\033[H'
    fi
}

_menu_pause() {
    [ -t 0 ] || return 0
    printf '\nPress Enter to return to the menu…'
    IFS= read -r _ || true
}

_menu_run() {
    # Always invoke through bash: repository copies are not necessarily marked
    # executable, while installed buyer copies are. This behaves identically
    # in both places and lets exec-heavy subcommands return to the menu.
    local rc
    if bash "$SCRIPT_PATH" "$@"; then
        return 0
    else
        rc=$?
        warn "Command failed (exit $rc)."
        return "$rc"
    fi
}

_menu_start() {
    ensure_probed
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ] && [ -f "$UNIT_PATH" ]; then
        _menu_run start
        return
    fi
    warn "The background service is not installed."
    if _confirm "Run the Docker engine in this terminal now?" 1; then
        _menu_run serve
    else
        hint "Run '$SCRIPT_NAME setup' to install the background service."
    fi
}

_menu_restart_if_running() {
    [ "$(_engine_state)" = "running" ] || return 0
    if ! _confirm "Restart the running engine to apply this change?" 1; then
        warn "The change is saved and will apply at the next restart."
        return 0
    fi
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ] && [ -f "$UNIT_PATH" ]; then
        _menu_run restart
    else
        _menu_run stop
        warn "The foreground engine was stopped. Start it again from menu option 5."
    fi
}

_ensure_configured_image() {
    # Model activation may switch backends on a mixed machine when the current
    # backend cannot place the selected model. Keep guided setup/menu one-step:
    # install the newly selected runtime before optimize/start tries to use it.
    local selected
    selected=$(_lucebox_config_get image.variant)
    [ -n "$selected" ] || selected=$(pick_variant)
    if docker image inspect "${IMAGE_BASE}:${selected}" >/dev/null 2>&1; then
        return 0
    fi
    if ! _confirm "Download the ${selected} inference image required by this model?" 1; then
        warn "The model is configured, but ${IMAGE_BASE}:${selected} is not installed."
        hint "Run '$SCRIPT_NAME pull' before starting the engine."
        return 1
    fi
    LUCEBOX_VARIANT="$selected" bash "$SCRIPT_PATH" pull
}

cmd_setup() {
    [ -t 0 ] && [ -t 1 ] \
        || die "guided setup needs a terminal — use '$SCRIPT_NAME --help' for non-interactive commands"
    ensure_probed
    _menu_clear
    print_logo
    printf '%bQuick setup%b\n\n' "$C_INFO" "$C_RST"
    cmd_check
    printf '\n'

    local variant
    variant=$(pick_variant)
    if [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ] \
       && [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ]; then
        printf 'This build has NVIDIA and AMD graphics. Which accelerator should run Lucebox?\n'
        printf '  1  NVIDIA / CUDA  %b(recommended for RTX + Strix builds)%b\n' "$C_DIM" "$C_RST"
        printf '  2  AMD / ROCm\n'
        printf 'Choice [1]: '
        local backend_choice
        IFS= read -r backend_choice || return 1
        case "$backend_choice" in
            2) variant=rocm ;;
            *) variant=$(_default_cuda_variant) ;;
        esac
    fi
    info "Selected backend: $variant"

    if docker image inspect "${IMAGE_BASE}:${variant}" >/dev/null 2>&1; then
        ok "Inference image is already installed (${IMAGE_BASE}:${variant})"
    else
        _confirm "Download the ${variant} inference image now?" 1 \
            || { warn "Setup stopped before the image download."; return 1; }
        LUCEBOX_VARIANT="$variant" bash "$SCRIPT_PATH" pull || return $?
    fi

    # Persist the accelerator choice only after its image is available; the
    # config writer lives in that image on buyer installations.
    LUCEBOX_VARIANT="$variant" bash "$SCRIPT_PATH" config set "variant=$variant" \
        || return $?

    printf '\n'
    bash "$SCRIPT_PATH" models select || return $?
    if [ -z "$(_lucebox_config_get model.preset)" ]; then
        warn "No model was selected; setup stopped without starting the engine."
        return 1
    fi
    _ensure_configured_image || return $?

    printf '\n'
    local optimized=0
    if _confirm "Enable recommended optimizations (may add a ~1.2 GB shared scorer)?" 1; then
        bash "$SCRIPT_PATH" optimize --yes || return $?
        optimized=1
    else
        hint "The safe base profile is active; optional scorer-based features remain off."
    fi

    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ]; then
        if [ ! -f "$UNIT_PATH" ]; then
            printf '\n'
            if _confirm "Install Lucebox as a background service?" 1; then
                bash "$SCRIPT_PATH" install || return $?
            fi
        fi
        if [ "$optimized" = "1" ] && [ -f "$UNIT_PATH" ] \
           && _confirm "Measure and calibrate this model on this GPU? (one-time; several minutes)" 1; then
            bash "$SCRIPT_PATH" calibrate \
                || warn "Calibration was skipped; the Automatic profile is unchanged."
        fi
        if [ -f "$UNIT_PATH" ] && _confirm "Start the inference engine now?" 1; then
            if [ "$(_engine_state)" = "running" ]; then
                bash "$SCRIPT_PATH" restart || return $?
            else
                bash "$SCRIPT_PATH" start || return $?
            fi
        fi
    else
        warn "Background services are unavailable on this host."
        hint "Use '$SCRIPT_NAME serve' to run the engine in the foreground."
    fi

    printf '\n'
    ok "Lucebox is configured"
    hint "API:    http://127.0.0.1:$DEFAULT_PORT/v1"
    hint "Menu:   $SCRIPT_NAME"
    hint "Status: $SCRIPT_NAME status"

    printf '\n'
    if _confirm "Link an installed AI harness now (without opening it)?" 1; then
        bash "$SCRIPT_PATH" connect --no-launch \
            || warn "The engine is ready; harness linking was not completed."
    fi
}

cmd_developer_menu() {
    local repo choice
    repo=$(_find_repo_root) \
        || { warn "Developer tools require a lucebox repository checkout."; return 1; }
    while true; do
        _menu_clear
        print_logo
        printf '%bDeveloper tools%b\n' "$C_INFO" "$C_RST"
        printf 'Repository: %s\n\n' "$repo"
        printf '  1  Build the native inference engine\n'
        printf '  2  Run the native inference engine\n'
        printf '  3  Choose and run a client harness\n'
        printf '  4  Package the CUDA + HIP buyer runtime\n'
        printf '  b  Back\n\n'
        printf 'Choose: '
        IFS= read -r choice || return 0
        case "$choice" in
            1) _menu_run build; _menu_pause ;;
            2) _menu_run native; _menu_pause ;;
            3) _menu_run harness; _menu_pause ;;
            4) _menu_run package-runtime; _menu_pause ;;
            b|B|q|Q) return 0 ;;
            *) warn "Choose 1–4 or b"; _menu_pause ;;
        esac
    done
}

_optimization_summary() {
    local mode model decode prefill kvflash spark draft_file active=""
    mode=$(_lucebox_config_get autotune.mode)
    model=$(_lucebox_config_get model.preset)
    decode=$(_lucebox_config_get dflash.speculative_decode)
    prefill=$(_lucebox_config_get dflash.prefill_mode)
    kvflash=$(_lucebox_config_get dflash.kvflash)
    spark=$(_lucebox_config_get dflash.spark)

    if [ -z "$mode" ]; then
        if [ -n "$(_lucebox_config_get dflash.max_ctx)" ]; then
            mode="custom"
        else
            mode="not configured"
        fi
    fi
    case "$decode" in false|0|no|off) ;; *)
        case "$model" in
            qwen3.6-27b|gemma-4-26b|gemma-4-31b|deepseek-v4-flash)
                draft_file=$(_lucebox_config_get model.draft_file)
                if [ -z "$draft_file" ]; then
                    case "$model" in
                        qwen3.6-27b) draft_file="dflash-draft-3.6-q4_k_m.gguf" ;;
                        gemma-4-26b) draft_file="gemma-4-26B-A4B-it-DFlash-q8_0.gguf" ;;
                        gemma-4-31b) draft_file="gemma-4-31B-it-DFlash-q8_0.gguf" ;;
                        deepseek-v4-flash)
                            draft_file="DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf"
                            ;;
                    esac
                fi
                [ -s "$DEFAULT_MODELS_DIR/draft/$draft_file" ] && active="DFlash"
                ;;
            laguna-xs.2)
                _model_artifact_ready "$DEFAULT_MODELS_DIR/draft/laguna-xs2-speculator" \
                    && active="DFlash"
                ;;
        esac
        ;;
    esac
    if [ -n "$prefill" ] && [ "$prefill" != "off" ]; then
        active="${active:+$active, }PFlash"
    fi
    if [ -n "$kvflash" ] && [ "$kvflash" != "off" ]; then
        active="${active:+$active, }KVFlash"
    fi
    case "$spark" in true|1|yes|on) active="${active:+$active, }Spark" ;; esac
    printf '%s (%s)' "$mode" "${active:-standard engine}"
}

_placement_summary() {
    local mode target targets draft remote_expert
    mode=$(_lucebox_config_get placement.mode)
    target=$(_lucebox_config_get placement.target_device)
    targets=$(_toml_array_to_csv \
        "$(_lucebox_config_get placement.target_devices)")
    draft=$(_lucebox_config_get placement.draft_device)
    remote_expert=$(_lucebox_config_get placement.remote_expert_device)
    if [ -n "$remote_expert" ]; then
        printf '%s + %s Spark experts' "${target:-$targets}" "$remote_expert"
    elif [ -n "$draft" ]; then
        printf '%s + %s draft/scorer' "${target:-$targets}" "$draft"
    elif [ -n "$targets" ]; then
        printf '%s target split' "$targets"
    elif [ -n "$target" ]; then
        printf '%s' "$target"
    else
        printf '%s' "${mode:-server default}"
    fi
}

cmd_menu() {
    local choice model variant state repo_hint optimization placement gpu_name gpu_count other_gpu
    local connector connector_label
    while true; do
        ensure_probed
        model=$(_lucebox_config_get model.preset)
        model="${model:-not selected}"
        variant=$(pick_variant)
        state=$(_engine_state)
        optimization=$(_optimization_summary)
        placement=$(_placement_summary)
        gpu_name="${LUCEBOX_HOST_GPU_NAME:-not detected}"
        gpu_count="$LUCEBOX_HOST_GPU_COUNT"
        other_gpu=""
        if _variant_is_rocm "$variant" && [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ]; then
            gpu_name="${LUCEBOX_HOST_AMD_GPU_NAME:-AMD GPU}"
            gpu_count="$LUCEBOX_HOST_AMD_GPU_COUNT"
            [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ] \
                && other_gpu="${LUCEBOX_HOST_GPU_NAME:-NVIDIA GPU}"
        elif [ "$LUCEBOX_HOST_HAS_NVIDIA_GPU" = "1" ] \
             && [ "$LUCEBOX_HOST_HAS_AMD_GPU" = "1" ]; then
            other_gpu="${LUCEBOX_HOST_AMD_GPU_NAME:-AMD GPU}"
        fi
        if _find_repo_root >/dev/null 2>&1; then
            repo_hint="available"
        else
            repo_hint="not a source checkout"
        fi
        connector=$(_connector_selected)
        if [ -n "$connector" ]; then
            connector_label=$(_connector_label "$connector")
            _connector_binary "$connector" >/dev/null 2>&1 \
                || connector_label="$connector_label (not found)"
        else
            connector_label="not selected"
        fi

        _menu_clear
        print_logo
        if [ "$gpu_count" -gt 1 ]; then
            printf '  GPU:          %s (primary of %s)\n' \
                "$gpu_name" "$gpu_count"
        else
            printf '  GPU:          %s\n' "$gpu_name"
        fi
        if [ -n "$other_gpu" ]; then
            printf '  Other GPU:    %s\n' "$other_gpu"
        fi
        printf '  Backend:      %s\n' "$variant"
        printf '  Model:        %s\n' "$model"
        printf '  Optimization: %s\n' "$optimization"
        printf '  Execution:    %s\n' "$placement"
        printf '  Engine:       %s\n' "$state"
        printf '  Harness:      %s\n\n' "$connector_label"

        printf '  1  Quick setup\n'
        printf '  2  Choose or download a model\n'
        printf '  3  Review optimizations\n'
        printf '  4  Calibrate and measure performance\n'
        printf '  5  Start the inference engine\n'
        printf '  6  Stop the inference engine\n'
        printf '  7  Status\n'
        printf '  8  Recent logs\n'
        printf '  9  Connect or open your harness\n'
        printf '  d  Developer tools  %b(%s)%b\n' "$C_DIM" "$repo_hint" "$C_RST"
        printf '  q  Quit\n\n'
        printf 'Choose: '
        IFS= read -r choice || return 0
        case "$choice" in
            1) cmd_setup; _menu_pause ;;
            2)
                if _menu_run models select \
                   && _ensure_configured_image; then
                    _menu_restart_if_running
                fi
                _menu_pause
                ;;
            3)
                if _menu_run optimize; then _menu_restart_if_running; fi
                _menu_pause
                ;;
            4) _menu_run calibrate; _menu_pause ;;
            5) _menu_start; _menu_pause ;;
            6) _menu_run stop; _menu_pause ;;
            7) _menu_run status; _menu_pause ;;
            8)
                if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ] && [ -f "$UNIT_PATH" ]; then
                    _menu_run logs -n 80 --no-pager
                else
                    _menu_run logs --tail 80
                fi
                _menu_pause
                ;;
            9) _menu_run connect; _menu_pause ;;
            d|D) cmd_developer_menu ;;
            q|Q|quit|exit) return 0 ;;
            *) warn "Choose 1–9, d, or q"; _menu_pause ;;
        esac
    done
}

usage() {
    cat <<EOF
$SCRIPT_NAME $VERSION — simple CLI for the Lucebox inference engine

Interactive:
  $SCRIPT_NAME            open the branded menu (when run in a terminal)
  menu                    open the menu explicitly
  setup                   guided backend, model, optimization, and service setup
  calibrate [--force]     measure this model/GPU and tune safe startup knobs
  connect [harness]       link an installed AI client to the local Lucebox API

Service management (via user systemd):
  install               install user systemd unit
  uninstall             stop, disable, remove the unit (keeps config + models)
  start | stop          systemctl --user start|stop lucebox
  enable | disable      systemctl --user enable|disable lucebox
  status                systemctl --user status lucebox
  logs [args]           journalctl --user -u lucebox  (default: -f)

Direct server invocation (foreground, no systemd):
  serve                 docker run the server in the foreground

Contributor workflow (inside a repository checkout):
  build [cuda|rocm]     build one native backend and its IPC companion
  build hybrid         build the paired CUDA + HIP runtime
  package-runtime [dir]
                        stage a relocatable factory runtime (default: dist/)
  native [cuda|rocm]    run the selected model with the native engine
  harness [name]        run Claude, Codex, OpenCode, Hermes, Pi, OpenClaw,
                        or Open WebUI against the native engine

Provisioning + workloads (delegated to the in-container Python CLI):
  check                 host + docker readiness report
  pull                  docker pull the auto-selected CUDA or ROCm image
  update                re-run the bootstrap installer to upgrade this script
  completion <shell>    print shell completion script (bash / zsh / fish)
  models select         numbered model picker; download + activate in one step
  models                list / download / activate model presets
  optimize              automatic or guided DFlash/PFlash/KVFlash/Spark setup
  config                read / write keys in .lucebox/config.toml
  print-run             print the docker-run command for the server

Misc:
  help, --help, -h      this message
  version, --version    print version

Environment overrides:
  LUCEBOX_IMAGE         image name without tag (default: ghcr.io/luce-org/lucebox-hub)
  LUCEBOX_VARIANT       image tag override (default: cuda13 on GB10, cuda128 on RTX 5090, cuda12 on other NVIDIA, rocm on AMD)
  LUCEBOX_PORT          host port for the server (default: 8080)
  LUCEBOX_CONTAINER     server container name (default: lucebox)
  LUCEBOX_MODELS        host model directory (default: \$XDG_DATA_HOME/lucebox/models)
  LUCEBOX_HOME          config/state directory (default: \$HOME/.lucebox)
  LUCEBOX_WRAPPER_SHA256
                        optional 64-hex checksum pin for install/update
  LUCEBOX_NO_EXEC=1     force docker-run for in-container subcommands even
                        when the container is up (equivalent to --no-exec)
  HF_TOKEN              propagated to \`models download\` for gated HF repos

Container routing:
  When the long-running '$CONTAINER_NAME' container is up, steady-state
  subcommands (config, models, check, print-run, print-serve-argv)
  'docker exec' into it instead of starting a fresh container. This avoids
  the ~1-3s docker-run cold-start AND shares the live server's network
  namespace so localhost:\$LUCEBOX_PORT reaches the server. Service-restarting
  commands (serve, pull, update, install, etc.) stay on the host-side /
  docker-run path. Pass --no-exec (or LUCEBOX_NO_EXEC=1) to force docker-run.
EOF
}

# ── dispatch ──────────────────────────────────────────────────────────────

main() {
    # Global flag pass: `--no-exec` anywhere before the subcommand forces the
    # docker-run path even if the container is up. Equivalent to
    # `LUCEBOX_NO_EXEC=1 lucebox ...`. We pop it out of argv up-front so the
    # rest of dispatch doesn't have to know about it.
    local args=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --no-exec) export LUCEBOX_NO_EXEC=1; shift ;;
            *) args+=("$1"); shift ;;
        esac
    done
    if [ "${#args[@]}" -gt 0 ]; then
        set -- "${args[@]}"
    else
        set --
    fi

    local cmd
    if [ $# -eq 0 ]; then
        if [ -t 0 ] && [ -t 1 ]; then cmd=menu; else cmd=help; fi
    else
        cmd="$1"
        shift
    fi
    case "$cmd" in
        # Branded interactive surface / guided first run.
        menu)             cmd_menu "$@" ;;
        setup)            cmd_setup "$@" ;;
        calibrate)        cmd_calibrate "$@" ;;
        connect)          cmd_connect "$@" ;;

        # Systemd surface
        install)          cmd_systemd_install "$@" ;;
        uninstall)        cmd_systemd_uninstall "$@" ;;
        start|restart|enable|disable)
                          cmd_systemctl_passthrough "$cmd" "$@" ;;
        stop)             cmd_stop "$@" ;;
        status)           cmd_status "$@" ;;
        logs)             cmd_logs "$@" ;;

        # Direct server
        serve)            cmd_serve "$@" ;;
        pull)             cmd_pull "$@" ;;

        # Native source-repository workflow.
        build)            cmd_native_build "$@" ;;
        package-runtime)  cmd_package_runtime "$@" ;;
        native)           cmd_native_serve "$@" ;;
        harness)          cmd_harness "$@" ;;

        # Self-update — re-runs the bootstrap installer against the channel
        # this script was installed from (LUCEBOX_INSTALLED_FROM).
        update)           cmd_update "$@" ;;

        # Host-only readiness check — pure shell, never enters the container.
        check)            cmd_check "$@" ;;

        # Shell completion — print a script the user sources into their rc
        # file. Bash and zsh share the bash-style emitter (zsh users add a
        # `bashcompinit; complete` shim); fish is native.
        completion)       cmd_completion "$@" ;;

        # Help / version
        help|--help|-h)   usage ;;
        version|--version) printf '%s\n' "$VERSION" ;;

        # Everything else → in-container Python CLI. cmd_route_to_container
        # picks between `docker exec` into the live container (cheap, shares
        # the running server's network namespace) and `docker run` (cold,
        # isolated) based on container state + the safe-to-exec command set.
        *)                cmd_route_to_container "$cmd" "$@" ;;
    esac
}

main "$@"
