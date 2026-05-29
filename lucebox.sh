#!/usr/bin/env bash
# lucebox.sh — host-side wrapper for the lucebox-hub container.
#
# Two jobs:
#
#   1) Probe the host (driver, docker, NVIDIA Container Toolkit, VRAM, RAM,
#      systemd), select the CUDA 12 image, and
#      dispatch into the in-container Python CLI via `docker run`. The
#      Python CLI lives at /opt/lucebox-hub/lucebox/ inside the image and is
#      the single source of truth for orchestration logic — TOML config,
#      autotune rules + sweep, server bring-up, smoke tests, model downloads.
#
#   2) Manage a user-level systemd unit (~/.config/systemd/user/lucebox.service)
#      so the server can run as a long-lived service without keeping a shell
#      open. install/uninstall/start/stop/enable/disable/status/logs all
#      delegate to systemctl --user / journalctl --user.
#
# Install:
#   curl -fsSL https://raw.githubusercontent.com/Luce-Org/lucebox-hub/main/install.sh | bash
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

# Canonical source of `lucebox.sh`. The bootstrap installer (`install.sh`)
# rewrites this line at install time to record which URL the user actually
# installed from — `lucebox update` then re-pulls from the same channel
# without losing track of forks. Falls back to the Luce-Org main branch
# when nothing was baked in (e.g. someone curl'd the script directly).
LUCEBOX_INSTALLED_FROM="${LUCEBOX_INSTALLED_FROM:-https://raw.githubusercontent.com/Luce-Org/lucebox-hub/main/lucebox.sh}"

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
# Inline `# comment` is honored. Arrays / inline tables / multi-line
# strings aren't written by the Python persister, so we don't parse them.
_lucebox_config_get() {
    local dotted="$1" cfg
    cfg="$(_lucebox_config_path)"
    [ -f "$cfg" ] || return 0
    local section="${dotted%.*}"
    local key="${dotted##*.}"
    [ "$section" = "$dotted" ] && section=""
    awk -v want_section="$section" -v want_key="$key" '
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
            line = $0
            sub(/#.*$/, "", line)
            eq = index(line, "=")
            if (eq == 0) next
            k = substr(line, 1, eq - 1)
            v = substr(line, eq + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            if (k != want_key) next
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
: "${LUCEBOX_HOST_GPU_NAME:=}"
: "${LUCEBOX_HOST_GPU_COUNT:=0}"
: "${LUCEBOX_HOST_VRAM_GB:=0}"
: "${LUCEBOX_HOST_GPU_SM:=}"
: "${LUCEBOX_HOST_DRIVER_VERSION:=}"
: "${LUCEBOX_HOST_DRIVER_MAJOR:=0}"
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
# Tracks whether probe_host has actually run; pieces of the code that need
# fresh host facts (e.g. cmd_check, cmd_serve) gate on this. Default 0.
: "${_LUCEBOX_HOST_PROBED:=0}"

# ── output helpers ────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_INFO='\033[1;34m'; C_OK='\033[1;32m'; C_WARN='\033[1;33m'
    C_ERR='\033[1;31m'; C_DIM='\033[2m'; C_RST='\033[0m'
else
    C_INFO=''; C_OK=''; C_WARN=''; C_ERR=''; C_DIM=''; C_RST=''
fi

info()  { printf '%b[INFO]%b  %s\n' "$C_INFO" "$C_RST" "$*"; }
ok()    { printf '%b[OK]%b    %s\n' "$C_OK"   "$C_RST" "$*"; }
warn()  { printf '%b[WARN]%b  %s\n' "$C_WARN" "$C_RST" "$*"; }
err()   { printf '%b[ERROR]%b %s\n' "$C_ERR"  "$C_RST" "$*" >&2; }
hint()  { printf '       %b%s%b\n'  "$C_DIM"  "$*"     "$C_RST"; }
die()   { err "$*"; exit 1; }

# ── host probing ──────────────────────────────────────────────────────────
# Sets the LUCEBOX_HOST_* variables consumed by the in-container Python CLI
# (passed through with -e). The Python side trusts these and doesn't reprobe
# — it can't see the host's /proc anyway, only the container's.

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
    LUCEBOX_HOST_GPU_NAME=""
    LUCEBOX_HOST_GPU_COUNT=0
    LUCEBOX_HOST_VRAM_GB=0
    LUCEBOX_HOST_GPU_SM=""
    LUCEBOX_HOST_DRIVER_VERSION=""
    LUCEBOX_HOST_DRIVER_MAJOR=0

    if command -v nvidia-smi &>/dev/null; then
        local q
        if q=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
                          --format=csv,noheader,nounits 2>/dev/null) && [ -n "$q" ]; then
            LUCEBOX_HOST_GPU_VENDOR="nvidia"
            LUCEBOX_HOST_GPU_NAME=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $1}')
            local mem_mib
            mem_mib=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $2}')
            LUCEBOX_HOST_VRAM_GB=$((mem_mib / 1024))
            LUCEBOX_HOST_DRIVER_VERSION=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $3}')
            LUCEBOX_HOST_DRIVER_MAJOR=${LUCEBOX_HOST_DRIVER_VERSION%%.*}
            local cc
            cc=$(printf '%s\n' "$q" | head -1 | awk -F', ' '{print $4}')
            LUCEBOX_HOST_GPU_SM="${cc//./}"
            LUCEBOX_HOST_GPU_COUNT=$(printf '%s\n' "$q" | wc -l)
        fi
        # Multi-GPU enumeration for /props.host. The single-GPU vars
        # above (GPU_NAME / GPU_SM / VRAM_GB / DRIVER_VERSION) keep
        # describing GPU 0 for back-compat with cmd_check + autotune;
        # the full per-GPU CSV rides along separately so HOST_INFO can
        # emit the whole array.
        LUCEBOX_HOST_GPU_LIST_CSV=$(nvidia-smi \
            --query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,power.limit \
            --format=csv,noheader 2>/dev/null || echo "")
    fi
    # CUDA_VISIBLE_DEVICES from the caller's env (empty default = "all GPUs").
    LUCEBOX_HOST_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

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
    export LUCEBOX_HOST_GPU_NAME LUCEBOX_HOST_GPU_COUNT LUCEBOX_HOST_VRAM_GB
    export LUCEBOX_HOST_GPU_SM LUCEBOX_HOST_DRIVER_VERSION LUCEBOX_HOST_DRIVER_MAJOR
    export LUCEBOX_HOST_HAS_SYSTEMD LUCEBOX_HOST_IS_WSL
    export LUCEBOX_HOST_HAS_DOCKER LUCEBOX_HOST_DOCKER_VERSION
    export LUCEBOX_HOST_HAS_CTK
    export LUCEBOX_HOST_OS_PRETTY LUCEBOX_HOST_KERNEL LUCEBOX_HOST_WSL_VERSION
    export LUCEBOX_HOST_NVIDIA_CTK_VERSION LUCEBOX_HOST_CPU_MODEL
    export LUCEBOX_HOST_GPU_LIST_CSV LUCEBOX_HOST_CUDA_VISIBLE_DEVICES
    _LUCEBOX_HOST_PROBED=1
}

# Cheap idempotency wrapper. Anything that needs real host facts (vs the safe
# defaults seeded at script-load) calls this. Subcommands that go straight to
# `systemctl`/`journalctl` no longer need to remember to call probe_host.
ensure_probed() {
    [ "$_LUCEBOX_HOST_PROBED" = "1" ] || probe_host
}

pick_variant() {
    # CUDA 12.8 is the supported image variant for this branch. Effective
    # value goes through the same env > config.toml > default ladder as
    # everything else so `config set image.variant=...` propagates.
    _lucebox_resolve "${LUCEBOX_VARIANT:-}" image.variant "cuda12"
}

# ── prereq checks (host-only) ─────────────────────────────────────────────
# Print-and-exit on anything that needs root to install. The Python CLI does
# the richer reporting; this is the bare minimum to make `docker run` viable.

require_host_prereqs() {
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

    if ! command -v nvidia-smi &>/dev/null; then
        err "nvidia-smi not found — no NVIDIA driver detected"
        hint "Install the NVIDIA driver: https://www.nvidia.com/Download/index.aspx"
        missing=1
    elif ! nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
        err "nvidia-smi present but NVML calls fail — likely a driver/library mismatch"
        hint "Reboot, or reinstall the matching NVIDIA driver package"
        missing=1
    fi

    [ "$missing" = "0" ] || exit 1
}

require_ctk() {
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
# bench containers on the host daemon), mount $HOME at the same path (so
# paths look identical in and out), and pass host facts via env. When an
# NVIDIA GPU is detected we also pass --gpus all so the orchestrator can
# call nvidia-smi during profile snapshot export; without it nvidia_smi_csv (and
# any downstream power/utilization fields) come back empty.

DOCKER_SOCK_PATH="${DOCKER_HOST:-/var/run/docker.sock}"
DOCKER_SOCK_PATH="${DOCKER_SOCK_PATH#unix://}"

build_orchestrator_argv() {
    local variant="$1"; shift
    local tty=()
    if [ -t 0 ] && [ -t 1 ]; then
        tty=(-it)
    else
        tty=(-i)
    fi
    local argv=(docker run --rm "${tty[@]}")
    if [ "${LUCEBOX_HOST_GPU_VENDOR:-none}" = "nvidia" ]; then
        argv+=(--gpus all)
    fi
    argv+=(--name "${CONTAINER_NAME}-cli-$$")
    argv+=(--user "$(id -u):$(id -g)")
    if [ -S "$DOCKER_SOCK_PATH" ]; then
        argv+=(--group-add "$(stat -c '%g' "$DOCKER_SOCK_PATH")")
    fi
    argv+=(-v "$DOCKER_SOCK_PATH:/var/run/docker.sock")
    argv+=(-v "$HOME:$HOME")
    # Bind-mount the XDG models dir explicitly (host = container path) so
    # paths line up in/out. The $HOME mount above already covers it when
    # XDG_DATA_HOME is unset, but an explicit -v is required when the user
    # points XDG_DATA_HOME outside $HOME.
    mkdir -p "$DEFAULT_MODELS_DIR"
    argv+=(-v "$DEFAULT_MODELS_DIR:$DEFAULT_MODELS_DIR")
    argv+=(-w "$PWD")
    argv+=(-e "HOME=$HOME")
    # Host facts — Python side reads these instead of reprobing.
    local var
    for var in $(compgen -e | grep '^LUCEBOX_HOST_' || true); do
        argv+=(-e "$var=${!var}")
    done
    # User overrides for image/port/container name propagate too.
    argv+=(-e "LUCEBOX_IMAGE=$IMAGE_BASE")
    argv+=(-e "LUCEBOX_VARIANT=$variant")
    argv+=(-e "LUCEBOX_PORT=$DEFAULT_PORT")
    argv+=(-e "LUCEBOX_CONTAINER=$CONTAINER_NAME")
    # Always export the resolved models dir so the in-container CLI sees
    # the same path the wrapper mounts (don't gate on `LUCEBOX_MODELS` being
    # set — the XDG default needs to flow through too).
    argv+=(-e "LUCEBOX_MODELS=$DEFAULT_MODELS_DIR")
    [ -n "${HF_TOKEN:-}" ] && argv+=(-e "HF_TOKEN=$HF_TOKEN")

    argv+=("${IMAGE_BASE}:${variant}")
    # `lucebox` is the entrypoint subcommand handled by server/scripts/entrypoint.sh
    # — it execs `python -m lucebox` with whatever args we pass on.
    argv+=(lucebox "$@")
    printf '%s\n' "${argv[@]}"
}

# ── subcommand implementations ────────────────────────────────────────────

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
    require_host_prereqs
    ensure_probed
    require_ctk
    local variant
    variant=$(pick_variant)

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

    local orch_argv server_argv
    mapfile -t orch_argv < <(build_orchestrator_argv "$variant" print-serve-argv)

    if mapfile -t server_argv < <("${orch_argv[@]}" 2>/dev/null) \
       && [ "${#server_argv[@]}" -gt 0 ] \
       && [ "${server_argv[0]}" = "docker" ]; then
        info "Starting lucebox server (variant=$variant, from config.toml)"
        _serve_and_track "${server_argv[@]}"
        return $?
    fi

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
        --gpus all
        -p "$DEFAULT_PORT:8080"
        -v "$HOME:$HOME"
        -v "$fallback_models:/opt/lucebox-hub/server/models")
    local var
    for var in $(compgen -e | grep '^LUCEBOX_HOST_' || true); do
        fallback_argv+=(-e "$var=${!var}")
    done
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
        wait "$docker_pid" 2>/dev/null || true
    }
    trap _serve_stop HUP INT TERM EXIT
    wait "$docker_pid"
    local rc=$?
    trap - HUP INT TERM EXIT
    return $rc
}

cmd_systemd_install() {
    require_host_prereqs
    ensure_probed
    require_systemd "service install"
    local docker_bin
    docker_bin=$(command -v docker)

    mkdir -p "$(dirname "$UNIT_PATH")"
    # Capture the user's resolved env at install time so the unit launches
    # with the same image/variant/port/models the user expected when they
    # ran `lucebox install`. Systemd's user-session env is sparse — without
    # this block, the wrapper inside the unit would fall back to the
    # in-script defaults and silently pick a different image or models
    # directory than the user's interactive session uses.
    #
    # ExecStartPre cleans up any orphaned container with the target name
    # left behind by a previous crash (docker's `--rm` only fires on clean
    # exit — a SIGKILL or daemon restart leaves the name claimed, and the
    # next ExecStart would die with "name already in use" while systemd
    # reports a useless "exit code 125").
    cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Lucebox hub LLM inference server
Documentation=https://github.com/Luce-Org/lucebox-hub
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=exec
Restart=on-failure
RestartSec=10
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LUCEBOX_IMAGE=$IMAGE_BASE
Environment=LUCEBOX_VARIANT=$(pick_variant)
Environment=LUCEBOX_PORT=$DEFAULT_PORT
Environment=LUCEBOX_CONTAINER=$CONTAINER_NAME
Environment=LUCEBOX_MODELS=$DEFAULT_MODELS_DIR
ExecStartPre=-$docker_bin rm -f $CONTAINER_NAME
ExecStart=$SCRIPT_PATH serve
ExecStop=$docker_bin stop -t 30 $CONTAINER_NAME
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
                    active|activating) ;;
                    *) break ;;
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

cmd_logs() {
    require_systemd "logs"
    # Pure passthrough: any flags the user wants (-f, -n, --since, ...) go
    # straight to journalctl. Default is follow.
    if [ $# -eq 0 ]; then
        exec journalctl --user -u "$UNIT_NAME" -f
    fi
    exec journalctl --user -u "$UNIT_NAME" "$@"
}

cmd_pull() {
    # Pull has to run on the host. Delegating this into the container creates a
    # stale-image trap: docker may start an old local tag before the fresh tag
    # has been pulled.
    require_host_prereqs
    local variant
    variant=$(pick_variant)
    info "Pulling ${IMAGE_BASE}:${variant}"
    exec docker pull "${IMAGE_BASE}:${variant}"
}

cmd_update() {
    # Re-run the bootstrap installer against the channel we were installed
    # from. The installer is the source of truth for "how do you install
    # lucebox correctly" — chmod, atomic mv, validation, baking the source
    # URL back into the new copy so the channel is preserved across
    # upgrades. Keeping the logic in install.sh means it can evolve
    # independently (sha verify, signature check, etc.) and the installed
    # `lucebox update` picks those changes up on the next run.
    #
    # The installer URL is derived from LUCEBOX_INSTALLED_FROM by swapping
    # `lucebox.sh` → `install.sh` in the same directory, so forks don't
    # need a separate registration. Override the source channel via
    # $LUCEBOX_INSTALL_URL (e.g. to switch from canonical to a dev fork).
    local source_url installer_url target
    source_url="${LUCEBOX_INSTALL_URL:-$LUCEBOX_INSTALLED_FROM}"
    if [[ "$source_url" != */lucebox.sh ]]; then
        die "LUCEBOX_INSTALLED_FROM doesn't end in /lucebox.sh: $source_url"
    fi
    installer_url="${source_url%/lucebox.sh}/install.sh"
    target=$(realpath "$SCRIPT_PATH")

    info "Updating lucebox via $installer_url"
    info "  source: $source_url"
    info "  target: $target"

    # Pass the URLs through to install.sh via env. The installer reads
    # $LUCEBOX_INSTALL_URL (which we set to source_url) and
    # $LUCEBOX_INSTALL_DEST (the realpath of *this* file, so a symlinked
    # install replaces the actual file behind the link).
    LUCEBOX_INSTALL_URL="$source_url" \
    LUCEBOX_INSTALL_DEST="$target" \
        bash -c "$(curl -fsSL "$installer_url")" \
            || die "update failed (installer exited non-zero)"
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
    # verbs (config get/set/unset, models list/download). Adding a new
    # top-level command means adding it here too.
    local shell="${1:-}"
    case "$shell" in
        bash)
            cat <<'BASH'
# lucebox bash completion. Source from ~/.bashrc:
#   source <(lucebox completion bash)
_lucebox_complete() {
    local cur prev cmds config_verbs models_verbs completion_shells
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmds="install uninstall start stop restart enable disable status logs \
          serve pull update check completion config models autotune \
          profile smoke print-run help version"
    config_verbs="get set unset"
    models_verbs="list download"
    completion_shells="bash zsh fish"

    # Sub-app verbs / shell args.
    case "$prev" in
        config)     COMPREPLY=( $(compgen -W "$config_verbs"     -- "$cur") ); return ;;
        models)     COMPREPLY=( $(compgen -W "$models_verbs"     -- "$cur") ); return ;;
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
set -l __lucebox_cmds install uninstall start stop restart enable disable \
    status logs serve pull update check completion config models autotune \
    profile smoke print-run help version
for cmd in $__lucebox_cmds
    complete -c lucebox -n "not __fish_seen_subcommand_from $__lucebox_cmds" -a $cmd
end
complete -c lucebox -n "__fish_seen_subcommand_from config" -a "get set unset"
complete -c lucebox -n "__fish_seen_subcommand_from models" -a "list download"
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
    local mark detail
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

    # nvidia container toolkit
    case "$LUCEBOX_HOST_HAS_CTK" in
        runtime)            _row 1    "nvidia ctk"     "wired into docker (runtime)" ;;
        cdi)                _row 1    "nvidia ctk"     "wired via CDI (nvidia.com/gpu)" ;;
        installed-unwired)  _row warn "nvidia ctk"     "installed but not registered with docker — sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" ;;
        none|*)             _row 0    "nvidia ctk"     "not installed — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" ;;
    esac

    # nvidia-smi + driver
    if [ "$LUCEBOX_HOST_GPU_VENDOR" = "nvidia" ]; then
        if [ "$LUCEBOX_HOST_DRIVER_MAJOR" -ge "$MIN_DRIVER_CUDA12" ]; then
            _row 1 "nvidia driver" "$LUCEBOX_HOST_DRIVER_VERSION (≥ $MIN_DRIVER_CUDA12 required for cuda12)"
        else
            _row 0 "nvidia driver" "$LUCEBOX_HOST_DRIVER_VERSION (< $MIN_DRIVER_CUDA12 — cuda12 image will fail)"
        fi
    elif command -v nvidia-smi &>/dev/null; then
        _row 0 "nvidia driver" "nvidia-smi present but NVML calls fail — driver/library mismatch, try reboot"
    else
        _row 0 "nvidia driver" "nvidia-smi not found — install the NVIDIA driver"
    fi

    # GPU detail
    if [ "$LUCEBOX_HOST_GPU_VENDOR" = "nvidia" ]; then
        _row 1 "gpu" "$LUCEBOX_HOST_GPU_NAME × $LUCEBOX_HOST_GPU_COUNT (sm_$LUCEBOX_HOST_GPU_SM, ${LUCEBOX_HOST_VRAM_GB} GB VRAM)"
        # cuda12 image arch coverage: sm_75;80;86;89;90;120 (see docker-bake.hcl)
        case "$LUCEBOX_HOST_GPU_SM" in
            75|80|86|89|90|120) _row 1    "cuda12 arch" "sm_$LUCEBOX_HOST_GPU_SM covered by image" ;;
            "")                 _row warn "cuda12 arch" "compute_cap not detected" ;;
            *)                  _row warn "cuda12 arch" "sm_$LUCEBOX_HOST_GPU_SM not in image arch list (75;80;86;89;90;120)" ;;
        esac
    fi

    # systemd
    if [ "$LUCEBOX_HOST_HAS_SYSTEMD" = "1" ]; then
        _row 1 "user systemd" "available (needed for '$SCRIPT_NAME install')"
    elif [ "$LUCEBOX_HOST_IS_WSL" = "1" ]; then
        _row warn "user systemd" "WSL detected — set 'systemd=true' under [boot] in /etc/wsl.conf, then 'wsl --shutdown'"
    else
        _row warn "user systemd" "not available — '$SCRIPT_NAME install' (service unit) won't work; '$SCRIPT_NAME serve' (foreground) will"
    fi

    # image we'd pull — marked ✗ when the host clearly can't run cuda12
    # (no nvidia driver, or no CTK wired into docker). It's still useful
    # to print the line so the user knows what would be pulled, but a
    # green ✓ would be misleading.
    if [ "$LUCEBOX_HOST_GPU_VENDOR" != "nvidia" ]; then
        _row 0 "image" "${IMAGE_BASE}:${variant} — requires NVIDIA driver"
    elif [ "$LUCEBOX_HOST_HAS_CTK" = "none" ] || [ "$LUCEBOX_HOST_HAS_CTK" = "installed-unwired" ]; then
        _row 0 "image" "${IMAGE_BASE}:${variant} — needs NVIDIA Container Toolkit wired into docker"
    else
        _row 1 "image" "${IMAGE_BASE}:${variant}"
    fi
    # RAM / cores (informational)
    _row 1 "host" "${LUCEBOX_HOST_NPROC} cpus, ${LUCEBOX_HOST_RAM_GB} GB RAM"
}

cmd_in_container() {
    # Generic dispatcher: anything that isn't a systemd action goes here.
    # Runs the in-container Python CLI with the supplied argv.
    require_host_prereqs
    ensure_probed
    # CTK isn't strictly required for every subcommand (e.g. `config get`
    # or `autotune` only touch local files), but the server-spawning
    # subcommands need it.
    # Letting docker error its own way is fine for the no-CTK case.
    local variant
    variant=$(pick_variant)
    local argv
    mapfile -t argv < <(build_orchestrator_argv "$variant" "$@")
    exec "${argv[@]}"
}

usage() {
    cat <<EOF
$SCRIPT_NAME $VERSION — host-side wrapper for the lucebox-hub container

Service management (via user systemd):
  install               install user systemd unit
  uninstall             stop, disable, remove the unit (keeps config + models)
  start | stop          systemctl --user start|stop lucebox
  enable | disable      systemctl --user enable|disable lucebox
  status                systemctl --user status lucebox
  logs [args]           journalctl --user -u lucebox  (default: -f)

Direct server invocation (foreground, no systemd):
  serve                 docker run the server in the foreground

Provisioning + workloads (delegated to the in-container Python CLI):
  check                 host + docker readiness report
  pull                  docker pull the cuda12 image
  update                re-run the bootstrap installer to upgrade this script
  completion <shell>    print shell completion script (bash / zsh / fish)
  models                list / download / activate model presets
  config                read / write keys in .lucebox/config.toml
  autotune              compute (and optionally apply) VRAM-tier DFLASH_* defaults
                        — `autotune --sweep` empirically picks a per-tier winner
  smoke                 hit /v1/chat/completions on a running server
  profile               run luce-bench snapshot via the running container
  print-run             print the docker-run command for the server

Misc:
  help, --help, -h      this message
  version, --version    print version

Environment overrides:
  LUCEBOX_IMAGE         image name without tag (default: ghcr.io/luce-org/lucebox-hub)
  LUCEBOX_VARIANT       image tag to pull/run (default: cuda12)
  LUCEBOX_PORT          host port for the server (default: 8080)
  LUCEBOX_CONTAINER     server container name (default: lucebox)
  LUCEBOX_MODELS        host model directory (default: \$XDG_DATA_HOME/lucebox/models
  HF_TOKEN              propagated to `models download` for gated HF repos
EOF
}

# ── dispatch ──────────────────────────────────────────────────────────────

main() {
    local cmd="${1:-help}"
    [ $# -gt 0 ] && shift
    case "$cmd" in
        # Systemd surface
        install)          cmd_systemd_install "$@" ;;
        uninstall)        cmd_systemd_uninstall "$@" ;;
        start|stop|restart|enable|disable|status)
                          cmd_systemctl_passthrough "$cmd" "$@" ;;
        logs)             cmd_logs "$@" ;;

        # Direct server
        serve)            cmd_serve "$@" ;;
        pull)             cmd_pull "$@" ;;

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

        # Everything else → in-container Python CLI
        *)                cmd_in_container "$cmd" "$@" ;;
    esac
}

main "$@"
