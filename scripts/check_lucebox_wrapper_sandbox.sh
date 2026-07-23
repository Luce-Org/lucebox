#!/usr/bin/env bash
# Exercise the host-side lucebox.sh installer/wrapper from an isolated prefix.
#
# The script intentionally runs from a throwaway HOME, XDG_CONFIG_HOME,
# LUCEBOX_HOME, model directory, and working directory. That catches accidental
# dependencies on the checkout or the user's real ~/.lucebox while keeping the
# test reproducible enough to paste into a bug report.

set -euo pipefail

IMAGE="${LUCEBOX_TEST_IMAGE:-ghcr.io/easel/lucebox-hub}"
VARIANT="${LUCEBOX_TEST_VARIANT:-integration-props-uv-squared-clean-cuda12}"
WRAPPER_SOURCE="${LUCEBOX_TEST_WRAPPER_SOURCE:-local}"
RUN_PULL="${LUCEBOX_TEST_RUN_PULL:-1}"
RUN_CONTAINER_CLI="${LUCEBOX_TEST_RUN_CONTAINER_CLI:-1}"
KEEP_SANDBOX="${LUCEBOX_TEST_KEEP_SANDBOX:-0}"

ROOT=""
LOG=""

usage() {
    cat <<EOF
Usage: $0 [--source local|URL] [--image IMAGE] [--variant TAG] [--no-pull] [--no-container-cli] [--keep]

Defaults:
  --source        local
  --image         $IMAGE
  --variant       $VARIANT

Environment aliases:
  LUCEBOX_TEST_WRAPPER_SOURCE, LUCEBOX_TEST_IMAGE, LUCEBOX_TEST_VARIANT,
  LUCEBOX_TEST_RUN_PULL=0, LUCEBOX_TEST_RUN_CONTAINER_CLI=0,
  LUCEBOX_TEST_KEEP_SANDBOX=1
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --source) WRAPPER_SOURCE="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --no-pull) RUN_PULL=0; shift ;;
        --no-container-cli) RUN_CONTAINER_CLI=0; shift ;;
        --keep) KEEP_SANDBOX=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() {
    echo "[FAIL] $*" >&2
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        echo "[FAIL] transcript: $LOG" >&2
    fi
    exit 1
}

note() {
    printf '[INFO] %s\n' "$*"
}

pass() {
    printf '[PASS] %s\n' "$*"
}

assert_file() {
    [ -f "$1" ] || die "missing file: $1"
    pass "file exists: $1"
}

assert_contains() {
    local file="$1"
    local pattern="$2"
    if ! grep -Fq "$pattern" "$file"; then
        echo "----- $file -----" >&2
        sed -n '1,220p' "$file" >&2 || true
        echo "-----------------" >&2
        die "expected '$pattern' in $file"
    fi
    pass "$file contains: $pattern"
}

run_logged() {
    note "run: $*"
    {
        printf '\n===== %s =====\n' "$*"
        "$@"
        printf '===== exit=0 =====\n'
    } 2>&1 | tee -a "$LOG"
}

run_logged_capture() {
    local out="$1"
    shift
    note "run: $* > $out"
    {
        printf '\n===== %s > %s =====\n' "$*" "$out"
        "$@"
        local rc=$?
        printf '===== exit=%s =====\n' "$rc"
        return "$rc"
    } 2>&1 | tee "$out" | tee -a "$LOG" >/dev/null
}

cleanup() {
    if [ -n "$ROOT" ] && [ "$KEEP_SANDBOX" != "1" ]; then
        rm -rf "$ROOT"
    elif [ -n "$ROOT" ]; then
        note "kept sandbox: $ROOT"
        note "transcript: $LOG"
    fi
}
trap cleanup EXIT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lucebox-wrapper-sandbox.XXXXXX")"
LOG="$ROOT/transcript.log"

HOME_DIR="$ROOT/home"
BIN_DIR="$ROOT/bin"
XDG_DIR="$ROOT/xdg"
MODELS_DIR="$ROOT/models"
WORK_DIR="$ROOT/work"
mkdir -p "$HOME_DIR" "$BIN_DIR" "$XDG_DIR" "$MODELS_DIR" "$WORK_DIR"

note "sandbox: $ROOT"
note "transcript: $LOG"

case "$WRAPPER_SOURCE" in
    local)
        cp "$REPO_ROOT/lucebox.sh" "$BIN_DIR/lucebox"
        ;;
    http://*|https://*)
        curl -fsSL "$WRAPPER_SOURCE" -o "$BIN_DIR/lucebox"
        ;;
    *)
        cp "$WRAPPER_SOURCE" "$BIN_DIR/lucebox"
        ;;
esac
chmod +x "$BIN_DIR/lucebox"

FIRST_LINE="$(head -n 1 "$BIN_DIR/lucebox")"
[ "$FIRST_LINE" = "#!/usr/bin/env bash" ] || die "unexpected shebang: $FIRST_LINE"
pass "wrapper has expected shebang"

export HOME="$HOME_DIR"
export XDG_CONFIG_HOME="$XDG_DIR"
export LUCEBOX_HOME="$HOME_DIR/.lucebox"
export LUCEBOX_MODELS="$MODELS_DIR"
export LUCEBOX_IMAGE="$IMAGE"
export LUCEBOX_VARIANT="$VARIANT"
export LUCEBOX_CONTAINER="lucebox-sandbox"
export LUCEBOX_PORT="18080"
export PATH="$BIN_DIR:$PATH"

cd "$WORK_DIR"
[ "$PWD" = "$WORK_DIR" ] || die "failed to enter sandbox workdir"
pass "working directory isolated: $PWD"

run_logged_capture "$ROOT/version.out" lucebox version
assert_contains "$ROOT/version.out" "0.2.0"

run_logged_capture "$ROOT/help.out" lucebox help
assert_contains "$ROOT/help.out" "LUCEBOX_VARIANT"
assert_contains "$ROOT/help.out" "LUCEBOX_IMAGE"

docker manifest inspect "${IMAGE}:${VARIANT}" >/dev/null
pass "image manifest exists: ${IMAGE}:${VARIANT}"

if [ "$RUN_PULL" = "1" ]; then
    run_logged_capture "$ROOT/pull.out" lucebox pull
    assert_contains "$ROOT/pull.out" "${IMAGE}:${VARIANT}"
fi

if [ "$RUN_CONTAINER_CLI" = "1" ]; then
    run_logged_capture "$ROOT/check.out" lucebox check
    # Sparse persistence: `config set` creates config.toml with only the
    # named key. Replaces the old `configure --overwrite` path.
    run_logged_capture "$ROOT/config-image.out" lucebox config set "image=$IMAGE"
    run_logged_capture "$ROOT/config-variant.out" lucebox config set "variant=$VARIANT"
    assert_file "$LUCEBOX_HOME/config.toml"
    [ "$(stat -c '%u' "$LUCEBOX_HOME/config.toml")" = "$(id -u)" ] \
        || die "config.toml is not owned by the invoking user"
    pass "config.toml ownership matches invoking user"
    assert_contains "$LUCEBOX_HOME/config.toml" "registry = \"$IMAGE\""
    assert_contains "$LUCEBOX_HOME/config.toml" "variant = \"$VARIANT\""

    run_logged_capture "$ROOT/print-run.out" lucebox print-run
    assert_contains "$ROOT/print-run.out" "${IMAGE}:${VARIANT}"
    assert_contains "$ROOT/print-run.out" "$MODELS_DIR:/opt/lucebox-hub/dflash/models"
    if grep -Fq "$REPO_ROOT" "$ROOT/print-run.out"; then
        die "print-run leaked repository path: $REPO_ROOT"
    fi
    pass "print-run did not reference repository checkout"
fi

# Exercise `lucebox install` without allowing it to call real systemctl,
# loginctl, docker, or nvidia-smi. The generated user unit must land under the
# sandbox XDG_CONFIG_HOME and point ExecStart at the sandbox-installed wrapper.
SHIM_DIR="$ROOT/shims"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/docker" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  info) exit 0 ;;
  version) echo "25.0.0"; exit 0 ;;
  stop) exit 0 ;;
  *) echo "docker shim: $*" >&2; exit 0 ;;
esac
EOF
cat > "$SHIM_DIR/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=name,memory.total,driver_version,compute_cap"*)
    echo "Fake GPU, 24576, 555.42.01, 8.6"; exit 0 ;;
  *"--query-gpu=name"*)
    echo "Fake GPU"; exit 0 ;;
  *) echo "Fake GPU"; exit 0 ;;
esac
EOF
cat > "$SHIM_DIR/systemctl" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = "--user" ] && [ "$2" = "show-environment" ]; then exit 0; fi
if [ "$1" = "--user" ] && [ "$2" = "daemon-reload" ]; then exit 0; fi
echo "systemctl shim: $*" >&2
exit 0
EOF
cat > "$SHIM_DIR/loginctl" <<'EOF'
#!/usr/bin/env bash
echo "Linger=no"
EOF
chmod +x "$SHIM_DIR/docker" "$SHIM_DIR/nvidia-smi" "$SHIM_DIR/systemctl" "$SHIM_DIR/loginctl"

PATH="$SHIM_DIR:$BIN_DIR:$PATH" run_logged_capture "$ROOT/install.out" lucebox install
UNIT="$XDG_CONFIG_HOME/systemd/user/lucebox.service"
assert_file "$UNIT"
assert_contains "$UNIT" "ExecStart=$BIN_DIR/lucebox serve"
assert_contains "$UNIT" "ExecStop=$SHIM_DIR/docker stop -t 30 lucebox-sandbox"
assert_contains "$ROOT/install.out" "Installed $UNIT"

pass "sandbox wrapper check completed"
note "summary: image=${IMAGE}:${VARIANT} wrapper_source=${WRAPPER_SOURCE}"
