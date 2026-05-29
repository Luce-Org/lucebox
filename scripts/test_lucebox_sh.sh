#!/usr/bin/env bash
# scripts/test_lucebox_sh.sh — smoke tests for the host-side wrapper +
# every other bash script we ship.
#
# Catches regressions like:
#   * syntax errors (bash -n)
#   * shellcheck error-level findings across every shipped bash script
#   * `set -u` violations in command paths that don't need docker/nvidia —
#     each subcommand dispatch is exercised in isolation to verify no
#     LUCEBOX_HOST_* or DFLASH_* read fires before the helper that should
#     populate it has run.
#   * missing dispatch handlers (help, version, check, usage)
#   * stale references to subcommands removed from main's case
#
# The wrapper is shell + has zero non-coreutils deps for the host-only
# commands, so this script doesn't need docker/nvidia/systemd present —
# probe_host degrades cleanly when those aren't found, and the
# formatter must render fine for the "everything is missing" case too.
#
# Run from anywhere:  scripts/test_lucebox_sh.sh

set -euo pipefail

# Resolve repo root + script under test.
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/.." && pwd))"
SCRIPT="$ROOT/lucebox.sh"
ENTRYPOINT="$ROOT/server/scripts/entrypoint.sh"
INSTALLER="$ROOT/install.sh"

if [ ! -f "$SCRIPT" ]; then
    echo "FAIL: lucebox.sh not found at $SCRIPT" >&2
    exit 1
fi

fail=0
pass=0
report() {
    if [ "$1" = "ok" ]; then
        printf '  \033[1;32m✓\033[0m %s\n' "$2"
        pass=$((pass + 1))
    else
        printf '  \033[1;31m✗\033[0m %s\n' "$2"
        if [ -n "${3:-}" ]; then
            printf '    %s\n' "$3"
        fi
        fail=$((fail + 1))
    fi
}

# Helper: run the wrapper with strict bash, capture stdout+stderr, check for
# (a) zero exit code, (b) substring match. NO_COLOR is set so colour codes
# don't pollute substring matches.
assert_runs() {
    local label="$1" cmd="$2" expect="${3:-}"
    local out rc
    out=$(NO_COLOR=1 bash -c "$cmd" 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        report fail "$label" "exit $rc; output: $(printf '%s' "$out" | head -3)"
        return
    fi
    if [ -n "$expect" ] && ! grep -qF "$expect" <<<"$out"; then
        report fail "$label" "missing expected substring '$expect'; got: $(printf '%s' "$out" | head -3)"
        return
    fi
    report ok "$label"
}

# Helper: run a subcommand whose successful completion would normally need
# docker / nvidia / systemd. We only care that the bash dispatch up to the
# point of the missing dependency does NOT trip `set -u`. Exit code is
# allowed to be non-zero; what we forbid is a raw "unbound variable" /
# "syntax error" / "line N:" leak in the captured output.
#
# Wrapped in `timeout` so subcommands that exec into a follow-style binary
# (logs → journalctl -f, status when systemd is healthy, etc.) don't hang
# the test runner on a dev box where the underlying tools succeed.
assert_no_set_u_leak() {
    local label="$1"
    shift
    local out
    out=$(NO_COLOR=1 timeout 5 bash "$@" 2>&1 || true)
    # The "line N:" pattern is anchored to a script-path prefix to avoid
    # false positives from journalctl output ("systemd[1385106]:") which
    # contains a similar shape but isn't a bash error. Bash always emits
    # the source filename before the line number, e.g.
    #   /tmp/lbh-flat/lucebox.sh: line 200: VAR: unbound variable
    if grep -qE 'unbound variable|syntax error|\.sh: line [0-9]+:' <<<"$out"; then
        report fail "$label" "raw bash error leaked: $(head -3 <<<"$out")"
    else
        report ok "$label"
    fi
}

echo "[test_lucebox_sh] running against $SCRIPT"

# ── 1. shellcheck ─────────────────────────────────────────────────────────
# Run shellcheck across every bash script we ship (the wrapper, the
# in-container entrypoint, and every helper under scripts/). Error-level
# findings fail the build; warnings are informational only — those have
# been triaged and the SC2034/SC2155/SC2164 hits in sweep_ds4_2case.sh
# aren't user-visible bugs.
SHELLCHECK_TARGETS=(
    "$SCRIPT"
    "$ENTRYPOINT"
    "$INSTALLER"
)
# Add every scripts/*.sh except this one (don't recurse into our own tests).
while IFS= read -r -d '' f; do
    [ "$f" = "${BASH_SOURCE[0]}" ] && continue
    SHELLCHECK_TARGETS+=("$f")
done < <(find "$ROOT/scripts" -maxdepth 1 -name '*.sh' -type f -print0 2>/dev/null)
SHELLCHECK_TARGETS+=("${BASH_SOURCE[0]}")

if command -v shellcheck >/dev/null 2>&1; then
    sc_out=$(shellcheck --severity=error "${SHELLCHECK_TARGETS[@]}" 2>&1) || sc_rc=$?
    sc_rc="${sc_rc:-0}"
    if [ "$sc_rc" -eq 0 ]; then
        report ok "shellcheck --severity=error (${#SHELLCHECK_TARGETS[@]} files)"
    else
        report fail "shellcheck --severity=error" "$(printf '%s' "$sc_out" | head -10)"
    fi
else
    report fail "shellcheck not installed" "install via 'apt-get install -y shellcheck' (Ubuntu) or 'brew install shellcheck'"
fi

# ── 2. Syntax / parse ─────────────────────────────────────────────────────
if bash -n "$SCRIPT"; then report ok "bash -n lucebox.sh parses cleanly"
else report fail "bash -n lucebox.sh"; fi
if bash -n "$ENTRYPOINT"; then report ok "bash -n entrypoint.sh parses cleanly"
else report fail "bash -n entrypoint.sh"; fi

# ── 3. Trivial subcommands (zero-exit expected) ───────────────────────────
assert_runs "help"     "bash '$SCRIPT' help"     "host-side wrapper"
assert_runs "--help"   "bash '$SCRIPT' --help"   "host-side wrapper"
assert_runs "-h"       "bash '$SCRIPT' -h"       "host-side wrapper"
assert_runs "version"  "bash '$SCRIPT' version"  ""
assert_runs "--version" "bash '$SCRIPT' --version" ""

# ── 4. check — host-only, must run to completion even without docker/nvidia.
#    This is the path that broke last time (multi-byte glyph + set -u).
assert_runs "check"    "bash '$SCRIPT' check"    "host readiness report"

# ── 5. systemd-surface subcommands — every one of these used to crash with
# `LUCEBOX_HOST_HAS_SYSTEMD: unbound variable` because cmd_systemctl_passthrough
# / cmd_logs / cmd_systemd_uninstall reached require_systemd without first
# calling probe_host. The fix routes through require_systemd → probe_host
# when the var is unset; these tests pin that invariant.
#
# On the bare runner there is no user systemd, no installed unit, and no
# docker — so every command is expected to exit non-zero with a CLEAN error
# message. What we forbid is a raw bash "unbound variable" leak.
for sub in start stop restart enable disable status install uninstall; do
    assert_no_set_u_leak "$sub dispatch (no set -u leak)" "$SCRIPT" "$sub"
done
# `logs` is special: it execs `journalctl -f` which streams every historical
# journal record for the unit. On a dev box where the lucebox service has
# actually run, that stream contains every past error — including the very
# bugs this test exists to prevent — and we'd false-positive on them. Pass
# `-n 0 --no-pager` so we only see new entries (none, in the test window).
assert_no_set_u_leak "logs dispatch (no set -u leak)" "$SCRIPT" logs -n 0 --no-pager

# ── 6. server-spawning subcommands — exercise the dispatch up to where
# the missing docker daemon stops them. `serve` is intentionally skipped
# because on a host with a working docker + the cuda12 image already
# pulled, it would actually exec into the container — at which point
# we'd be testing the image's entrypoint, not the wrapper. `pull` just
# execs `docker pull`, so we still smoke its host-side dispatch.
assert_no_set_u_leak "pull dispatch (no set -u leak)" "$SCRIPT" pull

# ── 7. Unknown subcommand → cmd_in_container fallback path. Same rule:
# clean error, no raw bash leak.
assert_no_set_u_leak "unknown subcommand dispatch" "$SCRIPT" no-such-subcommand

# ── 8. Pre-populated LUCEBOX_HOST_* env (simulates an already-probed host
# whose vars are passed in from a parent process). Useful in CI matrices
# where we want to mock a "good host" without nvidia-smi/docker on PATH.
out=$(
    NO_COLOR=1 \
    LUCEBOX_HOST_HAS_SYSTEMD=0 \
    LUCEBOX_HOST_HAS_DOCKER=0 \
    LUCEBOX_HOST_HAS_CTK=none \
    LUCEBOX_HOST_GPU_VENDOR=none \
    LUCEBOX_HOST_GPU_NAME="" \
    LUCEBOX_HOST_GPU_COUNT=0 \
    LUCEBOX_HOST_VRAM_GB=0 \
    LUCEBOX_HOST_GPU_SM="" \
    LUCEBOX_HOST_DRIVER_VERSION="" \
    LUCEBOX_HOST_DRIVER_MAJOR=0 \
    LUCEBOX_HOST_NPROC=1 \
    LUCEBOX_HOST_RAM_GB=0 \
    LUCEBOX_HOST_IS_WSL=0 \
    LUCEBOX_HOST_DOCKER_VERSION="" \
    timeout 5 bash "$SCRIPT" start 2>&1 || true
)
if grep -qE 'unbound variable|syntax error' <<<"$out"; then
    report fail "start with pre-populated LUCEBOX_HOST_* env" "leak: $(head -3 <<<"$out")"
else
    report ok "start with pre-populated LUCEBOX_HOST_* env"
fi

# ── 8b. PIN the top-of-script LUCEBOX_HOST_* safe-default seeds. Even with
# probe_host short-circuited to a no-op (the worst-case bug recurrence: a
# future refactor accidentally deletes the call from a dispatch path) the
# wrapper must not leak `unbound variable` on `start`. We achieve "probe_host
# is a no-op" by exporting `_LUCEBOX_HOST_PROBED=1` so ensure_probed skips
# the real probe — equivalent to a future refactor that calls ensure_probed
# but mis-implements the gate.
out=$(
    NO_COLOR=1 \
    _LUCEBOX_HOST_PROBED=1 \
    timeout 5 bash "$SCRIPT" start 2>&1 || true
)
if grep -qE 'unbound variable|syntax error' <<<"$out"; then
    report fail "start with probe_host bypassed (seed defaults must catch this)" "leak: $(head -3 <<<"$out")"
else
    report ok "start with probe_host bypassed (seed defaults intact)"
fi

# Same for every other systemd-surface subcommand, since the seed defaults
# are the only thing keeping these safe under `set -u` if probe_host is ever
# bypassed.
for sub in stop restart enable disable status install uninstall logs; do
    out=$(
        NO_COLOR=1 \
        _LUCEBOX_HOST_PROBED=1 \
        timeout 5 bash "$SCRIPT" "$sub" -n 0 --no-pager 2>&1 || true
    )
    if grep -qE 'unbound variable|syntax error' <<<"$out"; then
        report fail "$sub with probe_host bypassed" "leak: $(head -3 <<<"$out")"
    else
        report ok "$sub with probe_host bypassed"
    fi
done

# ── 8c. Install path writes a robust unit file. Use a sandbox HOME so we
# don't clobber the developer's real ~/.config/systemd/user/lucebox.service,
# and verify the generated unit contains the Environment= / ExecStartPre=
# hardening that Bug 2 ("systemctl start succeeds but no container") added.
# The install runs in a host with no real systemd (the sandbox doesn't have
# `systemctl --user`), so we pre-seed LUCEBOX_HOST_HAS_SYSTEMD=1 to slip past
# the require_systemd gate, then stub out the `systemctl` binary itself so
# daemon-reload is a no-op.
test_install_writes_robust_unit() {
    local label="install writes hardened unit file"
    local sandbox shim_dir
    sandbox=$(mktemp -d)
    shim_dir="$sandbox/bin"
    mkdir -p "$shim_dir"
    # Stub systemctl + docker + nvidia-smi + loginctl so the install's
    # require_host_prereqs and daemon-reload calls all succeed.
    for binname in systemctl docker nvidia-smi loginctl; do
        cat > "$shim_dir/$binname" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  ps|version) exit 0 ;;
  show-user) echo "Linger=no" ;;
  --query-gpu=*) echo "Fake, 24576, 550.00, 8.9" ;;
esac
exit 0
STUB
        chmod +x "$shim_dir/$binname"
    done
    local out rc unit_path
    unit_path="$sandbox/.config/systemd/user/lucebox.service"
    out=$(
        set +e
        HOME="$sandbox" \
        XDG_CONFIG_HOME="$sandbox/.config" \
        XDG_DATA_HOME="$sandbox/.local/share" \
        PATH="$shim_dir:$PATH" \
        LUCEBOX_HOST_HAS_SYSTEMD=1 \
        LUCEBOX_HOST_HAS_DOCKER=1 \
        LUCEBOX_HOST_HAS_CTK=runtime \
        LUCEBOX_HOST_GPU_VENDOR=nvidia \
        _LUCEBOX_HOST_PROBED=1 \
        NO_COLOR=1 \
        timeout 10 bash "$SCRIPT" install 2>&1
        echo "RC=$?"
    )
    rc=$(grep -oE 'RC=[0-9]+$' <<<"$out" | tail -1 | sed 's/^RC=//')
    rc="${rc:-99}"
    if [ "$rc" != "0" ]; then
        report fail "$label" "exit $rc; output: $(head -10 <<<"$out")"
        rm -rf "$sandbox"
        return
    fi
    if [ ! -f "$unit_path" ]; then
        report fail "$label" "unit file not written at $unit_path"
        rm -rf "$sandbox"
        return
    fi
    # Required hardening — each line is a Bug-2 root-cause defence:
    #   ExecStartPre=…docker rm -f …   → clear orphaned container name
    #   Environment=PATH=…             → systemd user-session PATH is sparse
    #   Environment=LUCEBOX_IMAGE=…    → pin the image the user installed against
    local missing=""
    for needle in \
        "ExecStartPre=" \
        "Environment=PATH=" \
        "Environment=LUCEBOX_IMAGE=" \
        "Environment=LUCEBOX_VARIANT=" \
        "Environment=LUCEBOX_PORT=" \
        "Environment=LUCEBOX_MODELS=" \
        ; do
        grep -qF "$needle" "$unit_path" || missing="$missing $needle"
    done
    if [ -n "$missing" ]; then
        report fail "$label" "unit missing required directives:$missing"
        rm -rf "$sandbox"
        return
    fi
    report ok "$label"
    rm -rf "$sandbox"
}
test_install_writes_robust_unit

# ── 9. entrypoint.sh dispatch — confirm the in-container dispatch routes
# trivial subcommands (shell, an unknown passthrough) without firing
# `set -u` on DFLASH_* / DRAFT_* vars that only get assigned on the
# serve path. We can't fully exec the serve path here (it needs nvidia
# and the compiled binary) but we can confirm the early dispatch is clean.
#
# Each `exec` would actually try to run the underlying binary, which we
# don't have — so we shim it by overriding `exec` via a wrapper script.
# Easier: just confirm `bash -n` parses and run a tiny subset.
out=$(NO_COLOR=1 SUBCMD=help bash -c "
    cd '$ROOT'
    # Simulate 'docker run ... lucebox-hub:cuda12 shell echo ok' — entrypoint
    # gets SUBCMD=shell and execs /bin/bash with the rest of argv. We replace
    # exec via PATH so we don't actually exec.
    tmpdir=\$(mktemp -d)
    trap 'rm -rf \$tmpdir' EXIT
    cat > \$tmpdir/uv <<'STUB'
#!/usr/bin/env bash
echo \"uv stub: \$*\"
exit 0
STUB
    chmod +x \$tmpdir/uv
    PATH=\$tmpdir:\$PATH bash $ENTRYPOINT shell -c 'echo entrypoint-shell-dispatched'
" 2>&1 || true)
if grep -qE 'unbound variable|syntax error' <<<"$out"; then
    report fail "entrypoint shell dispatch (no set -u leak)" "leak: $(head -5 <<<"$out")"
else
    report ok "entrypoint shell dispatch (no set -u leak)"
fi

# ── 10. entrypoint.sh serve-path under `set -u` — drive the REAL
# server/scripts/entrypoint.sh through its full draft-resolution block by
# sandboxing it with a synthetic DFLASH_DIR layout and a `dflash_server`
# shim that captures argv instead of execing the native binary. The
# `DRAFT_FAMILY_GLOB: unbound variable` bug fired precisely here — the
# previous version of this test inlined the block instead of sourcing
# the real file, and silently passed even when the shipped script was
# broken. So this test invokes server/scripts/entrypoint.sh directly.
test_entrypoint_serve_path() {
    local label="$1" target_name="$2" draft_file="$3"
    local sandbox draft_dir models_dir bin_dir shim_dir
    sandbox=$(mktemp -d)
    models_dir="$sandbox/models"
    draft_dir="$models_dir/draft"
    bin_dir="$sandbox/build"
    shim_dir="$sandbox/bin"
    mkdir -p "$draft_dir" "$bin_dir" "$shim_dir"
    # Synthetic target (must be a real file at least 5 GB to pass the
    # auto-detect block, OR we set DFLASH_TARGET explicitly to skip it).
    touch "$models_dir/$target_name"
    touch "$draft_dir/$draft_file"
    # `dflash_server` shim — print argv and exit 0 instead of running.
    cat > "$bin_dir/dflash_server" <<'STUB'
#!/usr/bin/env bash
printf '[shim] dflash_server'
for a in "$@"; do printf ' %q' "$a"; done
printf '\n'
exit 0
STUB
    chmod +x "$bin_dir/dflash_server"
    # `nvidia-smi` shim — pretend we have a 24 GB GPU so the autotune
    # block runs but doesn't pick the under-12-GB warn tier.
    cat > "$shim_dir/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=memory.total"*) echo 24576 ;;
  -L|*-L*) echo "GPU 0: Fake (UUID: 0)" ;;
  *) echo "ok" ;;
esac
exit 0
STUB
    chmod +x "$shim_dir/nvidia-smi"

    local out rc
    out=$(
        set +e
        PATH="$shim_dir:$PATH" \
        DFLASH_DIR="$sandbox" \
        DFLASH_SERVER_BIN="$bin_dir/dflash_server" \
        DFLASH_TARGET="$models_dir/$target_name" \
        DFLASH_DRAFT="$draft_dir" \
            timeout 10 bash "$ENTRYPOINT" serve 2>&1
        echo "RC=$?"
    )
    rc=$(grep -oE 'RC=[0-9]+$' <<<"$out" | tail -1 | sed 's/^RC=//')
    rc="${rc:-99}"
    rm -rf "$sandbox"
    if grep -qE 'unbound variable|syntax error' <<<"$out"; then
        report fail "$label" "leak: $(head -5 <<<"$out")"
    elif [ "$rc" != "0" ]; then
        report fail "$label" "exit $rc; output: $(head -5 <<<"$out")"
    elif ! grep -qF "[shim] dflash_server" <<<"$out"; then
        report fail "$label" "shim never executed; output: $(head -5 <<<"$out")"
    else
        report ok "$label"
    fi
}

# Exercise three branches of the family-glob logic: qwen3.6 + gemma-4 (the
# two families with family-specific globs) and an unknown target that
# triggers the empty-FAMILY_GLOBS fallback to the generic glob list.
test_entrypoint_serve_path "entrypoint serve: qwen3.6 family match" \
    "Qwen3.6-27B-Q4_K_M.gguf" "dflash-draft-3.6-test.gguf"
test_entrypoint_serve_path "entrypoint serve: gemma-4-31b family match" \
    "gemma-4-31B-it-Q8_0.gguf" "gemma-4-31b-dflash-q8.gguf"
test_entrypoint_serve_path "entrypoint serve: generic fallback" \
    "Mystery-Model-7B.gguf" "model.gguf"

# ── 11. entrypoint.sh serve-path with MULTIPLE target-sized GGUFs in
# models/. The single-candidate fixture in test 10 doesn't exercise the
# auto-detect path that picks "first alphabetically" when more than one
# target ≥5 GB lives in the models dir — that path is what the sindri
# decode sweep tripped over after the user added the qwen3.6-moe preset
# (commit 4b6bced) alongside the existing Qwen3.6-27B target. The crash
# manifested as `DRAFT_FAMILY_GLOB: unbound variable`, and the partial
# fix in a87bb93 didn't survive a recurrence.
#
# Uses sparse files (`truncate -s 6G`) so the test stays cheap on disk —
# the 6 GB virtual size is enough to clear the find ... -size +5G filter
# without consuming actual blocks. Skip if truncate is missing (e.g.
# minimal busybox CI image).
test_entrypoint_multi_target() {
    local label="$1"
    shift
    if ! command -v truncate &>/dev/null; then
        report ok "$label (skipped: truncate not available)"
        return
    fi
    local sandbox draft_dir models_dir bin_dir shim_dir
    sandbox=$(mktemp -d)
    models_dir="$sandbox/models"
    draft_dir="$models_dir/draft"
    bin_dir="$sandbox/build"
    shim_dir="$sandbox/bin"
    mkdir -p "$draft_dir" "$bin_dir" "$shim_dir"
    # Two qwen3.6-shaped targets ≥5 GB each — exactly the layout that
    # broke on sindri (Qwen3.6-27B + Qwen3.6-35B-A3B-UD-Q4_K_M).
    truncate -s 6G "$models_dir/Qwen3.6-27B-Q4_K_M.gguf"
    truncate -s 6G "$models_dir/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    touch "$draft_dir/dflash-draft-3.6-test.gguf"
    cat > "$bin_dir/dflash_server" <<'STUB'
#!/usr/bin/env bash
printf '[shim] dflash_server'
for a in "$@"; do printf ' %q' "$a"; done
printf '\n'
exit 0
STUB
    chmod +x "$bin_dir/dflash_server"
    cat > "$shim_dir/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=memory.total"*) echo 24576 ;;
  -L|*-L*) echo "GPU 0: Fake (UUID: 0)" ;;
  *) echo "ok" ;;
esac
exit 0
STUB
    chmod +x "$shim_dir/nvidia-smi"

    local out rc
    out=$(
        set +e
        # NOTE: deliberately NOT setting DFLASH_TARGET — the test must
        # exercise the auto-detect block (line ~151). The explicit-config
        # workaround from the bug report would skip the bug entirely.
        PATH="$shim_dir:$PATH" \
        DFLASH_DIR="$sandbox" \
        DFLASH_SERVER_BIN="$bin_dir/dflash_server" \
        DFLASH_DRAFT="$draft_dir" \
            timeout 10 bash "$ENTRYPOINT" serve 2>&1
        echo "RC=$?"
    )
    rc=$(grep -oE 'RC=[0-9]+$' <<<"$out" | tail -1 | sed 's/^RC=//')
    rc="${rc:-99}"
    rm -rf "$sandbox"
    if grep -qE 'unbound variable|syntax error' <<<"$out"; then
        report fail "$label" "leak: $(grep -E 'unbound variable|syntax error' <<<"$out" | head -3)"
    elif [ "$rc" != "0" ]; then
        report fail "$label" "exit $rc; output: $(head -5 <<<"$out")"
    elif ! grep -qF "[shim] dflash_server" <<<"$out"; then
        report fail "$label" "shim never executed; output: $(head -10 <<<"$out")"
    elif ! grep -qF "Multiple candidate targets" <<<"$out"; then
        report fail "$label" "multi-target warn missing — did the auto-detect block fire?"
    else
        report ok "$label"
    fi
}

# Drive the regression: the sindri layout that broke (post-moe-preset).
test_entrypoint_multi_target "entrypoint serve: multi-target auto-detect (no DRAFT_FAMILY_GLOB leak)"

# Also drive the DFLASH_DRAFT-is-a-file path. The init at entrypoint.sh:257
# sits inside `if [ -d "$DFLASH_DRAFT" ]; then` — when DRAFT is a file the
# block is skipped, and any future read of DRAFT_FAMILY_GLOB outside the
# block would trip set -u. The defensive `:-` guard at the read site is
# meant to survive that refactor; this test guarantees it.
test_entrypoint_draft_is_file() {
    local label="$1"
    local sandbox draft_dir models_dir bin_dir shim_dir
    sandbox=$(mktemp -d)
    models_dir="$sandbox/models"
    draft_dir="$models_dir/draft"
    bin_dir="$sandbox/build"
    shim_dir="$sandbox/bin"
    mkdir -p "$draft_dir" "$bin_dir" "$shim_dir"
    touch "$models_dir/Qwen3.6-27B-Q4_K_M.gguf"
    # DFLASH_DRAFT points at a FILE (not a directory).
    touch "$draft_dir/dflash-draft-3.6-test.gguf"
    cat > "$bin_dir/dflash_server" <<'STUB'
#!/usr/bin/env bash
printf '[shim] dflash_server'
for a in "$@"; do printf ' %q' "$a"; done
printf '\n'
exit 0
STUB
    chmod +x "$bin_dir/dflash_server"
    cat > "$shim_dir/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=memory.total"*) echo 24576 ;;
  -L|*-L*) echo "GPU 0: Fake (UUID: 0)" ;;
  *) echo "ok" ;;
esac
exit 0
STUB
    chmod +x "$shim_dir/nvidia-smi"

    local out rc
    out=$(
        set +e
        PATH="$shim_dir:$PATH" \
        DFLASH_DIR="$sandbox" \
        DFLASH_SERVER_BIN="$bin_dir/dflash_server" \
        DFLASH_TARGET="$models_dir/Qwen3.6-27B-Q4_K_M.gguf" \
        DFLASH_DRAFT="$draft_dir/dflash-draft-3.6-test.gguf" \
            timeout 10 bash "$ENTRYPOINT" serve 2>&1
        echo "RC=$?"
    )
    rc=$(grep -oE 'RC=[0-9]+$' <<<"$out" | tail -1 | sed 's/^RC=//')
    rc="${rc:-99}"
    rm -rf "$sandbox"
    if grep -qE 'unbound variable|syntax error' <<<"$out"; then
        report fail "$label" "leak: $(grep -E 'unbound variable|syntax error' <<<"$out" | head -3)"
    elif [ "$rc" != "0" ]; then
        report fail "$label" "exit $rc; output: $(head -5 <<<"$out")"
    else
        report ok "$label"
    fi
}
test_entrypoint_draft_is_file "entrypoint serve: DFLASH_DRAFT is a file (no DRAFT_FAMILY_GLOB leak)"

# ── 12. entrypoint.sh writes HOST_INFO atomically on the serve path. The
# C++ server reads /opt/lucebox-hub/HOST_INFO into ServerConfig.host_info
# and surfaces it under /props.host. We can't write to /opt/lucebox-hub
# from the test runner, so override the path by sourcing the helpers and
# calling _build_host_info_json directly. The full entrypoint runs in
# test 10/11 already; this test pins the JSON shape independently.
test_entrypoint_host_info_json() {
    local label="$1"
    # Source the helper functions from the real entrypoint.sh.
    # shellcheck disable=SC1090
    source <(awk '/^_json_escape\(\) \{/,/^\}/' "$ENTRYPOINT")
    # shellcheck disable=SC1090
    source <(awk '/^_json_str_or_null\(\) \{/,/^\}/' "$ENTRYPOINT")
    # shellcheck disable=SC1090
    source <(awk '/^_json_int_or_null\(\) \{/,/^\}/' "$ENTRYPOINT")
    # shellcheck disable=SC1090
    source <(awk '/^_emit_gpu_array\(\) \{/,/^\}/' "$ENTRYPOINT")
    # shellcheck disable=SC1090
    source <(awk '/^_build_host_info_json\(\) \{/,/^\}/' "$ENTRYPOINT")

    local out
    LUCEBOX_HOST_OS_PRETTY="Ubuntu 22.04.3 LTS" \
    LUCEBOX_HOST_KERNEL="6.6.87.2-microsoft-standard-WSL2" \
    LUCEBOX_HOST_WSL_VERSION="wsl2" \
    LUCEBOX_HOST_DOCKER_VERSION="29.1.3" \
    LUCEBOX_HOST_DRIVER_VERSION="596.36" \
    LUCEBOX_HOST_NVIDIA_CTK_VERSION="1.16.2" \
    LUCEBOX_HOST_CPU_MODEL='Intel(R) Core(TM) Ultra 9 275HX' \
    LUCEBOX_HOST_NPROC=24 \
    LUCEBOX_HOST_RAM_GB=64 \
    LUCEBOX_HOST_GPU_LIST_CSV="0, GPU-abc, 00000000:01:00.0, NVIDIA RTX 5090, 12.0, 24576 MiB, 175.00 W" \
    LUCEBOX_HOST_CUDA_VISIBLE_DEVICES="0" \
        out=$(_build_host_info_json "lucebox.sh" "lucebox.sh" "2026-05-28T20:31:42Z")
    if ! python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert d['os_pretty']=='Ubuntu 22.04.3 LTS'; assert d['wsl_version']=='wsl2'; assert d['nvidia_ctk_version']=='1.16.2'; assert d['source']=='lucebox.sh'; assert d['gpus'][0]['vram_gb']==24; assert d['gpus'][0]['name']=='NVIDIA RTX 5090'" "$out" >/dev/null 2>&1; then
        report fail "$label (populated)" "JSON shape mismatch: $out"
        return
    fi
    # Now drive the unknown path: every LUCEBOX_HOST_* unset → nulls and source=unknown.
    out=$(env -i bash -c "
        set -u
        $(declare -f _json_escape _json_str_or_null _json_int_or_null _emit_gpu_array _build_host_info_json)
        _build_host_info_json 'unknown' 'entrypoint.sh' '2026-05-28T20:31:42Z'
    ")
    if ! python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert d['source']=='unknown'; assert d['gpus']==[]; assert d['os_pretty'] is None" "$out" >/dev/null 2>&1; then
        report fail "$label (unknown)" "JSON shape mismatch: $out"
        return
    fi
    report ok "$label"
}
test_entrypoint_host_info_json "entrypoint HOST_INFO JSON shape (populated + unknown)"

# ── install.sh end-to-end ─────────────────────────────────────────────────
# Drive install.sh against a file:// URL pointing at a fixture lucebox.sh,
# verify the installed copy has LUCEBOX_INSTALLED_FROM rewritten to the
# fetched URL — that's the contract that `lucebox update` depends on to
# preserve the user's channel across upgrades.
test_install_sh_bakes_source_url() {
    local label="$1"
    local tmp dest_dir dest_path src_url out rc
    tmp=$(mktemp -d -t lucebox-install.XXXXXX)
    # Use the real lucebox.sh as the "remote" file — `file://` works with
    # curl out of the box and exercises the same install.sh code path as
    # an https fetch would.
    src_url="file://$SCRIPT"
    dest_dir="$tmp/bin"
    dest_path="$dest_dir/lucebox"
    out=$(LUCEBOX_INSTALL_URL="$src_url" LUCEBOX_INSTALL_DEST="$dest_path" \
        NO_COLOR=1 bash "$INSTALLER" 2>&1) || rc=$?
    rc="${rc:-0}"
    if [ "$rc" -ne 0 ]; then
        rm -rf "$tmp"
        report fail "$label" "installer exited $rc; output: $(printf '%s' "$out" | head -3)"
        return
    fi
    if [ ! -x "$dest_path" ]; then
        rm -rf "$tmp"
        report fail "$label" "installed file missing or not executable at $dest_path"
        return
    fi
    if ! grep -q "^LUCEBOX_INSTALLED_FROM=\"$src_url\"$" "$dest_path"; then
        rm -rf "$tmp"
        report fail "$label" "LUCEBOX_INSTALLED_FROM not rewritten in installed copy"
        return
    fi
    rm -rf "$tmp"
    report ok "$label"
}
test_install_sh_bakes_source_url "install.sh bakes LUCEBOX_INSTALLED_FROM into installed copy"

# ── update dispatch ───────────────────────────────────────────────────────
# `lucebox update` must dispatch to cmd_update — verify it's wired in the
# main case statement and appears in --help. We can't actually run the
# update (it'd curl + replace this very script) so the test is parse-level.
test_update_subcommand_wired() {
    local label="$1"
    local out
    out=$(LUCEBOX_HOST_HAS_SYSTEMD=0 "$SCRIPT" --help 2>&1)
    if ! grep -q '^  update ' <<<"$out"; then
        report fail "$label" "update command missing from --help output"
        return
    fi
    if ! grep -q '^[[:space:]]*update)[[:space:]]*cmd_update' "$SCRIPT"; then
        report fail "$label" "update) → cmd_update dispatch not wired"
        return
    fi
    report ok "$label"
}
test_update_subcommand_wired "lucebox update subcommand is wired"

# ── IMAGE_BASE derived from install source ────────────────────────────────
# Source lucebox.sh in a subshell with LUCEBOX_INSTALLED_FROM pointing at
# various URLs, then check that IMAGE_BASE comes out right. Uses
# `set -e; return` early so we don't actually run the wrapper's main().
test_image_base_derives_from_install_url() {
    local label="$1" url expected got
    for case in \
        "https://raw.githubusercontent.com/easel/lucebox-hub/feat/lucebox-docker/lucebox.sh|ghcr.io/easel/lucebox-hub" \
        "https://raw.githubusercontent.com/Luce-Org/lucebox-hub/main/lucebox.sh|ghcr.io/luce-org/lucebox-hub" \
        "https://raw.githubusercontent.com/easel/lucebox-hub/601ab52/lucebox.sh|ghcr.io/easel/lucebox-hub" \
        "https://example.com/bogus|ghcr.io/luce-org/lucebox-hub"
    do
        url="${case%%|*}"
        expected="${case##*|}"
        # Extract the derivation function from the script and run it in
        # isolation — sourcing the whole script triggers main() and side
        # effects we don't want under a test harness.
        got=$(bash -c '
            '"$(sed -n "/^_lucebox_derive_image()/,/^}/p" "$SCRIPT")"'
            _lucebox_derive_image "$1"
        ' bash "$url")
        if [ "$got" != "$expected" ]; then
            report fail "$label" "url=$url expected=$expected got=$got"
            return
        fi
    done
    report ok "$label"
}
test_image_base_derives_from_install_url "IMAGE_BASE derived from LUCEBOX_INSTALLED_FROM (4 URL shapes)"

# ── config.toml reader + resolver ─────────────────────────────────────────
# Drive _lucebox_config_get + _lucebox_resolve against a fixture
# config.toml in a tmp $LUCEBOX_HOME. Verifies the wrapper agrees with
# the Python CLI on every scalar that lives in [image]/[runtime]/[paths].
test_config_toml_reader_and_resolve() {
    local label="$1" tmp got
    tmp=$(mktemp -d -t lucebox-cfg.XXXXXX)
    cat > "$tmp/config.toml" <<'TOML'
[image]
variant = "cuda13"
registry = "ghcr.io/myorg/forkedhub"

[runtime]
port = 9090
container_name = "luce-test"

[paths]
models = "/srv/models"

[dflash]
budget = 22
lazy = false
TOML

    # Exercise both helpers + the resolver via a subshell that sources
    # the relevant snippets out of lucebox.sh. Each case is a triple:
    # env_value | toml_key | default | expected
    local cases=(
        "|image.registry|ghcr.io/luce-org/lucebox-hub|ghcr.io/myorg/forkedhub"
        "|image.variant|cuda12|cuda13"
        "|runtime.port|8080|9090"
        "|runtime.container_name|lucebox|luce-test"
        "|paths.models|/var/lib/lucebox|/srv/models"
        "OVERRIDE|image.registry|ghcr.io/luce-org/lucebox-hub|OVERRIDE"
        "|missing.key|fallback-default|fallback-default"
    )
    local case env_value toml_key default expected
    for case in "${cases[@]}"; do
        IFS='|' read -r env_value toml_key default expected <<<"$case"
        got=$(LUCEBOX_HOME="$tmp" bash -c '
            '"$(sed -n "/^_lucebox_config_path()/,/^}/p" "$SCRIPT")"'
            '"$(sed -n "/^_lucebox_config_get()/,/^}/p" "$SCRIPT")"'
            '"$(sed -n "/^_lucebox_resolve()/,/^}/p" "$SCRIPT")"'
            _lucebox_resolve "$1" "$2" "$3"
        ' bash "$env_value" "$toml_key" "$default")
        if [ "$got" != "$expected" ]; then
            rm -rf "$tmp"
            report fail "$label" "env=$env_value key=$toml_key default=$default expected=$expected got=$got"
            return
        fi
    done
    rm -rf "$tmp"
    report ok "$label"
}
test_config_toml_reader_and_resolve "config.toml reader + env > toml > default resolution (7 cases)"

# ── cmd_serve under systemd: INVOCATION_ID short-circuits is-active ──────
# When systemd invokes the wrapper as a unit's ExecStart, it sets
# $INVOCATION_ID. The wrapper must NOT then refuse "already running under
# systemd" — that's a self-defeating check that turns into a restart loop.
# Verify the guard is present in the source (the actual behavior requires
# a running systemd unit to test end-to-end, which the harness can't do).
test_cmd_serve_invocation_id_guard() {
    local label="$1"
    if ! grep -q 'INVOCATION_ID' "$SCRIPT"; then
        report fail "$label" "INVOCATION_ID guard missing from cmd_serve preflight"
        return
    fi
    # The guard must be the AND-condition gating the is-active check.
    # If grep finds the is-active line WITHOUT INVOCATION_ID nearby,
    # the guard isn't wired correctly.
    if ! awk '
        /INVOCATION_ID/ { saw_guard = NR }
        /is-active --quiet "\$UNIT_NAME"/ {
            if (saw_guard && NR - saw_guard <= 3) found = 1
        }
        END { exit (found ? 0 : 1) }
    ' "$SCRIPT"; then
        report fail "$label" "INVOCATION_ID not adjacent to is-active check (guard not wired)"
        return
    fi
    report ok "$label"
}
test_cmd_serve_invocation_id_guard "cmd_serve has INVOCATION_ID guard on systemd is-active check"

# ── cmd_systemctl_passthrough: smart start ───────────────────────────────
# Verify the source has the "already active" + "restart loop" short
# circuits for the start action. Behavior-level testing requires a real
# unit; this is a source-level guarantee that the branches exist.
test_cmd_start_already_active_shortcircuit() {
    local label="$1"
    if ! grep -q 'is already active' "$SCRIPT"; then
        report fail "$label" "already-active short-circuit missing"
        return
    fi
    if ! grep -q 'is in restart-loop' "$SCRIPT"; then
        report fail "$label" "restart-loop short-circuit missing"
        return
    fi
    report ok "$label"
}
test_cmd_start_already_active_shortcircuit "lucebox start has already-active + restart-loop short-circuits"

# ── install.sh SHA-pin refusal + CHANNEL override ────────────────────────
# A SHA-pinned LUCEBOX_INSTALL_URL with no LUCEBOX_INSTALL_CHANNEL must
# refuse — otherwise `lucebox update` would re-fetch that frozen SHA
# forever. With CHANNEL set, the bake-in uses the channel URL, not the
# fetch URL.
test_install_sha_pin_refusal_and_channel_override() {
    local label="$1" tmp got rc
    tmp=$(mktemp -d -t lucebox-sha.XXXXXX)

    # Case 1: SHA-pinned URL without CHANNEL → must refuse
    LUCEBOX_INSTALL_URL="https://raw.githubusercontent.com/easel/lucebox-hub/abc1234567/lucebox.sh" \
    LUCEBOX_INSTALL_DEST="$tmp/lucebox1" \
    NO_COLOR=1 \
        bash "$INSTALLER" >/dev/null 2>&1 && rc=0 || rc=$?
    if [ "$rc" -eq 0 ]; then
        rm -rf "$tmp"
        report fail "$label" "SHA-pinned URL without CHANNEL should have refused (rc=$rc, got success)"
        return
    fi
    if [ -f "$tmp/lucebox1" ]; then
        rm -rf "$tmp"
        report fail "$label" "SHA-pinned URL refusal still wrote $tmp/lucebox1"
        return
    fi

    # Case 2: SHA-pinned URL WITH CHANNEL → installs, bakes CHANNEL
    LUCEBOX_INSTALL_URL="file://$SCRIPT" \
    LUCEBOX_INSTALL_CHANNEL="https://raw.githubusercontent.com/easel/lucebox-hub/feat/lucebox-docker/lucebox.sh" \
    LUCEBOX_INSTALL_DEST="$tmp/lucebox2" \
    NO_COLOR=1 \
        bash "$INSTALLER" >/dev/null 2>&1 || rc=$?
    got=$(grep '^LUCEBOX_INSTALLED_FROM=' "$tmp/lucebox2" 2>/dev/null || echo missing)
    if [ "$got" != 'LUCEBOX_INSTALLED_FROM="https://raw.githubusercontent.com/easel/lucebox-hub/feat/lucebox-docker/lucebox.sh"' ]; then
        rm -rf "$tmp"
        report fail "$label" "CHANNEL not baked; got: $got"
        return
    fi

    rm -rf "$tmp"
    report ok "$label"
}
test_install_sha_pin_refusal_and_channel_override "install.sh refuses SHA-pin without CHANNEL + honors CHANNEL override"

# ── lucebox completion ───────────────────────────────────────────────────
# The completion script must source cleanly and complete a known prefix.
test_completion_bash() {
    local label="$1" out
    out=$(LUCEBOX_HOST_HAS_SYSTEMD=0 bash -c '
        source <("$1" completion bash 2>/dev/null)
        COMP_WORDS=(lucebox conf)
        COMP_CWORD=1
        _lucebox_complete
        printf "%s\n" "${COMPREPLY[@]}"
    ' bash "$SCRIPT")
    if ! grep -qx 'config' <<<"$out"; then
        report fail "$label" "completion didn't suggest 'config' for prefix 'conf'; got: $(printf '%s' "$out" | tr '\n' ' ')"
        return
    fi
    report ok "$label"
}
test_completion_bash "lucebox completion bash completes a known prefix"

echo
if [ "$fail" -eq 0 ]; then
    echo "[test_lucebox_sh] $pass passed, 0 failed"
    exit 0
else
    echo "[test_lucebox_sh] $pass passed, $fail failed" >&2
    exit 1
fi
