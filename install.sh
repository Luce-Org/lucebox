#!/usr/bin/env bash
# install.sh — Bootstrap installer for the lucebox host wrapper.
#
# Canonical install (Luce-Org main, stable channel):
#
#   curl -fsSL https://raw.githubusercontent.com/Luce-Org/lucebox/main/install.sh | bash
#
# Install from a different fork / branch (dev channel). Note the env var
# is on the `bash` side of the pipe — `VAR=val curl … | bash` would attach
# it to the `curl` process, leaving `bash` with the canonical default:
#
#   curl -fsSL https://raw.githubusercontent.com/easel/lucebox-hub/feat/lucebox-docker/install.sh | \
#     LUCEBOX_INSTALL_URL=https://raw.githubusercontent.com/easel/lucebox-hub/feat/lucebox-docker/lucebox.sh bash
#
# The installer bakes the source URL into the installed `lucebox.sh` as
# `LUCEBOX_INSTALLED_FROM=...`, so `lucebox update` later re-pulls from the
# same channel without the user having to remember which fork they used.
#
# Override the install destination via $LUCEBOX_INSTALL_DEST (default
# $HOME/.local/bin/lucebox). This is what `lucebox update` uses to replace
# the file in place.

set -euo pipefail

LUCEBOX_INSTALL_URL="${LUCEBOX_INSTALL_URL:-https://raw.githubusercontent.com/Luce-Org/lucebox/main/lucebox.sh}"
DEST="${LUCEBOX_INSTALL_DEST:-$HOME/.local/bin/lucebox}"

# ── helpers ───────────────────────────────────────────────────────────────
C_OK=$'\033[1;32m' ; C_ERR=$'\033[1;31m' ; C_DIM=$'\033[2m' ; C_RST=$'\033[0m'
if [ ! -t 1 ] || [ "${NO_COLOR:-}" ]; then
    C_OK="" ; C_ERR="" ; C_DIM="" ; C_RST=""
fi
info() { printf '%s[install]%s %s\n' "$C_DIM" "$C_RST" "$*"; }
ok()   { printf '%s[install] ✓%s %s\n' "$C_OK"  "$C_RST" "$*"; }
die()  { printf '%s[install] ✗%s %s\n' "$C_ERR" "$C_RST" "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required (apt-get install curl)"

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

# ── decide what gets baked in as the persisted channel ───────────────────
# Do this before fetching so a SHA-pinned URL is refused for the intended
# reason even when that remote object is unavailable.
channel_url="${LUCEBOX_INSTALL_CHANNEL:-}"
if [ -z "$channel_url" ]; then
    # A full 40-char SHA is what `git rev-parse HEAD` and GitHub raw URLs use.
    # Shorter hex-like path segments may be branch names, so don't reject them.
    if [[ "$LUCEBOX_INSTALL_URL" =~ /[0-9a-fA-F]{40}/[^/]+\.sh$ ]]; then
        die "$(cat <<EOM
LUCEBOX_INSTALL_URL is SHA-pinned ($LUCEBOX_INSTALL_URL).
Persisting that as LUCEBOX_INSTALLED_FROM would freeze \`lucebox update\`
to that specific commit forever. Set LUCEBOX_INSTALL_CHANNEL to the
branch URL you want \`update\` to track, e.g.:

  curl -fsSL <sha-pinned>/install.sh | \\
    LUCEBOX_INSTALL_URL=<sha-pinned>/lucebox.sh \\
    LUCEBOX_INSTALL_CHANNEL=https://raw.githubusercontent.com/<org>/<repo>/<branch>/lucebox.sh \\
    bash
EOM
)"
    fi
    channel_url="$LUCEBOX_INSTALL_URL"
fi

# ── fetch ─────────────────────────────────────────────────────────────────
tmp=$(mktemp -t lucebox.XXXXXX) || die "couldn't create temp file"
# shellcheck disable=SC2064  # we want $tmp expanded now, not at trap time
trap "rm -f '$tmp' '$tmp.baked'" EXIT
info "fetching $LUCEBOX_INSTALL_URL"
curl --connect-timeout 10 --max-time 120 -fsSL "$LUCEBOX_INSTALL_URL" -o "$tmp" \
    || die "download failed from $LUCEBOX_INSTALL_URL"

# Release automation can pin the exact wrapper payload. This is optional for
# branch-channel installs, where the URL intentionally moves over time.
expected_sha="${LUCEBOX_WRAPPER_SHA256:-}"
if [ -n "$expected_sha" ]; then
    [[ "$expected_sha" =~ ^[0-9a-fA-F]{64}$ ]] \
        || die "LUCEBOX_WRAPPER_SHA256 must be exactly 64 hexadecimal characters"
    actual_sha=$(sha256_file "$tmp")
    expected_sha=$(printf '%s' "$expected_sha" | tr '[:upper:]' '[:lower:]')
    [ "$actual_sha" = "$expected_sha" ] \
        || die "wrapper checksum mismatch (expected $expected_sha, got $actual_sha)"
    ok "wrapper sha256 verified"
fi

# ── sanity check ──────────────────────────────────────────────────────────
# Refuse to install something that isn't recognizably lucebox.sh. Catches
# 404 pages, redirects to HTML, and accidental URL typos.
head -1 "$tmp" | grep -q '^#!/usr/bin/env bash$' \
    || die "downloaded file does not look like a bash script (got: $(head -1 "$tmp"))"
grep -q '^VERSION=' "$tmp" \
    || die "downloaded file is missing VERSION marker — not lucebox.sh?"

# Bake the channel URL into the file. Use a `|` delimiter since URLs
# contain `/`. The line is expected to exist in lucebox.sh with a `:-`
# default; we rewrite the whole assignment.
#
# The URL ends up inside a bash double-quoted literal in the installed
# script, so any of $ ` " \ in `channel_url` would break the installed
# file (or worse, allow command substitution to run at next sourcing).
# Validate that the URL is plain http(s)+ASCII-URL-safe characters; we
# don't expect arbitrary content here, only an upstream raw.github URL
# (or a forked equivalent). Escape the sed metachars (\&|) separately so
# the substitution itself round-trips.
case "$channel_url" in
    *['"$`\']*) die "channel URL contains unsafe characters: $channel_url" ;;
esac
escaped_url=$(printf '%s' "$channel_url" | sed 's/[\\&|]/\\&/g')
sed "s|^LUCEBOX_INSTALLED_FROM=.*|LUCEBOX_INSTALLED_FROM=\"$escaped_url\"|" "$tmp" > "$tmp.baked"
mv "$tmp.baked" "$tmp"
grep -Fqx "LUCEBOX_INSTALLED_FROM=\"$channel_url\"" "$tmp" \
    || die "failed to bake install source into the downloaded script"

# ── install ───────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$DEST")"
chmod +x "$tmp"
mv "$tmp" "$DEST"
trap - EXIT
ok "installed lucebox → $DEST"
info "  fetched from:    $LUCEBOX_INSTALL_URL"
info "  update channel:  $channel_url"
if [ "$LUCEBOX_INSTALL_URL" != "$channel_url" ]; then
    info "  (lucebox update will track the channel URL, not the fetch URL)"
fi

# ── PATH hint ─────────────────────────────────────────────────────────────
case ":${PATH:-}:" in
    *":$(dirname "$DEST"):"*) ;;
    *) info "  hint: add $(dirname "$DEST") to PATH so 'lucebox' is on the path" ;;
esac

cat <<EOF

Next:
  ${C_DIM}lucebox${C_RST}                  open the guided setup and inference menu

For automation:
  ${C_DIM}lucebox check${C_RST}            verify Docker + NVIDIA CUDA or AMD ROCm
  ${C_DIM}lucebox setup${C_RST}            run guided setup directly
  ${C_DIM}lucebox update${C_RST}           re-run this installer to fetch the latest lucebox.sh
EOF
