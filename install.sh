#!/usr/bin/env bash
# install.sh — Bootstrap installer for the lucebox host wrapper.
#
# Canonical install (Luce-Org main, stable channel):
#
#   curl -fsSL https://raw.githubusercontent.com/Luce-Org/lucebox-hub/main/install.sh | bash
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

LUCEBOX_INSTALL_URL="${LUCEBOX_INSTALL_URL:-https://raw.githubusercontent.com/Luce-Org/lucebox-hub/main/lucebox.sh}"
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

# ── fetch ─────────────────────────────────────────────────────────────────
tmp=$(mktemp -t lucebox.XXXXXX) || die "couldn't create temp file"
# shellcheck disable=SC2064  # we want $tmp expanded now, not at trap time
trap "rm -f '$tmp' '$tmp.bak'" EXIT
info "fetching $LUCEBOX_INSTALL_URL"
curl -fsSL "$LUCEBOX_INSTALL_URL" -o "$tmp" \
    || die "download failed from $LUCEBOX_INSTALL_URL"

# ── sanity check ──────────────────────────────────────────────────────────
# Refuse to install something that isn't recognizably lucebox.sh. Catches
# 404 pages, redirects to HTML, and accidental URL typos.
head -1 "$tmp" | grep -q '^#!/usr/bin/env bash$' \
    || die "downloaded file does not look like a bash script (got: $(head -1 "$tmp"))"
grep -q '^VERSION=' "$tmp" \
    || die "downloaded file is missing VERSION marker — not lucebox.sh?"

# ── decide what gets baked in as the persisted channel ───────────────────
# `lucebox update` reads LUCEBOX_INSTALLED_FROM from the installed copy and
# re-fetches from it. Persisting a SHA-pinned URL is a footgun — every
# future update would re-install the same frozen SHA forever, defeating
# the point of `update`. So:
#
#   1. If $LUCEBOX_INSTALL_CHANNEL is set, that's the persisted URL
#      (caller takes responsibility for picking a real branch URL).
#   2. Else if LUCEBOX_INSTALL_URL has a 40-char hex SHA segment, refuse
#      to persist it — tell the user to set LUCEBOX_INSTALL_CHANNEL.
#      Common case: someone curl'd from /raw/<sha>/ to bypass a stale CDN
#      cache during dev; they meant for updates to track the branch.
#   3. Else persist LUCEBOX_INSTALL_URL as-is (branch or canonical main).
channel_url="${LUCEBOX_INSTALL_CHANNEL:-}"
if [ -z "$channel_url" ]; then
    if [[ "$LUCEBOX_INSTALL_URL" =~ /[0-9a-fA-F]{7,40}/[^/]+\.sh$ ]]; then
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

# Bake the channel URL into the file. Use a `|` delimiter since URLs
# contain `/`. The line is expected to exist in lucebox.sh with a `:-`
# default; we rewrite the whole assignment.
escaped_url=$(printf '%s' "$channel_url" | sed 's/[\\&|]/\\&/g')
sed "s|^LUCEBOX_INSTALLED_FROM=.*|LUCEBOX_INSTALLED_FROM=\"$escaped_url\"|" "$tmp" > "$tmp.baked"
mv "$tmp.baked" "$tmp"
grep -q "^LUCEBOX_INSTALLED_FROM=\"$escaped_url\"$" "$tmp" \
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
  ${C_DIM}lucebox check${C_RST}            verify host prereqs (docker + NVIDIA CTK + driver)
  ${C_DIM}lucebox install${C_RST}          install the user systemd unit
  ${C_DIM}lucebox start${C_RST}            start the server
  ${C_DIM}lucebox update${C_RST}           re-run this installer to fetch the latest lucebox.sh
EOF
