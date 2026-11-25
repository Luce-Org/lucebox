#!/usr/bin/env bash
# Drift guard: the luce-bench wheel bundles share/model_cards/ as package data
# (lucebench/_model_cards/) via a hatch force-include, so the standalone PyPI
# build resolves the same cards the C++ server reads from disk. force-include
# copies the directory at build time, so the bundle *should* always match
# share/ — this builds the wheel and proves it, catching a broken/stale
# force-include path or a partial copy before it ships.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

( cd luce-bench && uv build --wheel -o "$tmp" >/dev/null )
whl="$(ls "$tmp"/*.whl | head -1)"
( cd "$tmp" && unzip -q "$whl" )

drift=0
# Forward: every share/ card must exist in the bundle and match.
for f in share/model_cards/*; do
    name="$(basename "$f")"
    b="$tmp/lucebench/_model_cards/$name"
    if [ ! -f "$b" ]; then
        echo "MISSING from bundle: $name"; drift=1; continue
    fi
    if ! diff -q "$f" "$b" >/dev/null 2>&1; then
        echo "DRIFT: $name (share/ vs bundled wheel differ)"; drift=1
    fi
done

# Reverse: catch extras / stale entries in the bundle that no longer
# exist in share/. Without this, a force-include glob that picks up
# moved or deleted files would ship undetected.
if [ -d "$tmp/lucebench/_model_cards" ]; then
    for b in "$tmp/lucebench/_model_cards"/*; do
        [ -e "$b" ] || continue
        name="$(basename "$b")"
        if [ ! -f "share/model_cards/$name" ]; then
            echo "EXTRA in bundle (not present in share/model_cards): $name"
            drift=1
        fi
    done
fi

if [ "$drift" = 0 ]; then
    echo "OK: bundled model cards are byte-identical to share/model_cards"
else
    echo "FAIL: model-card bundle drift detected"; exit 1
fi
