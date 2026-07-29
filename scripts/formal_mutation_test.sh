#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
mutation="$repo_root/formal/contracts/mutations/prefix-cache-bypass-selector.patch"

cd "$repo_root"
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
    echo "mutation sensitivity requires a committed production checkout" >&2
    exit 2
fi

base_sha="$(git rev-parse HEAD)"
temporary_root="$(mktemp -d)"
trap 'rm -rf -- "$temporary_root"' EXIT
mutated_repo="$temporary_root/repo"
results="$temporary_root/results"

git clone --quiet --no-local "$repo_root" "$mutated_repo"
git -C "$mutated_repo" checkout --quiet --detach "$base_sha"
git -C "$mutated_repo" apply "$mutation"
git -C "$mutated_repo" config user.email "formal-mutation@example.invalid"
git -C "$mutated_repo" config user.name "Formal Mutation Test"
git -C "$mutated_repo" add server/src/server/prefix_cache_state.h
git -C "$mutated_repo" commit --quiet \
    -m "test: bypass production free-slot selector"

set +e
LUCEBOX_FORMAL_RESULTS="$results" \
    "$mutated_repo/scripts/formal.sh" --base-sha "$base_sha"
verification_status=$?
set -e

if [[ "$verification_status" -ne 10 ]]; then
    if [[ -f "$results/summary.md" ]]; then
        cat "$results/summary.md" >&2
    fi
    echo "expected counterexample exit 10, got $verification_status" >&2
    exit 1
fi

python3 -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
results = {item["id"]: item for item in report["results"]}
if report.get("conclusion") != "counterexample":
    raise SystemExit("mutation did not produce a counterexample conclusion")
if results["prefix-cache-inline"]["status"] != "verified":
    raise SystemExit("unrelated inline contract did not remain verified")
abort = results["prefix-cache-abort-hole"]
if abort["status"] != "counterexample":
    raise SystemExit("abort-hole target did not reject the call-site mutation")
native = abort.get("assumptions", {}).get("native_test", {})
if native.get("status") != "counterexample":
    raise SystemExit("base-approved native regression did not catch the mutation")
' "$results/report.json"

cat "$results/summary.md"
echo "abort-hole call-site mutation sensitivity: PASS"
