#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

# id|patch|mutated production path|expected target|base-native source|control target
mutation_cases=(
    "prefix-cache-abort-hole|prefix-cache-bypass-selector.patch|server/src/server/prefix_cache_state.h|prefix-cache-abort-hole|server/test/test_prefix_cache_state.cpp|prefix-cache-inline"
    "prefix-cache-full-lifecycle|prefix-cache-full-bypass-free-slot.patch|server/src/server/prefix_cache_state.h|prefix-cache-full-lifecycle|server/test/test_full_prefix_cache_state.cpp|prefix-cache-inline"
    "spec-commit-exactness|spec-commit-invert-prefix-match.patch|server/src/common/spec_commit.h|spec-commit-exactness|server/test/test_spec_commit.cpp|"
    "kvflash-residency-map|kvflash-capacity-off-by-one.patch|server/src/common/kvflash_residency_map.h|kvflash-residency-map|server/test/test_kvflash_residency_map.cpp|"
)

cd "$repo_root"
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
    echo "mutation sensitivity requires a checkout with no tracked changes" >&2
    exit 2
fi

base_sha="$(git rev-parse HEAD)"
temporary_root="$(mktemp -d)"
trap 'rm -rf -- "$temporary_root"' EXIT

for mutation_case in "${mutation_cases[@]}"; do
    IFS='|' read -r case_id patch_name mutable_path target_id native_source control_id \
        <<< "$mutation_case"
    mutated_repo="$temporary_root/$case_id/repo"
    results="$temporary_root/$case_id/results"
    mutation="$repo_root/formal/contracts/mutations/$patch_name"
    mkdir -p "$(dirname "$mutated_repo")"

    git clone --quiet --no-local "$repo_root" "$mutated_repo"
    git -C "$mutated_repo" checkout --quiet --detach "$base_sha"
    if ! git -C "$mutated_repo" apply --check "$mutation"; then
        echo "$case_id: mutation patch no longer applies to exact HEAD" >&2
        exit 1
    fi
    git -C "$mutated_repo" apply "$mutation"
    git -C "$mutated_repo" config user.email "formal-mutation@example.invalid"
    git -C "$mutated_repo" config user.name "Formal Mutation Test"
    git -C "$mutated_repo" add "$mutable_path"
    git -C "$mutated_repo" commit --quiet \
        -m "test: inject $case_id mutation"

    # First prove that the immutable native regression is behavior-sensitive,
    # independently of whether ESBMC finds the same mutation first.
    native_binary="$temporary_root/$case_id/native-test"
    c++ -std=c++17 -O0 -Wall -Wextra -Werror \
        -I "$mutated_repo/server/src" \
        "$mutated_repo/$native_source" \
        -o "$native_binary"
    set +e
    "$native_binary"
    native_status=$?
    set -e
    if [[ "$native_status" -eq 0 ]]; then
        echo "$case_id: native regression survived mutation" >&2
        exit 1
    fi

    set +e
    LUCEBOX_FORMAL_RESULTS="$results" \
        "$mutated_repo/scripts/formal.sh" --base-sha "$base_sha"
    verification_status=$?
    set -e

    if [[ "$verification_status" -ne 10 ]]; then
        if [[ -f "$results/summary.md" ]]; then
            cat "$results/summary.md" >&2
        fi
        echo "$case_id: expected counterexample exit 10, got $verification_status" >&2
        exit 1
    fi

    python3 - "$results/report.json" "$target_id" "$control_id" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
target_id = sys.argv[2]
control_id = sys.argv[3]
results = {item["id"]: item for item in report["results"]}
if report.get("conclusion") != "counterexample":
    raise SystemExit("mutation did not produce a counterexample conclusion")
if results[target_id]["status"] != "counterexample":
    raise SystemExit(f"{target_id} did not reject its mutation")
if control_id and results[control_id]["status"] != "verified":
    raise SystemExit(f"unrelated control {control_id} did not remain verified")
native = results[target_id].get("assumptions", {}).get("native_test")
if native is not None and native.get("status") != "counterexample":
    raise SystemExit(f"{target_id} native result was not a counterexample")
PY

    cat "$results/summary.md"
    echo "$case_id mutation sensitivity: PASS"
done

echo "all formal mutation cases: PASS"
