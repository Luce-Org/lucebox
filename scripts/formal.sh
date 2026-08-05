#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
result_dir="${LUCEBOX_FORMAL_RESULTS:-$repo_root/.formal-results}"
base_sha=""
mode="all"
legacy=false

while (($#)); do
    case "$1" in
        --base-sha)
            base_sha="${2:?--base-sha requires a commit}"
            mode="pr"
            shift 2
            ;;
        --all)
            mode="all"
            shift
            ;;
        --nightly)
            mode="nightly"
            shift
            ;;
        --legacy)
            legacy=true
            shift
            ;;
        *)
            echo "usage: $0 [--base-sha SHA|--all|--nightly] [--legacy]" >&2
            exit 2
            ;;
    esac
done

cd "$repo_root"
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
    echo "formal planning requires a committed production checkout" >&2
    exit 2
fi

registry_image="$(
    python3 -c '
import pathlib
import tomllib
registry = pathlib.Path("formal/contracts/registry.toml")
print(tomllib.loads(registry.read_text())["toolchain"]["verifier_image"])
' < /dev/null
)"
verifier_image="${LUCEBOX_FORMAL_IMAGE:-$registry_image}"

mkdir -p "$result_dir"
result_dir="$(cd "$result_dir" && pwd)"

container_common=(
    --rm
    --network none
    --read-only
    --cap-drop ALL
    --security-opt no-new-privileges
    --pids-limit 512
    --memory 6g
    --cpus 2
    --user "$(id -u):$(id -g)"
    # The plan verifier compiles the immutable base-native regression here and
    # executes it against the read-only head workspace.
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=512m
    --volume "$repo_root:/workspace:ro"
    --workdir /workspace
)

if [[ "$legacy" == true ]]; then
    docker run \
        "${container_common[@]}" \
        --volume "$result_dir:/results:rw" \
        "$verifier_image" verify \
        --manifest /workspace/formal/manifest.toml \
        --base-sha "$base_sha" \
        --mode "$mode" \
        --out /results
    exit
fi

head_sha="$(git rev-parse HEAD)"
if [[ -n "$base_sha" ]]; then
    policy_sha="$(git rev-parse "${base_sha}^{commit}")"
else
    policy_sha="$head_sha"
fi
plan_dir="$(mktemp -d)"
trap 'rm -rf -- "$plan_dir"' EXIT

docker run \
    "${container_common[@]}" \
    --volume "$plan_dir:/plan:rw" \
    "$verifier_image" plan \
    --workspace /workspace \
    --base-policy formal/contracts/registry.toml \
    --base-sha "$policy_sha" \
    --head-sha "$head_sha" \
    --mode "$mode" \
    --out /plan

docker run \
    "${container_common[@]}" \
    --volume "$plan_dir:/plan:ro" \
    --volume "$result_dir:/results:rw" \
    "$verifier_image" verify \
    --plan /plan/plan.json \
    --workspace /workspace \
    --generated-root /plan \
    --out /results
