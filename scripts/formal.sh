#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
result_dir="${LUCEBOX_FORMAL_RESULTS:-$repo_root/.formal-results}"
base_sha=""
mode="pr"

while (($#)); do
    case "$1" in
        --base-sha)
            base_sha="${2:?--base-sha requires a commit}"
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
        *)
            echo "usage: $0 [--base-sha SHA] [--all|--nightly]" >&2
            exit 2
            ;;
    esac
done

cd "$repo_root"
manifest_image="$(
    python3 -c '
import pathlib
import tomllib
manifest = pathlib.Path("formal/manifest.toml")
print(tomllib.loads(manifest.read_text())["toolchain"]["verifier_image"])
' < /dev/null
)"
verifier_image="${LUCEBOX_FORMAL_IMAGE:-$manifest_image}"

mkdir -p "$result_dir"
result_dir="$(cd "$result_dir" && pwd)"

docker run --rm \
    --network none \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 512 \
    --memory 6g \
    --cpus 2 \
    --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nosuid,nodev,size=512m \
    --volume "$repo_root:/workspace:ro" \
    --volume "$result_dir:/results:rw" \
    --workdir /workspace \
    "$verifier_image" verify \
    --manifest /workspace/formal/manifest.toml \
    --base-sha "$base_sha" \
    --mode "$mode" \
    --out /results
