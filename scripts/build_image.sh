#!/usr/bin/env bash
# scripts/build_image.sh — Build the cuda12 lucebox-hub image with a
# git-derived version tag matching the Python packages' hatch-vcs scheme.
#
# Tagging:
#   * Untagged tree    → lucebox-hub:cuda12 (moving)
#   * Tagged `lucebox-v0.3.0` (clean checkout):
#                      → lucebox-hub:cuda12
#                      → lucebox-hub:0.3.0-cuda12
#   * Post-tag dev commit / dirty:
#                      → lucebox-hub:cuda12
#                      → lucebox-hub:0.3.0-3-gabc1234-cuda12   (-dirty suffix
#                        appended when working tree is dirty)
#
# Single source of truth is the `lucebox-v*` git tag — same regex hatch-vcs
# uses for the Python wheel version, so the image version always matches.
#
# Usage:
#   scripts/build_image.sh                # build, no load (cache-fill)
#   scripts/build_image.sh --load         # load into local docker
#   scripts/build_image.sh --push         # push to $REGISTRY (must end in /)
#
# Env overrides:
#   REGISTRY=ghcr.io/luce-org/   override image registry prefix
#   DFLASH_CUDA_ARCHES=120       narrow arch list (skip the multi-arch compile)

set -euo pipefail

# Resolve to the repo root: prefer git's view (works from any subdir), else
# fall back to one level up from this script (scripts/build_image.sh →
# repo root).
cd "$(git rev-parse --show-toplevel 2>/dev/null || (cd "$(dirname "$0")/.." && pwd))"

# Derive the version. `git describe --match 'lucebox-v*'` finds the nearest
# matching annotated tag and appends commits-past + sha if not on the tag
# itself. `--always` keeps it from failing on a brand-new clone with no
# tags. `--dirty` appends `-dirty` when the working tree differs from HEAD.
raw=$(git describe --tags --match 'lucebox-v*' --always --dirty 2>/dev/null || true)
# Strip the `lucebox-v` prefix when present (clean releases). On a fresh
# tree with no matching tags, `git describe --always` returns the short
# sha — keep it as-is so the image still gets a sensible pinned tag.
VERSION="${raw#lucebox-v}"

if [ -z "$VERSION" ]; then
    echo "[build_image] no git description available — tagging :cuda12 only"
else
    echo "[build_image] version=$VERSION (from git describe)"
fi

export VERSION
# `REGISTRY` composes into the bake tag as `${REGISTRY}lucebox-hub:…`, so
# the value MUST end with `/` (e.g. `ghcr.io/luce-org/`). Normalize a
# trailing slash so callers that forget it (`REGISTRY=ghcr.io/luce-org`)
# get the right image name rather than `ghcr.io/luce-orglucebox-hub:…`.
REGISTRY="${REGISTRY:-}"
if [ -n "$REGISTRY" ] && [ "${REGISTRY: -1}" != "/" ]; then
    REGISTRY="${REGISTRY}/"
fi
export REGISTRY
export DFLASH_CUDA_ARCHES="${DFLASH_CUDA_ARCHES:-75;80;86;89;90;120}"

# /props.build identity (baked into /opt/lucebox-hub/IMAGE_INFO and surfaced
# at /props.build). All best-effort for local builds:
#   * GIT_SHA   — `git rev-parse HEAD` (full sha; empty on a non-git tree)
#   * BUILD_TIME — UTC ISO 8601 timestamp
#   * IMAGE_TAG  — falls back to "cuda12" (the moving tag) unless caller
#     pre-exported one. CI replaces this with the metadata-action version.
export GIT_SHA="${GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || true)}"
export BUILD_TIME="${BUILD_TIME:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
export IMAGE_TAG="${IMAGE_TAG:-${VERSION:+${VERSION}-}cuda12}"

exec docker buildx bake cuda12-local "$@"
