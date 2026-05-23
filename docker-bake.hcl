# docker-bake.hcl — Lucebox hub prebuild matrix.
#
# Single CUDA 12 image from one Dockerfile. Additional CUDA stacks are
# intentionally omitted.
#
#   docker buildx bake cuda12-local   # CUDA 12.8.1, tagged lucebox-hub:cuda12
#   docker buildx bake cuda12         # CI target; tags come from metadata-action
#                                 # Arches: sm_75;80;86;89;90;120
#
# Pre-Turing arches (Pascal sm_60/61, Volta sm_70) are intentionally
# excluded — dflash's kernels assume sm_75+ with no fallback below
# (dflash/CMakeLists.txt:276).
#
# The CI `cuda12` target takes tags from docker/metadata-action. The local
# `cuda12-local` target keeps a simple lucebox-hub:cuda12 tag for manual builds.
#
# Override the registry/tag via env: `TAG=v0.1 REGISTRY=ghcr.io/luce-org/ \
#   docker buildx bake cuda12-local`.

variable "REGISTRY" { default = "" }
variable "TAG"      { default = "" }

# Fat-binary CUDA arch list. Defaults to all supported arches so the
# released image runs on every consumer/datacenter GPU we target. Local
# dev builds can narrow this to the host's compute capability to skip the
# 5-6× CUDA template recompile cost:
#
#   DFLASH_CUDA_ARCHES=120 docker buildx bake cuda12-local --load
#
# (RTX 5090 / 5090 Laptop = 120, RTX 4090 = 89, RTX 3090 = 86, H100 = 90,
# A100 = 80, RTX 2080 Ti = 75.) Use a semicolon-separated list to include
# multiple arches.
variable "DFLASH_CUDA_ARCHES" { default = "75;80;86;89;90;120" }

# Image name prefix. With the default empty REGISTRY/TAG you get
# `lucebox-hub:cuda12`.
function "image_tag" {
    params = [variant]
    result = TAG == "" ? "${REGISTRY}lucebox-hub:${variant}" : "${REGISTRY}lucebox-hub:${TAG}-${variant}"
}

group "default" {
    targets = ["cuda12-local"]
}

# CI integration. docker/metadata-action in .github/workflows/docker.yml
# emits a bake-file that defines a `docker-metadata-action` target carrying
# tags + labels derived from the ref. Both build targets inherit from it.
# Local `docker buildx bake` invocations do not provide the metadata-action
# file, so this empty target keeps inheritance valid.
target "docker-metadata-action" {}

# ── CUDA 12.8 ───────────────────────────────────────────────────────────────
# CUDA 12.8 matches the uv-managed PyTorch cu128 stack and carries current-gen
# consumer Blackwell sm_120 coverage. Thor/GB10 variants stay out of this
# build matrix.
target "_cuda12-base" {
    context    = "."
    dockerfile = "Dockerfile"
    args = {
        CUDA_VERSION        = "12.8.1"
        UBUNTU_VERSION      = "22.04"
        DFLASH_CUDA_ARCHES  = DFLASH_CUDA_ARCHES
    }
}

target "cuda12" {
    inherits = ["_cuda12-base", "docker-metadata-action"]
}

target "cuda12-local" {
    inherits = ["_cuda12-base"]
    tags = [image_tag("cuda12")]
}
