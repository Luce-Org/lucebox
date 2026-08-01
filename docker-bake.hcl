# docker-bake.hcl — Lucebox hub prebuild matrix.
#
# CUDA 12 covers Turing through Hopper on the broad CUDA-12 driver floor.
# CUDA 12.8 covers x86 Blackwell sm_120, while CUDA 13 provides the native
# arm64/sm_121 image required by GB10/DGX Spark.
#
#   scripts/build_image.sh            # version-derived local build (preferred)
#   docker buildx bake cuda12-local   # raw local build; tagged lucebox-hub:cuda12
#   docker buildx bake cuda12         # CI target; tags come from metadata-action
#                                 # Arches: sm_75;80;86;89;90
#
# Pre-Turing arches (Pascal sm_60/61, Volta sm_70) are intentionally
# excluded — dflash's kernels assume sm_75+ with no fallback below
# (dflash/CMakeLists.txt:276).
#
# The CI `cuda12` target takes tags from docker/metadata-action. The local
# `cuda12-local` target tags `lucebox-hub:cuda12` (moving) and, when
# `VERSION` is set, also tags `lucebox-hub:<version>-cuda12` (pinned).
# `scripts/build_image.sh` is the recommended driver: it computes the
# version from `git describe --tags --match 'lucebox-v*'` so the image
# carries the same git-derived version as the Python packages (hatch-vcs).
#
# Override the registry / version via env: `VERSION=0.3.0 \
#   REGISTRY=ghcr.io/luce-org/ docker buildx bake cuda12-local`.

variable "REGISTRY" { default = "" }
# `VERSION` should be the bare version (e.g. `0.2.7.dev0+gabc1234`) so the
# image tag composes cleanly. Empty means "no pinned tag, just the moving
# variant tag" — keeps `docker buildx bake cuda12-local` working with zero
# config.
variable "VERSION"  { default = "" }
# `TAG` is the legacy override (pre-VERSION). Still honored for back-compat
# but new callers should use `VERSION`.
variable "TAG"      { default = "" }

# Fat-binary CUDA arch list for the broad-driver image. Local
# dev builds can narrow this to the host's compute capability to skip the
# 5-6× CUDA template recompile cost:
#
#   DFLASH_CUDA_ARCHES=86 docker buildx bake cuda12-local --load
#
# (RTX 4090 = 89, RTX 3090 = 86, H100 = 90, A100 = 80, RTX 2080 Ti = 75.)
# Use a semicolon-separated list to include multiple arches.
variable "DFLASH_CUDA_ARCHES" { default = "75;80;86;89;90" }

# RTX 5090 is the only currently qualified x86 sm_120 target. Keeping it in a
# narrow CUDA 12.8 image avoids raising the driver floor for every older GPU.
variable "DFLASH_CUDA128_ARCHES" { default = "120" }

# GB10 is an arm64 Blackwell part with compute capability 12.1. nvcc gained
# sm_121 support after CUDA 12.8, so it intentionally lives in a separate
# CUDA 13 image instead of raising the driver floor for existing x86 users.
variable "DFLASH_CUDA13_ARCHES" { default = "121" }

# Fat-binary HIP/gfx arch list for the rocm variant (semicolon-separated).
# Default is gfx1151 (Strix Halo, the lucebox appliance iGPU) only, to keep the
# build tractable. Widen for a broadly-runnable released image, e.g.:
#   DFLASH_HIP_ARCHES="gfx1151;gfx1100;gfx1200;gfx1201;gfx942;gfx90a" docker buildx bake rocm
# (gfx1151 Strix Halo, gfx1100 RX7900/RDNA3, gfx1200 RDNA4 RX9060,
# gfx1201 RDNA4 RX9070/Radeon AI PRO R9700, gfx942 MI300, gfx90a MI200.)
# Note: gfx1200 and gfx1201 are NOT code-object compatible — the R9700 needs
# gfx1201 explicitly.
variable "DFLASH_HIP_ARCHES" { default = "gfx1151" }

# ROCm base-image tag for the rocm variant. gfx1151 needs >= 6.4.1. Default
# stays 6.4.1 (7.2.x has shown intermittent problems on Strix Halo), but on a
# ROCm 7.x HOST driver the 6.4.x userspace can segfault at model load — set
# ROCM_VERSION=7.2.2 there. Keep the base aligned with the host driver (see
# Dockerfile.rocm).
variable "ROCM_VERSION" { default = "6.4.1" }

# Image identity stamped into /opt/lucebox-hub/IMAGE_INFO at build time and
# surfaced under /props.build at runtime (git_sha, image_tag, build_time).
# CI sets all three from the workflow context; local builds get a best-
# effort `git rev-parse` for GIT_SHA + empty IMAGE_TAG/BUILD_TIME (those
# come from CI metadata-action and the workflow timestamp, neither of
# which is available offline). Empty values turn into JSON null at /props.
variable "GIT_SHA"    { default = "" }
variable "IMAGE_TAG"  { default = "" }
variable "BUILD_TIME" { default = "" }

# Image tag list. Default (no VERSION / no TAG) emits just the moving
# `lucebox-hub:cuda12`. With VERSION set we also emit a pinned
# `lucebox-hub:<version>-cuda12`. Both point at the same image so users
# can pull either form. TAG (legacy) still works as a single-tag override.
#
# Docker tag charset is [A-Za-z0-9_.-], so PEP 440 local-version segments
# (e.g. `0.2.7.dev0+gabc1234` from hatch-vcs on a post-tag dev commit)
# need their `+` replaced before they can be used as a tag. We map `+` →
# `-` so the pinned tag becomes e.g. `0.2.7.dev0-gabc1234-cuda12`.
sanitized_version = regex_replace(VERSION, "\\+", "-")
function "image_tags" {
    params = [variant]
    result = TAG != "" ? ["${REGISTRY}lucebox-hub:${TAG}-${variant}"] : (VERSION != "" ? ["${REGISTRY}lucebox-hub:${variant}", "${REGISTRY}lucebox-hub:${sanitized_version}-${variant}"] : ["${REGISTRY}lucebox-hub:${variant}"])
}

group "default" {
    targets = ["cuda12-local"]
}

# Build every published variant locally (CUDA 12, CUDA 12.8, CUDA 13, and
# ROCm). CI builds these as a matrix; this group is the local equivalent.
group "all" {
    targets = ["cuda12-local", "cuda128-local", "cuda13-local", "rocm-local"]
}

# CI integration. docker/metadata-action in .github/workflows/docker.yml
# emits a bake-file that defines a `docker-metadata-action` target carrying
# tags + labels derived from the ref. Both build targets inherit from it.
# Local `docker buildx bake` invocations do not provide the metadata-action
# file, so this empty target keeps inheritance valid.
target "docker-metadata-action" {}

# ── CUDA 12 (broad driver compatibility) ───────────────────────────────────
# CUDA 12.0 is the oldest toolkit that builds every qualified pre-Blackwell
# architecture and runs on the CUDA-12 minor-compatibility driver floor
# (r525+). This is the default for RTX 20/30/40, A-series, A100, and H100.
target "_cuda12-base" {
    context    = "."
    dockerfile = "Dockerfile"
    args = {
        CUDA_VERSION        = "12.0.1"
        UBUNTU_VERSION      = "22.04"
        DFLASH_CUDA_ARCHES  = DFLASH_CUDA_ARCHES
        # /props.build identity. CI passes these as env vars from the
        # workflow context; local builds rely on the variables' defaults
        # (empty strings → JSON null at /props.build.*).
        GIT_SHA             = GIT_SHA
        IMAGE_TAG           = IMAGE_TAG
        BUILD_TIME          = BUILD_TIME
    }
}

target "cuda12" {
    inherits = ["_cuda12-base", "docker-metadata-action"]
}

target "cuda12-local" {
    inherits = ["_cuda12-base"]
    tags = image_tags("cuda12")
}

# ── CUDA 12.8 / x86 Blackwell ──────────────────────────────────────────────
# sm_120 first appeared in CUDA 12.8. It has a newer driver floor, so it must
# not share the image selected for older NVIDIA cards.
target "_cuda128-base" {
    context    = "."
    dockerfile = "Dockerfile"
    args = {
        CUDA_VERSION        = "12.8.1"
        UBUNTU_VERSION      = "22.04"
        DFLASH_CUDA_ARCHES  = DFLASH_CUDA128_ARCHES
        GIT_SHA             = GIT_SHA
        IMAGE_TAG           = IMAGE_TAG
        BUILD_TIME          = BUILD_TIME
    }
}

target "cuda128" {
    inherits = ["_cuda128-base", "docker-metadata-action"]
}

target "cuda128-local" {
    inherits = ["_cuda128-base"]
    tags = image_tags("cuda128")
}

# ── CUDA 13 / GB10 arm64 ──────────────────────────────────────────────────
target "_cuda13-base" {
    context    = "."
    dockerfile = "Dockerfile"
    platforms  = ["linux/arm64"]
    args = {
        CUDA_VERSION        = "13.0.1"
        UBUNTU_VERSION      = "24.04"
        DFLASH_CUDA_ARCHES  = DFLASH_CUDA13_ARCHES
        GIT_SHA             = GIT_SHA
        IMAGE_TAG           = IMAGE_TAG
        BUILD_TIME          = BUILD_TIME
    }
}

target "cuda13" {
    inherits = ["_cuda13-base", "docker-metadata-action"]
}

target "cuda13-local" {
    inherits = ["_cuda13-base"]
    tags = image_tags("cuda13")
}

# ── ROCm / HIP ───────────────────────────────────────────────────────────────
# AMD GPU build from Dockerfile.rocm: gfx1151 (Strix Halo) by default, widen via
# DFLASH_HIP_ARCHES for a broadly-runnable image. Block-Sparse-Attention is
# CUDA-only and disabled in this variant (see Dockerfile.rocm).
target "_rocm-base" {
    context    = "."
    dockerfile = "Dockerfile.rocm"
    args = {
        ROCM_VERSION      = ROCM_VERSION
        UBUNTU_VERSION    = "22.04"
        DFLASH_HIP_ARCHES = DFLASH_HIP_ARCHES
        GIT_SHA           = GIT_SHA
        IMAGE_TAG         = IMAGE_TAG
        BUILD_TIME        = BUILD_TIME
    }
}

target "rocm" {
    inherits = ["_rocm-base", "docker-metadata-action"]
}

target "rocm-local" {
    inherits = ["_rocm-base"]
    tags = image_tags("rocm")
}
