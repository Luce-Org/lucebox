# syntax=docker/dockerfile:1.7

# ─── Stage 1: builder ───────────────────────────────────────────────────────
# CUDA_VERSION / UBUNTU_VERSION / DFLASH_CUDA_ARCHES are build args so the
# same Dockerfile can be repinned later. The prebuilt image is the
# CUDA 12.8 path:
#   • lucebox-hub:cuda12  — CUDA 12.8.1, sm_75;80;86;89;90;120
# See docker-bake.hcl for the canonical invocation.
ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Fat-binary CUDA arch list, semicolon-separated. Defaults cover the CUDA 12.8
# image. dflash-supported arches in this image:
#   75  Turing      RTX 2080 Ti
#   80  Ampere      A100
#   86  Ampere      RTX 3090, A40, A10
#   89  Ada         RTX 4090, L40
#   90  Hopper      H100
#   120 Blackwell   RTX 5090, RTX 5090 Laptop
# Thor and GB10 prebuilt-image coverage is intentionally omitted.
# Pre-Turing arches (sm_60/61/70/72) are intentionally excluded — dflash's
# BF16/WMMA paths have no fallback below sm_75. Each arch adds ~50-200 MB
# of fat-binary kernel code and ~3-5 min of nvcc time per .cu translation
# unit.
ARG DFLASH_CUDA_ARCHES="75;80;86;89;90;120"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        ninja-build \
        pkg-config \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# CUDA driver stub. nvidia/cuda:*-devel images ship the driver stub at
# /usr/local/cuda/lib64/stubs/libcuda.so but not as libcuda.so.1. ld follows
# the NEEDED reference inside libggml-cuda.so by SONAME (libcuda.so.1) when
# linking executables, so without this symlink + ld.so.conf entry the
# test_dflash link step fails with `undefined reference to cuMem*`.
# At runtime the host driver provides the real libcuda.so.1 via
# --gpus all; the stub is only for build-time symbol resolution.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

WORKDIR /src

# COPY ordering is structured to keep the CUDA build cached across
# Python-only edits. The cmake build only depends on dflash/{CMakeLists,
# include, src, test, hip_compat, deps}. Everything else (Python scripts,
# workspace pyproject manifests, lockfile, READMEs) is copied later so
# editing server.py / bench_*.py / lucebox sources doesn't invalidate the
# ~25-minute CUDA template-instantiation layer below.

# C++ build inputs only — sources, headers, submodules, build script.
# Note: upstream rename (PR #281) moved dflash/ → server/. Source layout
# uses server/; submodule binding names still write `dflash/deps/...`
# inside .gitmodules (arbitrary identifiers; only paths matter).
COPY server/CMakeLists.txt /src/server/CMakeLists.txt
COPY server/include /src/server/include
COPY server/src /src/server/src
COPY server/test /src/server/test
COPY server/hip_compat /src/server/hip_compat
COPY server/deps /src/server/deps

# Submodules (`server/deps/llama.cpp`, `server/deps/Block-Sparse-Attention`)
# must be populated on the host before `docker build` — `.git/` is excluded
# by .dockerignore so we cannot re-fetch them inside the image. ggml's own
# CMakeLists also asserts this and errors with the right command if missing,
# but failing here gives a clearer message before nvcc spins up.
RUN test -f /src/server/deps/llama.cpp/ggml/CMakeLists.txt \
    || (echo "ERROR: server/deps/llama.cpp submodule not initialised. Run on host:" >&2 \
        && echo "       git submodule update --init --recursive" >&2 \
        && exit 1)

# Configure + build. `DFLASH27B_USER_CUDA_ARCHITECTURES` pins the arch list
# through dflash's own logic (skips its auto-extend rules that depend on
# nvcc version inspection); `CMAKE_CUDA_ARCHITECTURES` also gets set so the
# vendored ggml-cuda subproject picks up the same list.
# CMAKE_BUILD_WITH_INSTALL_RPATH=ON embeds CMakeLists.txt's $ORIGIN-relative
# CMAKE_INSTALL_RPATH (`$ORIGIN/deps/llama.cpp/ggml/src`, etc.) into the
# binary at link time, instead of the default absolute build-tree paths.
# Without this the binary loses its ggml shared libs after COPY to the
# runtime stage (`libggml.so.0: cannot open shared object file`).
RUN cmake -S /src/server -B /src/server/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DDFLASH27B_USER_CUDA_ARCHITECTURES="${DFLASH_CUDA_ARCHES}" \
        -DCMAKE_CUDA_ARCHITECTURES="${DFLASH_CUDA_ARCHES}" \
    && cmake --build /src/server/build --target test_dflash dflash_server --parallel

# Prune the build tree to only what the runtime stage needs: the native server,
# test_dflash, and the ggml shared libs their embedded rpath
# ($ORIGIN/deps/...) looks up. Drops ~1 GB per image of CMakeFiles/,
# libdflash27b.a (statically linked into the binaries), ninja state,
# compile_commands.json, and the template-instance .o tree from ggml-cuda.
RUN cd /src/server/build \
    && find . -mindepth 1 -maxdepth 1 \
            ! -name test_dflash ! -name dflash_server ! -name deps -exec rm -rf {} + \
    && find deps -mindepth 1 -type f ! -name 'lib*.so*' -delete \
    && find deps -depth -type d -empty -delete

# Python sources, workspace manifests, lockfile, READMEs — everything the
# runtime stage needs to COPY but the cmake build does not. Editing any
# of these reuses the cached CUDA layers above and only re-runs the
# runtime stage's uv sync (~70s) instead of the full ~25-minute build.
COPY pyproject.toml uv.lock README.md /src/
COPY server/pyproject.toml server/README.md /src/server/
COPY server/scripts /src/server/scripts
COPY lucebox /src/lucebox
COPY luce-bench /src/luce-bench
COPY harness /src/harness
COPY optimizations/pflash /src/optimizations/pflash
COPY optimizations/megakernel /src/optimizations/megakernel

# ─── Stage 2: runtime ───────────────────────────────────────────────────────
# Runtime image: ships nvidia driver libs but no nvcc / dev headers. Matches
# the builder's CUDA version so the test_dflash binary's libcudart SONAME
# resolves at runtime against the same major.minor.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Image identity baked in at build time and read by dflash_server at startup
# to populate /props.build (git_sha / image_tag / build_time). All three are
# wired from docker-bake.hcl, which sources them from CI metadata or local
# `git`. Missing args leave the corresponding fields empty in IMAGE_INFO,
# which dflash_server surfaces as JSON null at /props.build.* — that's the
# expected behavior on a `docker build` run without bake.
ARG GIT_SHA=""
ARG IMAGE_TAG=""
ARG BUILD_TIME=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        docker.io \
        libgomp1 \
        pciutils \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python 3.12 (required by the workspace) and resolves the
# lucebox-dflash + pflash members declared in pyproject.toml.
RUN curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh

# Install Python to a world-readable location, not /root/.local/share/uv/
# (the default). The container runs as the host UID for bind-mount sanity
# (so config.toml files in $HOME are user-owned, not root-owned), and a
# non-root UID cannot traverse into root's home to exec python. Same for
# the uv cache — must be world-readable so non-root reads from it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_TOOL_DIR=/opt/uv/tools

WORKDIR /opt/lucebox-hub

# Workspace files for uv sync (root pyproject + lock + README + workspace
# member manifests). Each is a leaf file or small dir so layers stay tiny.
# The in-container entrypoint lives at server/scripts/entrypoint.sh and
# dispatches to either the dflash server, the lucebox Python CLI, or the
# benchmark. The host-side `lucebox.sh` is the supported way to drive this
# image; the Python CLI inside owns all orchestration logic.
COPY --from=builder /src/pyproject.toml /src/uv.lock /src/README.md /opt/lucebox-hub/
COPY --from=builder /src/optimizations/pflash /opt/lucebox-hub/optimizations/pflash
COPY --from=builder /src/optimizations/megakernel/pyproject.toml \
                   /src/optimizations/megakernel/README.md \
                   /opt/lucebox-hub/optimizations/megakernel/
# The lucebox Python CLI ships in /opt/lucebox-hub/lucebox/ as a uv workspace
# member; entrypoint.sh execs `python -m lucebox` for any host-wrapper
# subcommand other than `serve` / `benchmark` / `shell`.
COPY --from=builder /src/lucebox /opt/lucebox-hub/lucebox

# luce-bench is the standalone benchmark harness, in-tree as a workspace
# member (see 490ff95 "absorb luce-bench into the monorepo"). Required so
# the runtime stage's `uv sync` can resolve `luce-bench = { workspace = true }`
# in the root pyproject.toml without falling over.
COPY --from=builder /src/luce-bench /opt/lucebox-hub/luce-bench

# harness is the "run X against a Lucebox server" abstraction — workspace
# member at harness/ that profile.py imports for `python -m harness.bench`.
# Required so the runtime stage's `uv sync` can resolve the workspace dep.
COPY --from=builder /src/harness /opt/lucebox-hub/harness

# server: ship the entrypoint/benchmark scripts, the pyproject + README that uv
# resolves against, and the pruned build tree (binaries + .so files from the
# prune step in the builder stage). Source code, headers, tests, and submodule
# sources stay in the builder.
COPY --from=builder /src/server/scripts /opt/lucebox-hub/server/scripts
COPY --from=builder /src/server/pyproject.toml /src/server/README.md \
                   /opt/lucebox-hub/server/
COPY --from=builder /src/server/build /opt/lucebox-hub/server/build

# Model-card sidecars resolved at startup. The server's search path
# (model_card.cpp) looks at <binary>/../share/model_cards first, so
# placing them at /opt/lucebox-hub/server/share/model_cards/ makes
# them discoverable without DFLASH_MODEL_CARDS_DIR. Copied directly
# from the build context (no builder roundtrip needed — these are
# static JSON, not compiled).
COPY share/model_cards /opt/lucebox-hub/server/share/model_cards

RUN test -x /opt/lucebox-hub/server/build/test_dflash \
    && test -x /opt/lucebox-hub/server/build/dflash_server \
    && test -f /opt/lucebox-hub/server/share/model_cards/qwen3.6-27b.json \
    && chmod +x /opt/lucebox-hub/server/scripts/entrypoint.sh

# Image identity for /props.build. dflash_server reads this file at startup
# (path: /opt/lucebox-hub/IMAGE_INFO, three lines: git_sha, image_tag,
# build_time). Override the path with $DFLASH_IMAGE_INFO_PATH for tests.
# All three args may be empty in non-bake builds — the empty lines that
# results in are detected at read time and surface as JSON null in /props.
RUN printf '%s\n%s\n%s\n' "$GIT_SHA" "$IMAGE_TAG" "$BUILD_TIME" \
        > /opt/lucebox-hub/IMAGE_INFO

# Register the ggml lib dir with ld.so so libggml-cpu.so (loaded transitively
# by libggml.so) resolves. CMakeLists.txt sets a `$ORIGIN/deps/...` RUNPATH
# uniformly across all linked artefacts — correct for test_dflash in
# server/build/, broken for the .so files in deps/llama.cpp/ggml/src/ which
# would need a plain `$ORIGIN`. ld.so.conf side-steps the RPATH bug without
# patching every shared lib.
RUN printf '%s\n%s\n' \
        /opt/lucebox-hub/server/build/deps/llama.cpp/ggml/src \
        /opt/lucebox-hub/server/build/deps/llama.cpp/ggml/src/ggml-cuda \
        > /etc/ld.so.conf.d/lucebox-ggml.conf \
    && ldconfig

# Resolve Python deps for the lucebox CLI and remaining Python benchmark harness.
# Megakernel is an optional extra and is intentionally skipped — its CUDA
# extension would require nvcc + matching torch headers in this stage.
# `--no-cache` keeps wheels from being persisted in the layer; hardlink mode
# means the venv files live alongside the cache during the install but the
# cache is gone by the time the layer commits, so we don't double-pay.
ENV UV_LINK_MODE=hardlink \
    UV_NO_CACHE=1
# --no-editable: install workspace members (luce-bench, lucebox, harness,
# pflash, lucebox-hub) as proper wheels rather than source-linked editable
# installs. Without this, hatch-vcs's build hook re-fires at runtime when
# `uv run` re-checks env consistency and tries to write `_version.py` into
# the root-owned workspace source dirs, which fails as a non-root user.
# With non-editable wheels the venv is self-contained and the build hook
# only runs once, here, with root.
RUN uv sync --no-dev --frozen --no-editable 2>/dev/null \
    || uv sync --no-dev --no-editable

# Host wrapper CLI containers run as the invoking host uid so bind-mounted
# config/profile files are not left root-owned. Keep the uv-managed
# interpreter, the python install, and the workspace readable/executable
# for that non-root uid. UV_PYTHON_INSTALL_DIR redirects the python
# install to /opt/uv/python (set as ENV before the sync above); we still
# chmod the venv + workspace + uv-install dir so the non-root user can
# reach interpreter, scripts, and the writable directories the runtime
# might touch.
RUN chmod -R a+rX /opt/lucebox-hub/.venv /opt/lucebox-hub /opt/uv

# Models live in server/models/ — bind-mount or volume them in.
# Example:
#   docker run --rm --gpus all -p 8080:8080 \
#       -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
#       lucebox-hub
# The VOLUME declaration keeps the path out of the image layer cache; the
# bind mount above replaces it with the host directory at run time.
VOLUME ["/opt/lucebox-hub/server/models"]

ENV DFLASH_HOST=0.0.0.0 \
    DFLASH_PORT=8080 \
    DFLASH_BIN=/opt/lucebox-hub/server/build/test_dflash \
    DFLASH_SERVER_BIN=/opt/lucebox-hub/server/build/dflash_server

EXPOSE 8080

ENTRYPOINT ["/opt/lucebox-hub/server/scripts/entrypoint.sh"]
