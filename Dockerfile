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
COPY dflash/CMakeLists.txt /src/dflash/CMakeLists.txt
COPY dflash/include /src/dflash/include
COPY dflash/src /src/dflash/src
COPY dflash/test /src/dflash/test
COPY dflash/hip_compat /src/dflash/hip_compat
COPY dflash/deps /src/dflash/deps

# Submodules (`dflash/deps/llama.cpp`, `dflash/deps/Block-Sparse-Attention`)
# must be populated on the host before `docker build` — `.git/` is excluded
# by .dockerignore so we cannot re-fetch them inside the image. ggml's own
# CMakeLists also asserts this and errors with the right command if missing,
# but failing here gives a clearer message before nvcc spins up.
RUN test -f /src/dflash/deps/llama.cpp/ggml/CMakeLists.txt \
    || (echo "ERROR: dflash/deps/llama.cpp submodule not initialised. Run on host:" >&2 \
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
RUN cmake -S /src/dflash -B /src/dflash/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DDFLASH27B_USER_CUDA_ARCHITECTURES="${DFLASH_CUDA_ARCHES}" \
        -DCMAKE_CUDA_ARCHITECTURES="${DFLASH_CUDA_ARCHES}" \
    && cmake --build /src/dflash/build --target test_dflash dflash_server --parallel

# Prune the build tree to only what the runtime stage needs: the native server,
# test_dflash, and the ggml shared libs their embedded rpath
# ($ORIGIN/deps/...) looks up. Drops ~1 GB per image of CMakeFiles/,
# libdflash27b.a (statically linked into the binaries), ninja state,
# compile_commands.json, and the template-instance .o tree from ggml-cuda.
RUN cd /src/dflash/build \
    && find . -mindepth 1 -maxdepth 1 \
            ! -name test_dflash ! -name dflash_server ! -name deps -exec rm -rf {} + \
    && find deps -mindepth 1 -type f ! -name 'lib*.so*' -delete \
    && find deps -depth -type d -empty -delete

# Python sources, workspace manifests, lockfile, READMEs — everything the
# runtime stage needs to COPY but the cmake build does not. Editing any
# of these reuses the cached CUDA layers above and only re-runs the
# runtime stage's uv sync (~70s) instead of the full ~25-minute build.
COPY pyproject.toml uv.lock README.md /src/
COPY dflash/pyproject.toml dflash/README.md /src/dflash/
COPY dflash/scripts /src/dflash/scripts
COPY lucebox /src/lucebox
COPY pflash /src/pflash
COPY megakernel /src/megakernel

# ─── Stage 2: runtime ───────────────────────────────────────────────────────
# Runtime image: ships nvidia driver libs but no nvcc / dev headers. Matches
# the builder's CUDA version so the test_dflash binary's libcudart SONAME
# resolves at runtime against the same major.minor.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

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

WORKDIR /opt/lucebox-hub

# Workspace files for uv sync (root pyproject + lock + README + workspace
# member manifests). Each is a leaf file or small dir so layers stay tiny.
# The in-container entrypoint lives at dflash/scripts/entrypoint.sh and
# dispatches to either the dflash server, the lucebox Python CLI, or the
# benchmark. The host-side `lucebox.sh` is the supported way to drive this
# image; the Python CLI inside owns all orchestration logic.
COPY --from=builder /src/pyproject.toml /src/uv.lock /src/README.md /opt/lucebox-hub/
COPY --from=builder /src/pflash /opt/lucebox-hub/pflash
COPY --from=builder /src/megakernel/pyproject.toml /src/megakernel/README.md \
                   /opt/lucebox-hub/megakernel/
# The lucebox Python CLI ships in /opt/lucebox-hub/lucebox/ as a uv workspace
# member; entrypoint.sh execs `python -m lucebox` for any host-wrapper
# subcommand other than `serve` / `benchmark` / `shell`.
COPY --from=builder /src/lucebox /opt/lucebox-hub/lucebox

# dflash: ship the entrypoint/benchmark scripts, the pyproject + README that uv
# resolves against, and the pruned build tree (binaries + .so files from the
# prune step in the builder stage). Source code, headers, tests, and submodule
# sources stay in the builder.
COPY --from=builder /src/dflash/scripts /opt/lucebox-hub/dflash/scripts
COPY --from=builder /src/dflash/pyproject.toml /src/dflash/README.md \
                   /opt/lucebox-hub/dflash/
COPY --from=builder /src/dflash/build /opt/lucebox-hub/dflash/build

# Model-card sidecars resolved at startup. The server's search path
# (model_card.cpp) looks at <binary>/../share/model_cards first, so
# placing them at /opt/lucebox-hub/dflash/share/model_cards/ makes
# them discoverable without DFLASH_MODEL_CARDS_DIR. Copied directly
# from the build context (no builder roundtrip needed — these are
# static JSON, not compiled).
COPY share/model_cards /opt/lucebox-hub/dflash/share/model_cards

RUN test -x /opt/lucebox-hub/dflash/build/test_dflash \
    && test -x /opt/lucebox-hub/dflash/build/dflash_server \
    && test -f /opt/lucebox-hub/dflash/share/model_cards/qwen3.6-27b.json \
    && chmod +x /opt/lucebox-hub/dflash/scripts/entrypoint.sh

# Register the ggml lib dir with ld.so so libggml-cpu.so (loaded transitively
# by libggml.so) resolves. CMakeLists.txt sets a `$ORIGIN/deps/...` RUNPATH
# uniformly across all linked artefacts — correct for test_dflash in
# dflash/build/, broken for the .so files in deps/llama.cpp/ggml/src/ which
# would need a plain `$ORIGIN`. ld.so.conf side-steps the RPATH bug without
# patching every shared lib.
RUN printf '%s\n%s\n' \
        /opt/lucebox-hub/dflash/build/deps/llama.cpp/ggml/src \
        /opt/lucebox-hub/dflash/build/deps/llama.cpp/ggml/src/ggml-cuda \
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
RUN uv sync --no-dev --frozen 2>/dev/null \
    || uv sync --no-dev

# Host wrapper CLI containers run as the invoking host uid so bind-mounted
# config/profile files are not left root-owned. Keep the uv-managed
# interpreter and workspace readable/executable for that non-root uid.
RUN chmod -R a+rX /opt/lucebox-hub/.venv /opt/lucebox-hub

# Models live in dflash/models/ — bind-mount or volume them in.
# Example:
#   docker run --rm --gpus all -p 8080:8080 \
#       -v "$PWD/dflash/models:/opt/lucebox-hub/dflash/models" \
#       lucebox-hub
# The VOLUME declaration keeps the path out of the image layer cache; the
# bind mount above replaces it with the host directory at run time.
VOLUME ["/opt/lucebox-hub/dflash/models"]

ENV DFLASH_HOST=0.0.0.0 \
    DFLASH_PORT=8080 \
    DFLASH_BIN=/opt/lucebox-hub/dflash/build/test_dflash \
    DFLASH_SERVER_BIN=/opt/lucebox-hub/dflash/build/dflash_server

EXPOSE 8080

ENTRYPOINT ["/opt/lucebox-hub/dflash/scripts/entrypoint.sh"]
