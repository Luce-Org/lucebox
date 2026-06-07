# Copilot Instructions — Lucebox Hub

## What is this repo

Local LLM inference engine with hand-tuned CUDA/HIP kernels for specific consumer GPUs. Speculative decoding, speculative prefill, and fused megakernels. Reference hardware: RTX 3090 (sm_86).

### Components

- **`server/`** — DFlash: C++/CUDA speculative-decoding server. OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`). Built with CMake on top of vendored ggml (`server/deps/llama.cpp` submodule) — no PyTorch or libllama at runtime. Supports multiple model architectures dispatched at startup via `general.architecture` in the GGUF (qwen35, qwen36, laguna, gemma4).
- **`optimizations/megakernel/`** — Fused 24-layer CUDA megakernel for Qwen 3.5-0.8B (18 DeltaNet + 6 Attention layers, single persistent dispatch). Python + CUDAExtension (`setup.py` links against torch C++ libs). Research proof-of-concept, batch-size-1 only.
- **`optimizations/pflash/`** — PFlash: speculative prefill compression. A small drafter scores token importance, then the target only prefills spans that matter. The algorithm lives in `server/` C++; this directory is the Python bench harness (NIAH case generation, daemon protocol driver).
- **`harness/`** — Client launchers and regression tests. Shell scripts that spawn `dflash_server` and run real clients (Codex, Claude Code, OpenCode, Hermes, etc.). Auto-installs client CLIs under `.harness-work/`.

## Build commands

```bash
# ── Prerequisites ──
# System deps (Ubuntu 22.04/24.04): build-essential cmake git git-lfs nvcc
sudo bash server/scripts/setup_system.sh  # idempotent, configures nvcc on PATH

# ── Submodules (required before CMake) ──
git submodule update --init --recursive

# ── Python workspace (uv 0.11+ is canonical; single .venv at repo root) ──
uv sync                       # dflash + pflash deps (pulls torch from cu128 index)
uv sync --extra megakernel    # second pass: compiles megakernel CUDA extension against the venv's torch

# ── C++/CUDA server (CUDA 12+, CMake 3.18+) ──
cmake -B server/build -S server -DCMAKE_BUILD_TYPE=Release
cmake --build server/build --target dflash_server -j

# ── Megakernel bench ──
uv run --directory megakernel python final_bench.py
```

### CMake options

| Option | Default | Notes |
|--------|---------|-------|
| `CMAKE_CUDA_ARCHITECTURES` | `75;86` (auto-extended) | Set to match your GPU. 86=3090, 89=4090, 120=5090/Spark, 110=Thor |
| `DFLASH27B_GPU_BACKEND` | `cuda` | Set to `hip` for AMD ROCm builds |
| `DFLASH27B_FA_ALL_QUANTS` | `ON` | All FA KV-quant pairs (3× longer compile; set OFF for fast iteration) |
| `DFLASH27B_ENABLE_BSA` | `ON` | Block-Sparse Attention for PFlash (requires sm_80+) |
| `DFLASH27B_TESTS` | `ON` | Build C++ test binaries |

### Key CMake targets

| Target | Purpose |
|--------|---------|
| `dflash_server` | Production HTTP server binary |
| `test_dflash` | Speculative-decode daemon binary (driven by Python scripts via stdin/stdout) |
| `test_server_unit` | C++ unit tests (run via ctest) |
| `test_vs_oracle` | Numerics correctness test (needs GPU + model files) |
| `test_generate` | Autoregressive generation correctness |
| `test_flash_attn_sparse` | Flash attention sparse kernel test |
| `test_flashprefill_kernels` | PFlash CUDA kernel tests |
| `pflash_daemon` | PFlash compression daemon binary |

### Stale build directory

If cmake was previously run without CUDA (or with different settings), wipe the build directory first (`rm -rf server/build`) to avoid a stale compiler cache.

## Test commands

```bash
# ── C++ unit tests (no GPU model files needed) ──
cd server/build && ctest --output-on-failure -R server_unit --no-tests=error

# ── C++ GPU tests (require model files in server/models/) ──
./server/build/test_vs_oracle \
  --target server/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/draft/dflash-draft-3.6-q4_k_m.gguf

# Smoke tests (individual GPU loads)
./server/build/smoke_load_target --target server/models/Qwen3.6-27B-Q4_K_M.gguf
./server/build/smoke_load_draft --draft server/models/draft/dflash-draft-3.6-q4_k_m.gguf

# ── Python integration tests (spawn their own server or pass --url) ──
python server/scripts/test_server_prefix_cache.py
python server/scripts/test_server_prefix_cache.py --url http://localhost:8000
python server/scripts/test_multi_turn_prefix_cache.py
python server/scripts/test_full_compress_cache.py

# ── Python tests via pytest (single file or full suite) ──
uv run pytest server/tests/test_tokenizer.py        # single test file
uv run pytest server/tests/                          # full suite

# ── Megakernel correctness (includes output parity check vs reference) ──
uv run --directory megakernel python bench_pp_tg.py

# ── Workspace smoke (lockfile + frozen sync + import check) ──
bash scripts/check_uv_workspace.sh

# ── Harness benchmarks against a running server ──
python3 harness/client_test_runner.py bench \
  --url http://127.0.0.1:8000 --suite he,agent --n-sample 3
```

## Architecture notes

- **uv workspace**: Root `pyproject.toml` declares members `server`, `optimizations/megakernel`, `optimizations/pflash`. All share a single `.venv` at repo root. The megakernel is `no-build-isolation` — it must link against the venv's cu128 torch wheel, so install requires the two-pass flow (`uv sync` then `uv sync --extra megakernel`).
- **C++ server internals**: `dflash_server` is a standalone C++ HTTP daemon (`server/src/server/`). Core runtime in `server/src/common/` (DDTree verify, draft graphs, speculative decode loop, KV cache, layer splitting). Model-specific forward paths in `server/src/qwen35/`, `server/src/laguna/`, `server/src/gemma4/`. Python scripts in `server/scripts/` drive the daemon binary via stdin/stdout protocol or HTTP.
- **Server API surface**: OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses (`/v1/responses` for Codex), Anthropic Messages (`/v1/messages` for Claude Code), health check (`/health`), model listing (`/v1/models`).
- **Model files**: Never committed. Live in `server/models/` (gitignored). Downloaded via `hf download`. Default: Qwen3.6-27B Q4_K_M target + Lucebox Q4_K_M GGUF draft. The target path can also be set via `DFLASH_TARGET` env var.
- **GPU arch detection**: CMake auto-detects CUDA architectures from the installed toolkit. Override via `CMAKE_CUDA_ARCHITECTURES`. Megakernel uses `MEGAKERNEL_CUDA_ARCH` env var. On Volta/Turing (sm_70/75) BF16 draft weights auto-convert to FP16 at load.
- **HIP backend**: AMD GPU support (Strix Halo, RX 7900 XTX) via `DFLASH27B_GPU_BACKEND=hip`, ROCm 6+. Compatibility layer in `server/src/hip_compat/`.
- **Environment variables**: Server behavior controlled via `DFLASH_` / `DFLASH27B_` prefixed env vars (e.g., `DFLASH27B_KV_TQ3=1` for TQ3_0 KV cache, `DFLASH_FP_USE_BSA=1` for BSA dispatch, `DFLASH_TARGET_GPU=N`). Harness launchers use `DFLASH_SERVER_BIN`, `DFLASH_TARGET`, `DFLASH_DRAFT`, `MAX_CTX`, `BUDGET`, `VERIFY_MODE`.

## Conventions

- **Commit messages**: Conventional commits — `feat(megakernel):`, `fix(dflash):`, `perf(pflash):`, `docs(hub):`. Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `bench`, `chore`, `ci`.
- **One concern per PR**: Kernel/algorithm changes, docs, and build config go in separate commits or PRs.
- **Benchmarks required**: Kernel/algorithm PRs must include before/after numbers on the same hardware, same power limit, same warmup. Numbers without methodology don't get merged.
- **Correctness checks**: Run `bench_pp_tg.py` (megakernel) or `test_vs_oracle` (DFlash) to confirm changes don't regress output parity.
- **Python**: 3.12 (pinned in `.python-version`). Use `uv` for dependency management (not raw pip, though legacy `pip install` flow still works for individual subprojects).
- **C++ standard**: C++17.
- **No closed-source deps**: Everything must be reproducible from public sources.
- **Power methodology**: Efficiency numbers (tok/J) measure accelerator power only via NVML, following Hazy Research's Intelligence Per Watt methodology. Default sweet spot: `sudo nvidia-smi -pl 220` on RTX 3090.

## CI

GitHub Actions on PRs to `main` (`.github/workflows/ci.yml`):

1. **`uv workspace`** — `uv lock --check`, sync without torch, import smoke test.
2. **`build`** — Full CMake build (sm_86, BSA off, FA_ALL_QUANTS off for speed), C++ unit tests via `ctest -R server_unit`, two-pass megakernel compile (sm_75 then sm_86), extension import verification.

## Running the server

```bash
# Download default models (~18 GB)
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir server/models/
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir server/models/draft/

# Run with DDTree speculative decode
./server/build/dflash_server server/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/draft/dflash-draft-3.6-q4_k_m.gguf \
  --ddtree --ddtree-budget 22 --fa-window 2048 --port 8080
```
