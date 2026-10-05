# DFlash Developer Guide

## Prerequisites

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | NVIDIA Turing (sm_75, e.g. RTX 2080) | Ampere+ (sm_86, e.g. RTX 3090) |
| VRAM | 22 GB | 24 GB |
| OS | Ubuntu 22.04 (jammy) | Ubuntu 24.04 (noble) |

> **Note:** FlashPrefill and BSA (Block-Sparse Attention) require **sm_80+** (Ampere or newer).
> On Turing (sm_75) the drafter falls back to ggml's `flash_attn_ext`.

### System packages

```
build-essential  cmake  git  git-lfs  nvcc (CUDA Toolkit)
```

A setup script is provided that installs everything (run as root):

```bash
sudo bash server/scripts/setup_system.sh
```

This installs build tools, `hf` (via pipx), and the CUDA Toolkit.

### Python

- **Python 3.11+** (tested with 3.11.2)
- Virtual environment recommended

```bash
python3 -m venv venv
source venv/bin/activate
```

### Python packages

Install the required packages:

```bash
pip install fastapi uvicorn transformers pydantic starlette
```

For running tests:

```bash
pip install pytest
```

---

## Building the C++ daemon

Luce uses **CMake** with CUDA. The build produces `test_dflash`, the speculative-decoding
daemon that the Python server drives via stdin/stdout.

```bash
cd server

# Initialize the remaining submodule (Block-Sparse-Attention)
git submodule update --init --recursive

# Configure
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build the daemon binary
cmake --build build --target test_dflash -j
```

The binary lands at `server/build/test_dflash`.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_CUDA_ARCHITECTURES` | `75;86` (auto-extended) | Target GPU architectures |
| `LUCE_FA_ALL_QUANTS` | `ON` | Build all FA KV-quant pairs (3× longer compile) |
| `LUCE_ENABLE_BSA` | `ON` | Block-Sparse Attention for spec-prefill (needs sm_80+) |
| `LUCE_TESTS` | `ON` | Build C++ numerics tests |

---

## Model files

Download models before running the server:

```bash
# Target model (Q4_K_M quantized Qwen3.6-27B)
hf download <repo-id> --local-dir server/models/

# Draft model (0.98 GB default Qwen3.6 GGUF draft)
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir server/models/draft/
```

Expected layout:

```
server/models/
├── Qwen3.6-27B-Q4_K_M.gguf          # --target (GGUF)
└── draft/
    └── dflash-draft-3.6-q4_k_m.gguf   # --draft  (GGUF)
```

The target path can also be set via the `LUCE_TARGET` environment variable.

---

## Running the server

```bash
cd server
./build/luce_server models/Qwen3.6-27B-Q4_K_M.gguf --port 8080
```

### Server CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Port |
| `--target` | `models/Qwen3.6-27B-Q4_K_M.gguf` | Target GGUF model |
| `--draft` | `models/draft` | Draft model directory |
| `--bin` | `build/test_dflash` | Path to the daemon binary |
| `--budget` | `22` | DDTree speculation budget |
| `--max-ctx` | `16384` | Maximum context length |
| `--kv-f16` | off | Force F16 KV cache |
| `--cache-type-k` / `--ctk` | auto | KV cache type for keys (f16/q4_0/q8_0/tq3_0/...) |
| `--cache-type-v` / `--ctv` | auto | KV cache type for values |
| `--fa-window` | auto | Sliding window size for flash attention (0 = full) |
| `--tokenizer` | auto (from GGUF) | HuggingFace tokenizer ID |
| `--prefix-cache-slots` | `4` | Number of prefix-cache slots |
| `--prefill-cache-slots` | `4` | Number of prefill-cache slots |
| `--daemon` | off | Run as background daemon |

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List models (OpenAI + Codex format) |
| `POST /v1/chat/completions` | OpenAI Chat Completions API |
| `POST /v1/responses` | OpenAI Responses API (Codex) |
| `POST /v1/messages` | Anthropic Messages API |

---

## Tests

### C++ unit tests (no GPU needed)

The tests that build and pass on a GPU-less machine carry the ctest label
`cpu` (listed in `_luce_cpu_ctest_names` / `_luce_cpu_cppunit_targets` in
`CMakeLists.txt`). One target builds just those binaries and runs them; the
hosted CI job runs the same target:

```bash
cmake --build server/build --target check-cpu
```

### C++ tests (require GPU + model files)

After building:

```bash
cd server/build

# Numerics tests
./test_vs_oracle --target ../models/Qwen3.6-27B-Q4_K_M.gguf \
                 --draft ../models/draft/dflash-draft-3.6-q4_k_m.gguf

# Smoke tests
./smoke_load_target --target ../models/Qwen3.6-27B-Q4_K_M.gguf
./smoke_load_draft --draft ../models/draft/dflash-draft-3.6-q4_k_m.gguf
./smoke_draft_graph --draft ../models/draft/dflash-draft-3.6-q4_k_m.gguf
```

### Integration tests (require running server)

The Python tests live in `server/test/python/` and are driven by pytest. Unit tests
run anywhere; tests that need a live `luce_server` are marked `server`, tests
that need local model files/binaries are marked `model`.

```bash
# From the repo root — everything that doesn't need hardware:
pytest -m "not server and not model and not slow"

# Server tests against a running server:
pytest server/test/python/test_server_smoke.py -v --base-url http://localhost:8080

# Or let pytest spawn luce_server itself:
pytest server/test/python/test_server_smoke.py -v --launch models/Qwen3-0.6B-BF16.gguf

# Parallel-serving tests need the slot count; with --launch it also adds
# --paged-attention --max-concurrency N to the spawned server:
pytest server/test/python/test_server_parallel.py -v --launch <model.gguf> --max-concurrency 3
```

The cache tests always spawn their own server with the flags they need (they
never reuse `--base-url`), stop it when the module finishes, and skip when the
model files or the `luce_server` binary are missing:

```bash
pytest server/test/python/test_server_prefix_cache.py -v
pytest server/test/python/test_multi_turn_prefix_cache.py -v
pytest server/test/python/test_full_compress_cache.py -v
pytest server/test/python/test_prefill_cache.py -v
```

| Variable / option | Used by |
|---|---|
| `--base-url` / `LUCE_TEST_SERVER_URL` | external server for `server` tests |
| `--launch` / `LUCE_TEST_MODEL` | model for the server spawned for each test module |
| `--server-bin` / `LUCE_SERVER_BIN` | every spawned server and the CLI tests |
| `--server-extra-args` / `LUCE_SERVER_EXTRA_ARGS` | extra flags for the `--launch`ed server |
| `--max-concurrency` / `LUCE_MAX_CONCURRENCY` | parallel tests (1–64; they skip without it) |
| `LUCE_TARGET`, `LUCE_DRAFT` | target GGUF and draft for the cache tests |
| `LUCE_PREFILL_DRAFTER` | pFlash drafter GGUF (`test_full_compress_cache.py`) |
| `LUCE_TOKENIZER_MODEL`, `LUCE_TOKENIZER_HARNESS` | `test_tokenizer.py` |

---

## Project structure

```
server/
├── CMakeLists.txt              # C++ build (cmake)
├── include/                    # C++ headers
├── src/                        # C++ sources (target/draft graph, KV cache, FlashPrefill)
├── test/                       # unit/, smoke/, bench/ (C++), python/ (pytest), fixtures/ + shared helpers
├── deps/
│   ├── llama.cpp/              # Vendored ggml snapshot + extracted helpers
│   └── Block-Sparse-Attention/ # BSA kernels (submodule)
├── models/                     # Model files (not in git)
│   ├── Qwen3.6-27B-Q4_K_M.gguf
│   └── draft/dflash-draft-3.6-q4_k_m.gguf
├── scripts/
│   ├── run.py                  # CLI text generation
│   └── setup_system.sh         # System dependency installer
├── README.md
└── DEVELOPER.md                # This file
```

---

## Using with OpenAI Codex CLI

The server natively supports the **Responses API** (`/v1/responses`) used by
[OpenAI Codex](https://github.com/openai/codex).

### Configuration

Create `~/.codex/config.toml`:

```toml
model = "luce-dflash"
model_provider = "luce"

[model_providers.luce]
name = "Luce"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
supports_websockets = false
```

No `env_key` is needed for local use.

### Running

```bash
# Start the server
./build/luce_server models/Qwen3.6-27B-Q4_K_M.gguf --port 8080

# In another terminal
codex --provider luce "Explain this codebase"
```
