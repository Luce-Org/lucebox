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

DFlash uses **CMake** with CUDA. The build produces `test_dflash`, the speculative-decoding
daemon that the Python server drives via stdin/stdout.

```bash
cd dflash

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
| `DFLASH27B_FA_ALL_QUANTS` | `ON` | Build all FA KV-quant pairs (3× longer compile) |
| `DFLASH27B_ENABLE_BSA` | `ON` | Block-Sparse Attention for spec-prefill (needs sm_80+) |
| `DFLASH27B_TESTS` | `ON` | Build C++ numerics tests |

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

The target path can also be set via the `DFLASH_TARGET` environment variable.

---

## Running the server

```bash
cd dflash
./build/dflash_server models/Qwen3.6-27B-Q4_K_M.gguf --port 8080
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

These scripts start their own server subprocess and need the server binary + models:

```bash
cd server/scripts
python test_server_prefix_cache.py
python test_multi_turn_prefix_cache.py
python test_full_compress_cache.py
```

Or run against an already-running server:

```bash
python test_server_prefix_cache.py --url http://localhost:8000
python test_multi_turn_prefix_cache.py --url http://localhost:8000
```

---

## Project structure

```
server/
├── CMakeLists.txt              # C++ build (cmake)
├── include/                    # C++ headers
├── src/                        # C++ sources (target/draft graph, KV cache, FlashPrefill)
├── test/                       # C++ test sources (test_dflash.cpp, smoke_*, test_*)
├── deps/
│   ├── llama.cpp/              # Vendored ggml snapshot + extracted helpers
│   └── Block-Sparse-Attention/ # BSA kernels (submodule)
├── models/                     # Model files (not in git)
│   ├── Qwen3.6-27B-Q4_K_M.gguf
│   └── draft/dflash-draft-3.6-q4_k_m.gguf
├── scripts/
│   ├── run.py                  # CLI text generation
│   ├── test_server_prefix_cache.py    # Integration test (--url or auto-spawn)
│   ├── test_multi_turn_prefix_cache.py # Integration test (--url or auto-spawn)
│   ├── test_full_compress_cache.py    # Integration test
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
model_provider = "dflash"

[model_providers.dflash]
name = "DFlash"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
supports_websockets = false
```

No `env_key` is needed for local use.

### Running

```bash
# Start the server
./build/dflash_server models/Qwen3.6-27B-Q4_K_M.gguf --port 8080

# In another terminal
codex --provider dflash "Explain this codebase"
```
