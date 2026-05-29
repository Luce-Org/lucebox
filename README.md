<p align="center">
  <img src="assets/banner.png" alt="Lucebox" width="85%">
</p>

<p align="center">
  <a href="https://lucebox.com"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=f5c842&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://huggingface.co/Lucebox"><img src="https://img.shields.io/badge/HuggingFace-f5c842?style=for-the-badge&logo=huggingface&logoColor=f5c842&labelColor=090909" alt="HuggingFace"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=f5c842&labelColor=090909" alt="Discord"></a>
  <a href="https://lucebox.com/blog"><img src="https://img.shields.io/badge/Blog-f5c842?style=for-the-badge&logo=rss&logoColor=f5c842&labelColor=090909" alt="Blog"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-e8e8ed?style=for-the-badge&labelColor=090909" alt="Apache 2.0"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=76b900&labelColor=090909" alt="CUDA 12+"></a>
  <a href="https://rocm.docs.amd.com/projects/HIP/en/latest/"><img src="https://img.shields.io/badge/HIP-7%2B-ed1c24?style=for-the-badge&logo=amd&logoColor=ed1c24&labelColor=090909" alt="HIP 7+"></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-e8e8ed?style=for-the-badge&logo=cplusplus&logoColor=e8e8ed&labelColor=090909" alt="C++17"></a>
</p>

<p align="center">
  <strong>Local LLM inference server built for speed. Custom kernels, speculative prefill & decoding.</strong><br/>
  Each optimization in our engine is for specific model family and hardware target.
</p>

---

## Quick start

The prebuilt Docker image covers CUDA 12.8-compatible NVIDIA GPUs. The
host wrapper `lucebox` probes your driver + GPU, selects the `:cuda12`
image, and either runs the server foreground or
manages it as a user systemd service. All orchestration logic — config,
autotune, benchmarks, smoke tests, model download — lives in a typed
Python CLI inside the image.

```bash
# 1. Install the host wrapper. Zero deps beyond docker + nvidia-smi. No uv
#    or Python required on the host. The installer bakes the source URL
#    into the installed copy, so `lucebox update` later re-pulls from the
#    same channel (canonical, dev fork, branch — your call).
curl -fsSL https://raw.githubusercontent.com/Luce-Org/lucebox-hub/main/install.sh | bash

# Install from a fork / branch instead — the URL is recorded for future
# `lucebox update` calls. The env var goes on `bash`, not on `curl`,
# since each side of the pipe is a separate process:
#   curl -fsSL https://raw.githubusercontent.com/easel/lucebox-hub/feat/lucebox-docker/install.sh | \
#     LUCEBOX_INSTALL_URL=https://raw.githubusercontent.com/easel/lucebox-hub/feat/lucebox-docker/lucebox.sh bash

# 2. Sanity check: driver, docker, NVIDIA Container Toolkit, VRAM, systemd.
lucebox check

# 3. Pull the image (~14 GB).
lucebox pull

# 4. Fetch the default target + DFlash draft (~17 GB) via the container —
#    no host-side huggingface-cli install needed. With no args, recommends
#    a preset based on detected VRAM and auto-activates it in config.toml.
lucebox models download

# 5. (Optional) Persist VRAM-tiered DFLASH_* defaults to config.toml:
lucebox autotune --apply

# 6. Run the server. Either foreground:
lucebox serve
#    …or install + start as a user systemd service:
lucebox install        # writes ~/.config/systemd/user/lucebox.service
lucebox start          # systemctl --user start lucebox
lucebox status         # journalctl-style status
lucebox logs           # follow the journal

# 7. Use it.
curl http://localhost:8080/v1/models
```

Prefer raw docker? `lucebox print-run` emits the exact `docker run`
command without executing — copy, tweak, paste. Or skip the wrapper
entirely:

```bash
docker run --rm --gpus all -p 8080:8080 \
    -v "$PWD/models:/opt/lucebox-hub/server/models" \
    ghcr.io/luce-org/lucebox-hub:cuda12
```

The container falls back to VRAM-tiered autotune when env vars aren't
supplied — ~112K ctx with TQ3_0 KV on a 24 GB card, full 128K on 32+ GB.

### Switching models

`lucebox models download <preset>` fetches the named preset's GGUFs
and (with `--activate` or as the first-install path) flips
`model.preset` in `config.toml`. Today's registry:

| Preset         | Target GGUF                                                        | DFlash draft                                              |
|----------------|--------------------------------------------------------------------|-----------------------------------------------------------|
| `qwen3.6-27b`  | `unsloth/Qwen3.6-27B-GGUF` · `Qwen3.6-27B-Q4_K_M.gguf`              | `spiritbuun/Qwen3.6-27B-DFlash-GGUF`                      |
| `gemma-4-26b`  | `bartowski/google_gemma-4-26B-A4B-it-GGUF` · `…-Q4_K_M.gguf`        | `Lucebox/gemma-4-26B-A4B-it-DFlash-GGUF`                  |
| `gemma-4-31b`  | `bartowski/google_gemma-4-31B-it-GGUF` · `…-Q4_K_M.gguf`            | `Lucebox/gemma-4-31B-it-DFlash-GGUF`                      |
| `laguna-xs.2`  | `Lucebox/Laguna-XS.2-GGUF` · `laguna-xs2-Q4_K_M.gguf`               | target-only (speculator is safetensors, fetch manually)   |

```bash
# List everything in the registry — repos, files, descriptions.
lucebox models list

# Switch an existing install from qwen to gemma-4-26b:
lucebox models download gemma-4-26b --activate     # ~18 GB
lucebox start                                      # systemd
# or `lucebox serve` for foreground
```

`models download … --activate` writes `[model] preset = "<name>"` into
`~/.lucebox/config.toml` (along with `target_file` / `draft_file`);
`lucebox serve` reads those and passes `DFLASH_TARGET` (+ `DFLASH_DRAFT`
when a draft GGUF exists) to the container so the entrypoint never has
to guess which GGUF to load on hosts that have multiple model families
on disk. Without `--activate` the entrypoint's autodetect runs (single-
GGUF models dir → picked automatically).

### Hardware coverage

| GPU                              | sm   | cuda12 |
|----------------------------------|------|:------:|
| RTX 2080 Ti                      | 75   | ✓      |
| A100                             | 80   | ✓      |
| RTX 3090 / A40 / A10             | 86   | ✓      |
| RTX 4090 / L40                   | 89   | ✓      |
| H100                             | 90   | ✓      |
| RTX 5090 / RTX 5090 Laptop       | 120  | ✓      |

Pre-Turing GPUs (Pascal sm_60/61, Volta sm_70) aren't supported — dflash's
kernels assume sm_75+ with no fallback below.

### Configuration

`lucebox autotune --apply` writes the 11 DFLASH_* keys into
`~/.lucebox/config.toml` with VRAM-tiered heuristics; or use
`lucebox config set dflash.<key>=<value>` to set individual knobs.
For a tuned config, run `lucebox autotune --sweep` after `pull`. The
sweep builds a per-VRAM-tier bracket of candidate `DFLASH_*` configs
(see `lucebox.autotune.candidate_configs`), cycles the live server
through each one via `lucebox restart`, captures decode tokens/sec
via `luce-bench snapshot --level level1`, and persists the highest-
tps cell back into `config.toml`. Snapshots land under
`~/.local/share/lucebox/profile-snapshots/sweep/` so they remain
queryable later by `luce-bench report`. Pre-sweep `config.toml` is
backed up to `.sweep-backup` and restored on interrupt or failure.

Override per-run via `-e VAR=value` on `docker run` when you want to
test a single point instead of the full bracket.

On WSL2, 24 GB-class NVIDIA GPUs default to a safer `DFLASH_MAX_CTX=65536`
and `DFLASH_BUDGET=16`. Stress testing on a 3090 Ti showed that `114688/22`
can leave only a few hundred MiB of VRAM headroom under repeated tool traffic,
which is not enough for CUDA/VMM scratch allocations. Use
`lucebox autotune --sweep` to prove higher settings before keeping them.

| Env var                       | Default         | What it does
|-------------------------------|-----------------|--------------
| `DFLASH_PORT`                 | `8080`          | HTTP port
| `DFLASH_MAX_CTX`              | autotuned       | Force a specific context length
| `DFLASH_BUDGET`               | `22`            | DDTree tree budget (8 on AMD RDNA3)
| `DFLASH_PREFIX_CACHE_SLOTS`   | `1`             | System-prompt prefix cache snapshots
| `DFLASH_CACHE_TYPE_K`         | auto            | Explicit K-cache type override
| `DFLASH_CACHE_TYPE_V`         | auto            | Explicit V-cache type override
| `DFLASH_PREFILL_MODE`         | `off`           | `auto` / `always` for pFlash long-prompt speedups
| `DFLASH_PREFILL_DRAFTER`      | unset           | Qwen3-0.6B BF16 GGUF for pFlash
| `DFLASH_TARGET`               | auto-detected   | Override the target `.gguf` path
| `DFLASH_DRAFT`                | `models/draft/` | Override the DFlash draft dir/file
| `LUCEBOX_IMAGE`               | `ghcr.io/luce-org/lucebox-hub` | Override the image repository
| `LUCEBOX_VARIANT`             | `cuda12`        | Override the image tag for release-candidate builds

CLI reference: [`lucebox`](lucebox) (host) and
[`lucebox/`](lucebox/) (Python package inside the container).

### Available tags

| Tag                            | Notes
|--------------------------------|-------
| `:cuda12`                      | rolling latest CUDA 12.8 image
| `:vX.Y.Z-cuda12`               | pinned to a specific release
| `:X.Y-cuda12`                  | latest patch in a minor series
| `:sha-<short>-cuda12`          | exact commit

### Building from source

Megakernel isn't in the Docker images yet (its CUDA extension links against
a `torch.utils.cpp_extension` wheel at build time and has to be compiled in
your venv). For megakernel benchmarks, dflash kernel development, or
running dflash with a non-default arch list, see
[`optimizations/megakernel/README.md`](optimizations/megakernel/README.md),
[`server/README.md`](server/README.md), and
[`optimizations/pflash/README.md`](optimizations/pflash/README.md).

---

## Inference Engine Optimizations

Each one is self-contained with setup instructions and benchmark notes.

<p align="center">
  <a href="optimizations/megakernel/"><img src="assets/cards/megakernel_card.png" alt="Megakernel" width="46%"></a>
  &nbsp;&nbsp;
  <a href="server/"><img src="assets/cards/dflash_card.png" alt="DFlash 27B" width="46%"></a>
</p>

<p align="center">
  <a href="optimizations/pflash/"><img src="assets/cards/pflash_card.png" alt="PFlash speculative prefill" width="46%"></a>
</p>

---

## Supported Models

All speedups measured vs vendored llama.cpp (`-fa 1`, matching KV quant). Combined = geometric mean √(TTFT × decode) where both phases benched; otherwise the single-phase speedup.

| Model | Speedup |
|-------|:-------:|
| Qwen 3.5-0.8B (Megakernel) | **~2×** |
| Qwen 3.5-27B Q4_K_M (DFlash + DDTree) | **3.43×** |
| Qwen 3.6-27B Q4_K_M (DFlash + PFlash) | **~5.6×** |
| Qwen 3.6-27B Q4_K_M (DFlash + DDTree) | **4.84×** |
| Laguna-XS.2 33B-A3B Q4_K_M (DFlash + PFlash) | **5.4×** @ 128K |
| Qwen 3.5-27B Q4_K_M (DFlash + PFlash, HIP) | **~2.6×** |
| Gemma-4-26B-A4B Q4_K_M (DFlash) | **1.31×** |

## Supported Machines

Reference target: **RTX 3090 (Ampere sm_86)** — all headline numbers. Other NVIDIA archs auto-detected by CMake / `setup.py`; AMD HIP backend separate ([Strix Halo section](#amd-strix-halo-hip-backend)).

| Arch | GPUs | Min CUDA | Status | Bench |
|------|------|:--------:|--------|:-----:|
| Ampere `sm_86` | RTX 3090, A-series | 12.0 | ✅ reference | [megakernel](optimizations/megakernel/RESULTS.md#rtx-3090-pp520-tg128) · [dflash](server/RESULTS.md) |
| Blackwell `sm_120` | RTX 5090 | 12.8 | ✅ 205 tok/s, 4.84× | [↗](server/RESULTS.md#rtx-5090-blackwell-sm_120sm_120a-32-gb) |
| Blackwell `sm_121` | DGX Spark / GB10 | 12.9 | ✅ megakernel NVFP4 | [↗](optimizations/megakernel/RESULTS.md#nvidia-dgx-spark-gb10-sm_121a) |
| Turing `sm_75` | RTX 2080 Ti | 12.0 | ✅ 53 tok/s DFlash | [↗](server/RESULTS.md#rtx-2080-ti-turing-sm_75-22-gb) |
| Ada `sm_89` | RTX 40xx | 12.0 | 🟡 community WSL2 bench | [↗](server/RESULTS.md#rtx-4090-ada-sm_89-24-gb--wsl2-community) |
| Blackwell `sm_110` | Jetson AGX Thor | 13.0 | 🟡 builds, unbenched | — |
| Volta `sm_70` / Pascal `sm_61` | V100, P40 | 12.0 | 🟡 fallback paths, unbenched | — |
| RDNA3.5 `gfx1151` | Ryzen AI MAX+ 395 / Strix Halo | ROCm 6+ | ✅ 37 tok/s HIP | [↗](#amd-strix-halo-hip-backend) |
| RDNA3 `gfx1100` | Radeon RX 7900 XTX | ROCm 6+ | ✅ 50 tok/s HIP | [↗](server/docs/HIP_PERF_PLAN.md) |

Build needs CMake 3.18+, PyTorch 2.0+, `--recurse-submodules` for `Luce-Org/llama.cpp@luce-dflash`. Power-tune: `sudo nvidia-smi -pl 220` (3090 sweet spot, re-sweep for other cards).

## Supported Harnesses

[`harness/`](harness/) contains RTX 3090 client launchers and regression tests
for Lucebox server compatibility. Run Lucebox inside Claude Code, Codex,
OpenCode, Hermes, Pi, OpenClaw, or Open WebUI, or check if a server change
still works with those clients.

<table>
<tr>
<td width="50%" valign="middle">

<a href="harness/"><img src="harness/assets/hero.png" alt="Lucebox client harness experiments on RTX 3090" width="100%" /></a>

</td>
<td width="50%" valign="middle">

| Client | Launcher |
|--------|----------|
| Claude Code | [`run_claude_code.sh`](harness/clients/run_claude_code.sh) |
| Codex | [`run_codex.sh`](harness/clients/run_codex.sh) |
| OpenCode | [`run_opencode.sh`](harness/clients/run_opencode.sh) |
| Hermes | [`run_hermes.sh`](harness/clients/run_hermes.sh) |
| Pi | [`run_pi.sh`](harness/clients/run_pi.sh) |
| OpenClaw | [`run_openclaw.sh`](harness/clients/run_openclaw.sh) |
| Open WebUI | [`run_openwebui.sh`](harness/clients/run_openwebui.sh) |

</td>
</tr>
</table>

All launchers spawn the native C++ HTTP server (`dflash_server`). Override defaults via env vars:

```bash
DFLASH_SERVER_BIN=server/build/dflash_server \
MAX_CTX=32768 BUDGET=22 VERIFY_MODE=ddtree \
harness/clients/run_codex.sh
```

## 01 · Megakernel Qwen3.5 0.8B on RTX 3090

Single-kernel CUDA inference for Qwen 3.5-0.8B on RTX 3090. All 24 layers run in one persistent dispatch.

```bash
# 1. clone + enter
git clone https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub

# 2. install via the workspace (Python 3.12, CUDA 12+, PyTorch 2.0+).
#    Weights stream from HF on first run.
uv sync --extra megakernel          # builds the CUDA extension; torch is auto-installed first, then setup.py compiles

# 3. run the benchmark (prefill pp520 + decode tg128 vs llama.cpp BF16 + PyTorch HF)
uv run --directory megakernel python final_bench.py
```

> Don't have `uv`? Install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or see [astral.sh/uv](https://astral.sh/uv/). The legacy `python -m venv` + `pip install -e . --no-build-isolation` flow still works from inside `optimizations/megakernel/`.

| Method | Prefill pp520 | Decode tg128 | tok/J |
|--------|:-------------:|:------------:|:-----:|
| **Megakernel** `@220W` | **21,347** | **413** | **1.87** |
| llama.cpp BF16 `@350W` | 11,247 | 267 | 0.76 |
| PyTorch HF | 7,578 | 108 | n/a |

Implementation notes: 82 blocks, 512 threads, cooperative grid sync, no CPU round trips between layers, and weights streamed from Hugging Face on first run.

[Full writeup →](optimizations/megakernel/README.md) · [Benchmarks →](optimizations/megakernel/RESULTS.md) · [Blog post →](https://lucebox.com/blog/megakernel)

> **Blackwell (RTX 5090, DGX Spark / GB10):** auto-detected by setup; NVFP4 decode path lands ~194 tok/s tg128 on GB10. See [optimizations/megakernel/README.md#blackwell-sm_120--sm_121a](optimizations/megakernel/README.md).

---

## 02 · DFlash DDtree Qwen3.5 & Qwen3.6 27B GGUF on RTX 3090

DFlash speculative decoding for Qwen3.5/Qwen3.6 27B GGUF targets on a single GPU. The default setup uses Qwen3.6-27B Q4_K_M plus a Q4_K_M GGUF DFlash draft.

- **Up to 207 tok/s** in the demo (207.6 tok/s DFlash vs 38.0 tok/s AR, 5.46×)
- **129.5 tok/s mean** on the HumanEval 10-prompt bench
- **3.43× faster than autoregressive** (+15% over chain speculative decoding)
- **2.8× faster than SGLang AWQ** on the same hardware
- **Up to 256K context in 24 GB** via TurboQuant TQ3_0 KV cache (128K Q4_0 bench: 134.78 tok/s at ctx=131072)

```bash
# 1. clone with submodules (pulls the pinned Luce-Org/llama.cpp@luce-dflash fork)
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub

# 2. install Python deps via the workspace (creates one shared .venv at the
#    repo root).
uv sync

# 3. build the C++/CUDA decoder (CUDA 12+, CMake 3.18+)
# Default compiles for Pascal/Volta/Turing/Ampere (60/61/62/70/75/86; +120 on CUDA 12.8+, +sm_121/DGX Spark on CUDA 12.9+, +sm_110/Thor on CUDA 13.0+) so the binary runs on every supported card.
# 3090-only users can add -DCMAKE_CUDA_ARCHITECTURES=86 to skip the other archs and build faster (~3 min).
cmake -B server/build -S server -DCMAKE_BUILD_TYPE=Release
cmake --build server/build --target test_dflash -j
cmake --build server/build --target test_generate -j
cmake --build server/build --target dflash_server -j

# 4. fetch weights: ~16 GB Q4_K_M target + 1.84 GB Lucebox Q8_0 GGUF DFlash draft
uv run hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir server/models/
uv run hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q8_0.gguf --local-dir server/models/draft/

# 5a. one-shot streaming generate
uv run --directory server python scripts/run.py --prompt "def fibonacci(n):"

# 5b. or reproduce the paper-style bench (HumanEval + GSM8K + Math500, ~15 min)
uv run --directory server python scripts/bench_llm.py
```

| Benchmark | AR (tok/s) | DFlash+DDTree (tok/s) | Speedup |
|-----------|:----------:|:---------------------:|:-------:|
| **HumanEval** | 37.8 | **129.5** | **3.43×** |
| Math500 | 37.7 | 110.5 | 2.93× |
| GSM8K | 37.7 | 96.2 | 2.55× |

**Why GGUF/Q4_K_M:** on 24 GB GPUs, the target, draft, DDTree verify state, and KV cache need to fit together. The default Qwen3.6 setup uses a ~16 GB Q4_K_M target and a quantized GGUF draft.

Algorithms used:
- [**DFlash**](https://arxiv.org/abs/2602.06036) (z-lab, 2026): block-diffusion draft conditioned on target hidden states.
- [**DDTree**](https://arxiv.org/abs/2604.12989) (Ringel et al., 2026): tree-structured verify that beats chain verify at the same compute budget.

Implemented here:
- C++/CUDA decode engine on top of ggml (no libllama, no Python runtime, Q4_K_M target path).
- Three custom CUDA kernels for tree-aware SSM state rollback: `ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`.
- DDTree budget swept for RTX 3090 + Q4_K_M target: **budget=22** is the sweet spot.
- TQ3_0 KV cache (TurboQuant 3.5 bpv, default) + sliding `target_feat` ring to fit up to 256K context in 24 GB (Q4_0 available as legacy, tops out near 128K).

### Running on other GPUs (4090, 5090, DGX Spark / GB10, Jetson AGX Thor)

Supported out of the box; the build just needs the right CUDA toolkit. `server/CMakeLists.txt` already auto-adds Blackwell archs when your nvcc is new enough, so the main quickstart above works as-is on newer cards.

| GPU | Arch | Min CUDA | Status |
|-----|:----:|:--------:|--------|
| Tesla P40 Pascal | `sm_61` | 12.0 | supported with scalar F16 fallback; needs 24 GB for the 27B stack |
| Tesla V100 Volta | `sm_70` | 12.0 | supported with F16 WMMA kernels |
| RTX 3090 Ampere | `sm_86` | 12.0 | **reference, all numbers above** |
| RTX 2080 Ti Turing | `sm_75` | 12.0 | supported, 53 tok/s DFlash verified (FP16 draft) |
| RTX 4090 Ada | `sm_89` | 12.0 | should work, unverified, pass `-DCMAKE_CUDA_ARCHITECTURES=89` |
| RTX 5090 Blackwell consumer | `sm_120` | 12.8 | **205 tok/s DFlash, 4.84× vs AR** (Q4_K_M, budget=40) |
| DGX Spark / GB10 | `sm_121` (compute capability 12.1) | 12.9 | supported, auto-added by CMake |
| Jetson AGX Thor | `sm_110` | 13.0 | supported, auto-added by CMake |

Verify your target:
```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, 'sm_%d%d'%(p.major,p.minor), p.multi_processor_count,'SMs', round(p.total_memory/1e9,1),'GB')"
nvcc --version
```

**DGX Spark / GB10 quick start:**
```bash
# CUDA 12.9+ required for sm_121
nvcc --version  # must show >= 12.9
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/server
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake auto-adds sm_121
cmake --build build --target test_dflash -j
```

**Jetson AGX Thor quick start:**
```bash
# CUDA 13.0+ required for sm_110 / AGX Thor.
nvcc --version
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/server
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake auto-adds the Thor arch your nvcc supports
cmake --build build --target test_dflash -j
```

**Retune per GPU:**
- **DDTree `budget=22`** tuned for 3090 + Q4_K_M + 24 GB. On the RTX 5090, budget=40 is optimal (swept). On GB10 (128 GB unified), re-sweep — larger tree = more verify throughput until memory bandwidth saturates. `scripts/bench_llm.py --budget N` has the sweep hooks.
- **TQ3_0 KV cache + sliding `target_feat` ring** was shaped by 24 GB (fits up to 256K context on a 3090). On GB10 (128 GB unified) / 5090 (32 GB) you can push context further or skip quantization entirely and keep F16 KV.
- **Perf numbers** (207 tok/s demo, 129.5 HumanEval, 2.8× vs SGLang AWQ) are RTX 3090 @ stock. RTX 5090 numbers (205 tok/s HumanEval, 4.84×) are in [RESULTS.md](server/RESULTS.md). Ada/GB10/Thor not yet swept, PRs with `RESULTS.md` entries welcome.

[Full writeup →](server/README.md) · [Benchmarks →](server/RESULTS.md) · [Blog post →](https://lucebox.com/blog/dflash27b)

---

## 03 · PFlash speculative prefill on RTX 3090

Speculative prefill for long prompts. A Qwen3-0.6B BF16 drafter scores token importance, then the 27B target prefills only the retained spans. Runtime is C++/CUDA through the dflash binaries; no PyTorch is required at serving time.

- **~10.4× TTFT** on 128K context: **24.8 s** dflash daemon vs **~257 s** llama.cpp (FA on, Q4_0 KV).
- **10.0× TTFT** on 64K context: **13.5 s** dflash vs **134.95 s** llama.cpp.
- **NIAH single-needle retrieved** at every measured context (32K → 128K), `keep_ratio=0.05`, `DFLASH_FP_ALPHA=0.85`.

```bash
# 1. build dflash + BSA kernel (sm_80+ required for BSA, ~10 min cold compile)
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/server
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_CUDA_ARCHITECTURES=86 \
                    -DDFLASH27B_ENABLE_BSA=ON
cmake --build build --target test_dflash test_flashprefill_kernels -j

# 2. fetch weights: 27B Q4_K_M target + 0.6B BF16 drafter (GGUF) + DFlash spec-decode draft
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --local-dir models/
hf download spiritbuun/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir models/draft/

# 3. run the daemon: compress (drafter scoring) + generate (target spec decode)
DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85 \
./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf models/draft/dflash-draft-3.6-q4_k_m.gguf --daemon
# stdin protocol: `compress <ids.bin> <keep_x1000> <drafter.gguf>` →
#                 stream of compressed token ids, then `generate <…>` →
#                 stream of generated tokens.
```

| Source S | dflash TTFT | llama.cpp baseline | Speedup | NIAH |
|----------|:-----------:|:------------------:|:-------:|:----:|
| **64K**  | **13.5 s** | 134.95 s (FA off, dense) | **10.0×** | ✅ |
| **128K** | **24.8 s** | ~257 s (FA on, Q4_0 KV)  | **~10.4×** | ✅ |

Daemon stdin commands: `compress` runs the drafter with FlashPrefill block-sparse attention and returns the compressed token-id stream; `generate` runs the target on that stream with normal speculative decode + DDTree. `park` / `unpark` / `free drafter` swap weights in and out of VRAM so target + drafter coexist on a 24 GB card.

**Runtime tunables** (full list in [`server/src/flashprefill.h`](server/src/flashprefill.h)):
```
DFLASH_FP_USE_BSA=1     # dispatch sparse FA forward through BSA (sm_80+)
DFLASH_FP_ALPHA=0.85    # block-selection threshold; higher = stricter = fewer K-blocks per Q-row
DFLASH_FP_PROFILE=1     # log mean / score / select / forward stage timings
```

**What's ours, what isn't.** Algorithms are from [Cross-Family Speculative Prefill (Liu et al., ICLR 2026)](https://arxiv.org/abs/2603.02631) for the scoring + selection layer and [FlashPrefill (Fan et al., 2026)](https://arxiv.org/abs/2603.06199) for the drafter sparse-attention forward. What we built:
- C++/CUDA daemon-resident speculative prefill in front of a quantized GGUF target — no PyTorch, no Triton, no per-request subprocess.
- BSA wired without `libtorch` via a 3-header ATen/c10 stub set under `server/deps/bsa_stubs/`.
- Custom Qwen3-0.6B forward (`qwen3_0p6b_*`) so the drafter runs through the same ggml allocator as the 27B target.
- 4 CUDA kernels (`flashprefill_kernels.cu`) for the FlashPrefill `mean_K / score / select / sparse_fwd` algorithm.

[Full writeup →](optimizations/pflash/README.md) · [Daemon-side build / tunables →](server/docs/SPEC_PREFILL.md) · [Blog post →](https://lucebox.com/blog/pflash)

---

## AMD Strix Halo (HIP backend)

**Same DFlash + PFlash stack on an AMD iGPU.** PR #119 ports the Phase 2 rocWMMA flashprefill kernels to HIP. End-to-end on a single Ryzen AI MAX+ 395 box (Radeon 8060S iGPU, gfx1151, 128 GiB LPDDR5X-8000 unified): **37.0 tok/s** DFlash decode on Qwen3.5-27B Q4_K_M, **27.6 s** TTFT at 16K context with NIAH retrieval intact. That is **3.08×** decode and **2.24×** prefill over llama.cpp HIP AR on the same iGPU. End-to-end wall clock at a realistic 16K prompt + 1K generation workload: **2.66×** faster than vanilla llama.cpp.

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub/server

# Build for gfx1151 (Strix Halo). Swap the arch for gfx1100 / gfx1201.
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
  -DDFLASH27B_HIP_SM80_EQUIV=ON
cmake --build build --target test_dflash -j
```

`DFLASH27B_HIP_SM80_EQUIV=ON` enables the rocWMMA Phase 2 flashprefill kernels (the path that delivers the prefill speedup). `OFF` falls back to ggml's `flash_attn_ext` (slower but no rocwmma headers needed).

**Per-arch DDTree tuning**: gfx1151 (Strix Halo iGPU, bandwidth-bound on LPDDR5X) peaks at `--ddtree-budget=22`. gfx1100 (7900 XTX, GDDR6) prefers `budget=8` per the [PR #156 cross-arch perf plan](https://github.com/Luce-Org/lucebox-hub/pull/156). Run `scripts/bench_he.py --ddtree-budget N` to verify on your card.

**Drafter recipe for max decode**: target = Qwen3.5-27B Q4_K_M, drafter = same gen quantized to Q8_0 via `server/scripts/quantize_draft_q8.py`. The matching Q8_0 GGUF on the unsloth Qwen3.6 target needs `DFLASH27B_DRAFT_SWA=2048` for sliding-window correctness.

[Blog post →](https://lucebox.com/blog/amd) · [PR #119 →](https://github.com/Luce-Org/lucebox-hub/pull/119) · [PR #156 cross-arch perf plan →](https://github.com/Luce-Org/lucebox-hub/pull/156)

---

## Why this exists

Local AI should be a default, not a privilege: private data, no per-token bill, no vendor lock-in. The hardware to run capable models already sits on desks. The software to run those chips well doesn't.

General-purpose frameworks dominated the last decade because hand-tuning kernels per chip was too expensive to justify. One stack, decent on everything, great on nothing. Most of the silicon's capability stays on the floor.

AI-assisted development flips that calculus. Rewrites that took a quarter now fit in a release cycle. Lucebox is where we publish them, one chip and one model family at a time. Apache 2.0 source, full writeup, reproducible benchmarks.

---

---

## Request for Contributions

```
  ▮▮▮▮▮▮▮▮▮▮    HIP/CUDA kernel optimizations
  ▮▮▮▮▮▮▮▮▮▯    Speculative inference optimizations
  ▮▮▮▮▮▮▮▯▯▯    Support to new GPU/APU consumer cards
  ▮▮▮▮▮▮▮▯▯▯    Inference engine debugging
  ▮▮▮▮▮▮▯▯▯▯    Add new performance benchmarks
  ▮▮▮▮▮▯▯▯▯▯    Improvements for harnesses integration
```

---

## Citation

```bibtex
@software{lucebox_2026,
  title  = {Fast LLM speculative inference server for specific consumer hardware.},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub},
  year   = {2026}
}
```

---

## Community

- **Discord**: [discord.gg/yHfswqZmJQ](https://discord.gg/yHfswqZmJQ)
- **Website**: [lucebox.com](https://lucebox.com)
- **Issues**: [github.com/Luce-Org/lucebox-hub/issues](https://github.com/Luce-Org/lucebox-hub/issues)
- **Blog**: [lucebox.com/blog](https://lucebox.com/blog)

---

<p align="center">
  <sub><a href="LICENSE">Apache 2.0</a> · <a href="https://lucebox.com">Lucebox.com</a></sub>
</p>
