<p align="center">
  <a href="https://www.lucebox.com/"><img src="https://www.lucebox.com/lucebox-logo.png" alt="Lucebox" width="160"></a>
</p>

<p align="center">
  <a href="https://www.lucebox.com/"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=f5c842&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://huggingface.co/Lucebox"><img src="https://img.shields.io/badge/HuggingFace-f5c842?style=for-the-badge&logo=huggingface&logoColor=f5c842&labelColor=090909" alt="HuggingFace"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=f5c842&labelColor=090909" alt="Discord"></a>
  <a href="https://www.lucebox.com/blog/"><img src="https://img.shields.io/badge/Blog-f5c842?style=for-the-badge&logo=rss&logoColor=f5c842&labelColor=090909" alt="Blog"></a>
  <a href="#tutorials"><img src="https://img.shields.io/badge/Tutorials-f5c842?style=for-the-badge&logo=youtube&logoColor=f5c842&labelColor=090909" alt="Tutorials"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-e8e8ed?style=for-the-badge&labelColor=090909" alt="Apache 2.0"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12%2B-76b900?style=for-the-badge&logo=nvidia&logoColor=76b900&labelColor=090909" alt="CUDA 12+"></a>
  <a href="https://rocm.docs.amd.com/projects/HIP/en/latest/"><img src="https://img.shields.io/badge/HIP-7%2B-ed1c24?style=for-the-badge&logo=amd&logoColor=ed1c24&labelColor=090909" alt="HIP 7+"></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-e8e8ed?style=for-the-badge&logo=cplusplus&logoColor=e8e8ed&labelColor=090909" alt="C++17"></a>
</p>

<p align="center">
  <strong>Speculative inference for heterogeneous machines and consumer GPUs.</strong><br/>
  Custom kernels, speculative prefill and decoding, tuned for each model and hardware target.
</p>

---

## Inference Engine Optimizations

| Optimization | Measured setup | Result |
|---|---|---:|
| [DFlash2](https://www.lucebox.com/blog/qwen38-r9700) | Qwen 3.8 27B on one R9700 | **208.1 tok/s** average, **227.8 tok/s** peak |
| [DSpark](https://www.lucebox.com/blog/deepseek-v4-flash-0731) | DeepSeek V4 on Strix Halo, native top-6 | **32.7 tok/s** high-acceptance median; **27.9 tok/s** mixed-eval average |
| [PFlash + KVFlash](https://www.lucebox.com/blog/laguna-xs21) | Laguna XS 2.1 33B at 256K on RTX 3090 | **6.1×** prefill, **411 s to 67.3 s** |
| [Luce Spark](optimizations/spark/README.md) | Laguna XS.2 33B on RTX 3090 | **~100 tok/s** in **14.6 GiB** |
| [KVFlash](https://www.lucebox.com/blog/laguna-xs21) | Laguna XS 2.1 33B at 256K on RTX 3090 | **152.3 tok/s** with an 8K pool |
| [Heterogeneous execution](https://www.lucebox.com/#benchmark) | DeepSeek V4 on R9700 + Strix Halo | **86 tok/s** decode; **788 tok/s** prefill at 2K |
| [Paged attention](optimizations/paged_attention/README.md) | Qwen 3.6 27B concurrent serving | **1.35×** attention step; **82%** less KV memory |
| [Megakernel](optimizations/megakernel/RESULTS.md#rtx-3090-pp520-tg128) | Qwen 3.5 0.8B on RTX 3090 | **413 tok/s**, **1.87 tok/J** |

---

## Supported Models and Drafters

Model links open the exact weights used by the measured setup. Drafter links open the published quant, or the source checkpoint when conversion is required.

| Model and optimization | Phase | Speedup |
|---|:---:|:---:|
| [Qwen 3.5 0.8B BF16](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/model.safetensors-00001-of-00001.safetensors) + [Megakernel](optimizations/megakernel/README.md) | Prefill + decode | **1.9×** prefill; **1.55×** decode |
| [Qwen 3.8 27B UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ4_XS.gguf) + [DFlash2 source](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/model.safetensors), converted to Q8_0, on R9700 | Decode | **6.4×** vs Lucebox AR; **3.8×** vs llama.cpp with the same drafter |
| [Laguna XS 2.1 33B Q4_K_M](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF/blob/main/Laguna-XS-2.1-Q4_K_M.gguf) + PFlash/KVFlash with [Qwen3 0.6B Q8_0](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/blob/main/Qwen3-0.6B-Q8_0.gguf) | Prefill | **6.1×**, **411 s to 67.3 s** at 256K |
| [Laguna XS 2.1 33B Q4_K_M](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF/blob/main/Laguna-XS-2.1-Q4_K_M.gguf) + [DFlash Q4 drafter](https://huggingface.co/Lucebox/Laguna-XS-2.1-DFlash-GGUF/blob/main/laguna-xs21-dflash-q4.gguf) | Decode | **1.7×** at 256K |
| [Gemma 4 26B-A4B Q4_K_M](https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF/blob/main/google_gemma-4-26B-A4B-it-Q4_K_M.gguf) + [DFlash Q8_0 drafter](https://huggingface.co/Lucebox/gemma-4-26B-A4B-it-DFlash-GGUF/blob/main/gemma-4-26B-A4B-it-DFlash-q8_0.gguf) | Decode | **1.31×** |
| [Gemma 4 31B IT Q4_K_M](https://huggingface.co/bartowski/google_gemma-4-31B-it-GGUF/blob/main/google_gemma-4-31B-it-Q4_K_M.gguf) + [DFlash Q8_0 drafter](https://huggingface.co/Lucebox/gemma-4-31B-it-DFlash-GGUF/blob/main/gemma-4-31B-it-DFlash-q8_0.gguf) | Decode | **3.2×** |
| [DeepSeek V4 Flash ROCmFPX MIX Strix](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3/blob/main/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf) + [DSpark Q4RMFP4 drafter](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF/blob/main/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf) | Decode | Up to **1.81×** vs target-only, **32.7 vs 18.1 tok/s** |
| [Ling 3.0 Flash 124B-A5.1B Q4_K_M](https://huggingface.co/bloomer010/Ling-3.0-flash-GGUF) | Decode | **34.6 tok/s** median AR on DGX Spark |

## Tested Machines (GPU/APU)

The engine is not tied to one reference card. NVIDIA architectures are selected by CMake; HIP builds should target the device's exact `gfx` architecture.

| | Architecture | Hardware | Runtime | Details |
|:---:|---|---|---|---|
| <img src="assets/gpus/r9700.png" width="750" /> | RDNA4 `gfx1201` | Radeon AI PRO R9700 | ROCm 7.2 | [Qwen 3.8 R9700 quick start](#run-the-server) |
| <img src="assets/gpus/ryze395.png" width="750" /> | RDNA3.5 `gfx1151` | Ryzen AI MAX+ 395 / Strix Halo | ROCm 7.2 | [DeepSeek V4 Strix profile](server/docs/RECOMMENDED_SETUPS.md#deepseek-v4-on-strix-halo) |
| <img src="assets/gpus/7900xtx.png" width="750" /> | RDNA3 `gfx1100` | Radeon RX 7900 XT / XTX | ROCm 6+ | [DeepSeek V4 dual AMD profile](server/docs/DS4.md#radeon-rx-7900-xt--strix-halo-true-top-k-6) |
| <img src="assets/gpus/3090.png" width="750" /> | Ampere `sm_86` | RTX 3090 | CUDA 12+ | [Qwen 3.8 NVLink result](https://github.com/Luce-Org/lucebox/pull/637) and [Megakernel results](optimizations/megakernel/RESULTS.md#rtx-3090-pp520-tg128) |
| <img src="assets/gpus/5090.png" width="750" /> | Blackwell `sm_120` | RTX 5090 | CUDA 12.8+ | [Qwen 3.8 single-GPU result](https://github.com/Luce-Org/lucebox/pull/637) |
| <img src="assets/gpus/gb10.png" width="750" /> | Blackwell `sm_121` | DGX Spark / GB10 | CUDA 12.9 | [Qwen 3.5 NVFP4 results](optimizations/megakernel/RESULTS.md#nvidia-dgx-spark-gb10-sm_121a) |
| <img src="assets/gpus/4090.png" width="750" /> | Ada `sm_89` | RTX 4090 | CUDA 12+ | [Linux](server/RESULTS.md#rtx-4090-ada-sm_89-24-gb--cachyos-bare-metal-community) and [WSL2](server/RESULTS.md#rtx-4090-ada-sm_89-24-gb--wsl2-community) community runs |
| <img src="assets/gpus/2080ti.png" width="750" /> | Turing `sm_75` | RTX 2080 Ti | CUDA 12.0 | [DFlash results](server/RESULTS.md#rtx-2080-ti-turing-sm_75-22-gb) |
| <img src="assets/gpus/v100.png" width="750" /> | Volta `sm_70`, Pascal `sm_61` | V100, P40 | CUDA 12.0 | [CUDA quick start](server/README.md#quick-start) |
| Not pictured | Blackwell `sm_110` | Jetson AGX Thor | CUDA 13.0 | [Thor quick start](server/README.md#jetson-agx-thor-sm_110-cuda-130) |

### Single-device results

| Hardware | Model | Measured result |
|---|---|---|
| **R9700** | [Qwen 3.8 27B UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ4_XS.gguf) + [DFlash2 source](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/model.safetensors) | **208.1 tok/s** HumanEval average; **227.8 tok/s** best request |
| **Strix Halo** | [DeepSeek V4 ROCmFPX MIX Strix](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3/blob/main/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf) + [DSpark Q4RMFP4](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF/blob/main/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf) | **32.7 tok/s** high-acceptance median; **27.9 tok/s** across the fixed 30-prompt evaluation, using all six routed experts |
| **RTX 5090** | Qwen 3.8 27B | **110.6 tok/s** for a 26,758-token prompt and 1,024-token continuation ([PR #637](https://github.com/Luce-Org/lucebox/pull/637)) |

### Heterogeneous and parallel results

| Hardware | Configuration | Measured result |
|---|---|---|
| **2x RTX 3090 + NVLink** | Qwen 3.8 target tensor parallel + DFlash2 | **79.7 tok/s**, **2.16×** autoregressive decode ([PR #637](https://github.com/Luce-Org/lucebox/pull/637)) |
| **RX 7900 XT + Strix Halo** | DeepSeek V4 with all six experts + DSpark verification width 4 | **45.0 to 47.7 tok/s** decode; **111.2 tok/s** prefill at 132,981 tokens ([PR #604](https://github.com/Luce-Org/lucebox/pull/604#qualified-results)) |
| **R9700 + Strix Halo** | DeepSeek V4 across both AMD devices | **86 tok/s** decode; **788 tok/s** prefill at 2K |

These runs use different prompts, quantizations, and inference policies. They show which configurations work; they are not a cross-hardware ranking.

## Recommended Setups

See [Recommended server setups](server/docs/RECOMMENDED_SETUPS.md) for the model and hardware matrix, including single-GPU and mixed-GPU profiles.

The DS4 guide also documents the Strix long-context sparse-verifier profile and
Qwen3-0.6B PFlash integration. PFlash is lossy prompt compression; keep it off
for exact-retrieval and matched true-context benchmarks.

## Client Harnesses

[`harness/`](harness/) runs Lucebox through popular coding clients and checks server compatibility.

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

Set the server binary and model paths, then run a launcher:

```bash
DFLASH_SERVER_BIN=server/build/dflash_server \
DFLASH_TARGET=server/models/Qwen3.8-27B-UD-IQ4_XS.gguf \
DFLASH_DRAFT=server/models/draft/qwen38-dflash2-q8_0.gguf \
MAX_CTX=32768 \
harness/clients/run_codex.sh
```

See the [harness guide](harness/README.md) for setup, no-draft targets, and benchmarks.

## Quick Start With Docker

Prebuilt images on GHCR track `main`. Mount the weights and serve the OpenAI-compatible API on `:8000`.

<table>
<tr>
<td width="38%" valign="middle">

| GPU | Image tag |
|-----|-----------|
| NVIDIA (CUDA 12+) | `:cuda12` |
| AMD (ROCm 6+) | `:rocm` |

Put the target in `server/models/` and its matching drafter in `server/models/draft/`.

</td>
<td width="62%" valign="middle">

<img src="assets/docker.png" alt="Lucebox prebuilt Docker images for NVIDIA and AMD" width="100%" />

</td>
</tr>
</table>

Run the image for your GPU:

```bash
# NVIDIA
docker run --rm --gpus all -p 8000:8080 \
  -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:cuda12

# AMD
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -p 8000:8080 -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:rocm
```

## Run the Server

This quick start runs the R9700 profile above. The complete flag reference is in the [server guide](server/README.md#server-parameter-reference).

```bash
# build (ROCm 7.2+, RDNA4)
git clone --recurse-submodules https://github.com/Luce-Org/lucebox.git
cd lucebox
cmake -S server -B server/build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1201 \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON
cmake --build server/build-hip --target dflash_server -j"$(nproc)"

# target and DFlash2 drafter
mkdir -p models
huggingface-cli download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-IQ4_XS.gguf --local-dir models
huggingface-cli download incoai/Qwen3.8-27B-DFlash2 --local-dir models/dflash2
python server/scripts/convert_dflash_to_gguf.py \
  models/dflash2/model.safetensors models/qwen38-dflash2-f16.gguf
python server/scripts/quantize_dflash_draft.py \
  models/qwen38-dflash2-f16.gguf models/qwen38-dflash2-q8_0.gguf --scheme q8_0

# launch the measured profile
./server/build-hip/dflash_server models/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft models/qwen38-dflash2-q8_0.gguf \
  --draft-block-size 16 --max-ctx 131072 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --port 8216

curl -s http://127.0.0.1:8216/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python LRU cache."}],
       "max_tokens":256,"temperature":0}'
```

## Documentation

| Topic | Guide |
|---|---|
| Recommended model and hardware profiles | [Recommended setups](server/docs/RECOMMENDED_SETUPS.md) |
| Runtime parameters | [Server parameter reference](server/README.md#server-parameter-reference) |
| OpenAI Chat Completions, Responses, and Anthropic Messages | [API reference](server/docs/API.md) |
| CUDA, HIP, and mixed-device placement | [Mixed-backend guide](server/docs/MIXED_BACKEND.md) |
| DeepSeek V4 single-device and heterogeneous profiles | [DeepSeek V4 guide](server/docs/DS4.md) |
| Environment variables | [Environment reference](server/docs/ENVIRONMENT.md) |
| Server internals | [Architecture](server/docs/ARCHITECTURE.md) |
| Client integration and qualification | [Harness guide](harness/README.md) |

Benchmarks stay with each implementation: [DFlash](server/RESULTS.md), [PFlash](optimizations/pflash/), [Spark](optimizations/spark/), [KVFlash](optimizations/kvflash/), and [Megakernel](optimizations/megakernel/).

---

## Tutorials

Video tutorials for each optimization and the harness setup.

|   |   |   |
|:-:|:-:|:-:|
| **Luce Spark**<br>[▶ YouTube](https://www.youtube.com/watch?v=LB1aVj9lNhg) | **Luce DFlash**<br>[▶ YouTube](https://www.youtube.com/watch?v=vbPGvvSB8IQ) | **Luce Turboquant**<br>[▶ YouTube](https://www.youtube.com/watch?v=uTOOrfhrnBk) |
| **OpenClaw harness setup**<br>[▶ YouTube](https://www.youtube.com/watch?v=PysoxVGfvRE) | **Luce PFlash**<br>[▶ YouTube](https://www.youtube.com/watch?v=NWeKUL9Bc6Y) | **Luce Megakernel**<br>[▶ YouTube](https://www.youtube.com/watch?v=e6jY4goVIu0) |
| **Luce KVFlash**<br>[▶ YouTube](https://www.youtube.com/watch?v=8rTVCRWvRDo) |   |   |

---

## The Lucebox Machine

Local AI should be the default, not a privilege. Private data, no per-token bill, no vendor lock-in. Lucebox pairs the R9700 with Strix Halo and ships this open engine ready to run.

<p align="center">
  <a href="https://www.lucebox.com/"><img src="assets/lucebox.png" alt="Lucebox local AI PC" width="85%" /></a>
</p>

See the hardware and current benchmarks at [lucebox.com](https://www.lucebox.com/).

---

## Request for Contributions

We welcome focused contributions to CUDA and HIP kernels, speculative inference, support for more consumer GPUs and APUs, performance benchmarks, and client harnesses.

---

## Citation

```bibtex
@software{lucebox_2026,
  title  = {Lucebox: Speculative inference for heterogeneous consumer hardware},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox},
  year   = {2026}
}
```

---

## Community

- **Discord**: [discord.gg/yHfswqZmJQ](https://discord.gg/yHfswqZmJQ)
- **Website**: [lucebox.com](https://www.lucebox.com/)
- **Issues**: [github.com/Luce-Org/lucebox/issues](https://github.com/Luce-Org/lucebox/issues)
- **Blog**: [lucebox.com/blog](https://www.lucebox.com/blog/)

---

<p align="center">
  <sub><a href="LICENSE">Apache 2.0</a> · <a href="https://www.lucebox.com/">Lucebox.com</a></sub>
</p>
