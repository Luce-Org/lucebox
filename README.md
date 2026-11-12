<p align="center">
  <a href="https://www.lucebox.com/">
    <img src="assets/logo-mark.png" alt="Lucebox" width="160">
  </a>
</p>

<h3 align="center">Fast local inference on heterogeneous consumer hardware</h3>

<p align="center">
  <a href="#models">Models</a> · <a href="#quick-start"><b>Quick start</b></a> · <a href="#hardware">Hardware</a> · <a href="#documentation">Docs</a> · <a href="https://www.lucebox.com/blog/">Blog</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a>
</p>

<p align="center">
  <a href="https://www.lucebox.com/"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=f5c842&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://huggingface.co/Lucebox"><img src="https://img.shields.io/badge/HuggingFace-f5c842?style=for-the-badge&logo=huggingface&logoColor=f5c842&labelColor=090909" alt="HuggingFace"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=f5c842&labelColor=090909" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-e8e8ed?style=for-the-badge&labelColor=090909" alt="Apache 2.0"></a>
</p>

Lucebox is an open-source inference engine for AMD's R9700 and Ryzen AI MAX+, on their own or together in one machine. It runs Qwen 3.8, DeepSeek V4 and V4.1, and MOSS-Image, and it generates faster than llama.cpp, Strata and DwarfStar on the same hardware. It's the engine inside every [Lucebox](https://www.lucebox.com/).

<p align="center">
  <a href="https://www.lucebox.com/engine"><img src="assets/engine-head-to-head.png" alt="Lucebox against llama.cpp, Strata and DwarfStar: decode, prefill and image generation" width="100%" /></a>
  <sub>"Whole box" is the R9700 and the Ryzen AI MAX+ together. <a href="https://www.lucebox.com/engine">Full results</a></sub>
</p>

## Features

Lucebox is fast with:

- Speculative decoding on every model: DFlash2 for Qwen 3.8 27B, MTP for Flash-Next, DSpark for DeepSeek
- Two chips as one machine: attention, KV cache and the busiest experts on the R9700, the other experts on the Ryzen AI MAX+
- Paged attention and continuous batching for several users
- A prefix cache built for agents, up to 92.2% hit rate on [AgentX](https://github.com/Luce-Org/lucebox/pull/989)
- [PFlash](#long-prompts): long prompts compressed before prefill, first token 3.7x sooner on average
- [KVFlash](#long-prompts): KV cache paged to system memory, decode speed flat from 64K to 256K tokens
- HIP kernels tuned for RDNA4 (`gfx1201`) and RDNA3.5 (`gfx1151`), plus a CUDA build for NVIDIA

and easy to use with:

- OpenAI Chat Completions and Responses, and the Anthropic Messages API
- Claude Code, Codex, OpenCode, Hermes, [OMP](https://omp.sh/), Pi, OpenClaw and Open WebUI, out of the box
- Image input on Qwen 3.8 27B and DeepSeek V4 and V4.1, and image generation with MOSS-Image
- Launch profiles for each model, picked from the GPUs it finds

## Models

| Model | Runs on | Speed | Files |
|---|---|---|---|
| [Qwen 3.8 27B](https://www.lucebox.com/blog/qwen38-r9700) | R9700, up to 8 users | 95.4 tok/s | [target](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ4_XS.gguf), [drafter](https://huggingface.co/Lucebox/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q8_0.gguf) |
| [Qwen 3.8 27B Vision](https://www.lucebox.com/blog/vision-llm-inference) | R9700, up to 8 users | 8 users 2.4x faster than llama.cpp on a DGX Spark | [target](https://huggingface.co/Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF/blob/main/Qwen3.8-27B-IQ4_XS-pure.gguf), [vision](https://huggingface.co/Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF/blob/main/Qwen3.8-27B-mmproj-Q8_0.gguf) |
| [Qwen 3.8 Flash-Next](server/docs/QWEN4EXP.md) | R9700 + Ryzen AI MAX+, or the Ryzen AI MAX+ alone, up to 4 users | 81.5 tok/s | [target](https://huggingface.co/Lucebox/Qwen3.8-Flash-Next-GGUF/blob/main/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf), [MTP](https://huggingface.co/Lucebox/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) |
| [DeepSeek V4.1 Flash](server/docs/DS41.md) | R9700 + Ryzen AI MAX+ | 25.4 tok/s on 128 GB, 51.1 on 192 GB | [target](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF/blob/main/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf), drafter for [128 GB](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF/blob/main/DeepSeek-V4.1-Flash-1017-DSpark-draft-MXFP4-Q8.gguf) or [192 GB](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF/blob/main/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf) |
| [DeepSeek V4 Flash](server/docs/DS4.md) | Ryzen AI MAX+ | 27.9 tok/s | [target](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3/blob/main/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf), [drafter](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF/blob/main/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf) |
| [MOSS-Image 5B](optimizations/image_gen/README.md) | R9700 | 2.74 s per 1024x1024 image | [model](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_bf16.safetensors) |

Every model runs on both the 128 GB and the 192 GB Ryzen AI MAX+. All the language models take PFlash, and the Qwen models KVFlash.

## Hardware

Lucebox is built for two AMD chips, alone or in the same machine:

- Radeon AI PRO R9700 (RDNA4, `gfx1201`), 32 GB
- Ryzen AI MAX+ 395 or 495 (RDNA3.5 iGPU, `gfx1151`), 128 or 192 GB of unified memory

Together, the R9700 runs what every token touches and the iGPU holds what doesn't fit on the card:

| Model | On the R9700 | On the Ryzen AI MAX+ |
|---|---|---|
| Qwen 3.8 Flash-Next | attention, KV cache, dense layers, MTP drafter | the routed experts, 72 GiB |
| DeepSeek V4.1 Flash, 128 GB | dense layers, drafter, about 10 GiB of the busiest experts | the next busiest experts; the rest stream from the SSD |
| DeepSeek V4.1 Flash, 192 GB | dense layers, drafter, about 12 GiB of the busiest experts | every other expert, about 150 GiB |
| Qwen 3.8 27B | the whole model and its drafter | free for a second model |

The R9700 can sit in a PCIe slot or an OCuLink dock; a x4 link is enough.

| <img src="assets/gpus/r9700.png" width="180" /> | <img src="assets/gpus/ryze395.png" width="180" /> | <img src="assets/gpus/gb10.png" width="180" /> | <img src="assets/gpus/3090.png" width="180" /> |
|:---:|:---:|:---:|:---:|
| R9700, ROCm 7.2 | Ryzen AI MAX+ 395 / 495, ROCm 7.2 | DGX Spark (GB10), CUDA 12.9 | RTX 3090, CUDA 12+ |

The CUDA build runs on NVIDIA GPUs from Volta onward, and CI tests it on a DGX Spark. The HIP build also runs on the RX 7900 XT and XTX. Community results: [RTX 4090 and 2080 Ti](server/RESULTS.md), [RTX 5090 and two RTX 3090s](https://github.com/Luce-Org/lucebox/pull/637).

## Quick start

You need Linux with ROCm 7.2 or newer. Only have one of the two chips? See [one chip or another card](#one-chip-or-another-card).

**1. Build for both chips.**

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox.git
cd lucebox
cmake -S server -B server/build-hip -DCMAKE_BUILD_TYPE=Release \
  -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES="gfx1201;gfx1151"
cmake --build server/build-hip --target luce_server -j"$(nproc)"
```

For Flash-Next, add `-DGGML_HIP_GRAPHS=ON`.

**2. Give the iGPU its memory.** By default Linux lets the iGPU use only part of the RAM. Add these to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub` and reboot:

```text
# 128 GB: 96 GiB for the iGPU
amdgpu.gttsize=98304 ttm.pages_limit=25165824

# 192 GB: 176 GiB for the iGPU
amdgpu.gttsize=180224 ttm.pages_limit=46137344
```

**3. Check the GPU order.**

```bash
./server/build-hip/luce_server --list-devices
```

The profiles expect the R9700 as `hip:0` and the iGPU as `hip:1`. External cards sometimes show up the other way round; if so, prefix the commands below with `HIP_VISIBLE_DEVICES=1,0`.

**4. Start a model.** The server picks the profile that matches your GPUs and listens on port 8080.

```bash
# Qwen 3.8 Flash-Next: attention and the MTP drafter on the R9700, experts on the iGPU
hf download Lucebox/Qwen3.8-Flash-Next-GGUF \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf \
  MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --local-dir models/flash-next
./server/build-hip/luce_server \
  models/flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --draft models/flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --profile qwen-next-r9700-strix
```

```bash
# DeepSeek V4.1 Flash, 128 GB: busiest experts on the R9700, more on the iGPU, the rest on the SSD
hf download Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF \
  DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  DeepSeek-V4.1-Flash-1017-DSpark-draft-MXFP4-Q8.gguf --local-dir models/ds41
ulimit -l unlimited
./server/build-hip/luce_server models/ds41/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --draft models/ds41/DeepSeek-V4.1-Flash-1017-DSpark-draft-MXFP4-Q8.gguf

# DeepSeek V4.1 Flash, 192 GB: busiest experts on the R9700, every other expert on the iGPU
hf download Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF \
  DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf --local-dir models/ds41
./server/build-hip/luce_server models/ds41/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --draft models/ds41/DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf
```

```bash
# Qwen 3.8 27B: the model and its DFlash2 drafter on the R9700
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ4_XS.gguf --local-dir models
hf download Lucebox/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q8_0.gguf --local-dir models
./server/build-hip/luce_server models/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft models/Qwen3.8-27B-DFlash2-Q8_0.gguf
```

```bash
# DeepSeek V4 Flash: on the iGPU
hf download Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3 \
  DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf --local-dir models
hf download Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF \
  DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf --local-dir models
./server/build-hip/luce_server models/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf \
  --draft models/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf \
  --target-device auto --profile ds4-strix
```

Try it:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python LRU cache."}],"max_tokens":256}'
```

For more than one user, add:

| Model | Flags |
|---|---|
| Qwen 3.8 27B | `--paged-attention --max-concurrency 8 --kv-pool-tokens 204800` |
| Qwen 3.8 Flash-Next | `--max-concurrency 4` |
| DeepSeek V4.1 Flash | `--paged-attention --max-concurrency 4 --ds4-prefill sparse` in place of `--draft` |

### One chip or another card

On an R9700 alone, run the 27B as above. On a Ryzen AI MAX+ alone, run Flash-Next without `--profile` and DeepSeek V4 as above; DeepSeek V4.1 fits on the iGPU alone with 192 GB. [Recommended setups](server/docs/RECOMMENDED_SETUPS.md) has every single-chip launch.

With an RX 7900 XT or XTX next to the iGPU, add `gfx1100` to the build and set the devices yourself (`--target-device hip:0 --expert-device hip:1`); DeepSeek V4 has a [tested recipe](https://github.com/Luce-Org/lucebox/pull/604) for this pair. An NVIDIA card can split a model's layers with the iGPU through the [mixed CUDA and HIP build](server/docs/MIXED_BACKEND.md).

### Docker

Prebuilt images track `main`: `:rocm` for AMD, `:cuda12` for NVIDIA. Put the model in `server/models/` and its drafter in `server/models/draft/`; `luce_server` flags like `--profile` go after the image name.

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -p 8000:8080 -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:rocm

docker run --rm --gpus all -p 8000:8080 \
  -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:cuda12
```

For compose, see the header of [`server/scripts/entrypoint.sh`](server/scripts/entrypoint.sh).

## Long prompts

PFlash reads a long prompt with a small model first and gives the main model only the parts that matter for the question ([how it works](https://www.lucebox.com/blog/pflash-v2)). KVFlash keeps part of the KV cache on the GPU and the rest in system memory, so decode speed holds as the context grows ([how it works](https://www.lucebox.com/blog/kvflash)).

```bash
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-BF16.gguf --local-dir models/pflash
hf download Lucebox/pflash-scorer-head seg-scoring-head.gguf --local-dir models/pflash
hf download Lucebox/pflash-segment-probe segment-probe.gguf --local-dir models/pflash
hf download Lucebox/pflash-request-probe request-probe.gguf --local-dir models/pflash

# PFlash: add to any launch above
PFLASH_SCORING_HEAD_GGUF=models/pflash/seg-scoring-head.gguf \
PFLASH_SEGMENT_PROBE_GGUF=models/pflash/segment-probe.gguf \
PFLASH_REQUEST_PROBE_GGUF=models/pflash/request-probe.gguf \
./server/build-hip/luce_server <model and flags as above> \
  --prefill-drafter models/pflash/Qwen3.5-0.8B-BF16.gguf

# KVFlash: sizes itself from free GPU memory
./server/build-hip/luce_server <model and flags as above> \
  --prefill-drafter models/pflash/Qwen3.5-0.8B-BF16.gguf --kvflash auto
```

With both chips, `--prefill-drafter-device hip:1` runs the small model on the iGPU. PFlash starts at 16K tokens and `--pflash off` turns it off. Every setting is in the [PFlash reference](server/docs/PFLASH_CONFIGURATION.md).

## Coding agents

The server speaks the OpenAI and Anthropic APIs, so coding agents connect to it as they are. [`harness/`](harness/) has a launcher for each, and there's a [video of the OpenClaw setup](https://www.youtube.com/watch?v=PysoxVGfvRE):

```bash
LUCE_SERVER_BIN=server/build-hip/luce_server \
LUCE_TARGET=models/Qwen3.8-27B-UD-IQ4_XS.gguf \
LUCE_DRAFT=models/Qwen3.8-27B-DFlash2-Q8_0.gguf \
MAX_CTX=32768 \
harness/clients/run_codex.sh
```

## Documentation

| Topic | Guide |
|---|---|
| Model and hardware profiles | [Recommended setups](server/docs/RECOMMENDED_SETUPS.md) |
| DeepSeek V4.1 Flash | [DeepSeek V4.1 guide](server/docs/DS41.md) |
| DeepSeek V4 Flash | [DeepSeek V4 guide](server/docs/DS4.md) |
| Qwen 3.8 Flash-Next | [Flash-Next guide](server/docs/QWEN4EXP.md) |
| Long prompts | [PFlash reference](server/docs/PFLASH_CONFIGURATION.md), [prefix cache](server/docs/PREFIX_CACHE.md) |
| Images | [Image input](docs/image-input.md), [MOSS-Image](optimizations/image_gen/README.md) |
| APIs | [API reference](server/docs/API.md) |
| Flags and environment | [Server parameters](server/README.md#server-parameter-reference), [environment](server/docs/ENVIRONMENT.md) |
| CUDA, HIP and two-chip placement | [Mixed-backend guide](server/docs/MIXED_BACKEND.md) |
| How the server is built | [Architecture](server/docs/ARCHITECTURE.md), [engine components](server/docs/ENGINE_COMPONENTS.md) |

## Research

[`optimizations/`](optimizations/) holds the research the engine grew out of: the [Megakernel](optimizations/megakernel/), [Luce Spark](optimizations/spark/), and the first [PFlash](optimizations/pflash/) and [KVFlash](optimizations/kvflash/) experiments. Video walkthroughs: [Spark](https://www.youtube.com/watch?v=LB1aVj9lNhg), [DFlash](https://www.youtube.com/watch?v=vbPGvvSB8IQ), [Turboquant](https://www.youtube.com/watch?v=uTOOrfhrnBk), [PFlash](https://www.youtube.com/watch?v=NWeKUL9Bc6Y), [Megakernel](https://www.youtube.com/watch?v=e6jY4goVIu0), [KVFlash](https://www.youtube.com/watch?v=8rTVCRWvRDo).

## The Lucebox machine

Lucebox Zero 495 pairs the R9700 with the Ryzen AI MAX+ PRO 495 (128 or 192 GB) and ships with this engine ready to run: private data, no per-token bill.

<p align="center">
  <a href="https://www.lucebox.com/"><img src="assets/lucebox.png" alt="Lucebox Zero 495" width="85%" /></a>
</p>

## Contributing

We welcome work on HIP and CUDA kernels, speculative decoding, new consumer GPUs and APUs, benchmarks and client harnesses. Open an issue or join us on [Discord](https://discord.gg/yHfswqZmJQ).

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

<p align="center">
  <sub><a href="LICENSE">Apache 2.0</a> · <a href="https://www.lucebox.com/">lucebox.com</a> · <a href="https://www.lucebox.com/blog/">Blog</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://github.com/Luce-Org/lucebox/issues">Issues</a></sub>
</p>
