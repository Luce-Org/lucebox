<p align="center">
  <img src="assets/banner.png" alt="Lucebox" width="85%">
</p>

<p align="center">
  <a href="https://lucebox.com"><img src="https://img.shields.io/badge/lucebox.com-f5c842?style=for-the-badge&logo=safari&logoColor=f5c842&labelColor=090909" alt="lucebox.com"></a>
  <a href="https://huggingface.co/Lucebox"><img src="https://img.shields.io/badge/HuggingFace-f5c842?style=for-the-badge&logo=huggingface&logoColor=f5c842&labelColor=090909" alt="HuggingFace"></a>
  <a href="https://discord.gg/yHfswqZmJQ"><img src="https://img.shields.io/badge/Discord-f5c842?style=for-the-badge&logo=discord&logoColor=f5c842&labelColor=090909" alt="Discord"></a>
  <a href="https://lucebox.com/blog"><img src="https://img.shields.io/badge/Blog-f5c842?style=for-the-badge&logo=rss&logoColor=f5c842&labelColor=090909" alt="Blog"></a>
  <a href="#tutorials"><img src="https://img.shields.io/badge/Tutorials-f5c842?style=for-the-badge&logo=youtube&logoColor=f5c842&labelColor=090909" alt="Tutorials"></a>
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

## Inference Engine Optimizations

Each one is self-contained with setup instructions and benchmark notes.

<p align="center">
  <a href="optimizations/megakernel/"><img src="assets/cards/megakernel_card.png" alt="Megakernel" width="46%"></a>
  &nbsp;&nbsp;
  <a href="server/"><img src="assets/cards/dflash_card.png" alt="DFlash 27B" width="46%"></a>
</p>

<p align="center">
  <a href="optimizations/pflash/"><img src="assets/cards/pflash_card.png" alt="PFlash speculative prefill" width="46%"></a>
  &nbsp;&nbsp;
  <a href="optimizations/spark/"><img src="assets/cards/spark_card.png" alt="Luce Spark MoE expert offload" width="46%"></a>
</p>

<p align="center">
  <a href="optimizations/kvflash/"><img src="assets/cards/kvflash_card.png" alt="Luce KVFlash paged KV cache" width="46%"></a>
</p>

---

## Supported Models & Drafters

All speedups measured vs vendored llama.cpp (`-fa 1`, matching KV quant). Combined = geometric mean √(TTFT × decode) where both phases benched; otherwise the single-phase speedup. Drafters published on [huggingface.co/Lucebox](https://huggingface.co/Lucebox).

<table>
<tr>
<td valign="top">

| Model | Speedup |
|-------|:-------:|
| Qwen 3.5-0.8B (Megakernel) | **~2×** |
| Qwen 3.6-27B + PFlash | **~5.6×** |
| Qwen 3.6-27B + DDTree | **4.84×** |
| Laguna-XS-2.1 33B + PFlash | **8.2×** @256K |
| Laguna-XS-2.1 33B + DFlash | **1.7×** @256K |
| Qwen 3.6-27B HIP | **~2.6×** |
| Gemma-4-26B-A4B | **1.31×** |

</td>
<td valign="top">

| Drafter | Phase |
|---------|:-----:|
| [`Qwen3.6-27B`](https://huggingface.co/Lucebox/Qwen3.6-27B-DFlash-GGUF) | decode |
| [`gemma-4-26B-A4B`](https://huggingface.co/Lucebox/gemma-4-26B-A4B-it-DFlash-GGUF) | decode |
| [`gemma-4-31B`](https://huggingface.co/Lucebox/gemma-4-31B-it-DFlash-GGUF) | decode |
| [`Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | prefill |

</td>
</tr>
</table>

## Tested Machines (GPU/APU)

Reference target: **RTX 3090 (Ampere sm_86)** — all headline numbers. Other NVIDIA archs auto-detected by CMake / `setup.py`; AMD HIP backend separate ([Strix Halo section](#amd-strix-halo-hip-backend)).

| | Arch | GPU | Min CUDA / ROCm | Status | Bench |
|:---:|------|-----|:---------------:|--------|:-----:|
| <img src="assets/gpus/3090.png" width="750" /> | Ampere `sm_86` | RTX 3090, A-series | CUDA 12.0 | ✅ reference | [megakernel](optimizations/megakernel/RESULTS.md#rtx-3090-pp520-tg128) · [dflash](server/RESULTS.md) |
| <img src="assets/gpus/5090.png" width="750" /> | Blackwell `sm_120` | RTX 5090 | CUDA 12.8 | ✅ 205 tok/s, 4.84× | [↗](server/RESULTS.md#rtx-5090-blackwell-sm_120sm_120a-32-gb) |
| <img src="assets/gpus/gb10.png" width="750" /> | Blackwell `sm_121` | DGX Spark / GB10 | CUDA 12.9 | ✅ megakernel NVFP4 | [↗](optimizations/megakernel/RESULTS.md#nvidia-dgx-spark-gb10-sm_121a) |
| <img src="assets/gpus/2080ti.png" width="750" /> | Turing `sm_75` | RTX 2080 Ti | CUDA 12.0 | ✅ 53 tok/s DFlash | [↗](server/RESULTS.md#rtx-2080-ti-turing-sm_75-22-gb) |
| <img src="assets/gpus/4090.png" width="750" /> | Ada `sm_89` | RTX 40xx | CUDA 12.0 | 🟡 community WSL2 bench | [↗](server/RESULTS.md#rtx-4090-ada-sm_89-24-gb--wsl2-community) |
| — | Blackwell `sm_110` | Jetson AGX Thor | CUDA 13.0 | 🟡 builds, unbenched | — |
| <img src="assets/gpus/v100.png" width="750" /> | Volta `sm_70` / Pascal `sm_61` | V100, P40 | CUDA 12.0 | 🟡 fallback paths, unbenched | — |
| <img src="assets/gpus/ryze395.png" width="750" /> | RDNA3.5 `gfx1151` | Ryzen AI MAX+ 395 / Strix Halo | ROCm 6+ | ✅ 37 tok/s HIP | [↗](server/README.md#amd-hip-backend-strix-halo-rx-7900-xtx) |
| <img src="assets/gpus/7900xtx.png" width="750" /> | RDNA3 `gfx1100` | Radeon RX 7900 XTX | ROCm 6+ | ✅ 50 tok/s HIP | [↗](server/README.md#amd-hip-backend-strix-halo-rx-7900-xtx) |
| — | RDNA4 `gfx1201` | Radeon AI PRO R9700 | ROCm 6.4+ | ✅ 55 tok/s HIP | [↗](server/README.md#amd-hip-backend-strix-halo-rx-7900-xtx) |

`server/` (DFlash) builds with CMake 3.18+ and vendors the required `ggml` sources directly; only `Block-Sparse-Attention` remains a git submodule. No PyTorch is needed for `server/`. `optimizations/megakernel/` is the only component requiring PyTorch 2.0+ (CUDAExtension links against torch C++ libs). Power-tune: `sudo nvidia-smi -pl 220` (3090 sweet spot, re-sweep for other cards).

## Quick Start On Harnesses

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
DFLASH_TARGET=server/models/Qwen3.6-27B-Q4_K_M.gguf \
DFLASH_DRAFT=server/models/draft/dflash-draft-3.6-q4_k_m.gguf \
MAX_CTX=32768 BUDGET=22 VERIFY_MODE=ddtree \
harness/clients/run_codex.sh
```

For no-draft targets such as Gemma, set only `DFLASH_TARGET` or pass
`DRAFT=none`; the harness will not attach the default Qwen draft to a custom
target.

Launcher scripts install missing real-client CLIs automatically under
`.harness-work/`. To preinstall them yourself:

```bash
python3 harness/client_test_runner.py install --clients codex,hermes,openwebui
```

For direct TPS/TTFT numbers against a running server:

```bash
python3 harness/client_test_runner.py bench \
  --url http://127.0.0.1:8000 \
  --suite he,agent \
  --n-sample 3
```

## Quick Start With Docker

Prebuilt images on GHCR track `main`. No CUDA toolkit or build needed. Pull the image, mount weights and serve. OpenAI-compatible API on `:8000`.

<table>
<tr>
<td width="38%" valign="middle">

| GPU | Image tag |
|-----|-----------|
| NVIDIA (CUDA 12+) | `:cuda12` |
| AMD (ROCm 6+) | `:rocm` |

Drop a GGUF model target into `server/models/` first, then
`:8000/v1/chat/completions`. Full tutorial in the
[Docker blog](https://lucebox.com/blog/docker).

</td>
<td width="62%" valign="middle">

<a href="https://lucebox.com/blog/docker"><img src="assets/docker.png" alt="Lucebox prebuilt Docker images for NVIDIA and AMD" width="100%" /></a>

</td>
</tr>
</table>

**Install and run:**

```bash
# 1. Pull the image for your GPU
docker pull ghcr.io/luce-org/lucebox-hub:cuda12   # NVIDIA
docker pull ghcr.io/luce-org/lucebox-hub:rocm     # AMD

# 2. Download a target model into server/models/ and the DFlash draft
#    into server/models/draft/ (the entrypoint only auto-discovers the
#    draft there; without it the server runs slower, target-only)
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf \
  --local-dir server/models/
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf \
  --local-dir server/models/draft/

# 3a. NVIDIA (CUDA 12+)
docker run --rm --gpus all -p 8000:8080 \
  -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:cuda12

# 3b. AMD (ROCm 6+, Strix Halo / RX 7900)
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -p 8000:8080 -v "$PWD/server/models:/opt/lucebox-hub/server/models" \
  ghcr.io/luce-org/lucebox-hub:rocm
```

Then hit `:8000/v1/chat/completions` (OpenAI-compatible).

## Run the Server

Default: Qwen 3.6-27B Q4_K_M target + Lucebox Q4_K_M DFlash drafter on RTX 3090. DDTree budget=22, TQ3_0 KV cache, full attention. OpenAI-compatible HTTP on `:8000`.

```bash
# build (CUDA 12+, CMake 3.18+)
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub && cd lucebox-hub
cmake -B server/build -S server -DCMAKE_BUILD_TYPE=Release
cmake --build server/build --target dflash_server -j

# default weights (~18 GB)
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir server/models/
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir server/models/draft/

# run (TQ3_0 KV auto-enabled; set =0 to disable)
DFLASH27B_KV_TQ3=1 \
./server/build/dflash_server server/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/draft/dflash-draft-3.6-q4_k_m.gguf \
  --ddtree --ddtree-budget 22 --port 8000
```

### Making requests

For the fastest, deterministic responses send `temperature: 0` (greedy decoding gives the
highest spec-decode acceptance):

```bash
curl :8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dflash",
  "messages": [{"role": "user", "content": "Write quicksort in Python."}],
  "temperature": 0
}'
```

Requests that omit `temperature` use the model card's sampling (Qwen3.6: `temperature: 1.0`,
`top_p: 0.95`, `top_k: 20`).

### Server flags

**Core**

| Flag | Default | Effect |
|---|---|---|
| `--draft <path>` | — | DFlash draft GGUF, required for speculative decode |
| `--port N` | `8000` | HTTP port |
| `--host H` | `127.0.0.1` | Bind address |
| `--max-ctx N` | auto-fit | KV cache size; oversizing slows prefill (FA stride over unused KV) |
| `--max-tokens N` | model-card | Generation cap |
| `--model-name S` | filename | OpenAI `model` field |
| `--chat-template-file <path>` | autodetect | Override Jinja template |

**Decode (DFlash + DDTree)**

| Flag | Default | Effect |
|---|---|---|
| `--ddtree` | off (chain) | Enable tree verify |
| `--ddtree-budget N` | `22` | Tree size. 22 on 3090 (default), 40 on 5090, re-sweep on GB10 |
| `--fa-window N` | `0` / `2048` (full attention) | Sliding FA window. Leave at 0: a finite window breaks tool calls (the full-attention layers lose the system prompt/tools). |
| `--draft-residency {auto,persistent,request-scoped}` | `auto` | When draft weights are evicted from VRAM. `request-scoped` parks/frees them after each request's draft work (frees VRAM for the target on tight GPUs); `persistent` keeps them resident across requests; `auto` preserves current behavior while honoring the low-VRAM / `--lazy-draft` hint. Reported at `/props.runtime.draft_residency`. |
| `--lazy-draft` | off | Legacy alias for `--draft-residency=request-scoped` (defer draft load until first request, release after) |

**GPU draft top-K & verify-argmax (DFlash)**

The draft-token top-K extraction and the per-step verify argmax used to run on the CPU, each requiring a full `vocab × n_tokens` logits copy from device to host (D2H) every speculation step. These two env flags move both onto the GPU, reading the logits in place on the device buffer and skipping the bulk D2H. Both are **on by default in the server** (the `test_dflash` harness defaults `DFLASH_GPU_VERIFY_ARGMAX` to off, see the table below) and take effect on **both CUDA and HIP/ROCm builds**: the draft top-K uses a custom device kernel (`geometric_draft_topk_cuda.cu`, the same source compiled directly for HIP) and the verify argmax reads an in-graph `ggml_argmax` node, so neither depends on a CUDA-only path. Each path validates its result and **falls back to the legacy CPU computation automatically** on any failure (e.g. an out-of-range index), so disabling them is only needed for debugging or A/B comparison.

| Env | Default | Effect |
|---|---|---|
| `DFLASH_GPU_DRAFT_TOPK=1` | `1` (on) | Compute the draft model's top-K vocab indices (K in 1–8) and log-sum-exp directly on the logits device buffer. `=0` forces the legacy CPU top-K (full-vocab D2H + CPU heap extract). Use `=0` to isolate the kernel when debugging or to baseline its speedup. |
| `DFLASH_GPU_VERIFY_ARGMAX` | on (server) / `0` (test harness) | Per-step verify argmax. In the server it is on by default and a simple on/off; `=0` forces the legacy CPU path. In `test_dflash` it is a tri-state with these values: <br>• `0` — legacy CPU path: full `vocab × N` D2H + CPU argmax (default in the test harness). <br>• `1` — GPU fast-path: read the in-graph batched GPU argmax (N int32s), no bulk D2H. <br>• `2` — run **both** the CPU and GPU paths and report any per-step mismatches (validation mode; guards against the historical tree-verify `-1`/tie regression). |

To reproduce the benchmark: baseline `DFLASH_GPU_DRAFT_TOPK=0 DFLASH_GPU_VERIFY_ARGMAX=0`, optimized `DFLASH_GPU_DRAFT_TOPK=1 DFLASH_GPU_VERIFY_ARGMAX=1`, both via `python server/scripts/bench_llm.py --bench HumanEval`.

| Env | Default | Effect |
|---|---|---|
| `DFLASH_TOPK_SPLIT=N` | auto-tuned | Override the split-K factor (blocks per draft position) for the GPU top-K kernel; auto-tune aims for ~240 total blocks across the device. Useful to re-sweep on a GPU with a different SM count. |
| `DFLASH_TOPK_PROFILE=1` | off | Print per-launch CUDA event timing (partial pass + combine pass) for the GPU top-K kernel to stderr. |

**GPU sampler (DFlash)**

The CPU `sample_logits` chain (repetition/frequency/presence penalty → softmax(temp) → top_p nucleus → multinomial draw) requires a full vocab-wide D2H logits copy every token. `geometric_sampler_cuda.cu` ports penalty application, the softmax reductions, and the draw onto the GPU, reading logits straight off the device tensor in the qwen35 decode loop (skipping that D2H). It's **on by default** at runtime on CUDA builds (opt out with `DFLASH_GPU_SAMPLE=0`).

Coverage is config-dependent, chosen by measurement rather than what's merely *possible* on the GPU:
- **Greedy and plain temperature/penalty sampling** (no `top_k`/`top_p` truncation) run entirely on the GPU — one kernel launch does penalties, softmax, and the multinomial draw, then a 4-byte D2H copy for the result.
- **Pure `top_p` nucleus sampling** (no `top_k`) is GPU-*assisted*: the GPU computes penalties+softmax and hands back the normalized probability vector, and the CPU does the nucleus search (`std::nth_element`-based binary search — O(vocab), not the O(vocab log vocab) a full sort would cost) on that already-computed vector.
- **`top_k` (with or without `top_p`)** always stays on the CPU. Its cost is already cheap — `partial_sort` scales with `k`, not vocab — so a GPU round trip (kernel launch + D2H copy) measured as a net *regression*, not just a non-win.

Per-call sampler-only latency at the Qwen3 vocab (151,936), measured on an RTX 3090 (GPU column reflects `DFLASH_GPU_SAMPLE=1`, CPU column reflects `=0`):

| Config | CPU | GPU | Speedup |
|---|---|---|---|
| greedy (temp=0) | 746 µs | 139 µs | ~5.4× |
| temp=0.8 (no truncation) | 1092 µs | 235 µs | ~4.6× |
| temp=0.8 + top_p=0.9 | 4915 µs | 3504 µs | ~1.4× (GPU-assisted) |
| temp=0.8 + top_k=40 | 283 µs | 273 µs | ~1.0× (top_k stays CPU-only either way) |

| Env / flag | Default | Effect |
|---|---|---|
| `DFLASH_GPU_SAMPLE=0` | on | Opt out of the GPU `sample_logits` path at runtime (on by default on CUDA builds). Falls back to the CPU chain per call when the config is unsupported or on any CUDA error. |
| `DFLASH_GPU_SAMPLER` (CMake option) | `ON` | Build-time switch; compiles `src/common/geometric_sampler_cuda.cu` into `dflash_common`. Configure with `-DDFLASH_GPU_SAMPLER=OFF` to drop the kernel entirely. |
| `--samp=temp,top_p,top_k,rep_pen,seed[,freq,pres]` **(for `test_dflash`)** | greedy | Exercise the sampler chain (and its GPU port, gated by `DFLASH_GPU_SAMPLE`) in the positional (non-daemon) harness instead of greedy decode. Same field order as the daemon's ` samp=` request-line tail. |
| `DFLASH_SAMP=temp,top_p,top_k,rep_pen,seed[,freq,pres]` **(for `bench_llm.py`)** | off (greedy) | Forward the same sampler tail to every DFlash bench call instead of greedy decode. AR (`test_generate`) is greedy-only and ignores this. |
| `DFLASH_N_SAMPLE=N` **(for `bench_llm.py`)** | `10` | Overrides how many prompts are drawn per benchmark dataset. |

End-to-end repro: `DFLASH_SAMP=0.8,1.0,0,1.1,42 python server/scripts/bench_llm.py --bench HumanEval` (GPU sampler on by default) vs the same command with `DFLASH_GPU_SAMPLE=0` (CPU-only).

**Prefill compression (PFlash)**

| Flag / env | Default | Effect |
|---|---|---|
| `--prefill-compression {off,auto,always}` | `off` | When to score+compress the prompt |
| `--prefill-threshold N` | `32000` | In `auto`, the prompt-token count above which a single-shot prompt is compressed. Also the per-message minimum that an aged message must exceed before FlowKV compresses it on multi-turn requests. Lower it (e.g. `1024`) if you want FlowKV to act on shorter history. |
| `--prefill-keep-ratio F` | `0.05` | Fraction of source tokens kept (0.02 @128K, 0.10 @32K) |
| `--prefill-curve T:R [T:R ...]` | off (flat keep-ratio) | Piecewise keep-ratio curve, linear-interpolated over `(tokens, ratio)` breakpoints, e.g. `10000:0.5 40000:0.2 100000:0.1` (2× compression @10K, 5× @40K, 10× @100K+). Overrides `--prefill-keep-ratio`; per-session bandit override still wins. |
| `--prefill-drafter <gguf>` | required if on | Drafter weights (Qwen3-0.6B BF16 GGUF) |
| `--prefill-skip-park` | off | Keep drafter resident across requests (more VRAM, faster) |
| `PFLASH_FREEZE_HOT_WINDOW=N` | `2` | FlowKV: how many of the most recent messages stay verbatim. Everything older than this window (but after the system prompt) is compressed once and cached. Larger = more recent context kept uncompressed. |
| `DFLASH_FP_USE_BSA=1` | `0` | Dispatch sparse FA through BSA (sm_80+); required for headline 10.4× |
| `DFLASH_FP_ALPHA=0.85` | `0.12` | Block-selection threshold; higher = stricter = fewer K-blocks |
| `DFLASH_FP_PROFILE=1` | `0` | Per-stage timing log |

When compression is on, the request path picks one of three modes automatically, so they never stack: the first turn is sent verbatim (the system prompt stays as a stable cache anchor), multi-turn continuations use **FlowKV** (only the aged history is compressed, recent turns kept verbatim, so the disk prefix cache from `--prefix-cache-slots` keeps hitting), and a single oversized prompt with no prior turns uses whole-prompt PFlash. With `--prefill-compression off` the request path is identical to a build without compression.

**KV cache**

| Flag / env | Default | Effect |
|---|---|---|
| `--cache-type-k <t>` / `--cache-type-v <t>` | env-driven | Per-side quant override: `f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0,tq3_0` |
| `DFLASH27B_KV_TQ3=1` | (default) | Preset TQ3_0 K+V (3.5 bpv, fits 256K @ 24 GB) |
| `DFLASH27B_KV_Q4=1` | off | Q4_0 K+V (4.5 bpv, legacy, ~128K ceiling) |
| `--prefix-cache-slots N` | — | Live prefix-cache slot count |
| `--kv-cache-dir <path>` | — | Persist prefix cache to disk |
| `--kv-cache-budget N` | — | On-disk cache size cap |

**Bounded KV residency (KVFlash)**

Pages the attention KV cache through a fixed pool of GPU slots; cold 64-token chunks live in host RAM, bit-exact and recallable. Decode speed stops depending on context length and resident KV stays pool-sized at any context. Off by default; works on every model family. Drafter-scored residency is the default on every family: the server finds the Qwen3-0.6B drafter next to the model (or via `--prefill-drafter`) and lazy-loads it as the relevance scorer that decides which chunks stay resident — non-qwen targets (laguna, gemma4) bridge the tokenizer gap by re-tokenizing the context text for the drafter. LRU is the fallback when no drafter is present, or the explicit choice via `--kvflash-policy lru`. Per-model numbers in [Luce KVFlash →](optimizations/kvflash/README.md).

| Flag / env | Default | Effect |
|---|---|---|
| `--kvflash <tokens\|auto>` | off | Resident pool size. `auto` sizes from the GPU: half of free VRAM after weights and reserves, at the model's KV density, capped where decode speed stays near the flat optimum (default 16384, override `DFLASH_KVFLASH_MAX_POOL`) and at `--max-ctx`. Explicit values are rounded to 256, clamped to `--max-ctx`, floored at the protected minimum so eviction always has a victim. |
| `--kvflash-policy {drafter,lru,qk}` | `drafter` | Residency policy. `lru` opts out of the drafter probe/load (recency-only paging, no extra VRAM). `qk` (qwen35 only) scores residency from the target model's own pooled keys against the decode query, matching drafter-grade recall at a fraction of the rescore cost with no extra model resident. |
| `--kvflash-tau N` | `64` | Reselect interval floor (drafter policy only); the effective interval grows with history to cap rescore overhead. |
| `DFLASH_KVFLASH=N` | off | Env equivalent of `--kvflash`. |
| `DFLASH_KVFLASH_TAU=N` | `64` | Env equivalent of `--kvflash-tau`. |

**Thinking budget**

| Flag | Default | Effect |
|---|---|---|
| `--think-max-tokens N` | model-card | Max tokens inside `<think>…</think>` |
| `--default-max-tokens N` | model-card | Default response cap |
| `--hard-limit-reply-budget N` | `4096` | Hard ceiling; injects `</think>` close near limit |
| `--reasoning-effort-{low,medium,high,x-high,max} N` | model-card | OpenAI-style effort tiers |

**Multi-GPU / IPC**

| Flag / env | Default | Effect |
|---|---|---|
| `--target-device <dev>` | `cuda:0` | Target backend (e.g. `cuda:0`, `hip:0`) |
| `--draft-device <dev>` | same as target | Draft backend; mixed backend needs `--draft-ipc-bin` |
| `--target-gpu N` | `0` | Target GPU index |
| `--draft-gpu N` | same as target | Draft GPU index; offload draft to a second GPU |
| `--target-devices <list>` / `--target-layer-split` | single GPU | Layer-split target across GPUs |
| `--draft-ipc-bin <path>` | — | Out-of-process draft binary (mixed CUDA/HIP) |
| `--peer-access` | off | Enable P2P between target GPUs |
| `--chunk N` | backend default | Prefill ubatch size |
| `--no-cors` | CORS on | Disable CORS headers |
| `DFLASH_TARGET_GPU=N` | `0` | Env var equivalent of `--target-gpu` |
| `DFLASH_DRAFT_GPU=N` | same as target | Env var equivalent of `--draft-gpu` |
| `DFLASH_MODEL_NAME=<name>` | `dflash` | Env var equivalent of `--model-name`; sets the `/v1/models` id and selects the matching `share/model_cards/<name>.json` |

**MoE expert offload (Spark)**

For MoE targets (`laguna`, `qwen35`/`qwen36`) whose experts don't fit in VRAM. `--spark` self-tunes the hot/cold expert split, a bounded GPU cache, and the placement profile from live traffic; decode stays near the all-GPU ceiling via the default single-graph fused path. See [Luce Spark →](optimizations/spark/README.md).

| Flag / env | Default | Effect |
|---|---|---|
| `--spark` | off | One-flag autotune: enable the bounded expert cache, size it from the VRAM target, auto-load and keep persisting a placement profile (`<model>.gguf.spark.csv`). |
| `--spark-vram <GiB>` | whole card | Total VRAM Spark may use; it sizes the hot tier + cache + KV under this cap. |
| `DFLASH_SPARK=1` | off | Env equivalent of `--spark`. |
| `DFLASH_SPARK_VRAM_MB=N` | — | Env equivalent of `--spark-vram` (in MB). |
| `DFLASH_<ARCH>_EXPERT_CACHE=1` | off | Bounded GPU expert cache (`<ARCH>` = `LAGUNA` or `QWEN35MOE`); cold-miss falls toward 0 after warmup. |
| `DFLASH_<ARCH>_CACHE_SLOTS=N` | auto | Cache slots per layer. |
| `DFLASH_LAGUNA_NO_SINGLE_GRAPH=1` | off | Fall back to per-layer decode instead of the default single-graph fused hybrid. |

[DFlash benchmarks →](server/RESULTS.md) · [DFlash blog →](https://lucebox.com/blog/dflash27b) · [PFlash benchmarks →](optimizations/pflash/README.md) · [PFlash blog →](https://lucebox.com/blog/pflash) · [Per-machine quick starts (DGX Spark, Jetson Thor, HIP) →](server/README.md#quick-start)

---

## Run Megakernel Bench (Qwen 3.5-0.8B)

Separate Python bench; 24 layers fused into one persistent CUDA dispatch.
**413 tok/s decode, 21,347 prefill, 1.87 tok/J @220W** vs llama.cpp BF16.

```bash
uv sync --extra megakernel
uv run --directory megakernel python final_bench.py
```

| Method | Prefill pp520 | Decode tg128 | tok/J |
|--------|:-------------:|:------------:|:-----:|
| **Megakernel** `@220W` | **21,347** | **413** | **1.87** |
| llama.cpp BF16 `@350W` | 11,247 | 267 | 0.76 |
| PyTorch HF | 7,578 | 108 | n/a |

[Setup →](optimizations/megakernel/) · [Bench →](optimizations/megakernel/RESULTS.md) · [Blog →](https://lucebox.com/blog/megakernel)

> **Blackwell (RTX 5090, DGX Spark / GB10):** auto-detected by setup; NVFP4 decode path lands ~194 tok/s on GB10. See [optimizations/megakernel/README.md#blackwell-sm_120--sm_121a](optimizations/megakernel/README.md).

---

## Tutorials

Video tutorials for each optimization and the harness setup.

|   |   |   |
|:-:|:-:|:-:|
| **Luce Spark**<br>[▶ YouTube](https://www.youtube.com/watch?v=LB1aVj9lNhg) | **Luce DFlash**<br>[▶ YouTube](https://www.youtube.com/watch?v=vbPGvvSB8IQ) | **Luce Turboquant**<br>[▶ YouTube](https://www.youtube.com/watch?v=uTOOrfhrnBk) |
| **Luce Harness setup**<br>[▶ YouTube](https://www.youtube.com/watch?v=PysoxVGfvRE) | **Luce PFlash**<br>[▶ YouTube](https://www.youtube.com/watch?v=NWeKUL9Bc6Y) | **Luce Megakernel**<br>[▶ YouTube](https://www.youtube.com/watch?v=e6jY4goVIu0) |
| **Luce KVFlash**<br>[▶ YouTube](https://www.youtube.com/watch?v=8rTVCRWvRDo) |   |   |

---

## Why this exists

Local AI should be the default, not a privilege. Private data, no per-token bill, no vendor lock-in. The hardware to run capable models already sits on desks. The software to get real throughput out of it does not.

Nothing was built for local AI inference. Most machines bolt a stock GPU onto a desktop CPU and run a stock runtime, never tuning the kernels to the silicon underneath. On the same 27B model, a DGX Spark or Mac Studio leaves four to six times the real throughput on the table. General-purpose frameworks won the last decade because hand-tuning per chip cost more than it returned: one stack, decent on everything, great on nothing. Speculative decoding, speculative prefill, fused megakernels, and calibrated MoE expert offload turn idle silicon into 3-10× speedups, but they stay locked to BF16 weights on data-center GPUs. Consumer cards inherit the leftovers.

**See the benchmarks and the machine at [lucebox.com](https://lucebox.com).**

<p align="center">
  <a href="https://lucebox.com"><img src="assets/lucebox.png" alt="Lucebox local AI PC" width="85%" /></a>
</p>

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
