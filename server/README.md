<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce DFlash</h1>

<p align="center">
  <strong>GGUF DFlash speculative decoding for Qwen3.5/Qwen3.6 27B.</strong><br/>
  C++/CUDA runtime on top of ggml. Default path: Qwen3.6-27B Q4_K_M target + Lucebox Q8_0 GGUF DFlash draft.<br/>
  Qwen3.5 reference: 129.5 tok/s mean on HumanEval (3.43x vs AR); best demo run: 207.6 tok/s vs 38.0 tok/s AR (5.46x).<br/><br/>
  <a href="https://lucebox.com/blog/dflash27b">Blog post</a> · <a href="RESULTS.md">Benchmarks</a> · <a href="https://discord.gg/yHfswqZmJQ">Discord</a> · <a href="https://lucebox.com">lucebox.com</a>
</p>

<p align="center">
  <img src="demo.gif" width="600" />
</p>

---

```
                   AR (tok/s)   DFlash (tok/s)   Speedup
HumanEval             37.78        129.52          3.43x
Math500               37.71        110.51          2.93x
GSM8K                 37.65         96.15          2.55x
```

## Overview

DFlash is a speculative decoder. A small draft proposes multiple tokens, and the target verifies them in one forward pass. DDTree verifies a tree of candidates instead of a single chain, which improves acceptance length at the same target compute budget.

This repo provides the GGUF target path and runtime pieces needed to run that stack on consumer GPUs:

- C++/CUDA decode loop on top of ggml, without libllama or PyTorch at runtime.
- Qwen3.5/Qwen3.6 `qwen35` GGUF target support.
- DFlash draft loading from GGUF or safetensors.
- DDTree verify with tree-aware SSM rollback kernels.
- TQ3_0 and asymmetric K/V cache quantization for long context.

The default setup fits on a 24 GB RTX 3090: ~16 GB Q4_K_M target, 1.84 GB GGUF draft, DDTree verify state, and KV cache.

## Results

Qwen3.5-27B Q4_K_M, concurrency=1, n_gen=256, 10 prompts/dataset:

| Task      | AR tok/s | DFlash+DDTree tok/s | AL   | Speedup |
|-----------|:--------:|:-------------------:|:----:|:-------:|
| HumanEval | 37.78    | **129.52**          | 8.31 | **3.43×** |
| Math500   | 37.71    | **110.51**          | 7.04 | **2.93×** |
| GSM8K     | 37.65    | **96.15**           | 6.14 | **2.55×** |

AR = autoregressive (`test_generate`). DFlash+DDTree = tree verify at budget=22 with fast rollback (`test_dflash`). AL = Acceptance Length, average committed tokens per draft/verify step. Reproduce via `python3 scripts/bench_llm.py`.

**Up to 256K context on 24 GB** via TQ3_0 KV cache (3.5 bpv, default; Q4_0 legacy path tops out near 128K) + sliding `target_feat` ring (4096 slots). TQ3 = ~9.7× memory saving vs F16; Q4_0 = 8×.

| Prompt length | KV     | Prefill time | Decode tok/s (FA window=2048) |
|:-------------:|:------:|:------------:|:------------:|
| 520 (HE)      | Q8_0   | 0.06 s       | ~104 (window inactive)        |
| 32K           | Q8_0   | 38 s         | ~95 (interp.)                 |
| 64K           | Q4_0   | 126 s        | **91** (PR #26)               |
| 128K          | TQ3_0  | ~10 min      | ~85–95 (sliding active)       |

Prefill numbers assume `--max-ctx` sized to the prompt (auto-fit in `run.py` / `bench_llm.py`). Oversizing — e.g. `--max-ctx=131072` on a 32K prompt — triggers FA stride over unused KV and slows prefill ~27× at that ratio.

HE 10-prompt bench mean in 128K mode (ctx=131072, ddtree-budget=16, FA window=2048): **134.78 tok/s** at AL 8.33.

Decode tok/s assume the default sliding-window flash attention (`--fa-window 2048`, lossless: 100% acceptance at all window sizes). Disable with `--fa-window 0` for full attention; expect ~25 tok/s at 60K+. Tune the window via `python3 scripts/run.py --fa-window N` or `--fa-window N` on `test_dflash`/`dflash_server` (sweet spot 1024–2048; bigger windows trade speed for marginally tighter attention).

Set `DFLASH27B_KV_TQ3=1` (TQ3_0, 3.5 bpv, default) or `DFLASH27B_KV_Q4=1` (Q4_0, 4.5 bpv, legacy) to enable. Full sweep in [RESULTS.md](RESULTS.md).

### Asymmetric K/V quantization

The cache now supports independent quantization types for keys and values, optimizing memory-asymmetric workloads. Set `DFLASH27B_KV_K=<type>` and `DFLASH27B_KV_V=<type>` via environment or CLI flags. Supported types (case-insensitive): `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `tq3_0`.

**Supported (K, V) pairs:**
- K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0} × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
- K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}

Unsupported pairs abort at allocation with a printed list. Precedence (high→low): per-axis `_KV_K`/`_KV_V` override legacy shorthand (`--kv-tq3` / `--kv-q4` / `--kv-f16`); legacy shorthand last-wins among themselves. Default: `q8_0` for both.

**Environment variables:**
```bash
DFLASH27B_KV_K=q8_0 DFLASH27B_KV_V=q4_0 ./test_dflash …
```

**CLI flags on `test_dflash` / `test_generate`:**
```bash
./test_dflash … -ctk q8_0 -ctv q4_0
./test_dflash … --cache-type-k=q8_0 --cache-type-v=q4_0
```

**Target layer-split harness on `test_dflash`:**
```bash
./test_dflash target.gguf draft.safetensors prompt.bin 128 out.bin \
  --target-gpus=1,2,3,4 --target-layer-split=1,1,1,1

./test_dflash target.gguf draft.safetensors prompt.bin 128 out.bin \
  --draft-gpu=0 --target-gpus=0,1 --target-layer-split=1,3 \
  --target-split-load-draft

./test_dflash target.gguf draft.gguf prompt.bin 128 out.bin \
  --draft-gpu=0 --target-gpus=0,1 --target-layer-split=1,1 \
  --target-split-dflash
```

When more than one target GPU is listed, `test_dflash` runs a non-daemon
target-only layer-split harness. Each target GPU loads only its assigned
contiguous layer range; `output_norm` and `output.weight` stay on the last
target GPU. `--target-split-load-draft` additionally loads the DFlash draft
on `--draft-gpu` (`.safetensors` or quantized draft `.gguf`), captures target
features into a draft-side mirror, and runs a small draft forward smoke. This
allows capacity checks where the draft and a target layer range share one GPU
before serving integration. `--target-split-dflash` runs the same split target
placement through a chain DFlash decode loop and reports acceptance length.

**CLI flags on `scripts/run.py`:**
```bash
python3 scripts/run.py --ctk q8_0 --ctv q4_0 --prompt "hello"
python3 scripts/run.py --cache-type-k q8_0 --cache-type-v q4_0 --prompt "hello"
```

**TQ3 semantics under asymmetry:** TQ3_0 is K-side-driven. The FWHT rotation is applied to the query and inverse-applied to the attention output only when `K=TQ3_0`. V's type is independent of rotation. The 256-stride context alignment is triggered if *either* K or V is TQ3_0.

Legacy `--kv-tq3` / `--kv-q4` / `--kv-f16` flags continue to work as symmetric shorthand for backward compatibility.

## Qwen3.6-27B Target

Qwen3.6-27B is the default integration path. It uses the same `qwen35` target architecture as Qwen3.5, and the docs use the Lucebox GGUF DFlash draft by default.

```bash
# 1. target
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/

# 2. matched 3.6 draft (GGUF, used by default by scripts/run.py and dflash_server)
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q8_0.gguf --local-dir models/draft/

# 3. bench
DFLASH_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf python3 scripts/bench_he.py --n-gen 128
```

The default draft path is discovered under `models/draft/`. Scripts prefer `dflash-draft-*.gguf`, then any `.gguf`, then `model.safetensors`. Explicit `.gguf` and safetensors drafts still work via `DFLASH_DRAFT` / `--draft`; qwen35-compatible targets remain swappable via `DFLASH_TARGET` / `--target`.

## Native C++ HTTP server

`dflash_server` serves the same client-facing local API surface used by the
harnesses. It supports `/health`,
`/v1/models`, OpenAI Chat Completions including streaming and tool metadata,
OpenAI Responses for Codex, Anthropic Messages for Claude Code, and Open WebUI
model metadata.

Build it with the rest of the CUDA runtime:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dflash_server -j
```

Run it directly:

```bash
./build/dflash_server models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft models/draft/dflash-draft-3.6-q8_0.gguf \
  --host 127.0.0.1 --port 18080 \
  --max-ctx 32768 --max-tokens 512 \
  --fa-window 2048 \
  --ddtree --ddtree-budget 22 \
  --model-name luce-dflash
```

Then point OpenAI-compatible clients at `http://127.0.0.1:18080/v1`, or probe
the server with:

```bash
python3 ../harness/client_test_runner.py probe \
  --url http://127.0.0.1:18080 \
  --clients all
```

On fragile external RTX links, use the same conservative NVIDIA profile as the
RTX mixed-hardware notes before running long prompts.

Reference measurements from the Qwen3.6 bring-up on RTX 3090:

| Target | Draft | Bench | AL | Accept | Mean tok/s |
|---|---|---|---:|---:|---:|
| Qwen3.5-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash | HumanEval (README config) | 8.33 | ~65% | 134.78 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash (mismatch) | HumanEval (10 prompts, n_gen=128) | 4.74 | 30.6% | 73.67 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.6-27B-DFlash safetensors | HumanEval (10 prompts, n_gen=128) | 5.05 | 32.3% | 77.77 |
| Qwen3.6-27B Q4_K_M | z-lab/Qwen3.5-27B-DFlash (mismatch) | Math (10 prompts, n_gen=128) | 3.63 | 23.7% | 57.00 |

Full `bench_llm.py` suite on Qwen3.6-27B UD-Q4_K_XL, 10 prompts, n_gen=256, RTX 3090 24 GB, auto-fit `--max-ctx`:

| Bench | AR tok/s | DFlash tok/s | AL | Speedup |
|---|---:|---:|---:|---:|
| HumanEval | 34.90 | 78.16 | 5.94 | **2.24×** |
| GSM8K | 34.89 | 59.65 | 4.43 | **1.71×** |
| Math500 | 35.13 | 69.77 | 5.15 | **1.99×** |
| **Mean** | 34.97 | 69.19 | 5.17 | **1.98×** |

## Laguna-XS.2 target (experimental, Poolside MoE)

[Poolside Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) is a 40-layer MoE LLM with 256 experts (top-8) plus an always-on shared expert, per-layer head counts `[48,64,64,64]×10`, and a per-layer SWA pattern (window 512). It is **architecturally distinct from `qwen35`**, so dflash adds a hand-rolled CUDA forward path (`Path A`, ggml-only — no libllama dependency) that mirrors the qwen35 stack. The Q4_K_M GGUF lands at 18.77 GiB on a single RTX 3090; tok_embd stays CPU-only (110 MiB) to keep the GPU budget under 24 GB.

### Single binary, single server

`test_dflash` peeks `general.architecture` from the target GGUF at startup
and dispatches by arch:

  - `qwen35` / `qwen36` → existing DFlash + DDTree pipeline (no change).
  - `laguna` → `dflash::common::run_laguna_daemon()` (no spec-decode, no DDTree).

The daemon stdin/stream-fd protocol is identical, so `dflash_server`
drives both arches end-to-end. The only thing the user changes is the model path.

### Build + run

```bash
cmake --build build --target test_dflash test_laguna_daemon pflash_daemon -j

# 19 GB Q4_K_M target + 1.2 GB Qwen3-0.6B BF16 drafter + tokenizers
hf download Lucebox/Laguna-XS.2-GGUF laguna-xs2-Q4_K_M.gguf --local-dir models/
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --local-dir models/
hf download poolside/Laguna-XS.2 --local-dir models/Laguna-XS-2 \
    --include 'tokenizer*' '*.json'

# OpenAI-compatible HTTP server.
# dflash_server routes to run_laguna_daemon() when arch=laguna.
./build/dflash_server models/laguna-xs2-Q4_K_M.gguf \
    --max-ctx 16384 --port 8000

curl -sN http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"luce-dflash","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'

# Smoke (loader only, no forward)
./build/smoke_load_target_laguna models/laguna-xs2-Q4_K_M.gguf

# Variable-N TTFT bench (DFLASH_KV_TYPE=q4_0 for ctx > 32K, DFLASH_CHUNK=2048 default)
DFLASH_KV_TYPE=q4_0 ./build/bench_laguna_ttft models/laguna-xs2-Q4_K_M.gguf '4096,16384,65536'

# NIAH single-needle, with PFlash compression. The driver still spawns the
# standalone test_laguna_daemon binary so it can run without dflash_server.
python3 scripts/laguna_pflash_niah.py \
    --target models/laguna-xs2-Q4_K_M.gguf \
    --drafter models/Qwen3-0.6B-BF16.gguf \
    --laguna-tok models/Laguna-XS-2 \
    --drafter-tok Qwen/Qwen3-0.6B \
    --pflash-bin ./build/pflash_daemon \
    --laguna-bin ./build/test_laguna_daemon \
    --ctx 131072 --depth 0.5 --keep 0.10 --target-kv q4_0
```

### NIAH retrieval (RTX 3090, depth=0.5, q4_0 KV target, Q8_0 KV at 4K)

| Context | KV   | keep | drafter (s) | target prefill (s) | end-to-end TTFT | NIAH |
|--------:|:----:|:----:|------------:|-------------------:|----------------:|:----:|
|   4 096 | Q8_0 | 0.10 |        1.54 |               0.39 |          1.92 s |  ✅  |
|  65 536 | Q4_0 | 0.10 |        ~5   |                ~6  |           ~11 s |  ✅  |
|  65 536 | Q4_0 | 0.20 |        ~5   |                ~8  |           ~13 s |  ✅  |
|  65 536 | Q4_0 | 0.30 |        ~5   |                ~10 |           ~15 s |  ✅  |
|  65 536 | Q4_0 | 0.50 |        ~5   |                ~17 |           ~22 s |  ✅  |
| 131 072 | Q4_0 | 0.10 |       11.11 |               4.79 |         15.91 s |  ✅  |
| 131 072 | Q4_0 | 0.20 |       11.20 |              13.55 |         24.75 s |  ✅  |
| 131 072 | Q4_0 | 0.30 |       11.41 |              26.43 |         37.84 s |  ✅  |

The 131K `keep=0.10` run depends on token-boundary repair in `scripts/laguna_pflash_niah.py::cross_tok_compressed`. The driver recovers kept-token positions, groups consecutive runs, expands each run to whitespace boundaries, and decodes the union once. That keeps multi-token needles intact after compression.

### Real-prompt NIAH (code corpus filler)

The table above uses synthetic uniform filler. Pass `--filler-file <path>` to
use a real corpus instead (file or directory; directories are recursively
concatenated). On `server/src` (1.3 MiB of C++/CUDA, ctx=16K, depth=0.5):

| keep | drafter compressed | NIAH |
|:----:|-------------------:|:----:|
| no-compress | (full haystack)        | ✅ PASS |
| 0.10 |    1486 / 15278     | ❌ FAIL |
| 0.15 |    2254 / 15278     | ❌ FAIL |
| 0.20 |    3022 / 15278     | ❌ FAIL |
| 0.30 |    4558 / 15278     | ✅ PASS |
| 0.50 |    7630 / 15278     | ✅ PASS |

No-compress passes, so the model itself reads the needle out of code; the
failure mode at low `keep` is the drafter dropping the needle line. Code has
a denser distribution of "important" tokens than synthetic filler (every
identifier, syntax token, and semantic span looks informative to the
attention-based scorer), so the needle line doesn't stand out as an outlier
until retention is around 3× the synthetic-filler threshold (~0.10 → ~0.30).

Production implication: route by prompt source. Synthetic / prose prompts
tolerate `keep=0.10`; code-heavy prompts want `keep ≥ 0.30` to keep recall
intact. A NIAH-aware drafter (or a hybrid scorer that boosts low-frequency
tokens) is the path to bring code recall to the same ratio as prose.

### Sampler

`test_laguna_daemon` is greedy by default. When `scripts/laguna_serve.py` (or any caller) appends ` samp=temp,top_p,top_k,rep_pen,seed` to a `generate` line, the daemon strips it and runs a CPU sampler chain (rep_penalty → top_k → softmax(temp) → top_p → draw) over the prompt + emitted history. Verified that greedy / temp=2.0 seed=42 / temp=2.0 seed=43 / top_p=0.5 produce four distinct outputs on the same prompt.

### Outstanding

- **No Laguna spec-decode draft published yet.** Current decode is autoregressive only (~111 tok/s on RTX 3090). When a matched draft lands, the DFlash + DDTree machinery already in `test_dflash` ports across.
- **Prefix cache + in-process PFlash compression** are disabled on the laguna path. Both require `SNAPSHOT` / `RESTORE` / `FREE_SNAPSHOT` and `compress` / `park` / `unpark` commands inside `run_laguna_daemon`. Tracked as follow-ups; the qwen35 path uses them today.
- **Path B (in-process Python drafter)** is not used here. Path A keeps the dflash daemon ggml-only.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub
cd lucebox-hub/dflash

# Build (CUDA 12+, CMake 3.18+, sm_60+ GPU including Pascal; CUDA 13+ required for Jetson AGX Thor sm_110)
# Pass -DCMAKE_CUDA_ARCHITECTURES matching your GPU. Common values:
#   60;61 = Pascal P100/P40 (scalar flashprefill fallback, no WMMA)
#   70 = V100 (F16 WMMA kernels, BF16 draft → FP16 at load)
#   75 = 2080 Ti (F16 WMMA kernels, auto-converts BF16 draft → FP16 at load)
#   86 = RTX 3090 / A40 (native BF16 WMMA)
#   89 = RTX 4090
#   90 = H100
#   120 = Blackwell / DGX Spark
#   110 = Jetson AGX Thor (CUDA 13+)
# Omitting the flag falls back to the CMake-set default ("60;61;62;70;75;86"),
# which compiles Pascal (scalar), Volta/Turing (F16 WMMA), and Ampere+ (BF16 WMMA)
# flashprefill paths.
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target test_dflash -j

# Fetch models: ~16 GB target + 1.84 GB Lucebox Q8_0 GGUF DFlash draft.
# Quickstart pins to Qwen3.6-27B (latest release). For Qwen3.5-27B swap in
# unsloth/Qwen3.5-27B-GGUF + z-lab/Qwen3.5-27B-DFlash; arch is identical so
# no rebuild is needed.
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q8_0.gguf --local-dir models/draft/

# Streaming one-shot generate (run.py defaults to models/Qwen3.6-27B-Q4_K_M.gguf;
# override with --target or DFLASH_TARGET=... env var).
python3 scripts/run.py --prompt "def fibonacci(n):"

# Multi-turn chat REPL
python3 examples/chat.py

# OpenAI-compatible HTTP server (drop-in for Open WebUI / LM Studio / Cline).
./build/dflash_server models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft models/draft/dflash-draft-3.6-q8_0.gguf --port 8000

# Reproduce paper numbers
python3 scripts/bench_llm.py                                 # HE + GSM8K + Math500
python3 scripts/bench_he.py --n-gen 256 --ddtree-budget 22   # minimal HE bench
```

**Long-context mode (up to 256K):**
```bash
DFLASH27B_KV_TQ3=1 DFLASH27B_PREFILL_UBATCH=16 \
  build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf \
  models/draft/dflash-draft-3.6-q8_0.gguf /tmp/long_prompt.bin 64 /tmp/out.bin \
  --fast-rollback --ddtree --ddtree-budget=16 --max-ctx=4096   # align_up(prompt + n_gen + 64, 256); raise up to 262144 for long prompts
```

**Requirements:** NVIDIA sm_60+ GPU (Pascal P40 24GB, V100 32GB, 2080 Ti, 3090, A10, A40, 4090) or Jetson AGX Thor sm_110, CUDA 12+ (CUDA 13+ required for Thor), 22+ GB VRAM, ~80 GB disk. Pascal GPUs use the scalar flashprefill fallback (no WMMA); P100 (12/16 GB) cannot fit the 27B target + draft under the 22 GB minimum. On Volta (SM 7.0) and Turing (SM 7.5), BF16 draft weights are auto-converted to FP16 at load time for tensor core acceleration.

## How it works

**Block-diffusion draft.** Each step, the draft sees `[last_target_token, MASK×15]` plus the last 5 captured target hidden states. It denoises the masks in a single forward, producing 16 candidate tokens conditioned on real target features. Structurally stronger than chain EAGLE: every position conditions on the same captured context, not its own noisy predictions.

**DDTree tree verify.** Instead of one chain of 16 candidates, a best-first tree of up to 22 nodes spans the top-K branches at each position. One target forward verifies the whole tree via a causal mask derived from parent pointers. Budget=22 is the sweet spot where draft accuracy plateaus. Chain pre-seed matters: pure best-first construction with greedy verify on a quantized target can rescue an inferior suffix; the `chain_seed=true` flag in `build_ddtree` recovered AL from ~4 to ~9.

**Per-step rollback, kernel-free.** Before verify, the target's recurrent state (SSM intermediate, conv window, KV cache) is snapshotted; after accept, restored to the committed prefix. Three custom CUDA kernels keep rollback off the critical path:

| Kernel | Purpose |
|--------|---------|
| `ggml_gated_delta_net_tree_persist` | Direct-writes SSM intermediates into a persistent buffer, skipping a 9 ms `ggml_cpy` per step |
| `ggml_ssm_conv_tree` | Tree-aware conv state gather: each sibling reads its K-1 window along the DDTree parent chain, not DFS order |
| Sliding `target_feat` ring | 4096-slot ring via `(pos % cap)`, enables 128K without holding 6.6 GB of captured features |

Prefill and decode share one graph builder; chain mode is just DDTree with `budget=n_spec+1` and no branching.

## Architecture note

Qwen3.5-27B is **not** a dense transformer. llama.cpp calls the arch `qwen35`:

- 64 layers. Every 4th is full softmax attention, the rest are **Gated DeltaNet** (linear attention with learned recurrence)
- M-RoPE, dimension sections `[11, 11, 10, 0]`
- 24 Q heads, 4 KV heads, key/value length 256
- SSM state cache alongside the KV cache

The DeltaNet primitive is already a first-class ggml op (`ggml_gated_delta_net`). Our fork of llama.cpp adds three tree-mode variants (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`) so DDTree verify can roll back SSM state in place, without a replay forward. The full engine (graph builders + decode loop + rollback + kernels) is ~2000 lines.

## Why not llama.cpp / vLLM / z-lab?

- **llama.cpp**: runs Qwen3.5-27B via GGUF but has no DFlash integration. Chain EAGLE isn't enough; block diffusion + DDTree needs a custom decode loop that bypasses `llama_decode`.
- **vLLM / SGLang**: Qwen3.5-27B in BF16 is 54 GB, so a single 24 GB card forces a quantized path. GGUF for this arch is broken on SGLang as of 2026-04 and vLLM is dropping GGUF support. AWQ runs on SGLang as plain autoregressive at 46.6 tok/s but can't host the BF16 draft + DDTree tree state alongside it on 24 GB. Q4_K_M GGUF is the only format that fits the full spec-decode stack, this repo runs it at 129.5 tok/s mean on HumanEval, **2.8× faster** than SGLang AWQ autoregressive on the same hardware.
- **z-lab reference**: vLLM / SGLang integrations ship DFlash as a speculative-decoding method, but only on BF16 weights benchmarked on NVIDIA B200 (54+ GB VRAM). No GGUF path.

## Scope and limits

Current scope:

- **Batch size 1**, single-user local inference target (Ollama / LM Studio use case)
- **Target family**: Qwen3.5/Qwen3.6 `qwen35` GGUF targets. Other architectures need their own graph builder.
- **Draft formats**: DFlash GGUF drafts and z-lab-style safetensors drafts.
- **Optional sampling**: `temperature`, `top_p`, `top_k`, `seed`, and `frequency_penalty` are honored on the OpenAI endpoint. The DDTree verify skeleton stays argmax (preserves accept rate); only the *committed* token at each verify step is drawn from a small CPU sampler chain (rep-pen → top-k → top-p → temp → multinomial). `temperature=0` (default) keeps the path bit-exact greedy. Full Leviathan-style rejection sampling on the tree is still a future addition.
- **Backends**: CUDA is the primary path. CUDA sm_60+ is supported: Pascal (sm_60-69) uses scalar F16 fallback, Volta/Turing (sm_70-75) use F16 WMMA kernels, and Ampere+ use native BF16 WMMA. HIP support exists for the documented AMD path. No Metal.
- **Quantized target**: Q4_K_M fits the full stack on 24 GB. Higher target quantizations may improve acceptance if they fit.

Correctness: `test_vs_oracle` validates the draft graph at cos sim 0.999812 vs the PyTorch reference. The target graph matches llama.cpp's `models/qwen35.cpp` semantically and produces bit-identical output to `test_generate` in autoregressive mode.

## Contributing

Open an issue or PR against `Luce-Org/lucebox-hub`. Good first picks:

- **Leviathan-style rejection sampling** on each DDTree branch (the current implementation samples only the committed token; full prob-matching across the tree is the next step)
- **Full llama.cpp integration**: new arch, `llama-speculative-dflash.cpp`, `llama-cli` / `llama-server` wiring

## Citation

```bibtex
@software{luce_dflash_2026,
  title  = {Luce DFlash: GGUF port of block-diffusion speculative decoding for Qwen3.5-27B on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox-hub/tree/main/dflash},
  year   = {2026}
}

@article{dflash2026,
  title   = {DFlash: Block-Diffusion Speculative Decoding},
  author  = {z-lab},
  journal = {arXiv:2602.06036},
  year    = {2026}
}

@article{ddtree2026,
  title   = {Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author  = {Ringel, Liran and Romano, Yaniv},
  journal = {arXiv:2604.12989},
  year    = {2026}
}
```

---

Apache 2.0 · [Lucebox](https://lucebox.com) · [Discord](https://discord.gg/yHfswqZmJQ)

Inspired by [z-lab/DFlash](https://arxiv.org/abs/2602.06036), [liranringel/ddtree](https://github.com/liranringel/ddtree), [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

---

## Using with OpenAI Codex CLI

The DFlash server natively supports the **Responses API** (`/v1/responses`), which is the
only wire protocol used by [OpenAI Codex](https://github.com/openai/codex).

### 1. Start the DFlash server

```bash
./build/dflash_server models/Qwen3.5-27B-Q4_K_M.gguf \
  --draft models/Qwen3.5-3B-f16.safetensors \
  --ddtree --ddtree-budget 22 --port 8080
```

### 2. Configure Codex

Create or edit `~/.codex/config.toml`:

```toml
model = "luce-dflash"
model_provider = "dflash"

[model_providers.dflash]
name = "DFlash"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
supports_websockets = false
```

No `env_key` is needed — the local server accepts any token.

### 3. Run Codex

```bash
codex --provider dflash "Explain this codebase"
```

### Supported features

| Feature | Status |
|---------|--------|
| Responses API (`POST /v1/responses`) | ✅ |
| Function/tool calling | ✅ |
| Streaming (SSE) | ✅ |
| Codex models endpoint | ✅ |
| Reasoning / thinking | ✅ (effort: low / medium) |
| WebSockets | ❌ (not needed) |
