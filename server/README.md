<p align="left">
  <a href="../README.md">← lucebox</a>
</p>

<p align="center">
  <img src="hero.png" width="600" />
</p>

<h1 align="center">Luce DFlash</h1>

<p align="center">
  <strong>GGUF DFlash speculative decoding for Qwen3.5/Qwen3.6 27B.</strong><br/>
  C++/CUDA runtime on top of ggml. Default path: Qwen3.6-27B Q4_K_M target + Lucebox Q4_K_M GGUF DFlash draft.<br/>
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

For one listener with Qwen on R9700 and DS4 on Strix Halo, see
[model load balancing](docs/MODEL_LOAD_BALANCING.md).

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

Decode tok/s assume the default sliding-window flash attention (`--fa-window 2048`, lossless: 100% acceptance at all window sizes). Disable with `--fa-window 0` for full attention; expect ~25 tok/s at 60K+. Tune the window via `python3 scripts/run.py --fa-window N` or `--fa-window N` on `test_dflash`/`luce_server` (sweet spot 1024–2048; bigger windows trade speed for marginally tighter attention).

Set `LUCE_KV_TQ3=1` (TQ3_0, 3.5 bpv, default) or `LUCE_KV_Q4=1` (Q4_0, 4.5 bpv, legacy) to enable. Full sweep in [RESULTS.md](RESULTS.md).

### Asymmetric K/V quantization

The cache now supports independent quantization types for keys and values, optimizing memory-asymmetric workloads. Set `LUCE_KV_K=<type>` and `LUCE_KV_V=<type>` via environment or CLI flags. Supported types (case-insensitive): `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `tq3_0`.

**Supported (K, V) pairs:**
- K ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0} × V ∈ {F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ3_0}
- K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}

Unsupported pairs abort at allocation with a printed list. Precedence (high→low): per-axis `_KV_K`/`_KV_V` override legacy shorthand (`--kv-tq3` / `--kv-q4` / `--kv-f16`); legacy shorthand last-wins among themselves. Default: `q8_0` for both.

**Environment variables:**
```bash
LUCE_KV_K=q8_0 LUCE_KV_V=q4_0 ./test_dflash …
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

# 2. matched 3.6 draft (GGUF, used by default by scripts/run.py and luce_server)
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir models/draft/

# 3. bench
LUCE_TARGET=models/Qwen3.6-27B-Q4_K_M.gguf python3 scripts/bench_he.py --n-gen 128
```

The default draft path is discovered under `models/draft/`. Scripts prefer `dflash-draft-*.gguf`, then any `.gguf`, then `model.safetensors`. Explicit `.gguf` and safetensors drafts still work via `LUCE_DRAFT` / `--draft`; qwen35-compatible targets remain swappable via `LUCE_TARGET` / `--target`.

## Native C++ HTTP server

`luce_server` serves the same client-facing local API surface used by the
harnesses. It supports `/health`,
`/v1/models`, OpenAI Chat Completions including streaming and tool metadata,
OpenAI Responses for Codex and OMP, Anthropic Messages for Claude Code, and
Open WebUI model metadata.

Build it with the rest of the CUDA runtime:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target luce_server -j
```

Run it directly:

```bash
./build/luce_server models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft models/draft/dflash-draft-3.6-q4_k_m.gguf \
  --host 127.0.0.1 --port 18080 \
  --max-ctx 32768 --max-tokens 512 \
  --fa-window 2048 \
  --ddtree --ddtree-budget 22 \
  --model-name luce-dflash
```

Use `--specla` to enable speculative linear-attention verification when the
target supports it. The runtime chooses a compatible proposal adapter; the
current monolithic Qwen3.5/Qwen3.6 path uses DDTree with tested defaults.
Hardware- or checkpoint-specific overrides remain available through
`--ddtree-budget`, `--ddtree-tau`, `--specla-top-k`, and `--draft-swa`.

### Compression proxy mode

`luce_server` can run as a **PFlash compression proxy** in front of any
OpenAI-compatible backend instead of doing local inference. When
`--prefill-upstream-base` is set, each request is compressed (PFlash) and
forwarded upstream: compressed requests are sent as a raw `prompt` to
`<base>/v1/completions` (the compressed text already carries chat-template
markup, so this avoids double-templating), while uncompressed requests pass
through to `<base>/v1/chat/completions`. Streaming and non-streaming responses
are rewritten back to the Chat Completions shape. With no upstream flags the
server is byte-identical to local-inference mode.

```bash
./build/luce_server models/Qwen3.6-27B-Q4_K_M.gguf \
  --pflash on --prefill-threshold 10000 \
  --prefill-drafter models/Qwen3.5-0.8B-BF16.gguf \
  --prefill-curve 10000:0.5 40000:0.2 100000:0.1 \
  --prefill-upstream-base http://127.0.0.1:8099 \
  --prefill-upstream-model my-upstream-model \
  --port 8080
```

New PFlash flags:

| Flag | Purpose |
|---|---|
| `--prefill-curve T:R [T:R ...]` | Piecewise keep-ratio curve. Linear interpolation over `(tokens, ratio)` breakpoints, e.g. `10000:0.5 40000:0.2 100000:0.1` (2× compression at 10K tokens, 5× at 40K, 10× at 100K+). Points may be separate arguments or one quoted list. Overrides `--prefill-keep-ratio`; a per-session bandit override still takes precedence. |
| `--prefill-upstream-base <URL>` | OpenAI-compatible upstream base URL. Enables proxy mode. |
| `--prefill-upstream-key <KEY>` | Bearer token sent to the upstream. |
| `--prefill-upstream-model <NAME>` | Model name sent on forwarded requests. |

Then point OpenAI-compatible clients at `http://127.0.0.1:18080/v1`, or probe
the server with:

```bash
python3 ../harness/client_test_runner.py probe \
  --url http://127.0.0.1:18080 \
  --clients all
```

On fragile external RTX links, use the same conservative NVIDIA profile as the
RTX mixed-hardware notes before running long prompts.

## Server parameter reference

The command shape is `luce_server <model.gguf> [options]`. The first positional argument selects the target weights. `--model-name` only changes the name reported by the API; it does not select a model file.

`luce_server --list-devices [model.gguf]` prints every GPU with its `backend:N` index, architecture, memory and whether another process is computing on it, and, given a model, the placement `--target-device auto` would choose.

`--target-device auto` places the model from the GPUs' state at startup. It skips a GPU another process holds: on Linux/AMD it reads `/sys/class/kfd/kfd/proc`, where a running server has queues on every GPU it spans before its weights load (free memory alone misses a busy Strix Halo). KFD IDs are matched to the full PCI address, including the function or GPU partition. The log and `--list-devices` report the owning host PID, process/model names when readable, and the GPUs it spans. Each GPU is sized by queried free memory; zero free bytes never falls back to total memory.

Placement order:
1. A discrete GPU that holds the model alone.
2. The model's supported multi-GPU technique when the available devices together hold it: DeepSeek V4/V4.1 expert parallel with `--expert-device`, or layer splitting with `--target-devices` for families with a layer-split adapter.
3. One available GPU that holds it, discrete first.
4. For backends with expert offload, the roomiest available GPU.

Otherwise auto stops before loading. It preserves requested single-device features: for example, Qwen paged attention or a vision projector, and Gemma/Laguna speculative decoding, prevent automatic layer splitting. A separately placed DeepSeek vision encoder (`--mmproj-device`) prevents automatic expert parallelism. DeepSeek paged expert parallelism requires a HIP `gfx1201` primary and `gfx1151` secondary. With `--profile`, auto chooses only the device, because a profile is qualified for one topology. On an R9700 + Strix Halo machine, the 91.5 GiB DeepSeek V4 model therefore loads expert parallel on both when both are free, on Strix Halo alone when the R9700 is busy, and not at all when Strix Halo is busy. The expert-parallel choice uses no tuning profile; `--profile ds4-r9700-strix` remains the qualified recipe for that topology.

Availability is a startup snapshot, not a GPU reservation: near-simultaneous launches can race. CUDA has no busy-process signal here and uses free memory only. KFD process IDs belong to the host PID namespace; in a container with a separate PID namespace, process labels and self-exclusion are not reliable unless `/proc` and the process share the host PID namespace.

DeepSeek's hybrid hot-expert budget reserves the same headroom floor as its post-load check (1.5 GiB on a discrete primary, 4 GiB on an integrated primary). KV memory remains charged to the primary even when all routed experts run on the secondary. All-cold placement does not require room for a hot expert round.

`--profile <name>` applies a qualified hardware and model configuration. The profiles live in the model cards (`share/model_cards/<model>.json`, field `profiles`): `qwen38-r9700` (Qwen 3.8 27B on an R9700), `qwen38-halo` (Qwen 3.8 27B on the Lucebox's Strix Halo, leaving the R9700 free), `ds4-strix` (DeepSeek V4 on Strix Halo), `ds4-r9700-strix` (DeepSeek V4 with experts split between an R9700 and Strix Halo), `ds41-halo` (DeepSeek V4.1 on an R9700, Strix Halo and an SSD), `ds41-r9700-gorgon` (DeepSeek V4.1 on an R9700 beside a 192 GB Ryzen AI Max+ PRO 495, every expert resident), `ds41-gorgon` (DeepSeek V4.1 fully resident on the 192 GB Ryzen AI Max+ PRO 495 alone; not yet measured on one) and `qwen-next-r9700-strix` (Qwen 3.8 Flash-Next split between an R9700 and Strix Halo). A model served alone gets the first of its card's default profiles whose qualified GPUs come first in device order, with the memory the profile needs: `qwen38-r9700` with an R9700 as `hip:0`; for DeepSeek V4.1, `ds41-r9700-gorgon` with an R9700 as `hip:0` and an iGPU with at least 160 GiB (KFD's figure, the GTT) as `hip:1`, `ds41-gorgon` with such an iGPU as `hip:0`, else `ds41-halo` with an R9700 and Strix Halo. `--profile none` turns that off. Flags on the command line replace the profile's value, `--no-<flag>` leaves a profile flag out (for example `--no-peer-access`), and environment variables that are already set keep theirs. The startup log lists what the profile applied, and `luce_server --list-devices <model.gguf>` names the default profile that would apply.

### Core server

| Option | Default | Purpose |
|---|---|---|
| `--draft <path>` | none | Draft model for speculative decode: DFlash for Qwen, Gemma and Laguna, DSpark for DeepSeek V4. |
| `--host <addr>` | `0.0.0.0` | Bind address. |
| `--port <N>` | `8080` | Listen port. |
| `--max-ctx <N>` | `8192` | Maximum context length. |
| `--max-tokens <N>` | model card | Legacy alias for `--default-max-tokens`. |
| `--default-max-tokens <N>` | model card or `16000` | Output cap when a request omits a token limit. |
| `--model-name <name>` | `luce` | API alias returned by `/v1/models` and responses. It does not change the loaded weights. |
| `--chat-template-file <path>` | model default | Jinja chat-template override. |
| `--no-cors` | CORS enabled | Disable CORS headers. |

### Speculative decode

| Option | Default | Purpose |
|---|---|---|
| `--draft-block-size <N>` | checkpoint metadata | Dense Qwen proposal width, from 2 through 32 and no more than twice the trained block size. |
| `--draft-swa <N>` | `0` | Draft sliding-attention window. |
| `--draft-residency auto\|persistent\|request-scoped` | `auto` | Control when draft weights remain resident. |
| `--lazy-draft` | off | Alias for request-scoped draft residency. |
| `--fast-rollback` | on | Enable speculative fast rollback. |
| `--no-fast-rollback` | off | Disable fast rollback. |
| `--specla` | off | Enable SpecLA when the target supports it. |
| `--specla-top-k <K>` | `4` | SpecLA draft-tree width. |
| `--ddtree` | off | Enable DDTree verification. |
| `--ddtree-budget <N>` | `22` | DDTree candidate budget. |
| `--ddtree-tau <T>` | off | Confidence margin; SpecLA defaults to 6. |
| `--verify-width <N>` | adaptive base 8 | Laguna chain-verification width. |
| `--adaptive-experts [tau]` | off | Enable expert-count gating with a default threshold of 0.80. |
| `--fa-window <N>` | `0` | Target attention window. Keep `0` for full attention and tool use at long context. |

### Device placement

| Option | Default | Purpose |
|---|---|---|
| `--target-device <backend:gpu\|auto>` | `auto:0` or `LUCE_TARGET_DEVICE` | Place the target on a CUDA or HIP device. `auto` uses available free memory and the model's supported topology, preserving requested features; see the placement policy above. |
| `--draft-device <backend:gpu>` | `auto:0` | Place the draft on a CUDA or HIP device. DeepSeek V4 and `--target-device auto` default to the target GPU. |
| `--expert-device <backend:gpu>` | none | DeepSeek V4: keep dense work and hot experts on the target and run the remaining routed experts on this GPU in the same process. |
| `--target-devices <list>` | one device | Select multiple target devices, such as `cuda:0,cuda:1`. |
| `--target-split-mode layer\|tensor` | `layer` | Select the multi-GPU target strategy. |
| `--target-layer-split <weights>` | none | Optional layer-split weights. |
| `--target-split-fast-rollback` | off | Use exact F32 checkpoints for supported local Qwen layer splits. |
| `--peer-access` | off | Enable peer access between target GPUs. |
| `--chunk <N>` | `512` | Chunked-prefill batch size. |
| `--draft-ipc-bin <path>` | none | Remote draft daemon for mixed backends. |
| `--draft-ipc-work-dir <path>` | none | Remote draft scratch directory. |
| `--draft-ipc-ring-cap <N>` | runtime default | Remote draft feature-ring capacity. |
| `--target-shard-ipc-bin <path>` | none | Remote target-shard daemon. |
| `--target-shard-ipc-work-dir <path>` | none | Remote target-shard scratch directory. |

See [recommended setups](docs/RECOMMENDED_SETUPS.md) for tested combinations and [mixed-backend placement](docs/MIXED_BACKEND.md) for build requirements.

### DeepSeek V4

| Option | Default | Purpose |
|---|---|---|
| `--ds4-fused-decode` | off | Enable single-graph GPU decode. |
| `--ds4-fused-verify-f16-kv` | off | Reuse the F16 MLA cache during batched verification. |
| `--ds4-expert-top-k <N>` | `0`, model default | Set routed experts per token. The current DeepSeek V4 profile uses `6`. |
| `--ds4-prefill exact\|dense\|sparse` | `exact` | Select prefill mode. Dense and sparse modes can change generated tokens. |

See the [current six-expert Strix Halo profile](https://www.lucebox.com/blog/deepseek-v4-flash-0731) and the [DeepSeek V4 implementation guide](docs/DS4.md).

### KV cache, paging, and concurrency

| Option | Default | Purpose |
|---|---|---|
| `--cache-type-k <type>` | model and backend default | K-cache type: `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, or `tq3_0`. |
| `--cache-type-v <type>` | model and backend default | V-cache type using the same choices. |
| `--kvflash <tokens\|auto>` | off | Enable bounded KV residency. |
| `--kvflash-policy drafter\|lru\|qk` | `drafter` | Choose the KVFlash residency policy. |
| `--kvflash-tau <N>` | `64` | Drafter-policy reselect interval. |
| `--prefix-cache-slots <N>` | `32` | In-memory prefix-cache slots; `0` disables. |
| `--prefix-cache-max-mib <auto\|MiB>` | `auto` | Resident RAM limit for single-sequence prefix snapshots (Qwen and DeepSeek4); `auto` keeps room for three snapshots at `--max-ctx`, at most a quarter of available memory; `0` is unlimited. See [docs/PREFIX_CACHE.md](docs/PREFIX_CACHE.md#configuration). |
| `--concurrent-prefix-cache-max-mib <MiB>` | auto | Resident RAM limit for copied concurrent paged prefix checkpoints. `auto` sizes it when the scheduler starts: 2 x `--max-concurrency` + 1 checkpoints at `--max-ctx`, at least 4096 MiB (the former default); above that, at most 1/4 of available memory. `0` is unlimited. |
| `--agent-turn-cache` | off | Replay a generated tool-call turn the next request renders differently, so prefix caching still extends through it. |
| `--prefill-cache-slots <N>` | `0` | Full-prompt cache slots. |
| `--paged-attention` | off | Enable paged KV for supported Qwen targets (16-token blocks) or DeepSeek4 on Strix Halo and R9700 plus Strix Halo (128-token pages). DeepSeek4 paged serving is AR-only. |
| `--max-concurrency <N>` | `1` | Maximum concurrent decode sequences. Qwen supports up to 64; DeepSeek4 supports up to 6. Values above 1 enable paged attention. |
| `--admission-coalesce-ms <N>` | `20` | Idle-to-busy batching window from 0 through 1000 ms. |
| `--kv-pool-tokens <N>` | auto | Shared physical K/V capacity for concurrent serving, rounded to the backend page size. |
| `--decode-kv-offload-mb <auto\|N>` | `auto` | Automatically size the RAM budget for active KV suspension; `N` sets a per-model cap in MiB, `0` disables. Applies to concurrent serving. |
| `--kv-cache-dir <path>\|off` | `~/.cache/lucebox/kv-cache` | Disk session cache directory (`$XDG_CACHE_HOME/lucebox/kv-cache` when set); `off` or `--no-kv-cache` disables it. Snapshots are written when they leave RAM and at graceful shutdown; a crash loses what was only in RAM. See [docs/PREFIX_CACHE.md](docs/PREFIX_CACHE.md#disk-session-cache). |
| `--kv-cache-budget <MB>` | `131072` | Disk KV-cache size cap (hard cap, LRU eviction, clamped to the free space). |
| `--kv-cache-ttl <hours>` | `72` | Delete disk snapshots unused this long; `0` = never. |
| `--kv-cache-shutdown-flush <s>` | `120` | Longest a graceful shutdown spends writing RAM snapshots to disk; `-1` = no limit. |
| `--kv-cache-min-tokens <N>` | `512` | Minimum prefix length to persist. |
| `--kv-cache-interval <N>` | `10240` | Continued-checkpoint interval. |
| `--kv-cache-cold-max <N>` | `10240` | Cold-prefix limit for long prompts. |
| `--disk-prefix-cache off\|full\|auto\|auto:N\|N` | `full` | Default disk prefix policy. |
| `--disk-prefix-cache-compress` | off | Clamp FlowKV snapshots to the stable system prefix. Requires a prefill drafter. |

Decode-pressure suspension defaults to `auto`. After all models load, the
server shares 25% of available host RAM across eligible models, subtracting
explicit offload caps first. Each model's automatic cap is also bounded by
the payload needed to preserve up to `max_concurrency - 1` contexts at their
individual context/pool limit. Repeated suspensions may save more than one
pool's payload as resident requests reuse pages and continue growing. These
are allocation caps; no RAM is reserved until a request is suspended.

Linux uses `MemAvailable`, bounded by the current cgroup v2 and its visible
ancestors' remaining memory limits. Windows uses available physical RAM.
Unknown memory reporting (including cgroup v1 memory control) disables
automatic offload; an explicit MiB cap remains available. The resolved byte
cap is logged at startup and exposed in `/props` under
`runtime.continuous_batching.decode_kv_offload_bytes`. Available memory is
checked again before each checkpoint allocation when reporting is available.
This is conservative sizing, not a reservation against other processes.

Use `--decode-kv-offload-mb 1024` to override the automatic value with a 1 GiB
per-model cap, or `--decode-kv-offload-mb 0` to disable it. Explicit nonzero
caps require concurrency greater than one. Automatic mode resolves to zero
for single-request or unsupported engines. Disabling recovery still checks
decode capacity and reduces speculation when necessary; if one-token decode
cannot fit, only the selected request fails, before model execution.

Before a decode step, the scheduler reserves its growth; if necessary it
first reduces speculation to one token, then suspends a newer request.
Copies finish before its blocks are released. The same slot retains
recurrent/compressor/draft state, sampler, pending token, and response
connection; existing SSE heartbeats continue during the pause. Parked
requests resume in admission order once resident requests drain, or earlier
when the pool has room for the checkpoint plus the cohort's next step; new
admissions wait while any request is parked.

A request that cannot be checkpointed within the RAM budget — or whose
checkpoint cannot be copied back — is not terminated: it parks without a
payload and resumes through ordinary chunked prefill over its retained
token history, which rebuilds paged KV and slot-local state together.
Requests whose context cannot fit the pool even alone, and unrecoverable
engine failures, still produce an error.

This is an in-process checkpoint: it neither survives a server restart nor
moves a request between models. RAM avoids disk I/O and checkpoint files.
Slot-local state remains on the device, so this recovers pool blocks, not
the whole request's device footprint. `/status/json` reports
`parked_requests` (suspended plus recompute-parked) and
`offloaded_kv_bytes` per model.

### PFlash

With a trained scoring head (`PFLASH_SCORING_HEAD_GGUF`), every PFlash setting
left unset takes the measured release profile: threshold 16384, the curve
`16384:0.315 32768:0.270 65536:0.270 120000:0.225`, budget selection over probe
segments, the request-probe query and the unified view for chats and agent
sessions (below). Without a head the built-in defaults
below apply. The [PFlash configuration reference](docs/PFLASH_CONFIGURATION.md)
lists both, the validated overrides and the active experiments.

| Option | Default | Purpose |
|---|---|---|
| `--pflash on\|off` | `off` | Turn PFlash prompt compression on. |
| `--prefill-threshold <N>` | `32000` (`16384` with a head) | Prompts from this many tokens are compressed; `0` compresses every prompt. |
| `--prefill-keep-ratio <F>` | `0.05` (the release curve with a head) | Fraction of source tokens kept. |
| `--prefill-curve T:R [T:R ...]` | none (the release curve with a head) | Piecewise keep-ratio curve; overrides the flat ratio. |
| `--prefill-drafter <path>` | none | PFlash drafter GGUF (Qwen3.5-0.8B). |
| `--prefill-skip-park` | off | Keep target and decode draft resident while PFlash runs. |
| `--prefill-drafter-device <backend:gpu>` | the `--draft-device` | Device the PFlash drafter scores on, in process: a GPU on the target's backend, with one explicit `--target-device` (not `auto` or `--target-devices`). On another GPU than the target, concurrent serving reserves nothing for it in the paged pool and a single request is not parked, so the target's GPU keeps its memory for context. |
| `--prefill-upstream-base <URL>` | none | Enable compression-proxy mode. |
| `--prefill-upstream-key <KEY>` | none | Bearer token for the upstream. |
| `--prefill-upstream-model <NAME>` | none | Model name forwarded upstream. |
| `PFLASH_SELECT_MODE=top_k` + `PFLASH_SELECT_TOPK <K>` | unset: legacy selection (`budget_only` with a head) | Rank rule: keep the K highest-scoring optional segments in score order instead of filling the keep ratio, with the keep-ratio budget still a hard ceiling (min(K segments, the budget)). Use it where the evidence is compact and sits in the first few ranks -- needle retrieval, passage QA, code -- so a small K reaches it for a fraction of the budget's tokens. Do not use it where the answer needs a whole document identified, since the evidence there spans many segments and K cuts it off. |
| `PFLASH_SELECT_QUERY_HEAD=1` (with `PFLASH_SELECT_QUERY_PARSER=latest_user`) | off (on with a head) | The question may come before pasted material. The user's leading prose block (up to the first paste-like line: a code fence, heading, separator, record/file header, tag, JSON or table data, indented or code-shaped line) is scored as an extra query window and kept verbatim; the trailing prose block after the last paste becomes the query window instead of the fixed tail. Caps: `PFLASH_SELECT_QUERY_HEAD_TOKENS` / `PFLASH_SELECT_QUERY_TAIL_TOKENS` (256); a longer block contributes its first and last cap/2 tokens. Blocks under 16 tokens are ignored; a message with no paste structure keeps today's window, except that a short opening paragraph ending in `?` before a long plain text counts as the head. |
| `PFLASH_SELECT_STRATEGY=fold` (with `PFLASH_SELECT_MODE=budget_only`) | `segments` | Fold selection: each of the top 20 segments (`PFLASH_SELECT_FOLD_K`) is completed to its smallest enclosing block -- a function or class in code (Python indentation or brace-matched C/C++/JS/TS/Go/Rust bodies), a numbered record such as `Document N:`, a markdown section, else a blank-line paragraph (`PFLASH_SELECT_FOLD_PARAGRAPHS=0` turns that last fallback off). A block over 1500 tokens (`PFLASH_SELECT_FOLD_CAP`) keeps its first 150 (`PFLASH_SELECT_FOLD_HEAD`) plus the segment. Code-like prompts (most non-blank lines inside functions; JSON and markdown are not) rank segments by mass (density x length), others by density. The keep-ratio budget stays the ceiling. Needs the head scorer; the drafter loads its vocabulary to read the prompt's structure. |

With compression on, `--max-ctx` is also the longest raw prompt PFlash takes,
and a longer one is refused with HTTP 400. A prompt that fits but leaves too
little room for its reply is compressed, and the reply is checked against the
compressed prompt. The drafter scores the whole raw prompt, and its scoring
memory grows with it, so set `--max-ctx` to what the card holds beside the
target: on a 32 GB R9700 with the 27B IQ4_XS target resident (q4_0 KV,
`--prefill-skip-park`), `--max-ctx 204800` fits, with a 204K-token prompt
peaking at 30.6 of 31.9 GiB. If the drafter does run out of memory, the
request is served from the raw prompt with full prefill (`[pflash-fallback]
... reason=compress_failed`).

With a Qwen3.5-0.8B drafter and strict budget selection
(`PFLASH_SELECT_MODE=budget_only`, `PFLASH_SELECT_CHUNK_SIZE`,
`PFLASH_SELECT_QUERY_TOKENS`), the drafter runs only its first fifteen
blocks and scores the context with block 15's NoPE Q/K projections as an
attention-mass scorer. Its 262K native context covers very long inputs. `PFLASH_SCORING_HEAD_GGUF` accepts a trained block-15 head
(schema `qwen3_5_0_8b_nope_qk_mass_v1`); `PFLASH_QWEN35_LEGACY_SCORER=1`
restores the previous all-layer running-max scorer. The Qwen3.5 attention
runs dense (`ggml_flash_attn_ext`); the block-sparse FlashPrefill kernels
still dispatch head dimension 128 only.

The scorer query of a chat is the prompt's last token: the end of the
generation prompt, where the model starts answering, having read the whole
request. Nothing is parsed out of the user's text, so the question can sit
anywhere in the message -- before a pasted document, in the middle of it, or
among the user's own sentences -- and the user's own words score far above
the material they paste. The tail (`PFLASH_SELECT_QUERY_TOKENS`) of the
latest user turn scores alongside it as a second query window at the same
weight: the last token reads the whole request, the user's own tokens match
literal strings -- an identifier, a described function -- that it does not
carry. Strict selection keeps the generation prompt and
the latest user turn's role header, and it runs on every turn of a
multi-turn chat. In an agent loop the assistant and tool turns after the
user's turn are scored like the rest of the conversation. A prompt without
chat markers scores the tail (`PFLASH_SELECT_QUERY_TOKENS`, default 8) of
its content.

The keep ratio applies to the droppable tokens only: what strict selection
keeps anyway (system and developer messages, tool definitions, the query and
its turn's envelope, the generation prompt) is added on top, so a long
system prompt no longer exhausts the budget. Auto mode compares
`--prefill-threshold` with the droppable tokens too. PFlash never compresses
the system prompt: one that alone would not fit the context fails the
request. Developer messages and tool definitions that would not fit lose
their pin and are scored like any other context.

Multi-turn chats keep a view: the prompt served for a turn is remembered,
and when the next request's prompt continues it (same tokens up to the old
generation prompt), PFlash serves that view plus the new turns instead of a
fresh compression, so the target restores its prefix-cache snapshot of the
view (taken at the start of its generation prompt) and prefills only what is
new. Segments the fresh selection keeps for the new question that the view
lacks are recalled as excerpts at the start of the new user turn. When the
view grows past twice the fresh prompt, or past the context, the fresh prompt
starts a new view. `PFLASH_CHAT_VIEW=0` serves the fresh compression every
turn.

What a turn adds is appended verbatim while it is small, the way full
prefill appends a follow-up; from `PFLASH_CHAT_COMPRESS_NEW_TOKENS` (default
16384) tokens of new material (a pasted document, a large tool output) only
what the fresh selection keeps of it is appended, and the view before it
stays cached. `PFLASH_CHAT_RECALL=0` turns recall off: a small follow-up is
then served without running the drafter at all. Recall takes what a fresh
selection for the new question keeps that the view lacks: a question on the
view's topic misses little and is served by appending it, and a question
that needs more than a third of a fresh selection -- the conversation moved
to other material -- starts a new view from that selection instead. Kept pieces that were not adjacent in the prompt are joined by a
paragraph break when neither side has one (`PFLASH_SELECT_PARAGRAPH_JOIN=0`
turns it off; the breaks do not count against the token ceiling).
`PFLASH_VIEW_TRACE_PATH`
appends each compressed request's served prompt as JSONL, for evidence
checks in evaluations.

Coding-agent sessions (with `PFLASH_ADMISSION=cache` and
`--pflash on`): in a conversation with tool activity (the
request defines `tools`, or its history has an assistant tool call or a tool
result) PFlash never compresses the agent's history or any tool output. By
default it compresses nothing in such a session: every request is served raw
with full prefill (`pflash_fallback.reason = keep_history`), and a raw prompt
over the context gets the context-overflow error full prefill gets.
`PFLASH_AGENT_PASTE=1` (default off) lets PFlash compress one thing: the
newest turn when it is a user message with at least `--prefill-threshold`
droppable tokens (a pasted document, on the first turn too). That content
alone is compressed, on its own keep budget, after the raw history (restored
from the prefix cache or prefilled) or after the conversation's view. A view
is then only continued: later turns and tool outputs are appended verbatim, a
new big paste is compressed as a region of its own, and recall adds the
passages of earlier pastes that the new question needs in the excerpt block
at the start of the newest user message, after the cached view. The view is
never rebuilt and never compressed again as a whole; when it cannot continue
(or no longer fits the context) the raw prompt is served. The switch is off
by default because follow-up questions after a compressed paste cost more
than full prefill, which keeps the whole raw prompt cached: a follow-up that
needs recall prefills the recalled passages again, and every user turn runs a
drafter selection (up to 48 s more per follow-up in a 10-turn coding session on
an R9700; the session took 281 s against 258 s with full prefill). Chats
without tools are not affected.

With a strict-selection environment whose scorer gives per-candidate scores
(the release profile, which also sets the `PFLASH_ADMISSION=cache` it needs),
chat and agent sessions are served by one mechanism, the unified view. Only the newest turn is ever paged: the user
messages since the last reply, or the tool results of the last tool call. A
message there with at least `--prefill-threshold` droppable tokens that can be
cut into lines becomes a region, and only what the drafter selection keeps of
it is served, in whole lines, with a note saying the content was shortened.
Tool outputs are paged for the tools in `PFLASH_REGION_TOOLS` (default `*`,
every tool; a comma-separated list names tools; empty pages none). Everything
before the newest turn is the view already served and stays cached: the
served prompt only grows by appending, and is never compressed again. A new
user question recalls, as excerpts at the start of its message, the lines of
earlier regions it needs that the view does not hold. Recall is gated: the
drafter scores the context for the question, and when the units the view does
not hold carry less than `PFLASH_RECALL_GATE` of that score mass (default
0.075, frozen from a development replay as the largest gate whose answers stay
within 3 of ungated recall; `off` recalls by the regions' caps alone) nothing
is appended; otherwise the best unheld units are recalled until what stays
unheld is below the gate. `PFLASH_RECALL=0` turns recall off. The request
probe's pins (the question it finds and its recency pin) stay whole, as in the
chat view. When the drafter pass fails, the newest content is served verbatim
and the view continues; `PFLASH_FAIL_CLOSED=1` refuses the request instead. A
re-read of a paged tool output is served verbatim. A view that would no longer fit the
context starts over from the raw prompt.

Requests the unified view does not take are handed back to the chat view
described above, which serves them as on its own: a request without a
messages array, a remote drafter, a turn structure or generation prompt it
cannot read, and a conversation it would start (no stored view continues it)
whose uncached history before the newest turn, with newest content that
cannot be cut into lines or has a line longer than its region's cap (one long
line of prose could only be dropped whole), reaches `--prefill-threshold`
(reason `uncached`,
or `reset_uncached` when a stored view could not continue). A conversation
handed back, or served raw after a PFlash failure, loses its unified views.
After an `uncached` or `reset_uncached` hand-back the unified view adopts what
the chat view served as the conversation's new view (no regions: what the
chat view left out of it cannot be recalled), so the next turn continues it
append-only and reuses its prefix. A tool session the chat view serves whole
is adopted the same way; after a PFlash failure served raw nothing is adopted
and the next turn starts a view of its own. The other hand-backs recur on every
request of their kind. The view trace records `pflash.unified = {decision:
handback, reason}` for these requests, and the unified stats count adoptions
(`adopt.<reason>`). Without a strict-selection environment, or with a
selection that gives no per-candidate scores (the legacy or split scorer,
`PFLASH_QWEN35_LEGACY_SCORER=1`, or the fold strategy), the unified view is
off (startup says so for the scorer) and every request takes the chat view.
`PFLASH_UNIFIED` is no longer read (startup warns when it is set).
`PFLASH_AGENT_PASTE`, `PFLASH_ARRIVE`, `PFLASH_CHAT_RECALL` and the other
chat-view knobs apply only to handed-back requests. `PFLASH_RECALL_SOURCE`
was removed.

Every turn of a multi-turn chat keeps its role header, and user turns (the
latest included) and assistant answers up to `PFLASH_CHAT_SKELETON_TOKENS`
(default 256 drafter tokens; 0 keeps headers only) stay whole: the
conversation's skeleton, as opposed to the material it quotes. Like
instructions, the skeleton is scored as context when it alone would not fit.
The last `PFLASH_CHAT_HISTORY_QUERIES` (default 3) earlier user turns score
the context alongside the current query, each through the last token of the
header of the reply that followed it (that turn's own prompt end), their
masses mixed in at weights 1/2, 1/4, 1/8, so what the conversation keeps
coming back to stays selected.

The drafter keeps a scoring session per conversation
(`PFLASH_DRAFTER_SESSIONS`, default 2, least recently used evicted; 0 scores
every prompt from scratch): the cache of blocks 0-14, the block-15 keys and
the probe logits of the prompt it last scored, with the recurrent state
checkpointed at the last 8192-token grid point at or before 64 tokens from
its end. A prompt that shares that prefix runs only its new tokens through
the drafter, from the checkpoint, and the new query scores against every
stored key. Every call keeps a fresh call's partition (ubatches and key
chunks counted from token 0), so a resumed selection is the one a fresh pass
would make; the query step reads its rows from this call or from an exact F32
store, and a request that cannot resume exactly is scored from token 0 with
the reason logged (`[qwen35-scorer] redo from token 0`).
With the default `--draft-residency auto` the drafter is released after each
compression; the sessions a call used are copied to host RAM first (plain
pageable memory; `PFLASH_SESSION_OFFLOAD_PINNED=1` uses the device's pinned
host memory instead, which on an R9700 raised peak VRAM by up to 1 GiB on
220K-token prompts, kept about 0.5 GiB of VRAM in use between requests and
slowed target prefill), only the positions a resume reads, and when the drafter loads again
only the session the new prompt can resume from comes back to the GPU. A
stored session is restored only under the drafter, head, probe and session
settings it was stored with; anything else drops it, and the call scores
from token 0.
`--draft-residency persistent` keeps the sessions on the GPU and copies
nothing.
A new session is sized with headroom (S + S/2 + 4096 tokens for an S-token
prompt) so later turns append without reallocating, but never past the
engine's `--max-ctx` or the drafter's trained context, and never below the
prompt itself.
A request's `pflash_query` string replaces the derived query and keeps its
whole span; it is meant for benchmarks. `PFLASH_SELECT_QUERY_PARSER=latest_user`
selects the benchmark parser, which finds the latest user message through
sentinel renders.

`PFLASH_SEGMENT_PROBE_GGUF` loads a segment probe (schema
`qwen3_5_0_8b_segment_probe_v1`): a 264K-parameter network on the same block-14
tap that scores every token for "a new unit of text starts here". With it
loaded, the context is cut at every boundary above the probe's threshold
(the query start and instruction-span edges are always cut; minimum and
maximum segment lengths come from the GGUF metadata) and the strict selector
ranks the resulting whole functions, classes, files or paragraphs by mass
density, skipping segments that do not fit the remaining budget, so a kept
piece is never a definition cut in half. It falls back to fixed chunks when
the probe finds fewer than four boundaries in a context.
`PFLASH_SELECT_SEGMENTS=auto|fixed|probe` and
`PFLASH_SELECT_SCORE=auto|sum|density` override the defaults (auto =
probe segments and density when a probe is loaded, fixed chunks and mass sum
otherwise); the compression trace records `segmentation`, `candidate_score`
and the segment spans.

The per-session adaptive keep ratio applies to this path unchanged: a request
carrying a `session_id` retains the session's ratio, the strict selector fills
its token budget from it, and the ratio is updated from the smoothed DFlash
acceptance rate after every turn where speculative decoding ran (below 75%
acceptance retain more, above 85% retain less, 0.5-1 point per turn, bounded
to 2.5-20%; `server/src/server/adaptive_keep_ratio.h`). A new session starts
from the configured ratio for its prompt length (`--prefill-keep-ratio` or
`--prefill-curve`), so the controller adapts around the real-use budget
instead of a fixed 10%. Acceptance is a proxy for compression quality: it
does not detect a dropped answer document directly, so the ratio curve and
the retention benchmarks remain the quality reference.

### Reasoning and MoE controls

| Option | Default | Purpose |
|---|---|---|
| `--think-max-tokens <N>` | model card | Reasoning-phase cap. |
| `--hard-limit-reply-budget <N>` | `4096` | Reserve output space by closing the reasoning phase near the limit; `0` disables. |
| `--reasoning-effort-low <N>` | model card | Token budget for low effort. |
| `--reasoning-effort-medium <N>` | model card | Token budget for medium effort. |
| `--reasoning-effort-high <N>` | model card | Token budget for high effort. |
| `--reasoning-effort-x-high <N>` | model card | Token budget for x-high effort. |
| `--reasoning-effort-max <N>` | model card | Token budget for max effort. |
| `--spark` | off | Enable self-tuning MoE expert placement. |
| `--spark-slots <N>` | auto | Set expert-cache slots per layer. |
| `--spark-vram <GiB>` | whole card | Cap total VRAM used by Spark. |
| `--freq` | off | Print expert-frequency analysis at shutdown. |
| `--collect-routing <path>` | none | Write binary routing data for predictor training. |

Burn-in switches, profiling controls, and compatibility environment variables are listed in the [environment reference](docs/ENVIRONMENT.md). The executable's usage text and `server_main.cpp` remain the source of truth for the current CLI.

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

## Hybrid MoE (hot/cold expert split)

For MoE targets whose experts don't fit in VRAM, dflash can split experts
across the GPU and CPU: the most-used (**hot**) experts stay resident on the
GPU, the rest (**cold**) live in host RAM and are evaluated on the CPU,
overlapped with the GPU hot path. This trades some decode/prefill speed for
VRAM headroom, and is computed automatically at load by a dynamic-placement
pass. It applies to both MoE arches: `qwen35`/`qwen36` and `laguna`.

**When it triggers.** If the experts fit in the available VRAM budget, all
experts load to GPU (no split, fastest path). Otherwise placement keeps as
many hot experts as the budget allows and routes the rest to CPU. You can also
shrink the budget manually to force a split (e.g. to free VRAM for a longer
context or a larger target).

### Spark: one-flag autotune (recommended)

`--spark` turns the split into a self-tuning command: it enables the bounded
GPU expert cache, sizes it from the VRAM target, loads a placement profile
next to the model (`<model>.gguf.spark.csv`) if present, and keeps refining it
from live traffic across restarts. Cap total VRAM with `--spark-vram <GiB>`, or
pin the cache ring directly with `--spark-slots <N>` (default: auto-sized from
the VRAM target).

```bash
luce_server models/laguna-xs2-Q4_K_M.gguf --spark                  # size to the card
luce_server models/laguna-xs2-Q4_K_M.gguf --spark --spark-vram 14  # cap total VRAM
luce_server models/laguna-xs2-Q4_K_M.gguf --spark --spark-slots 48 # pin 48 cache slots/layer
```

Under offload, laguna decodes the whole token in **one fused graph**
(`laguna_step_hybrid`), so throughput stays near the all-GPU ceiling (e.g.
~100 tok/s at 60% residency vs ~118 all-GPU on an RTX 3090); set
`LUCE_LAGUNA_NO_SINGLE_GRAPH=1` to fall back to the per-layer path.

### Budget knobs

| Env | Arch | Effect |
|---|---|---|
| `LUCE_EXPERT_BUDGET_MB N` | both | Cap hot-expert VRAM to `N` MB. Applies only when `N` is below the auto-computed budget; experts beyond it go cold (CPU). |
| `LUCE_EXPERT_BUDGET_PCT P` | laguna | Keep hot experts to `P`% (`0<P<100`) of total expert bytes. Applies only when below the auto budget. |
| `LUCE_MAX_CONTEXT N` | both | Override the max context used when sizing the KV cache (more KV = less VRAM left for hot experts). |

### Placement / tuning knobs (per arch)

Substitute `<ARCH>` = `LAGUNA` or `QWEN35MOE`:

| Env | Effect |
|---|---|
| `LUCE_<ARCH>_HOTNESS <file>` | Expert frequency/hotness file driving which experts are placed hot. |
| `LUCE_<ARCH>_TELEMETRY 1` | Log per-layer hot/cold FFN timing telemetry. |
| `LUCE_<ARCH>_SWAP_MAX N` | Max hot/cold promotions per request boundary (runtime re-placement); `0` disables swapping. |
| `LUCE_<ARCH>_SWAP_MIN_GAIN N` | Min observed-frequency gain before a cold expert is promoted to hot. |
| `LUCE_<ARCH>_NEXT_PLACEMENT_OUT <file>` | Dump the placement chosen this run (warm-start the hotness file next time). |
| `LUCE_QWEN35MOE_RUNTIME_STATS_OUT <file>` | (qwen only) Dump runtime routing-frequency stats. |

### Cache + single-graph knobs

Substitute `<ARCH>` = `LAGUNA` or `QWEN35MOE`. `--spark` sets these for you.

| Env | Effect |
|---|---|
| `LUCE_SPARK 1` | Enable the autotuning Spark path (set by `--spark`). |
| `LUCE_SPARK_VRAM_MB N` | Total VRAM target Spark sizes the hot tier + cache to (set by `--spark-vram`). |
| `LUCE_<ARCH>_EXPERT_CACHE 1` | Bounded GPU expert cache: swap selected cold experts into spare slots (LRU) so they are served on-GPU; cold-miss falls toward 0 after warmup. |
| `LUCE_<ARCH>_CACHE_SLOTS N` | Cache slots per layer (default: auto-sized from the VRAM target; `--spark-slots N` is the CLI equivalent). |
| `LUCE_LAGUNA_GPU_REMAP 1` | Serve the cache through the unified on-GPU FFN (required for the laguna cache to take effect). |
| `LUCE_LAGUNA_NO_SINGLE_GRAPH 1` | Fall back to per-layer decode instead of the default single-graph hybrid. |

### Example

```bash
# Force ~8 GB of hot experts on GPU; the rest run cold on the CPU.
LUCE_EXPERT_BUDGET_MB=8000 ./build/luce_server models/laguna-xs2-Q4_K_M.gguf --port 8000
# Startup log e.g.: "dynamic placement result: 4717 hot experts, 5267 cold experts"
```

### Caveat: reduced-stack prefill chunking

When a layer's hot-expert stack is **reduced** (i.e. a genuine split), the
ggml-cuda MMQ `mul_mat_id` kernel illegal-accesses for certain batch sizes on
**both** HIP/gfx1151 and CUDA/sm_86. As a guard, hybrid-split **prefill** is
sliced into ≤4-token sub-batches (forcing the stable MMVQ path); **decode**
(single-token) is unaffected. This costs some prefill throughput on split
layers and is removed once the kernel is fixed upstream.

## Laguna-XS.2 target (experimental, Poolside MoE)

[Poolside Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) is a 40-layer MoE LLM with 256 experts (top-8) plus an always-on shared expert, per-layer head counts `[48,64,64,64]×10`, and a per-layer SWA pattern (window 512). It is **architecturally distinct from `qwen35`**, so dflash adds a hand-rolled CUDA forward path (`Path A`, ggml-only — no libllama dependency) that mirrors the qwen35 stack. The Q4_K_M GGUF lands at 18.77 GiB on a single RTX 3090; tok_embd stays CPU-only (110 MiB) to keep the GPU budget under 24 GB.

### Single binary, single server

`test_dflash` peeks `general.architecture` from the target GGUF at startup
and dispatches by arch:

  - `qwen35` / `qwen36` → existing DFlash + DDTree pipeline (no change).
  - `laguna` → `luce::common::run_laguna_daemon()` (no spec-decode, no DDTree).

The daemon stdin/stream-fd protocol is identical, so `luce_server`
drives both arches end-to-end. The only thing the user changes is the model path.

### Build + run

```bash
cmake --build build --target test_dflash test_laguna_daemon pflash_daemon -j

# 19 GB Q4_K_M target + ~1.6 GB Qwen3.5-0.8B BF16 drafter + tokenizers
hf download Lucebox/Laguna-XS.2-GGUF laguna-xs2-Q4_K_M.gguf --local-dir models/
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-BF16.gguf --local-dir models/
hf download poolside/Laguna-XS.2 --local-dir models/Laguna-XS-2 \
    --include 'tokenizer*' '*.json'

# OpenAI-compatible HTTP server.
# luce_server routes to run_laguna_daemon() when arch=laguna.
./build/luce_server models/laguna-xs2-Q4_K_M.gguf \
    --max-ctx 16384 --port 8000

curl -sN http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'

# Smoke (loader only, no forward)
./build/smoke_load_target_laguna models/laguna-xs2-Q4_K_M.gguf

# Variable-N TTFT bench (LUCE_KV_TYPE=q4_0 for ctx > 32K, LUCE_CHUNK=2048 default)
LUCE_KV_TYPE=q4_0 ./build/bench_laguna_ttft models/laguna-xs2-Q4_K_M.gguf '4096,16384,65536'

# NIAH single-needle, with PFlash compression. The driver still spawns the
# standalone test_laguna_daemon binary so it can run without luce_server.
python3 scripts/laguna_pflash_niah.py \
    --target models/laguna-xs2-Q4_K_M.gguf \
    --drafter models/Qwen3.5-0.8B-BF16.gguf \
    --laguna-tok models/Laguna-XS-2 \
    --drafter-tok Qwen/Qwen3.5-0.8B \
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
- **Path B (in-process Python drafter)** is not used here. Path A keeps the luce daemon ggml-only.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox.git
cd lucebox/server

# Build (CUDA 12+, CMake 3.21+, sm_60+ GPU including Pascal; CUDA 13+ required for Jetson AGX Thor sm_110)
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
cmake --build build --target test_dflash luce_server -j

# Fetch models: ~16 GB target + 0.98 GB Lucebox Q4_K_M GGUF DFlash draft.
# Quickstart pins to Qwen3.6-27B (latest release). For Qwen3.5-27B swap in
# unsloth/Qwen3.5-27B-GGUF + z-lab/Qwen3.5-27B-DFlash; arch is identical so
# no rebuild is needed.
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/
hf download Lucebox/Qwen3.6-27B-DFlash-GGUF dflash-draft-3.6-q4_k_m.gguf --local-dir models/draft/

# Streaming one-shot generate (run.py defaults to models/Qwen3.6-27B-Q4_K_M.gguf;
# override with --target or LUCE_TARGET=... env var).
python3 scripts/run.py --prompt "def fibonacci(n):"

# Multi-turn chat REPL
python3 examples/chat.py

# OpenAI-compatible HTTP server (drop-in for Open WebUI / LM Studio / Cline).
./build/luce_server models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft models/draft/dflash-draft-3.6-q4_k_m.gguf --port 8000

# Reproduce paper numbers
python3 scripts/bench_llm.py                                 # HE + GSM8K + Math500
python3 scripts/bench_he.py --n-gen 256 --ddtree-budget 22   # minimal HE bench
```

**Long-context mode (up to 256K):**
```bash
LUCE_KV_TQ3=1 LUCE_PREFILL_UBATCH=16 \
  build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf \
  models/draft/dflash-draft-3.6-q4_k_m.gguf /tmp/long_prompt.bin 64 /tmp/out.bin \
  --fast-rollback --ddtree --ddtree-budget=16 --max-ctx=4096   # align_up(prompt + n_gen + 64, 256); raise up to 262144 for long prompts
```

**Requirements:** NVIDIA sm_60+ GPU (Pascal P40 24GB, V100 32GB, 2080 Ti, 3090, A10, A40, 4090) or Jetson AGX Thor sm_110, CUDA 12+ (CUDA 13+ required for Thor), 22+ GB VRAM, ~80 GB disk. Pascal GPUs use the scalar flashprefill fallback (no WMMA); P100 (12/16 GB) cannot fit the 27B target + draft under the 22 GB minimum. On Volta (SM 7.0) and Turing (SM 7.5), BF16 draft weights are auto-converted to FP16 at load time for tensor core acceleration.

### Verify your target

```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, 'sm_%d%d'%(p.major,p.minor), p.multi_processor_count,'SMs', round(p.total_memory/1e9,1),'GB')"
nvcc --version
```

### DGX Spark / GB10 (sm_121, CUDA 12.9+)

```bash
nvcc --version  # must show >= 12.9
git clone --recurse-submodules https://github.com/Luce-Org/lucebox.git && cd lucebox/server
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake auto-adds sm_121
cmake --build build --target test_dflash luce_server -j
```

On GB10 (128 GB unified), re-sweep `--ddtree-budget` (larger tree = more verify throughput until memory bandwidth saturates) and consider skipping KV quantization entirely (`--cache-type-k f16 --cache-type-v f16`).

### Jetson AGX Thor (sm_110, CUDA 13.0+)

```bash
nvcc --version  # must show >= 13.0
git clone --recurse-submodules https://github.com/Luce-Org/lucebox.git && cd lucebox/server
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CMake auto-adds Thor arch
cmake --build build --target test_dflash luce_server -j
```

### Per-GPU retune

- **DDTree `--ddtree-budget`**: 22 on 3090 + Q4_K_M + 24 GB (default). RTX 5090 = 40 (swept). GB10 = re-sweep.
- **TQ3_0 KV** sized for 24 GB. On 32 GB (5090) or 128 GB (GB10) you can push context further or skip quantization.
- Headline numbers (207 tok/s demo, 129.5 HumanEval, 2.8× vs SGLang AWQ) are RTX 3090 stock. RTX 5090 numbers (205 tok/s HE, 4.84×) are in [RESULTS.md](RESULTS.md). Ada / GB10 / Thor not yet swept, PRs welcome.

## AMD HIP backend (Strix Halo, RX 7900 XTX)

Same DFlash + PFlash stack on AMD GPUs. PR #119 ports the Phase 2 rocWMMA flashprefill kernels to HIP. End-to-end on a Ryzen AI MAX+ 395 box (Radeon 8060S iGPU, `gfx1151`, 128 GiB LPDDR5X-8000 unified): **37.0 tok/s DFlash decode** on Qwen3.5-27B Q4_K_M, **27.6 s TTFT @ 16K** with NIAH retrieval intact. **3.08× decode and 2.24× prefill over llama.cpp HIP AR** on the same iGPU. End-to-end wall clock at a 16K prompt + 1K generation workload: **2.66× faster** than vanilla llama.cpp.

**RDNA4 — Radeon AI PRO R9700 (`gfx1201`, 32 GB).** First-class RDNA4 target as of this build. Qwen3.6-27B Q4_K_M + DFlash draft (`dflash-draft-3.6-q4_k_m.gguf`), `--ddtree-budget=22`: **54.65 tok/s mean DFlash decode** across the 10-prompt HumanEval suite (`bench_he.py --n-gen 256`, AL 7.14, range 36.9–93.0 tok/s) on ROCm 7.1.1. The rocWMMA Phase 2 flashprefill kernels are numerically correct on RDNA4 — ROCm 7.1's rocWMMA handles the gfx12 WMMA operand-format change internally, so no kernel changes are needed (`test_flashprefill_kernels` PASS on `gfx1201`: max diff 5e-4, e2e `flash_prefill_forward_bf16` at S=8192 in 10.7 ms/iter). Note `gfx1200` (RX 9060) and `gfx1201` (RX 9070 / R9700) are **not** code-object compatible — build for `gfx1201` explicitly for the R9700.

For Qwen3.8-27B IQ4_XS with the Q8_0 DFlash2 drafter, the checkpoint declares
a block size of 8: one seed token plus seven speculative tokens. Greedy chain
verification keeps the output byte-identical to plain decode at any width, so
`--draft-block-size` accepts values from 2 up to 2x the checkpoint metadata;
widening only risks acceptance depth, and on this checkpoint it extrapolates:
completions stayed byte-identical across widths 8/10/12/16 while per-step
commits grew.

On the R9700 (`gfx1201`), chain verification with `--draft-block-size 16` is
the fastest measured code/math configuration: HumanEval ten-prompt decode went
from 143.6 tok/s (block 8) to 204.1 average with per-request peaks above 230,
and a five-prompt math set from 129.2 to 152.6. Prose stays on block 8 (low
acceptance pays the wider draft without commit gains). Widths past 16 measured
a step-time cliff with no commit gain and stay rejected. The block-8
`--specla --ddtree-budget 8` configuration is a quality-validated alternative
(the out-of-the-box default is the plain chain at the checkpoint's block 8):
163.9 versus 159.0 aggregate decode tok/s across repeated ten-prompt
HumanEval runs (+3.1%), HumanEval+ pass@1 145/164 versus 143/164, with all ten
short A/B replies and 133/164 full-suite replies byte-identical.

```bash
git clone --recurse-submodules https://github.com/Luce-Org/lucebox.git && cd lucebox/server

# Ubuntu/ROCm build dependencies used by ggml's HIP backend.
sudo apt-get update
sudo apt-get install hipblas-dev hipcub-dev rocblas-dev rocprim-dev rocwmma-dev

# Build for gfx1151 (Strix Halo). Swap arch for gfx1100 / gfx1201.
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DLUCE_GPU_BACKEND=hip \
  -DLUCE_HIP_ARCHITECTURES=gfx1151 \
  -DLUCE_HIP_SM80_EQUIV=ON
cmake --build build --target test_dflash -j
```

`LUCE_HIP_SM80_EQUIV=ON` enables the rocWMMA Phase 2 flashprefill kernels (path that delivers the prefill speedup). `OFF` falls back to ggml's `flash_attn_ext` (slower but no rocwmma headers needed). With `SM80_EQUIV=ON` the build also produces `test_flashprefill_kernels` (HIP) — run it on your card to validate the rocWMMA kernels numerically (`HIP_VISIBLE_DEVICES=<gpu> ./build/test_flashprefill_kernels`).

**Per-arch DDTree tuning:** `gfx1151` (Strix Halo iGPU, bandwidth-bound on LPDDR5X) peaks at `--ddtree-budget=22`. `gfx1100` (7900 XTX, GDDR6) prefers `budget=8` per the [PR #156 cross-arch perf plan](https://github.com/Luce-Org/lucebox-hub/pull/156). `gfx1201` (RDNA4 / R9700, GDDR6) prefers `budget=22` (`budget=8` is a ~9% regression). Run `scripts/bench_he.py --ddtree-budget N` to verify on your card.

> **Multi-GPU / distro note.** On a host with more than one AMD GPU, pin the bench to the target with `HIP_VISIBLE_DEVICES`. On distros that link PIE executables by default (e.g. Fedora's system ROCm under `/usr`), add `-DCMAKE_EXE_LINKER_FLAGS=-no-pie` to the `cmake` configure line, and point at the toolchain with `-DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr -DROCM_PATH=/usr` if ROCm lives under `/usr` rather than `/opt/rocm`.

**Drafter recipe for max decode:** target = Qwen3.5-27B Q4_K_M, drafter = same gen quantized to Q8_0 via `server/scripts/quantize_draft_q8.py`. For the unsloth Qwen3.6 target, pass `--qwen36-swa` when creating the Q8_0 draft so its GGUF embeds the required 2048-token, 4-of-5-layer sliding-window configuration. Older Qwen3.6 drafts without that metadata still need `LUCE_DRAFT_SWA=2048` at runtime.

See also: [`docs/HIP_PERF_PLAN.md`](docs/HIP_PERF_PLAN.md) (perf sweeps), [`docs/MIXED_BACKEND.md`](docs/MIXED_BACKEND.md) (mixed CUDA+HIP runs).

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

Open an issue or PR against `Luce-Org/lucebox`. Good first picks:

- **Leviathan-style rejection sampling** on each DDTree branch (the current implementation samples only the committed token; full prob-matching across the tree is the next step)
- **Full llama.cpp integration**: new arch, `llama-speculative-dflash.cpp`, `llama-cli` / `llama-server` wiring

## Citation

```bibtex
@software{luce_dflash_2026,
  title  = {Luce DFlash: GGUF port of block-diffusion speculative decoding for Qwen3.5-27B on consumer GPUs},
  author = {Lucebox},
  url    = {https://github.com/Luce-Org/lucebox/tree/main/server},
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
./build/luce_server models/Qwen3.5-27B-Q4_K_M.gguf \
  --draft models/Qwen3.5-3B-f16.safetensors \
  --ddtree --ddtree-budget 22 --port 8080
```

### 2. Configure Codex

Create or edit `~/.codex/config.toml`:

```toml
model = "luce-dflash"
model_provider = "luce"

[model_providers.luce]
name = "Luce"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
supports_websockets = false
```

No `env_key` is needed — the local server accepts any token.

### 3. Run Codex

```bash
codex --provider luce "Explain this codebase"
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

Tool-call history from OpenAI Chat Completions, Responses, and Anthropic Messages
is normalized into model-independent calls (IDs, names, and typed arguments).
The selected renderer then emits the model's native protocol: Qwen function
blocks, Bailing/Laguna argument blocks, DeepSeek V4/V4.1 DSML, or Gemma tool
calls and responses. Tool results retain their call IDs and resolve function
names from the corresponding calls, including out-of-order parallel results.
DeepSeek V4/V4.1 follow the reference encoder: results are reordered by call ID
into call order before rendering their ID-less result blocks, while intervening
user text stays in place. Missing or unknown result IDs use the reference's
rank-zero fallback, preserving arrival order among ties. Cold tool-call history
closes the thinking channel before assistant prose and calls; remembered raw
turns retain their existing reasoning. Other built-in renderers retain result
arrival order, so clients should send parallel results in call order when the
model protocol cannot uniquely identify the calls.

For built-in renderers, if every call has a non-empty ID and tool memory resolves
all IDs to the same remembered turn, the server replays that turn byte for byte
without duplicating assistant text or calls. Otherwise every call is reconstructed
in order. Missing IDs, partial cache misses, eviction, and server restarts do not
discard calls. Object arguments become named parameters; non-empty argument
strings that are invalid JSON or encode a non-object value are preserved as one
`arguments` parameter. Empty or omitted arguments render a parameterless call.

With `--chat-template-file`, the template always receives structured
`messages[].tool_calls`, object-valued `function.arguments`, and tool-result
`tool_call_id` / `name` fields. Tool definitions from all three APIs are exposed
as OpenAI-shaped `tools[].function` objects. Raw tool-memory replay and its
canonical-turn cache optimization are bypassed for custom templates: templates
such as Gemma's need the structured calls to locate and render the results.
Custom templates that previously read flat `tools[].name` or `input_schema`
must use `tools[].function.name` and `tools[].function.parameters` instead.
Custom templates remain limited to features supported by the bundled Jinja
interpreter; filter-mapped `map('upper')` (used by Gemma's template for union-type
schemas) is currently unsupported.
