# DeepSeek V4 Flash — DFlash Integration

This document describes the current DeepSeek V4 Flash implementation in
DFlash. DeepSeek4 supports a monolithic HIP backend, a layer-split backend,
and an in-process heterogeneous expert-parallel backend for a discrete GPU
paired with Strix Halo.

## Model Architecture

DeepSeek V4 Flash is a 43-layer MoE model with:

| Parameter | Value |
|-----------|-------|
| Hidden dim (`n_embd`) | 4096 |
| Attention heads | 64 (MLA: 1 KV head, low-rank Q/O projections) |
| Head dim | 512 (partial RoPE on 64 dims, YaRN scaling) |
| Experts per layer | 256 routed (top-6) + 1 shared |
| Expert FFN dim | 2048 |
| First 3 layers routing | Hash-based (token ID → expert table) |
| Remaining layers routing | Top-k over learned router + optional bias |
| KV Compression | Learned compressor (ratio-4 even, ratio-128 odd) |
| Indexer | Top-k scorer on ratio-4 layers for compressed KV selection |
| HC (Hierarchical Controller) | 4 parallel residual streams, Sinkhorn-normalized combine |

## Tool-call compatibility

The native DeepSeek V4 renderer requests `<function_call>` output. The parser
also accepts named bare-JSON fallbacks emitted by compatible checkpoints:
`{"function":"name","parameters":{...}}` and the legacy OpenAI
`{"function_call":{"name":"name","arguments":{...}}}` envelope. Named JSON
calls remain unambiguous when a request supplies more than one tool; ordinary
JSON that does not resolve to an allowed tool is preserved as assistant text.

## Reasoning effort

The native DeepSeek V4 renderer supports the official `low`, `high`, and `max`
reasoning encodings. `low` adds no prefix; `high` and `max` prepend their
official model-facing instruction before the system message. The server accepts
`reasoning.effort`, top-level `reasoning_effort`, and the official
`chat_template_kwargs` form:

```json
{"chat_template_kwargs":{"thinking":true,"reasoning_effort":"max"}}
```

For DeepSeek V4 Flash API compatibility, `medium` and `xhigh` map to `high`.
Lucebox's hyphenated `x-high` extension retains its own phase-1 budget tier
and uses the `max` model-facing encoding. Explicitly enabling thinking without
an effort uses DeepSeek's `high` default. Bare `chat_template_kwargs.thinking`
and `chat_template_kwargs.enable_thinking` toggles only control prompt rendering;
they do not activate the force-close budget envelope unless the request also
sets an explicit effort. Disabling thinking suppresses every effort prefix.

## Code Layout

| Area | Files |
|------|-------|
| Backend selection / init | `src/common/backend_factory.cpp`, `src/deepseek4/deepseek4_backend.{h,cpp}`, `src/deepseek4/deepseek4_layer_split_adapter.{h,cpp}` |
| Per-shard forward graph | `src/deepseek4/deepseek4_graph.cpp` |
| Model weights and metadata | `src/deepseek4/deepseek4_internal.h`, `src/deepseek4/deepseek4_loader.cpp` |
| HC pre/post CUDA kernel | `src/deepseek4/deepseek4_hc_cuda.cu`, `.h` |
| DSpark runtime | `src/deepseek4/deepseek4_dspark*.{h,cpp}` |
| Heterogeneous expert storage/evaluation | `src/common/moe_hybrid_{storage,ffn_eval}.*` |
| Remote target-shard daemon | `src/deepseek4/deepseek4_target_shard_ipc_daemon.cpp` |
| Shared target-shard IPC infrastructure | `src/common/target_shard_ipc.*`, `src/placement/remote_target_shard_config.h` |
| Backend IPC CLI entry | `src/ipc/backend_ipc_main.cpp` |

## Forward Pass (Layer-Split Path)

`deepseek4_step_layer_range()` drives per-shard execution over a contiguous layer range:

1. **Embedding + HC init** — On the first shard (`layer_begin == 0`), token embeddings are replicated into all HC streams to initialize the per-token HC state.
2. **Per-layer forward** — Each layer runs the HC-enabled sequence: HC pre (attention) → MLA attention → HC post (attention) → HC pre (FFN) → router + MoE FFN → HC post (FFN).
3. **Decode HC fast path** — For single-token decode (`n_tokens == 1`), the runtime reuses cached decode graphs. CUDA decode uses the cached backend HC graph path; HIP decode uses the direct HC-pre helper plus refreshed HC-post weights.
4. **Shard boundary handoff** — Non-final shards return the updated **full HC state tensor** (`n_tokens × n_hc × n_embd`) to the next shard.
5. **Tail shard completion** — The last shard resumes at its `layer_begin`, runs the remaining layers, then performs the final HC merge, RMSNorm, and `lm_head` projection to produce logits.

The layer-split path keeps MoE computation inside the shard that owns each
layer. The heterogeneous path is different: it partitions every layer's routed
experts between two local HIP backends and joins their results in process. It
does not use the retired per-expert IPC worker.

## Execution Modes

### Monolithic HIP

Single-device HIP launches use `DeepSeek4Backend`. Two explicit serving
options are available:

- `--ds4-fused-decode` enables the cached single-graph decode path. It keeps
  HC, attention, MoE, and the output projection on the GPU and avoids
  per-layer host round trips. On HIP this option requests a monolithic model
  load because the fused graph must reference every expert tensor directly.
  If that allocation fails, the backend logs the fallback and continues with
  hybrid expert placement and layered decode.
- Adaptive qtype-105/106 experts work in monolithic mode and across two GPUs
  using the same runtime. The loader gives each GPU's compact tensor the
  decode-table rows for the experts it owns. CPU expert offload and mixed
  CUDA/HIP peers keep the safe monolithic fallback.

- `--ds4-expert-top-k N` keeps the highest-ranked `N` routed experts and
  renormalizes their weights. `0` uses the model default. Reducing this value is an
  approximate inference policy and must be quality-validated for the target
  workload.

  **Do not reduce it on an adaptive (qtype-105/106) artifact without measuring.**
  Measured 2026-08-05 on DeepSeek-V4-Flash-0731, exact-copy fidelity over 20
  identifiers x 3 repeats at temperature 0 — the model is asked to echo a line of
  Python that is already in its prompt, so anything but a verbatim copy is a defect:

  | artifact | `--ds4-expert-top-k 6` (model default) | `--ds4-expert-top-k 4` |
  |---|---|---|
  | adaptive (105 down-experts) | 95.0% | **60.0%** |
  | uniform (104 down-experts) | 100% | 100% |

  The adaptive failures are not degraded paraphrases, they are degenerate output —
  `" 0 0 0 0 0 ..."`, repeated markdown fragments, empty strings — on prompts as
  trivial as "repeat this line". Deterministic, and reproducible across context
  sizes and with fusion on or off. The uniform artifact tolerates the same
  approximation perfectly, so top-4 leaves no error margin and the adaptive
  formats' different error profile crosses the threshold. Serve adaptive artifacts
  at the model default.

For the validated single-device Strix Halo profile:

```bash
./server/build-hip/dflash_server /opt/models/DeepSeek-V4-Flash.gguf \
  --target-device hip:0 \
  --ds4-fused-decode
```

### In-process heterogeneous expert parallel

The Lucebox path keeps dense target work, selected hot experts, and the local
DSpark drafter on the discrete GPU. Strix Halo owns the remaining routed
experts. Both owners are submitted for each MoE layer before an on-device join;
the target KV cache and sampler remain on the main GPU. This is route-level
expert parallelism, not layer pipelining or a second server process.

Build one HIP binary for both architectures. For the qualified R9700 + Strix
Halo machine:

```bash
cmake -S server -B server/build-hip-dual \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1151;gfx1201' \
  -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build server/build-hip-dual -j
```

The feature is still a burn-in profile and is disabled by default. The minimum
placement controls are:

```bash
export DFLASH_DS4_MOE_TP=1
export DFLASH_DS4_MOE_TP_INPROC=1
export DFLASH_DS4_MOE_TP_GPU=1       # Strix Halo
export DFLASH_EXPERT_BUDGET_MB=11700 # hot experts on the R9700
export DFLASH_DS4_DRAFT_GPU=0        # local drafter on the R9700
export LUCE_MMVQ_MAX_NCOLS=4

./server/build-hip-dual/dflash_server /path/to/deepseek4-target.gguf \
  --target-device hip:0 \
  --peer-access \
  --ds4-prefill sparse
```

Top-4 routing and sparse prefill are explicit approximations. Omit them when
the model-default top-6 route or exact prefill is required. The 2026-07-29
qualification used a fixed q=4 local drafter, two discarded warm-ups, R9700
`auto`, and Strix Halo `high`. A 1,956-token prompt measured 51.1 tok/s median
decode and 415.52 tok/s median sparse prefill. Those numbers require the full
qualified manifest, including the burn-in kernel switches; they are not a
claim for the minimal activation example above.

#### Radeon RX 7900 XT + Strix Halo, true top-k-6

The portable `scripts/serve_ds4_dual_rocm_128k.sh` profile captures the
qualified fixed-codebook ROCmFPx configuration for a 20 GiB gfx1100 discrete
GPU plus a 128 GiB gfx1151 Strix Halo. It keeps the model-default six routed
experts, places dense work and calibrated hot experts on gfx1100, and runs the
remaining experts and DSpark drafter on gfx1151. This profile is for uniform
or fixed-codebook expert tensors. Adaptive qtype-105/106 expert tensors remain
monolithic-only because their codebook registrations cannot follow a sliced
expert allocation.

Build for both targets with affine qtype-107 support enabled:

```bash
cmake -S server -B server/build-hip-dual \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1100;gfx1151' \
  -DDFLASH27B_ROCMFP2_AFFINE=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build server/build-hip-dual -j
```

Placement depends on the workload's routing distribution. Capture a profile
with representative requests, stop the server so it flushes the CSV, then use
that CSV for serving:

```bash
DFLASH_DS4_ROUTING_STATS_OUT=/tmp/ds4-routing.csv \
  server/scripts/serve_ds4_dual_rocm_128k.sh \
  /path/to/target.gguf /path/to/dspark.gguf

DFLASH_DS4_HOTNESS_CSV=/tmp/ds4-routing.csv \
  server/scripts/serve_ds4_dual_rocm_128k.sh \
  /path/to/target.gguf /path/to/dspark.gguf
```

The script defaults to `hip:0` for gfx1100, peer and draft device `1` for
gfx1151, a 10,200 MiB discrete-GPU expert budget, 135,168 context slots
(128 KiB prompt plus 4 KiB generation headroom), sparse prefill, and disabled
prefix/prefill caches. Every device, budget, context, build, host, and port
setting is environment-overridable at the top of the script.

The controlled decode workload is deliberately predictable: temperature zero,
true top-k-6, fixed DSpark q=4, one warm-up, three fresh 512-token `BETA`
sequence requests, and no prefix or prefill cache. It measures the upper-bound
speculative path, not typical conversational or agent throughput. Run the
checked-in harness against the qualified profile:

```bash
python3 server/scripts/bench_ds4_decode.py \
  --url http://127.0.0.1:8016 \
  --model dflash \
  --warmups 1 \
  --runs 3 \
  --max-tokens 512
```

The harness rejects partial generations, cache hits, and autoregressive
fallbacks, requires the measured outputs to be byte-identical, and records
server-reported decode timing, acceptance, and output SHA-256 values as JSON.
On a Radeon RX 7900 XT plus
Ryzen AI Max+ 395, a clean rebase onto upstream `main` using the script's
default 10,200 MiB expert budget and 135,168-token context produced 100%
speculative acceptance and one byte-identical output digest:

| Run | Decode time | Throughput |
|---:|---:|---:|
| 1 | 11.3772 s | 45.0 tok/s |
| 2 | 10.9023 s | 47.0 tok/s |
| 3 | 10.7302 s | 47.7 tok/s |

Retain that JSON with the target and draft GGUF SHA-256 values, build revision,
ROCm version, GPU identifiers, launch environment, and context length. An
earlier qualified build produced three 46.8 tok/s runs; its previous
same-hardware route/dispatch baseline was 37.7 tok/s, a 24.1% improvement.
The checked-in kernel test verifies actual affine MMQ dispatch on gfx1151 and
gfx12xx. gfx1100 uses the supported non-MMQ fallback and is not claimed as an
MMQ qualification.

Sparse-prefill records used several explicitly different approximation
profiles and must not be compared as if the launch settings were identical.
A true top-k-6 128 KiB sweep measured 111.2 tok/s at 132,981 tokens. Older
top-k-4 profiles measured 194.51 tok/s at 12,281 tokens, 158.9-164.3 tok/s
around 2K tokens, and 53.1 tok/s at 257,965 tokens. Sparse prefill is an
explicit approximation; applications that require reference-exact prompt
ingestion must use `--ds4-prefill exact` and remeasure.

#### CUDA 3090 + Strix Halo in one process

The mixed-vendor build links the selected target runtime normally and loads
the other vendor as an isolated backend module. CUDA and HIP device indices
have separate namespaces, so `cuda:0` and `hip:0` are a valid pair.
Cross-vendor activations are staged in host memory inside the process; native
peer access is intentionally not attempted.

```bash
cmake -S server -B server/build-cuda-hip \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_ENABLE_MIXED_CUDA_HIP=ON \
  -DDFLASH27B_CUDA_ARCHITECTURES=86 \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build server/build-cuda-hip -j
ctest --test-dir server/build-cuda-hip -R mixed_cuda_hip --output-on-failure
```

```bash
export DFLASH_DS4_MOE_TP=1
export DFLASH_DS4_MOE_TP_INPROC=1
export DFLASH_DS4_MOE_TP_BACKEND=cuda
export DFLASH_DS4_MOE_TP_GPU=0       # cuda:0 (RTX 3090)
export DFLASH_DS4_MOE_TP_CONCENTRATE_COLD=1
export DFLASH_DS4_TP_SCHEDULE_BRANCHES=1
export DFLASH_DS4_TP_TARGETED_JOIN_SPLIT=1
export GGML_BATCH_PEER_COPIES=1
# Start conservatively and tune from the startup placement and memory logs;
# the usable budget depends on the model, placement policy, and free VRAM.
export DFLASH_EXPERT_BUDGET_MB=85000
export DFLASH_DS4_DRAFT=/path/to/dspark-draft.gguf
export DFLASH_DS4_DRAFT_BACKEND=cuda
export DFLASH_DS4_DRAFT_GPU=0

./server/build-cuda-hip/dflash_server /path/to/deepseek4-target.gguf \
  --target-device hip:0 \
  --ds4-prefill sparse
```

The peer module is normally found beside the executable. Set
`DFLASH_CUDA_BACKEND_PATH` or `DFLASH_HIP_BACKEND_PATH` only when packaging it
elsewhere. Sparse/approximate DeepSeek4 prefill remains restricted to a HIP
target; CUDA-primary ROCmFP2 execution is not yet qualified. The mixed path is
burn-in functionality. On the qualified 3090 + Strix machine, the tuned top-4
performance profile held 48.1 tok/s median on the deterministic 128-token
workload. The all-6-expert reference-exact mode is a correctness profile, not
a throughput profile.

### Strix Halo concurrent serving

DeepSeek4 paged concurrency supports two resident HIP deployments:

- one monolithic Strix Halo (`gfx1151`) target with every expert on that
  device, through 6 lanes; and
- the in-process R9700 (`gfx1201`) target + Strix Halo expert-parallel
  deployment, qualified at concurrency 1–4 with the grouped expert setting
  below. The target keeps dense work and its selected experts while the
  secondary owns the remaining materialized experts. The runtime accepts up
  to 6 lanes, but concurrency 5–6 is not qualified for this configuration.

The heterogeneous mode is route-level expert parallelism. It is not an
explicit `--target-device hip:0,hip:1` layer split and does not use a remote
target shard or host-streamed experts.

The backend keeps raw MLA rows, compressed rows, and indexer state in a
persistent 128-token paged cache. The shared
HTTP scheduler performs admission, cancellation, slow-client isolation, and
fair continuous batching. DeepSeek4 lowers each scheduler plan into one exact
gathered graph with up to 6 independent lanes. Decode rows share the weight
pass; each selected prompt advances by one exact token because the graph must
not contain two rows from the same sequence.

```bash
hf download Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3 \
  DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf \
  --local-dir /path/to/models

cmake -S server -B server/build-hip \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
  -DDFLASH27B_SERVER=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build server/build-hip -j

./server/build-hip/dflash_server \
  /path/to/models/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf \
  --target-device hip:0 \
  --paged-attention \
  --max-concurrency 6 \
  --kv-pool-tokens 24576 \
  --max-ctx 4096 \
  --ds4-prefill exact \
  --prefix-cache-slots 0
```

For the R9700 + Strix Halo path, build `server/build-hip-dual` for
`gfx1151;gfx1201` as shown in
[In-process heterogeneous expert parallel](#in-process-heterogeneous-expert-parallel).
Then expose the R9700 first and select the static in-process expert split.
The grouped expert setting below is qualified at concurrency 1–4 with
ROCmFP2 gate/up and ROCmFP3 down weights:

```bash
export HIP_VISIBLE_DEVICES=<r9700-index>,<strix-index>
export DFLASH_DS4_MOE_TP=1
export DFLASH_DS4_MOE_TP_INPROC=1
export DFLASH_DS4_MOE_TP_GPU=1
export DFLASH_EXPERT_BUDGET_MB=11700
export DFLASH_DS4_TP_BATCH_SPLIT_COPIES=1
export DFLASH_DS4_TP_GROUPED_MMVQ=1

./server/build-hip-dual/dflash_server \
  /path/to/models/DeepSeek-V4-Flash.gguf \
  --target-device hip:0 \
  --peer-access \
  --paged-attention \
  --max-concurrency 4 \
  --kv-pool-tokens 24576 \
  --max-ctx 4096 \
  --ds4-prefill exact \
  --prefix-cache-slots 0
```

The heterogeneous paged cache is charged against the R9700 before selecting
resident experts. Increase `--kv-pool-tokens` only when the primary has enough
memory for the larger
shared history pool.

Paged concurrency fails closed for other primary/secondary architecture pairs,
CUDA or out-of-process expert ownership, explicit layer or remote target
splits, drafts/DSpark, DDTree, PFlash/KVFlash, fused decode, approximate
prefill, windowed attention, mutable expert caching, and prefix-cache parking.
There is no automatic fallback to a slower or asymmetric execution mode.

### Local single-shard

If the adapter decides all 43 layers fit on one CUDA GPU, it loads a single shard locally and no IPC daemon is involved.

### Local multi-shard

When the server is configured with an explicit local layer split across multiple GPUs, the adapter loads each contiguous shard locally and executes them in order.

### CUDA parent + Halo target-shard split

For heterogeneous setups, the CUDA-built server can keep the prefix layers on the CUDA GPU and launch the suffix shard on the Halo/HIP build through the existing target-shard IPC path:

```
┌─────────────────────────────────────────────────────────────┐
│  CUDA Parent                                                │
│  - Token embedding                                          │
│  - Layers [0, split)                                        │
│  - Maintains local KV/cache state for its layer range       │
│  - Emits updated HC-state tensor at the shard boundary      │
├─────────────────────────────────────────────────────────────┤
│  Halo Target Shard (IPC daemon)                             │
│  - Layers [split, 43)                                       │
│  - Resumes from boundary HC state                           │
│  - Final HC merge, RMSNorm, lm_head                         │
│  - Returns logits / sampled token to the parent             │
└─────────────────────────────────────────────────────────────┘
```

This path uses `TargetShardIpcSession`, `deepseek4_target_shard_ipc_daemon.cpp`, and `BackendIpcMode::DeepSeek4TargetShard` rather than the old expert-worker protocol.

## Shard Boundary State

The shard boundary transfers the **full HC state tensor**, not a separate expert-routing payload:

- **Boundary activation / HC state** — `[n_tokens × n_hc × n_embd]` floats. DeepSeek4 uses `n_hc = 4`, so the per-token boundary payload is `4 × 4096` floats.
- **Sequence position / token metadata** — enough information for the tail shard to continue cache updates and finish the forward pass.

KV cache tensors remain owned by the shard that owns the corresponding layer range.

## Auto-Split Behavior

If DeepSeek4 is started without an explicit target layer split, `DeepSeek4LayerSplitAdapter` computes the CUDA prefix automatically:

1. Read `DFLASH_DS4_CUDA_LAYERS`. If it is set to a positive value, that value becomes the number of prefix layers kept on CUDA.
2. Otherwise, query CUDA free memory.
3. Reserve a fixed **2 GiB** overhead for caches and safety margin.
4. Estimate roughly **1.9 GiB per DeepSeek4 layer**.
5. Clamp the result so at least one layer remains on each side (`1..42`), then assign the remaining `43 - N` layers to Halo.

The runtime logs the chosen split with a `[deepseek4-split] auto-split:` banner.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DFLASH_DS4_CUDA_LAYERS` | Override the auto-split heuristic and pin the first `N` DeepSeek4 layers to CUDA. The remaining `43 - N` layers run on the Halo shard. |
| `DFLASH_DS4_TIMING` | Enable DS4 timing logs for local, paged, and layer-split execution. Paged rounds report full-graph build, input upload, compute, and readback time; leave unset for normal runs. |
| `DFLASH_DS4_ROCTX` | HIP-only, default-off semantic ROCTX ranges for an external rocprof trace. The library is loaded dynamically only when set to `1`, `true`, `yes`, or `on`. |
| `DFLASH_DS4_SPEC` / `DFLASH_DS4_DRAFT` | Enable DSpark and select its GGUF. |
| `DFLASH_DS4_DRAFT_BACKEND` / `DFLASH_DS4_DRAFT_GPU` | Backend and device for the in-process drafter. |
| `DFLASH_DS4_MOE_TP` | Enable routed-expert partitioning. |
| `DFLASH_DS4_MOE_TP_INPROC` | Use two local GPU backends instead of an expert IPC worker. |
| `DFLASH_DS4_MOE_TP_BACKEND` | Cold expert backend (`cuda` or `hip`); mixed builds default to the peer runtime. |
| `DFLASH_DS4_MOE_TP_GPU` | Device index within the cold expert backend. |
| `DFLASH_DS4_MOE_TP_CONCENTRATE_COLD` | Cross-vendor burn-in mode: place complete cold expert layers on the peer to reduce joins. |
| `DFLASH_DS4_MOE_TP_PEER_HOT` | With a routing profile, place its hottest experts on the secondary owner. |
| `DFLASH_DS4_CROSS_VENDOR_OWNER_SUMS` | Reduce each owner's routed outputs locally before the final cross-vendor add. This changes floating-point association and is not the byte-identity mode. |
| `DFLASH_DS4_TP_SCHEDULE_BRANCHES` | Submit the two owner branches independently through the mixed scheduler. |
| `DFLASH_DS4_TP_TARGETED_JOIN_SPLIT` | Gather the peer result at the join without an extra peer fence per layer. |
| `DFLASH_DS4_COMP_PAD_STRIDE` | Exact compressed-KV padding bucket; wider buckets trade small masked work for fewer verifier graph captures. |
| `DFLASH_DS4_DISABLE_GROUPED_OUTPUT_PROJECTION` | Diagnostic fallback for runtimes that cannot preserve grouped projection metadata across a scheduler copy. |
| `DFLASH_CUDA_BACKEND_PATH` / `DFLASH_HIP_BACKEND_PATH` | Optional explicit peer backend module path. |
| `DFLASH_EXPERT_BUDGET_MB` | Main-GPU memory budget for hot experts. |
| `DFLASH_DS4_HOTNESS_CSV` | Optional per-layer routing profile for hot placement. |
| `DFLASH_DS4_TP_GROUPED_MMVQ` | Opt in to grouped expert MMVQ for `n_tokens > 1`, replacing tokenwise ROCmFP2 gate/up dispatch. The paged R9700 + Strix profile is qualified at concurrency 1–4; the flag itself does not enforce topology or lane limits. `DFLASH_MOE_TP_GROUPED_MMVQ` is the model-neutral name and takes precedence. |
| `DFLASH_DS4_TP_BATCH_SPLIT_COPIES` | Establish destination readiness once per scheduler split without combining backend copy dependencies. The qualified dual-ROCm launcher enables this exact path. |
| `GGML_BATCH_PEER_COPIES` | Additionally combine HIP peer-copy dependency publication. The old `GGML_CUDA_BATCH_PEER_COPIES` spelling remains an alias. Keep these event-batching variables unset for the exact qualified profile. |
| `DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT` | Use the routing profile and measured owner-rate ratio to minimize the predicted two-owner MoE critical path instead of maximizing aggregate hot-hit rate. Requires `DFLASH_DS4_HOTNESS_CSV`. |
| `DFLASH_DS4_TP_MAIN_TO_PEER_RATE` | Relative main/peer routed-expert rate used by critical-path placement. It must be finite and greater than zero; the default is `3.4`. |
| `DFLASH_DS4_TP_BALANCE_MIN_HOT` | Minimum hot experts retained on every routed layer by critical-path placement. Defaults to `0`. |
| `DFLASH_DS4_Q5_VERIFY` | Opt in to the AMD q=5 fused verifier. This also selects the qualified MMVQ width and verifier-cache defaults when they are not explicitly overridden. |
| `DFLASH_CUDA_MMVQ_FP4_X4` | Enable the four-column ROCmFP4 dense x4 kernel and permit the five-column x4+1 kernel. Defaults to `1` for monolithic `gfx1151` paged serving and opt-in HIP q=5 verification; set `0` to restore generic four- and five-column dispatch. |
| `DFLASH_CUDA_MMVQ_FP4_Q5_X4_PLUS1` | Enable five-column ROCmFP4 dense x4+1 dispatch when `DFLASH_CUDA_MMVQ_FP4_X4=1`. Defaults to `1` for monolithic `gfx1151` paged serving and the opt-in q=5 verifier on `gfx1201`; set `0` to restore the generic five-column kernel. |
| `DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24` | Enable packed 24-bit ROCmFP3 expert dispatch. Defaults to `1` for monolithic `gfx1151` paged serving; set `0` to restore generic ROCmFP3 expert dispatch. Other configurations require explicit opt-in. |
| `DFLASH_DS4_TP_FUSED_CACHE_SLOTS` | Number of heterogeneous verifier graph slots. Defaults to `2` for q<=4 and `9` for the opt-in q=5 verifier; each slot retains scheduler scratch on both GPUs. |
| `DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY` | Skip the expensive property scan only for a warmed verifier graph. Rebuilt scheduler generations are always validated. Leave unset for the conservative production profile. |
| `GGML_DS4_FA_SERIAL_INDEX_SCAN` | Restore the serial compressed-row mask scan for an indexed-attention A/B. By default, HIP scans contexts above 512 compressed rows in parallel. |
| `DFLASH_MOE_PREFILL_PERSISTENT_OWNER_ALLOC` | Long-prefill arena kill switch; set `0` to restore per-layer owner allocation. |

`DFLASH_DS4_TIMING` enables the existing timing banners:

- local and paged serving: `[deepseek4-timing]`. Paged rounds use `paged-prefill`, `paged-decode`, or `paged-mixed` and aggregate all lanes into `full_build`, `full_set`, `full_compute`, and `full_read`.
- parent / local shard: `[deepseek4-split-timing]`
- remote Halo shard: `[deepseek4-target-timing]`

The old per-expert IPC worker is retired. The `DFLASH_DS4_MOE_TP*` variables
above configure the in-process route-owner implementation.

### External ROCm traces

Set `DFLASH_DS4_ROCTX=1` when collecting an external rocprof marker trace.
The runtime emits balanced `ds4.prefill`, `ds4.spec_decode`, and
`ds4.layer_range` ranges with the applicable mode, token count, layer bounds,
and device. The marker layer does not use HIP events, synchronize a stream, or
calculate timings; rocprof remains the timing authority. Non-HIP builds emit
no markers or ROCTX library calls, and an unset or false value does not load
ROCTX.

Use the marker-trace option supported by the installed rocprof version (for
example, `rocprofv3 --marker-trace -- <lucebox command>`), then correlate these
host scopes with the kernel and memory-copy tracks. A target-machine trace is
still required to assess instrumentation perturbation.

## DSpark Speculative Decode

DeepSeek4 uses the shared DFlash DSpark head implementation together with a
DeepSeek4-specific three-layer drafter and fused target verification. The draft
GGUF carries its auxiliary projections under the existing `dflash.dspark.*`
tensor contract. DeepSeek4/MTP checkpoints store compatible heads under the
`mtp.2.*` namespace, which the converter maps as follows.

Supported DeepSeek4/MTP input tensors:

| DeepSeek4/MTP tensor | GGUF tensor |
|----------------------|-------------|
| `mtp.2.markov_head.markov_w1.weight` | `dflash.dspark.markov.w1` |
| `mtp.2.markov_head.markov_w2.weight` | `dflash.dspark.markov.w2` |
| `mtp.2.confidence_head.proj.weight` | `dflash.dspark.confidence.weight` |
| `mtp.2.confidence_head.proj.bias` | `dflash.dspark.confidence.bias` |

If the MTP confidence projection is bias-less, the converter writes a zero
bias so the GGUF loader still sees the pair it expects. The Markov head alone
is enough for DSpark greedy-chain correction; confidence gating remains
optional.

Example conversion with the DS4 MTP shard that contains the DSpark heads:

```bash
python server/scripts/convert_dflash_to_gguf.py \
  /path/to/dflash-draft/model.safetensors \
  /path/to/dflash-draft.gguf \
  --aux-heads /path/to/hf-ds4-flash-dspark/model-00048-of-00048.safetensors
```

Run the converted drafter against a DeepSeek4 target with:

```bash
export DFLASH_DS4_SPEC=1
export DFLASH_DS4_FUSED_VERIFY=1
export DFLASH_DS4_DRAFT=/path/to/dflash-draft.gguf
export DFLASH_DS4_SPEC_Q=4

./server/build-hip/dflash_server /path/to/deepseek4-target.gguf \
  --target-device hip:0 \
  --ds4-fused-verify-f16-kv \
  --ds4-fused-decode
```

`--ds4-fused-verify-f16-kv` feeds the persistent F16 MLA cache directly to
batched explicit verifier attention instead of converting the full cache to
F32 on every speculative step. Key-side accumulation remains F32 through 512
attention rows to preserve the short-context quality baseline. The option is
currently qualified only for a single HIP target and remains off by default.
It changes verifier floating-point inputs and can change generated tokens, so
re-run workload quality checks before enabling it for another checkpoint.

`DFLASH_DS4_FUSED_VERIFY=1` is the opt-in throughput profile. Its persistent
whole-model GPU graph uses stable padded reduction shapes, so near-tied greedy
logits can select a different token than the normal causal verifier even at
temperature 0. Leave it unset when comparing against the normal verifier, or
set `DFLASH_DS4_SEQ_VERIFY=1` for the slower token-at-a-time verification
diagnostic. `DFLASH_DS4_SPEC_REFERENCE_EXACT=1` combines sequential target
verification with full rollback snapshots for byte-identity checks. Neither
fused verification nor the separate
`--ds4-expert-top-k 4` approximation should be presented as byte-identical AR.

DSpark can verify against in-process heterogeneous expert placement. The
drafter remains local to its selected GPU backend; a failed draft load is
reported and falls back to normal autoregressive decode. The target cache and
sampler stay on the main backend while routed target experts execute on their
configured owners. `--ds4-expert-top-k 4` remains a separate approximate
policy; omit it to retain the model's default six routed experts.

### Silent AR fallback (check this before calling spec decode broken)

The DSpark verifier is greedy-only, so the server routes a request to plain
autoregressive decode — with the drafter loaded and `DSpark spec-decode
ENABLED` already printed at startup — whenever any of these hold:

- the effective sampler is non-greedy (`temperature > 0`, repetition or
  presence/frequency penalties),
- the request carries budget stop tokens,
- the request forces AR explicitly.

The sampler trap is the subtle one: when a request **omits** `temperature`,
the HTTP layer falls back to the model card's sampling defaults, and
`share/model_cards/deepseek-v4-flash-0731-rocmfpx.json` sets
`temperature: 1.0`. A temperature-less benchmark request therefore decodes
pure AR on any build that ships the card, while the same request engages
speculation on a tree without it — which looks exactly like a spec-decode
regression between the two binaries. It is not one; send
`"temperature": 0.0` explicitly. The tell in the log is `decode tokens=N
steps=N-1` (one step per token) instead of `[ds4-spec] gen=... steps=...`
lines; the server also now logs one `DSpark spec loaded but this request
decodes AR: ...` line naming the condition the first time it happens.

Speculative throughput is acceptance-bound, and acceptance is workload-bound:
on the gfx1151 host, code/math prompts (accept 0.83–0.90, ~2.4 committed
tokens per verify step) measured 24–27 tok/s where open-ended prose (accept
0.64–0.69, ~1.1 committed per step) measured 16–18 tok/s from the same server
under the same flags. Judge a spec-decode number against the acceptance it
was measured at, not against the headline figure from a different prompt mix
(see the gfx1151 numbers below for the measured governor/top-k envelope).

Measure on an otherwise idle box. Strix Halo decode is unified-memory
bandwidth bound, so an unrelated co-tenant compile on the same host dropped
the identical request from 18.2 to 3.8 tok/s — a 5x swing with no change to
the acceptance rate or the step count, which is what makes it easy to misread
as a model or kernel regression. Record `/proc/loadavg` alongside any tok/s
figure, and re-run anything anomalous before believing it.

### Verifier graph-cache safety

Every heterogeneous verifier slot owns a scheduler and its per-backend scratch
buffers. The production default is therefore two slots. Do not copy the old
12-slot benchmark override into a long-lived service without measuring free
VRAM across every intended context shape.

Native CUDA/HIP graph executables are cached below the ggml scheduler. Before a
DS4 slot is rebuilt, the runtime now synchronizes its backends and retires every
native graph key that points into that slot's metadata arena. Forced replay is
also generation-checked using the scheduler split uid. Consequently, an LRU
slot rebuilt at the same address cannot replay the previous shape's executable.

The required burn-in sequence is repeated requests at 2K, 4K, 8K, and 16K in
one process, followed by another 2K request to force additional eviction. Run
with `DFLASH_DS4_TP_FUSED_CACHE_SLOTS=2`; first qualify with forced replay
unset, then repeat with it enabled as a separate performance A/B.

### Experimental AMD q=5 verifier

`DFLASH_DS4_Q5_VERIFY=1` enables a five-row fused verifier on HIP. It handles
the shape that crosses two ratio-4 compressor boundaries, preserves five raw
SWA rows for rollback, and restores plus replays only the accepted prefix after
a partial rejection. q<=4 behavior is unchanged when the flag is absent.

The heterogeneous verifier keeps all five lanes in one
`[n_embd * n_hc, q]` tensor. Attention HC-pre, attention HC-post, FFN HC-pre,
FFN HC-post, drafter-feature capture, and output HC merge are batched across
the verifier width. This removes the former per-lane HC controller paths and
progressive concatenations without changing the verifier result. The split
HC-post kernel also accepts a token dimension and joins the two owner outputs
inside that batched kernel.

Critical-path placement models each routed layer as two concurrent branches:
the main branch includes its fixed shared-expert work and hot routed work,
while the peer branch executes the remaining routed work. The allocator adds
the next profiled expert only when its marginal reduction in
`max(main / main_to_peer_rate, peer)` is positive. The expert memory budget is
therefore an upper bound; leaving part of it unused is valid when another hot
expert would lengthen the predicted fork.

On the qualified R9700 + Strix Halo profile, leaving the related q=5 controls
unset selects `LUCE_MMVQ_MAX_NCOLS=5`, nine heterogeneous verifier slots, and
the ROCmFP4 x4+1 dense kernel on `gfx1201`.
The wider MMVQ ceiling avoids the slow small-matrix crossover, while nine slots
hold the recurring compressor phases without steady graph rebuilds. The x4+1
kernel decodes shared weights through the existing four-column vector path and
retains the original scalar accumulation for the fifth verifier column.
Explicit environment values still take priority.

Sparse heterogeneous prefill uses a reusable graph allocator with preferred
128 MiB backing chunks. This avoids depending on one large contiguous HIP
allocation on devices without virtual-memory-backed buffers. Individual
tensors remain unsplit and may exceed the preferred chunk size. Prompts ending
above 4K use a 1K-token prefill shape, and that cap remains sticky for later
requests in the process so a post-16K request cannot force a fragmented
1K-to-2K arena replacement. Reproducible decode graph caches are retired before
a necessary prefill-arena growth; persistent HC mirrors remain resident.

The exact qualification launch used:

```bash
export DFLASH_DS4_Q5_VERIFY=1
export DFLASH_DS4_SPEC_Q=5
export DFLASH_EXPERT_BUDGET_MB=14350
export DFLASH_DS4_HOTNESS_CSV=/path/to/ds4_moe_tp_hotness.csv
export DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT=1
export DFLASH_DS4_TP_MAIN_TO_PEER_RATE=4.4
export DFLASH_DS4_TP_BALANCE_MIN_HOT=0
```

The checked-in wrapper reproduces the full exact-context protocol and records
the manifest, response hashes, server log, ROCm state, and a two-second VRAM
trace:

```bash
TARGET_MODEL=/path/to/target.gguf \
DRAFT_MODEL=/path/to/dspark-draft.gguf \
HOTNESS_CSV=/path/to/ds4_moe_tp_hotness.csv \
CRITICAL_PATH_PLACEMENT=1 \
MAIN_TO_PEER_RATE=4.4 \
BALANCE_MIN_HOT=0 \
EXPERT_BUDGET_MB=14350 \
harness/qualification/deepseek4/qualify_ds4_q5_amd.sh
```

Its q=5 MMVQ width, verifier slots, and x4+1 controls default to `auto`, so the
run also verifies the platform defaults. `EXPECTED_SHA256` can override the
qualified deterministic-workload hash when intentionally testing another
compatible artifact.

At temperature zero, all 25 requests in the 2K -> 4K -> 8K -> 16K -> 2K
burn-in produced the same expected response hash. With the automatic q=5
MMVQ/cache/kernel defaults and the critical-path profile above, measured client
decode medians were 75.818, 74.530, 69.898, 62.685, and 76.703 tok/s. The final
2K measurements after repeated 16K prefill were 76.685-76.727 tok/s, confirming
that the bounded sticky arena recovers steady decode rather than merely
surviving the request. The placement retained 1,688 profiled hot experts, 23-63
per layer, and the full sweep peaked at 31.089 GiB on the reported 31.86 GiB
main GPU. Treat these as workload-specific burn-in measurements, not as a
portable default for unrelated memory layouts.

For an overlap trace, run the same wrapper with the delayed profiler launcher:

```bash
SERVER_BIN=harness/qualification/deepseek4/rocprof_server_wrapper.sh \
PROFILED_SERVER_BIN=/path/to/dflash_server \
ROCPROF_OUTPUT_DIR=/path/to/trace-output \
ROCPROF_START_SECONDS=180 \
ROCPROF_DURATION_SECONDS=90 \
harness/qualification/deepseek4/qualify_ds4_q5_amd.sh

harness/qualification/deepseek4/analyze_rocprof_overlap.py \
  /path/to/trace-output/trace_kernel_trace.csv
```

The analyzer reports per-owner busy time, simultaneous kernel-busy time,
time-binned overlap, and the kernels dominating each owner. Use a steady decode
window rather than model load or prefill when comparing placement changes.

In the post-batching trace, steady 2K decode windows placed only 16-22% of
either owner's kernel-busy time inside a simultaneously busy interval. The
unprofiled server timing attributed 63.7 ms of each 74.4 ms speculative step to
target verification; draft, head, snapshot, and apply work together accounted
for the remaining 10.7 ms. Changing the placement rate from 4.4 to 3.8 at the
same 14,350 MiB budget moved 116 experts to the peer but changed the measured
2K median by less than 0.1 tok/s. These measurements show that placement is
already near its local balance point. Further large gains require removing
split/copy dispatches or parallelizing work outside the routed-expert fork;
adding the two devices' headline bandwidths is not a valid throughput model
because attention, routing, HC boundaries, and every layer join remain ordered.

On HIP `gfx1151`, enabling DSpark defaults `LUCE_MMVQ_MAX_NCOLS` to `4` when
the variable is unset. This keeps the four-row verifier on MMVQ. On a 128 GiB
Strix Halo Radeon 8060S using ROCm 7.2.4, the rebased candidate measured 32.12
tok/s weighted at fixed q=4 and 31.94 tok/s with confidence-adaptive width,
versus 25.31 tok/s autoregressive. All three configurations scored 10/10 on the
same five GSM and five Math prompts. The run used `--ds4-expert-top-k 4`, the
platform `performance` profile, and the GPU `high` performance level; fixed
q=4 with the model-default six routed experts measured 28.26 tok/s. Those
throughput figures were taken on a UNIFORM artifact, where top-4 is harmless;
on an adaptive artifact the same setting costs 35 points of exact-copy fidelity
(see `--ds4-expert-top-k` above), so the 32.12 vs 28.26 tok/s trade is not
available there. Enabling
DSpark alone therefore does not guarantee 30 tok/s. Set
`LUCE_MMVQ_MAX_NCOLS` explicitly to override the platform default. AR, NVIDIA,
and other HIP architectures retain the shared dispatch default.

Adaptive width is automatic. When the draft artifact has a compatible
confidence projection, the runtime selects q=2, q=3, or q=4 from the cumulative
confidence of the proposed prefix. It adds the projection to the same fused
Markov graph and reads its scores in the existing token-id synchronization; no
additional host round trip is introduced. Artifacts without a compatible
confidence head transparently retain the existing acceptance-EWMA policy.

On the gfx1151 validation host, confidence-adaptive width retained 10/10
GSM+Math accuracy and measured 31.94 tok/s weighted, within 0.6% of fixed q=4
at 32.12 tok/s. These numbers are workload-specific; the confidence policy is
enabled only when DSpark is explicitly enabled and the draft artifact contains
a compatible confidence head.

## Example: CUDA + Halo Layer Split

Automatic split (CUDA prefix chosen from free memory, optional manual override via `DFLASH_DS4_CUDA_LAYERS`):

```bash
export DFLASH_DS4_CUDA_LAYERS=24   # optional

./server/build-cuda/dflash_server /opt/models/DeepSeek-V4-Flash.gguf \
  --target-device cuda:0 \
  --target-shard-ipc-bin $PWD/server/build-hip/backend_ipc_daemon \
  --target-shard-ipc-work-dir $PWD/server/target_shard_ipc \
  --port 8213
```

Explicit mixed-backend split using the generic target-shard flags:

```bash
./server/build-cuda/dflash_server /opt/models/DeepSeek-V4-Flash.gguf \
  --target-devices cuda:0,hip:0 \
  --target-layer-split 24,19 \
  --target-shard-ipc-bin $PWD/server/build-hip/backend_ipc_daemon \
  --target-shard-ipc-work-dir $PWD/server/target_shard_ipc \
  --port 8213
```

## Performance Notes

- **Split granularity is coarse and stable**: the boundary moves by whole layers.
- **Boundary traffic is HC-state traffic**: the remote handoff is the full HC-state tensor for the current token batch.
- **Decode has backend-specific HC paths**:
  - CUDA decode uses cached backend HC graphs.
  - HIP decode uses the direct HC-pre helper plus host-refreshed HC-post weights.
- **Auto-split is only a heuristic**: override `DFLASH_DS4_CUDA_LAYERS` when you want a reproducible split or when empirical throughput differs from the simple memory estimate.

## Build Targets

| Target | Backend | Purpose |
|--------|---------|---------|
| `dflash_server` | CUDA or HIP | Production server |
| `backend_ipc_daemon` | HIP | Remote Halo target shard for mixed-backend layer split |
| `test_deepseek4_unit` | CUDA | Unit tests (no model files needed) |
