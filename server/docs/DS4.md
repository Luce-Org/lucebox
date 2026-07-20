# DeepSeek V4 Flash — DFlash Integration

This document describes the current DeepSeek V4 Flash implementation in
DFlash. DeepSeek4 supports a monolithic HIP backend, a layer-split backend,
and an in-process heterogeneous expert-parallel backend for Lucebox systems
with an RDNA4 discrete GPU and a Strix Halo GPU. The DSpark speculative path
can run locally on a selected HIP device or in a separate HIP process.

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

## Code Layout

| Area | Files |
|------|-------|
| Backend selection / init | `src/common/backend_factory.cpp`, `src/deepseek4/deepseek4_backend.{h,cpp}`, `src/deepseek4/deepseek4_layer_split_adapter.{h,cpp}` |
| Per-shard forward graph | `src/deepseek4/deepseek4_graph.cpp` |
| Model weights and metadata | `src/deepseek4/deepseek4_internal.h`, `src/deepseek4/deepseek4_loader.cpp` |
| HC pre/post CUDA kernel | `src/deepseek4/deepseek4_hc_cuda.cu`, `.h` |
| Remote target-shard daemon | `src/deepseek4/deepseek4_target_shard_ipc_daemon.cpp` |
| DSpark runtime and remote draft daemon | `src/deepseek4/deepseek4_dspark*.{h,cpp}` |
| Heterogeneous expert storage/evaluation | `src/common/moe_hybrid_{storage,ffn_eval}.*`, `src/common/moe_expert_compute.*` |
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
layer. The heterogeneous expert-parallel path below is different: every layer
can dispatch routed experts concurrently to two local HIP backends.

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
- `--ds4-expert-top-k N` keeps the highest-ranked `N` routed experts and
  renormalizes their weights. `0` uses the model default. Reducing this value is an
  approximate inference policy and must be quality-validated for the target
  workload.

These options currently apply only to the monolithic HIP backend. For the
validated Strix Halo profile:

```bash
./server/build-hip/dflash_server /opt/models/DeepSeek-V4-Flash.gguf \
  --target-device hip:0 \
  --ds4-fused-decode \
  --ds4-expert-top-k 4
```

### Lucebox heterogeneous expert parallel

The in-process Lucebox path keeps dense target work, selected hot experts, and
the DSpark drafter on the R9700 (`hip:0`). It materializes the remaining
experts on Strix Halo (`hip:1`). At every MoE layer, routing is partitioned by
expert ownership, both HIP backends are submitted before the join, and the
owner results are reduced back into the target graph. Hot placement is bounded
by the R9700 memory budget and may use a measured routing profile.

This is route-level expert parallelism, not layer pipelining and not the
retired per-expert IPC worker. It avoids host IPC and preserves a single target
KV/cache owner. The optional DSpark worker remains available as a compatibility
mode, but the fastest two-GPU profile uses the local drafter on `hip:0`.

Build one HIP binary for both Lucebox architectures:

```bash
cmake -S server -B server/build-hip-dual \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1151;gfx1201' \
  -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build server/build-hip-dual -j
```

The promoted q=4 profile additionally enables fused target verification,
device-side owner joining, small-batch MoE MMVQ kernels, pinned rollback, and
DSpark context-KV reuse. `DFLASH_DS4_TOPK=4` is an explicit approximate
inference policy; omit it when model-default top-6 quality is required.

On 2026-07-19, the clean dual-architecture build produced steady 50.7 and
50.8 tok/s decode runs on the deterministic integer-list workload (the
53-token prompt includes the explicit default system message; 128 output
tokens, q=4, acceptance 1.00). Mean draft time was 6.1 ms and mean target
verification was 70.7--70.9 ms. With the same prompt and the same 1.00
acceptance, incorrectly splitting boundary-spanning verification into q=3 and
q=1 measured 37.5 tok/s and 98.5 ms verification. These figures establish the
execution-path A/B; they are workload-specific and are not a quality or
held-out benchmark claim.

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

### Local target + remote HIP DSpark draft

The speculative IPC mode keeps the complete DeepSeek4 target, KV cache, tied
embedding, DSpark head, and sampler in the parent. A separate
`BackendIpcMode::DeepSeek4DSparkDraft` process executes the DSpark proposal
graph on the selected HIP GPU. The parent currently retains its locally loaded
draft weights for metadata and fallback; removing that duplicate residency is
a separate optimization.

Each speculative step transfers:

- target feature captures in token-major
  `[n_tokens, n_target_layers, n_embd]` layout;
- the embedded seed/MASK proposal block;
- the proposal hidden states returned by the HIP worker.

All target-layer captures are uploaded as one feature block and one
acknowledgement. On POSIX hosts this mode defaults to the existing memfd-backed
shared payload transport; proposal input and output reuse the same mapping.
`stream` remains available as an A/B and compatibility fallback. This is shared
host IPC, not direct `hipIpcMemHandle` peer memory.

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
| `DFLASH_DS4_TIMING` | Enable DS4 timing logs for the layer-split parent and target-shard daemon. Useful for profiling prefill/decode breakdowns; leave unset for normal runs. |
| `DFLASH_DS4_SPEC` | Enable the DeepSeek4 DSpark speculative runtime. |
| `DFLASH_DS4_DRAFT` | DSpark draft GGUF loaded locally for metadata/head use and by the remote worker. |
| `DFLASH_DS4_DRAFT_IPC_BIN` | Backend IPC daemon executable used for the remote DSpark worker. |
| `DFLASH_DS4_DRAFT_IPC_GPU` | HIP device index for the remote DSpark proposal blocks. |
| `DFLASH_DS4_DRAFT_IPC_WORK_DIR` | Scratch directory for the child process. |
| `DFLASH_DS4_DRAFT_IPC_REQUIRED` | Fail initialization instead of falling back to a local draft when remote startup fails. |
| `DFLASH_DRAFT_IPC_TRANSPORT` | Payload transport: `auto`, `shared`, or `stream`. DeepSeek4 DSpark defaults to `auto`; other draft modes retain their existing default. |
| `DFLASH_DS4_MOE_TP` | Enable routed-expert partitioning between local and expert owners. |
| `DFLASH_DS4_MOE_TP_INPROC` | Use the in-process dual-HIP implementation instead of an IPC expert worker. |
| `DFLASH_DS4_MOE_TP_GPU` | HIP index of the expert-owner GPU; on Lucebox this is normally Strix Halo (`1`). |
| `DFLASH_EXPERT_BUDGET_MB` | R9700 memory budget used to select locally resident hot experts. |
| `DFLASH_DS4_HOTNESS_CSV` | Optional per-layer expert routing profile used for hot placement. |
| `DFLASH_DS4_FUSED_VERIFY` | Enable the persistent q-wide target verification graph. |
| `DFLASH_DS4_DENSE_TP_MASK` | Experimental row splitting for dense projections. This is automatically disabled when fused verification is enabled because split-buffer weights are not compatible with verifier graph replay. |
| `DFLASH_DS4_TP_DEVICE_JOIN` | Join expert-owner results on device instead of through a host reduction. |
| `DFLASH_DS4_DRAFT_CONTEXT_KV_CACHE` | Reuse DSpark context projection/KV state between speculative steps. |
| `DFLASH_DS4_ADAPTIVE_WIDTH` | Set to `1` to select q=2/3/4 from DSpark confidence; `0` keeps the configured fixed verify width. If unset, the legacy `/tmp/ds4_awidth` control remains supported. |
| `DFLASH_DS4_DRAFT_AHEAD` | On an independent in-process draft backend, launch the next proposal concurrently with target verification. The target still validates every candidate. |

`DFLASH_DS4_TIMING` enables the existing timing banners:

- parent / local shard: `[deepseek4-split-timing]`
- remote Halo shard: `[deepseek4-target-timing]`

The old per-expert IPC worker is retired. The `DFLASH_DS4_MOE_TP*` variables
above configure the newer in-process route-owner implementation.

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
  --ds4-fused-decode \
  --ds4-expert-top-k 4
```

`DFLASH_DS4_FUSED_VERIFY=1` is the opt-in throughput profile. Its persistent
whole-model GPU graph uses stable padded reduction shapes, so near-tied greedy
logits can select a different token than the normal causal verifier even at
temperature 0. Leave it unset when comparing against the normal verifier, or
set `DFLASH_DS4_SEQ_VERIFY=1` for the slower token-at-a-time verification
diagnostic. Neither fused verification nor the separate
`--ds4-expert-top-k 4` approximation should be presented as byte-identical AR.

DSpark can verify against the in-process hybrid expert placement. The target
cache and sampler remain in the parent, while routed target experts execute on
their configured owners. `--ds4-expert-top-k 4` (or the equivalent environment
setting) remains a separate approximate policy; omit it to retain the model's
default six routed experts.

On HIP `gfx1151`, enabling DSpark defaults `LUCE_MMVQ_MAX_NCOLS` to `4` when
the variable is unset. This keeps the four-row verifier on MMVQ. On a 128 GiB
Strix Halo Radeon 8060S using ROCm 7.2.4, the rebased candidate measured 32.12
tok/s weighted at fixed q=4 and 31.94 tok/s with confidence-adaptive width,
versus 25.31 tok/s autoregressive. All three configurations scored 10/10 on the
same five GSM and five Math prompts. The run used `--ds4-expert-top-k 4`, the
platform `performance` profile, and the GPU `high` performance level; fixed
q=4 with the model-default six routed experts measured 28.26 tok/s. Enabling
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
