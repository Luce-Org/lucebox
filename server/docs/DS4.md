# DeepSeek V4 Flash — DFlash Integration

This document describes the DeepSeek V4 Flash model backend in DFlash, covering the architecture mapping, the CUDA/Halo layer split, and the target-shard IPC path.

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
| Backend selection / init | `src/common/backend_factory.cpp`, `src/deepseek4/deepseek4_layer_split_adapter.{h,cpp}` |
| Per-shard forward graph | `src/deepseek4/deepseek4_graph.cpp` |
| Model weights and metadata | `src/deepseek4/deepseek4_internal.h`, `src/deepseek4/deepseek4_loader.cpp` |
| HC pre/post CUDA kernel | `src/deepseek4/deepseek4_hc_cuda.cu`, `.h` |
| Remote target-shard daemon | `src/deepseek4/deepseek4_target_shard_ipc_daemon.cpp` |
| Shared target-shard IPC infrastructure | `src/common/target_shard_ipc.*`, `src/placement/remote_target_shard_config.h` |
| Backend IPC CLI entry | `src/ipc/backend_ipc_main.cpp` |

## Forward Pass (Layer-Split Path)

`deepseek4_step_layer_range()` drives per-shard execution over a contiguous layer range:

1. **Embedding + HC init** — The first shard embeds tokens and initializes the HC state for new sequences.
2. **Per-layer forward** — Each layer still runs the normal DS4 sequence: HC pre (attention) → MLA attention → HC post (attention) → HC pre (FFN) → RMSNorm + router → MoE FFN → HC post (FFN).
3. **Shard boundary handoff** — At the split point, the parent sends the current hidden activation plus HC state to the next shard.
4. **Tail shard completion** — The last shard resumes at its `layer_begin`, runs the remaining layers, then performs the final HC merge, RMSNorm, and `lm_head` projection.

The MoE computation stays local to the shard that owns each layer. DeepSeek4 no longer partitions experts into hot/cold or forwards per-token expert work to a separate worker.

## Execution Modes

### Local layer split

DeepSeek4 always runs through `LayerSplitBackend`. When the server is configured with an explicit target layer split, the adapter loads each contiguous shard locally and executes them in order.

### CUDA parent + Halo target-shard split

For heterogeneous setups, the CUDA-built server can keep the prefix layers on the CUDA GPU and launch the suffix shard on the Halo/HIP build through the existing target-shard IPC path:

```
┌─────────────────────────────────────────────────────────────┐
│  CUDA Parent                                                │
│  - Token embedding                                          │
│  - Layers [0, split)                                        │
│  - Maintains local KV/cache state for its layer range       │
├─────────────────────────────────────────────────────────────┤
│  Halo Target Shard (IPC daemon)                             │
│  - Layers [split, 43)                                       │
│  - Final HC merge, RMSNorm, lm_head                         │
│  - Returns logits / sampled token to the parent             │
└─────────────────────────────────────────────────────────────┘
```

This path uses `TargetShardIpcSession`, `deepseek4_target_shard_ipc_daemon.cpp`, and `BackendIpcMode::DeepSeek4TargetShard` rather than the old expert-worker protocol.

## Shard Boundary State

The shard boundary transfers model state at the layer split, not per-expert routing payloads:

- **Hidden activation** — current `[n_tokens × n_embd]` activation at the split point.
- **HC state** — persistent `[n_hc × n_embd]` controller state shared across all layers.
  For DeepSeek4 this is `4 × 4096` floats, about 64 KiB.
- **Sequence position / token metadata** — enough information for the tail shard to continue its cache updates and finish the forward pass.

KV cache tensors remain owned by the shard that owns the corresponding layer range.

## Auto-Split Behavior

If DeepSeek4 is started without an explicit target layer split, `DeepSeek4LayerSplitAdapter` computes the CUDA prefix automatically:

1. Read `DFLASH_DS4_CUDA_LAYERS`. If it is set to a positive value, that value becomes the number of prefix layers kept on CUDA.
2. Otherwise, query CUDA free memory.
3. Reserve fixed overhead for embeddings, output head, cache growth, and safety margin.
4. Estimate roughly 500 MiB per DeepSeek4 layer.
5. Clamp the result so at least one layer remains on each side (`1..42`), then assign the remaining `43 - N` layers to Halo.

The runtime logs the chosen split with a `[deepseek4-split] auto-split:` banner.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DFLASH_DS4_CUDA_LAYERS` | Override the auto-split heuristic and pin the first `N` DeepSeek4 layers to CUDA. The remaining `43 - N` layers run on the Halo shard. |

DeepSeek4 no longer uses the old expert-split environment variables or expert-worker tuning knobs.

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

- **Split granularity is coarse and stable**: the boundary moves by whole layers, so execution avoids per-token expert ownership checks and per-request expert placement decisions.
- **Boundary traffic is small**: the HC state is only about 64 KiB, so the split cost is dominated by the hidden activation transfer and the tail-shard compute.
- **Auto-split is meant as a starting point**: override `DFLASH_DS4_CUDA_LAYERS` when you want a reproducible split or when empirical throughput differs from the simple memory estimate.

## Build Targets

| Target | Backend | Purpose |
|--------|---------|---------|
| `dflash_server` | CUDA | Production server / CUDA parent |
| `backend_ipc_daemon` | HIP | Remote Halo target shard for mixed-backend layer split |
| `test_deepseek4_unit` | CUDA | Unit tests (no model files needed) |
