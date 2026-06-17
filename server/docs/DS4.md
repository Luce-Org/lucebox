# DeepSeek V4 Flash — DFlash Integration

This document describes the DeepSeek V4 Flash model backend in DFlash, covering the architecture mapping, the hot/cold MoE expert split, and the IPC worker protocol.

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
| Backend lifecycle and init | `src/deepseek4/deepseek4_backend.cpp`, `.h` |
| Forward graph, routing, hybrid step | `src/deepseek4/deepseek4_graph.cpp` |
| Model weights and metadata | `src/deepseek4/deepseek4_internal.h`, `deepseek4_loader.cpp` |
| HC pre/post CUDA kernel | `src/deepseek4/deepseek4_hc_cuda.cu`, `.h` |
| Expert IPC client | `src/common/expert_ipc.cpp`, `.h` |
| Expert IPC daemon (worker) | `src/deepseek4/deepseek4_expert_ipc_daemon.cpp` |
| Hybrid expert storage | `src/common/moe_hybrid_storage.*` |
| Hybrid FFN evaluation | `src/common/moe_hybrid_ffn_eval.*` |
| MoE types and config | `src/common/moe_hybrid_types.h` |
| Backend IPC CLI entry | `src/ipc/backend_ipc_main.cpp` |

## Forward Pass (Hybrid Path)

`deepseek4_step_hybrid()` drives per-layer execution:

1. **HC pre (attention)** — Sinkhorn-normalize the 4 HC streams, produce a working vector. Uses a CUDA kernel (`deepseek4_cuda_hc_pre_mix`) for decode or CPU fallback for prefill.
2. **MLA attention** — Low-rank Q projection → RoPE → KV cache write → scaled dot-product over raw SWA + compressed KV + indexed KV → grouped low-rank output projection.
3. **HC post (attention)** — Update HC streams from attention output.
4. **HC pre (FFN)** — Same Sinkhorn mix for FFN sublayer.
5. **RMSNorm + Router** — Normalize, compute router logits, select top-6 experts (or hash-route for layers 0–2).
6. **Hybrid/Worker FFN** — `eval_ds4_hybrid_or_worker()` dispatches expert compute.
7. **HC post (FFN)** — Update HC streams from FFN output.
8. **Output HC + lm_head** — Final HC merge → RMSNorm → vocabulary projection.

## Execution Modes

### Full CUDA

All weights including experts on GPU. Only viable on ≥80 GB VRAM; automatically attempted first and falls back on OOM.

### Hybrid (Hot GPU / Cold CPU)

Non-expert weights on CUDA, experts split by placement:

- **Hot experts** — subset loaded into GPU VRAM, evaluated via cached ggml graphs.
- **Cold experts** — remaining experts in CPU memory, evaluated on a CPU backend.
- Results are summed on the parent.

Triggered by: `DFLASH_DS4_HYBRID=1` or automatic OOM fallback.

### CUDA Parent + HIP Worker Split (`hip_hot_cpu_cold`)

The primary multi-device mode for heterogeneous setups (e.g., RTX 3090 CUDA + Radeon 8060S HIP):

```
┌─────────────────────────────────────────────────────────────┐
│  CUDA Parent (RTX 3090)                                     │
│  - Non-expert weights (MLA, HC, embeddings, lm_head)        │
│  - Attention, HC mix, routing, RMSNorm                      │
│  - Cold expert tail (CPU memory, evaluated locally)          │
├─────────────────────────────────────────────────────────────┤
│  HIP Worker (Radeon 8060S via IPC daemon)                   │
│  - Hot experts loaded into HIP VRAM                         │
│  - Receives activations + selected IDs over IPC             │
│  - Returns weighted FFN output                              │
└─────────────────────────────────────────────────────────────┘
```

Enabled by:
```bash
DFLASH_DS4_EXPERT_SPLIT=hip_hot_cpu_cold
DFLASH_DS4_EXPERT_IPC=1
DFLASH_DS4_EXPERT_IPC_BIN=<path to HIP backend_ipc_daemon>
DFLASH_DS4_EXPERT_IPC_MODEL=<GGUF path>
DFLASH_DS4_EXPERT_WORKER_BUDGET_MB=61000
```

## Hot/Cold Expert Handling

### Placement Computation

Placement determines which experts are "hot" (on the accelerator) vs "cold" (on CPU):

1. **Uniform placement** (default) — First N experts per layer up to the memory budget. Simple, avoids any profiling dependency.
2. **Route-stat placement** — Top-N by observed routing frequency (`DFLASH_DS4_ROUTE_STATS_OUT` to collect, `DFLASH_DS4_PLACEMENT_IN` to load). Maximizes hit rate but can overload the worker.
3. **Explicit JSON** — `DFLASH_DS4_PLACEMENT_IN=<file>` loads a pre-computed placement (shared between parent and worker).

The `MoeHybridPlacement` struct records per-layer hot expert IDs and counts.

### Partitioning at Inference Time

In `eval_ds4_hybrid_or_worker()`, for each token the top-6 selected expert IDs are classified:

```
for each selected expert ID:
    if hot_local_by_global[id] >= 0  →  expert is "hot"
    else                              →  expert is "cold"

if worker_owns_hot_ids:
    local  = cold IDs  (evaluated on CPU by parent)
    remote = hot IDs   (sent to HIP worker)
else:
    local  = hot IDs   (evaluated on GPU by parent)
    remote = cold IDs  (sent to worker)
```

The `worker_owns_hot_ids` flag inverts ownership semantics for the split mode where hot experts live on the HIP worker rather than the CUDA parent.

### FFN Evaluation Paths

| Function | Purpose |
|----------|---------|
| `eval_moe_hybrid_ffn_single()` | Single-token: hot on GPU + cold on CPU, cached graphs |
| `eval_moe_hybrid_ffn_batched()` | Batched prefill: hot GPU + cold CPU concurrently |
| `eval_moe_hot_only_batched()` | All-hot batched (when full GPU fit) |
| `eval_moe_hybrid_ffn_gpu_resident()` | GPU-resident decode (data stays on GPU, only router IDs read to CPU) |

Graph caching: per-layer cached ggml graphs keyed by selected expert count avoid rebuild overhead. Fixed-slot mode (`DFLASH_MOE_FIXED_SLOT_GRAPHS=1`) pads to `n_expert_used` slots with zero-weighted dummies.

### Async Worker Overlap

`DFLASH_DS4_ASYNC_WORKER=1` enables pipelining:

1. Partition selected IDs into local and remote sets.
2. `eval_begin()` — send request to HIP worker (non-blocking write).
3. Evaluate local CPU cold + shared expert work.
4. `eval_end()` — block until worker responds.
5. Sum local + remote outputs.

This hides part of the HIP compute behind CPU cold work.

## Expert IPC Protocol

Communication uses temp-file payloads with stdin/stdout command framing:

**Request** (parent → worker):
- Write binary file: `ExpertIpcRequestHeader` + activation `[n_tokens × n_embd]` + selected IDs `[n_tokens × n_selected]` + weights `[n_tokens × n_selected]`.
- Send `eval <path>` command on the IPC stream.

**Response** (worker → parent):
- `ExpertIpcResponseHeader` (version 2) — status, dimensions, worker-side timing.
- Optional `ExpertIpcGraphTiming` extension (version 3, if `EXPERT_IPC_FLAG_GRAPH_TIMING` set) — per-graph build/compute/read breakdown.
- Output activation payload `[n_tokens × n_embd]`.

The `eval_begin()`/`eval_end()` split enables async overlap without protocol changes.

## Telemetry

`DFLASH_DS4_TIMING=1` emits `[deepseek4-timing]` aggregate logs per request:

| Field | Meaning |
|-------|---------|
| `ffn`, `hot`, `cold`, `combine`, `partition` | Parent-side hybrid FFN phases |
| `worker` | Parent-observed worker wall time |
| `worker_write`, `worker_wait`, `worker_read` | Parent IPC phases |
| `worker_req_read`, `worker_part`, `worker_resident`, `worker_miss_build`, `worker_miss` | Worker-side phases |
| `worker_req_kib`, `worker_resp_kib` | Payload volume |
| `ffn_hot_graph_build/hit`, `ffn_cold_graph_build/hit` | Parent graph cache |
| `worker_hot_graph_build/hit` | Worker graph cache |
| `hot_sel`, `cold_sel` | Routed expert counts by ownership |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DFLASH_DS4_HYBRID=1` | Force hybrid mode (skip full-load attempt) |
| `DFLASH_DS4_EXPERT_SPLIT=hip_hot_cpu_cold` | Enable CUDA + HIP + CPU three-way split |
| `DFLASH_DS4_EXPERT_IPC=1` | Use IPC worker for expert eval |
| `DFLASH_DS4_EXPERT_IPC_BIN` | Path to `backend_ipc_daemon` binary |
| `DFLASH_DS4_EXPERT_IPC_MODEL` | Worker GGUF model path |
| `DFLASH_DS4_EXPERT_IPC_GPU` | Worker GPU index |
| `DFLASH_DS4_EXPERT_IPC_SUDO=1` | Launch worker via sudo (for HIP access) |
| `DFLASH_DS4_EXPERT_WORKER_BUDGET_MB` | Worker hot expert memory budget |
| `DFLASH_DS4_HIP_HOT_PER_LAYER` | Override hot experts per layer |
| `DFLASH_DS4_EXPERT_WORKER_OFFSET` | Uniform placement first expert offset |
| `DFLASH_DS4_TIMING=1` | Enable timing telemetry |
| `DFLASH_DS4_ASYNC_WORKER=1` | Async worker overlap |
| `DFLASH_MOE_FIXED_SLOT_GRAPHS=1` | Fixed-slot cached FFN graphs |
| `DFLASH_DS4_ROUTE_STATS_OUT` | Write routing stats CSV |
| `DFLASH_DS4_PLACEMENT_IN` | Load placement JSON |
| `DFLASH_DS4_PLACEMENT_OUT` | Export computed placement JSON |
| `DFLASH_DS4_PLACEMENT_ONLY` | Compute and save placement, then exit |

## Example: CUDA + HIP Split

```bash
export DFLASH_DS4_EXPERT_SPLIT=hip_hot_cpu_cold
export DFLASH_DS4_EXPERT_IPC=1
export DFLASH_DS4_EXPERT_IPC_SUDO=1
export DFLASH_DS4_EXPERT_IPC_BIN=$PWD/server/build-hip/backend_ipc_daemon
export DFLASH_DS4_EXPERT_IPC_MODEL=/opt/models/DeepSeek-V4-Flash.gguf
export DFLASH_DS4_EXPERT_WORKER_BUDGET_MB=61000
export DFLASH_DS4_TIMING=1
export DFLASH_DS4_ASYNC_WORKER=1

./server/build/dflash_server /opt/models/DeepSeek-V4-Flash.gguf --port 8213
```

## Performance Notes

- **CPU cold tail is often cheaper than HIP hot**: Naive route-stat placement that maximizes HIP coverage can regress throughput because HIP FFN compute is the bottleneck, not CPU cold misses.
- **Async overlap is safe**: Hides ~3–4s of HIP latency behind CPU cold work with no correctness impact.
- **Fixed-slot graphs**: Eliminates graph rebuilds but increases per-token expert compute; keep opt-in.
- **Current bottleneck**: HIP resident FFN compute dominates worker-side cost. Future work should target HIP kernel optimization or cost-aware placement.

## Build Targets

| Target | Backend | Purpose |
|--------|---------|---------|
| `dflash_server` | CUDA | Production server (parent) |
| `backend_ipc_daemon` | HIP | Expert worker for split mode |
| `test_deepseek4_unit` | CUDA | Unit tests (no model files needed) |
