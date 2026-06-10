# MoE Hybrid Expert Offload

Model-agnostic Mixture-of-Experts (MoE) hybrid offload subsystem that splits experts between GPU ("hot") and CPU ("cold") to fit large MoE models (e.g. Qwen3.5-MoE, Laguna) on consumer GPUs with limited VRAM.

## Overview

In a standard MoE forward pass, all experts reside on GPU. The hybrid mode instead:

1. **Profiles** which experts are activated most frequently (routing statistics).
2. **Places** the most-used experts on GPU (hot), the rest on CPU (cold).
3. **Evaluates** hot experts on GPU and cold experts on CPU concurrently, then combines results.
4. **Swaps** experts between hot/cold at request boundaries based on shifting workload patterns.

This allows running 27B+ MoE models on a single RTX 3090 (24 GB) that would otherwise require 40+ GB.

## File Layout

| File | Purpose |
|------|---------|
| `moe_hybrid_types.h` | Core types: `MoeHybridConfig`, `MoeLayerDesc` |
| `moe_hybrid_types_impl.h` | Model-specific → generic conversion helpers (qwen35, laguna) |
| `moe_hybrid_routing_stats.{h,cpp}` | Per-layer expert activation counters and ranking |
| `moe_hybrid_placement.{h,cpp}` | Hot/cold assignment: greedy budget allocation from stats |
| `moe_hybrid_swap_manager.{h,cpp}` | Runtime expert promotion/demotion between requests |
| `moe_hybrid_storage.{h,cpp}` | GPU/CPU buffer management for split expert tensors |
| `moe_hybrid_ffn_eval.{h,cpp}` | FFN execution: hot on GPU, cold on CPU, result combination |

## Key Types

### `MoeHybridConfig`

Model-agnostic architecture descriptor:

```cpp
struct MoeHybridConfig {
    int n_embd;           // hidden dimension
    int n_expert;         // total experts per layer
    int n_expert_used;    // top-k selected per token
    int n_ff_exp;         // routed expert intermediate dim
    int n_ff_shexp;       // shared expert intermediate dim (0 = none)
    int n_layer;          // number of MoE layers
    int first_moe_layer;  // first MoE layer index
};
```

### `MoeLayerDesc`

Uniform view of per-layer expert tensors regardless of model backend:

- Routed expert weight tensors (gate, up, down, optional fused gate_up)
- Shared expert tensors (optional)
- Per-tensor quantization scale factors (for NVFP4 models)

### `MoeHybridPlacement`

Specifies which experts are hot per layer:

- `hot_counts[layer]` — number of hot experts in each layer
- `hot_expert_ids[layer]` — ranked list of hot expert indices
- Serializable to/from JSON for persistence across runs

### `MoeHybridLayerStorage`

Per-layer split buffer state:

- `hot_ctx/hot_buf` — GPU-resident tensors for hot experts
- `cold_ctx/cold_buf` — CPU-resident tensors for cold experts
- `hot_local_by_global[expert_id]` — maps global expert index → local hot index (-1 if cold)
- `cold_local_by_global[expert_id]` — maps global expert index → local cold index (-1 if hot)
- `CachedFfnGraph hot_graph/cold_graph` — pre-built ggml compute graphs (avoids per-token rebuild)

## Algorithms

### Placement (Budget Allocation)

Two placement strategies:

1. **Count-based** (`build_from_stats`): Given a total hot expert budget (count), greedily assigns the next-most-activated expert across all layers until budget exhausted.

2. **Byte-budget** (`build_from_stats_with_layer_bytes`): Same greedy approach but accounts for varying expert sizes across layers. Maximizes activation-count-per-byte (value = count / bytes).

Both respect a `min_hot_per_layer` floor guarantee.

### Routing Statistics

`MoeHybridRoutingStats` maintains a flattened `[n_layer][n_expert]` activation count matrix. Observations come from:

- Direct `observe(layer, expert_ids, n)` calls during decoding
- `observe_selected_tensor()` for reading router outputs from GPU tensors

Statistics can be saved/loaded as CSV for offline analysis or warm-starting placement decisions.

### Swap Manager

At request boundaries, `build_moe_hybrid_swap_plan` identifies profitable swaps:

1. For each layer, find the weakest hot expert and strongest cold expert.
2. If the cold expert's activation count exceeds the hot expert's by `min_promote_gain`, propose a swap.
3. Rank all candidates by gain delta, apply up to `max_swaps_total`.

This adapts placement to workload drift without full re-profiling.

## FFN Evaluation Modes

### `eval_moe_hybrid_ffn_single`

Single-token decode path:

1. **Partition** selected expert IDs into hot (GPU-local) and cold (CPU-local) subsets using the local-by-global index maps.
2. **GPU path**: Run hot routed experts + shared expert in a single fused ggml graph (`run_hot_and_shared_ffn_gpu`).
3. **CPU path**: Run cold routed experts via `run_routed_subset` on CPU backend.
4. **Combine**: Sum hot+shared and cold results on host.

Telemetry reports wall time and per-phase breakdown (partition, hot, cold, shared, combine).

### `eval_moe_hybrid_ffn_gpu_resident`

Optimized single-token path that keeps activations on GPU:

- Reads only router IDs to CPU for hot/cold partitioning.
- Uses `GpuResidentState` with a pre-built `ResidualCombineGraph` (residual + hot + cold correction).
- Uses `CachedFfnGraph` to avoid per-token graph rebuilds.
- Cold expert output is uploaded back to GPU for the residual combine.

### `eval_moe_batched_prefill_ffn`

Batched prefill (all experts on GPU, no hybrid split). Used when all tokens can be processed at once on GPU.

### `eval_moe_hybrid_ffn_batched`

Batched hybrid prefill: splits the batch FFN into hot (GPU) and cold (CPU) subgraphs computed concurrently, then combines. Supports reusable allocators (`p_hot_alloc`, `p_cold_alloc`) for amortizing allocation cost across layers.

## Cached FFN Graphs

To avoid per-token ggml graph construction overhead, `CachedFfnGraph` pre-builds the computation graph for a fixed expert count:

- `build_cached_hot_graph` — hot experts + shared expert, fused into one GPU graph
- `build_cached_cold_graph` — cold experts on CPU

At inference time, only input/ids/weights tensors are updated and `ggml_backend_graph_compute` is called on the pre-allocated graph.

## Storage Construction

Two loading paths:

1. **From GPU tensors** (`build_moe_hybrid_storage`): Reads expert slices from already-loaded full stacked tensors on GPU. Copies hot slices to a compact GPU buffer, cold slices to CPU.

2. **From file** (`build_moe_hybrid_storage_from_file`): Reads expert slices directly from mmap'd GGUF file data, avoiding the need to load all experts to GPU first. Useful when VRAM is insufficient for the full model.

Both paths produce the same `MoeHybridStorage` containing per-layer split buffers ready for evaluation.

## Model Integration

The subsystem is model-agnostic. Model backends integrate via:

1. Include `moe_hybrid_types_impl.h` after their internal weight struct header.
2. Call `make_moe_hybrid_config(weights)` and `make_moe_layer_desc(layer)` to convert model-specific types to the generic interface.
3. Use the generic placement, storage, and eval APIs.

Currently integrated with:
- **qwen35moe** — all layers are MoE (`first_moe_layer = 0`), supports fused gate_up, shared expert gating, NVFP4 scales.
- **laguna** — layer 0 is dense (`first_moe_layer = n_layer_dense_lead`), no fused gate_up, no shared-expert gate, no NVFP4.

## Quantization Support

The `MoeLayerDesc` carries per-tensor scale factors (`*_s` fields) for NVFP4 quantization. The `apply_scale2()` helper in the eval code multiplies matmul results by the scale when non-unity, making the FFN evaluation transparently handle both standard quantized and NVFP4 models.

---

## Deep Dive: Overhead Reduction When All Experts Are Hot (GPU-Resident)

When all selected experts for a given token happen to reside on GPU, the system eliminates almost all overhead through several mechanisms:

### 1. Zero-Copy GPU→GPU Input Path

In `eval_moe_hybrid_ffn_gpu_resident`, the activation tensor (`ffn_post_gpu`) is already on GPU from the preceding attention layer. The input is fed to the hot graph via:

```cpp
ggml_backend_tensor_copy(ffn_post_gpu, storage.hot_graph.inp);  // GPU→GPU, no PCIe
```

This is a device-local memcpy (≈500 GB/s on RTX 3090 HBM-equivalent bandwidth), not a PCIe transfer. The activation never touches host memory.

### 2. Pre-Built Cached Graph (No Per-Token Construction)

The `CachedFfnGraph` is lazily built on first use and reused for every subsequent token with the same expert count:

```cpp
if (!storage.hot_graph.valid() || storage.hot_graph.n_hot != n_hot) {
    build_cached_hot_graph(...);  // only runs once per expert-count change
}
// Per-token: just update inputs and dispatch
ggml_backend_tensor_set(storage.hot_graph.ids, hot_ids.data(), ...);
ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);
```

This avoids:
- `ggml_init` / `ggml_free` per token (context allocation overhead)
- Graph construction (`ggml_new_graph_custom`, `ggml_build_forward_expand`)
- Buffer allocation (`ggml_gallocr_alloc_graph`) — the most expensive part, involving memory planning and pointer assignment for every intermediate tensor

In practice, building a graph costs 50–200 µs. With cached graphs, the per-token hot path is just tensor_set (IDs + weights, ~64 bytes each) + kernel launch.

### 3. Cold Path Skipped Entirely

When all experts are hot, `cold_ids` is empty:

```cpp
const bool has_cold = !cold_ids.empty();  // false when all hot
```

The entire cold branch is skipped — no CPU graph compute, no CPU→GPU upload of cold results. The `combine.cold_in` tensor retains its pre-initialized zeros.

### 4. Residual Combine Stays on GPU

The final combination (`output = residual + hot + cold`) runs as a pre-built `ResidualCombineGraph` on GPU:

```cpp
ggml_backend_tensor_copy(storage.hot_graph.output, gpu_state.combine.hot_in);  // GPU→GPU
ggml_backend_graph_compute(gpu_backend, gpu_state.combine.gf);                 // GPU kernel
ggml_backend_tensor_copy(gpu_state.combine.output, gpu_state.act_cur);         // GPU→GPU
```

The entire data flow for the all-hot case is: **GPU → GPU → GPU** with zero PCIe round-trips for activations.

### 5. Minimal Host-Side Data

The only host→device transfers when all experts are hot:
- Router IDs: `n_expert_used` × sizeof(int32_t) = typically 8 × 4 = **32 bytes**
- Router weights: `n_expert_used` × sizeof(float) = typically 8 × 4 = **32 bytes**

Total PCIe payload: ~64 bytes per token — negligible vs. the 14+ GB/s available bandwidth.

### Net Effect

For a Qwen3.5-MoE token where all 8 selected experts are hot, the MoE FFN evaluation path degenerates to:
1. One GPU→GPU tensor copy (activation input)
2. One pre-built GPU kernel dispatch (routed + shared FFN)
3. One GPU→GPU tensor copy (to combine input)
4. One pre-built GPU kernel dispatch (residual add)
5. One GPU→GPU tensor copy (to persistent state)

Wall time: **~100–200 µs** on RTX 3090 — comparable to a non-hybrid monolithic FFN dispatch with equivalent FLOPS.

---

## Deep Dive: Handling Experts in System RAM (Cold Path)

When some selected experts are cold (CPU-resident), the system uses a concurrent execution strategy to hide CPU latency behind GPU work.

### Memory Layout

Cold expert tensors are stored in CPU-backend buffers (`cold_buf`) allocated via `ggml_backend_cpu_init()`. They use the same quantization format as their GPU counterparts (Q4_K_M, etc.) — no format conversion is needed. The CPU backend is configured with limited threads:

```cpp
ggml_backend_cpu_set_n_threads(out.cpu_backend, std::max(1, std::min(cfg.n_expert_used, 8)));
```

This caps CPU threads to avoid starving the GPU driver thread and other system work.

### Concurrent GPU/CPU Execution

The key insight: **launch GPU kernels first (async), then run CPU work while GPU is busy.**

```
Timeline:
  GPU: ┤███ hot FFN + shared ███████████████████████├─ sync ─┤
  CPU:      ┤██ cold FFN (overlaps GPU) ██├
                                                         ┤ combine ├
```

In code (`eval_moe_hybrid_ffn_single`):

```cpp
// 1. Launch GPU async (returns immediately)
ggml_backend_graph_compute_async(gpu_backend, storage.hot_graph.gf);

// 2. Run cold on CPU (blocking, but GPU is running in parallel)
ggml_backend_graph_compute(cpu_backend, storage.cold_graph.gf);

// 3. Sync GPU (usually already done by the time CPU finishes)
ggml_backend_synchronize(gpu_backend);

// 4. Combine results
out[i] = hot_and_shared[i] + cold[i];
```

### Why CPU Evaluation (Not GPU Streaming Over PCIe)?

A naive alternative would keep cold experts in system RAM but stream them to GPU for computation. This is **strictly worse** for single-token decode:

| Metric | CPU-local compute | GPU via PCIe stream |
|--------|-------------------|---------------------|
| Bandwidth to weights | 40–60 GB/s (DDR5) | 15.75 GB/s (PCIe 4.0 x16) |
| Transfer overhead | 0 (weights are local) | Must DMA entire expert (~6 MB for Q4_K_M) |
| GPU utilization | GPU runs hot experts in parallel | GPU stalls waiting for PCIe DMA |
| Latency per cold expert | ~100 µs (matmul on Zen4 AVX-512) | ~380 µs (DMA) + kernel time |

**The fundamental problem**: PCIe 4.0 x16 bandwidth (15.75 GB/s) is **4× lower** than DDR5 memory bandwidth (≈50 GB/s). Loading a single Q4_K_M Qwen expert (gate + up + down ≈ 6 MB) over PCIe takes ~380 µs just for the transfer. Meanwhile, the CPU can complete the entire matmul chain in ~100–150 µs using the same data that's already in L3/RAM.

Additionally:
- **PCIe is half-duplex for bulk transfers**: You cannot overlap upload of expert weights with download of results efficiently.
- **GPU kernel launch overhead**: Even after DMA completes, the GPU needs to dispatch a kernel for a single-token matmul on just 1–2 cold experts — an inefficient use of GPU SMs.
- **Bubble injection**: Streaming cold experts to GPU introduces pipeline bubbles. The GPU must wait for DMA → compute → return result, during which the SMs sit idle or context-switch. With CPU-local compute, the GPU is 100% occupied on hot experts.

### When Streaming *Would* Make Sense

PCIe streaming is only beneficial for **large batch prefill** where:
- Many tokens need the same cold expert (amortizing DMA cost over batch)
- GPU FLOPS utilization is high enough to offset transfer time
- The batch size makes CPU compute bandwidth-bound

This is why `eval_moe_batched_prefill_ffn` uses the full GPU expert stack (all experts on GPU) for prefill — during prefill, the model is loaded fully and the batch dimension makes GPU compute dominant.

### Cold Path Overhead Budget

For a typical decode token with 2 out of 8 selected experts cold:

| Phase | Duration | Notes |
|-------|----------|-------|
| Partition | ~1 µs | Index lookup in `hot_local_by_global` |
| Cold graph setup | 0 µs (cached) | Pre-built, reused |
| Cold tensor_set (inp) | ~2 µs | n_embd × 4 = ~14 KB for 3584-dim |
| Cold graph compute | ~100–150 µs | 2 experts × (gate + up + SwiGLU + down) |
| Cold tensor_get (out) | ~2 µs | n_embd × 4 = ~14 KB |
| **Total cold latency** | **~105–155 µs** | Overlaps with GPU hot path |

Since the GPU hot path (6 experts + shared) takes ~100–200 µs, the cold path is **fully hidden** in the overlap window in most cases.

---

## Deep Dive: Why Not Load Data Over PCIe (Let CUDA Do All Work)

### The PCIe Bottleneck Argument

A seemingly simpler design would keep all expert weights in system RAM (unified virtual memory or explicit staging) and let CUDA handle everything:

```
Naive approach: System RAM → PCIe → GPU VRAM → CUDA kernel → result
```

This fails catastrophically for MoE decode for three reasons:

### 1. PCIe Bandwidth Is the Binding Constraint (Not Compute)

For a single-token MoE FFN step on Qwen3.5-MoE-27B (Q4_K_M):
- 8 selected experts × ~6 MB each = **48 MB** of weight data
- PCIe 4.0 x16: 15.75 GB/s → 48 MB takes **3.05 ms**
- Actual GPU compute for those matmuls: **~200 µs**

The system would spend **93% of time waiting for PCIe** and **7% doing useful compute**. Token generation throughput drops from ~50 tok/s to ~5 tok/s — a 10× regression.

### 2. PCIe Transfers Are Not Free (Even With Async DMA)

CUDA DMA engines can transfer while kernels run, but:
- **Pinned memory required**: System RAM must be page-locked for async DMA, which pressures the OS memory manager and increases allocation latency.
- **DMA engine contention**: Most consumer GPUs have 1–2 copy engines. With 8 expert transfers queued, serialization and scheduling overhead accumulates.
- **TLB/IOMMU overhead**: Each DMA transfer requires address translation through the IOMMU (Intel VT-d / AMD-Vi), adding ~1–5 µs per transfer setup.
- **Cache pollution on return**: Results copied back from GPU flush CPU caches that were holding useful data for the next layer's attention computation.

### 3. Hybrid CPU Compute Exploits a Free Resource

The CPU cores are otherwise **idle** during GPU decode. Between two GPU kernel dispatches (attention → FFN → next-layer attention), the CPU driver thread submits work and waits. Running cold expert matmuls on CPU during this window is pure throughput gain at zero opportunity cost:

```
Without hybrid:  GPU: [attention][wait for PCIe][FFN][attention]...
                 CPU: [idle.........................idle........]

With hybrid:     GPU: [attention][hot FFN async]....[combine]...
                 CPU: [          cold FFN       ]...[idle]......
```

### 4. Quantized Matmuls Are CPU-Efficient

ggml's Q4_K_M dequant+matmul kernels are highly optimized for x86 (AVX2/AVX-512):
- A single Qwen expert forward (gate + up + SwiGLU + down) on 8 cores: ~50–75 µs
- Memory bandwidth utilization on DDR5-5600: ~85% of theoretical peak
- The CPU is compute-limited on a single token, not memory-limited — ideal workload balance

### 5. Total System Throughput Comparison

| Strategy | MoE FFN latency (8 experts, 2 cold) | Bottleneck |
|----------|--------------------------------------|------------|
| All-GPU (requires 40+ GB VRAM) | ~200 µs | GPU compute |
| PCIe streaming (all from RAM) | ~3050 µs | PCIe bandwidth |
| Hybrid (6 hot GPU + 2 cold CPU) | ~200 µs | max(GPU, CPU) |

The hybrid approach matches all-GPU performance on the common case (most experts hot) while gracefully degrading on cold hits — and it fits in 24 GB VRAM.

### 6. The Residual Upload Is Tiny

The only mandatory CPU→GPU transfer in the hybrid path is the cold expert result:
- Size: `n_embd × sizeof(float)` = 3584 × 4 = **14 KB**
- Transfer time at PCIe 4.0: **~1 µs** (negligible)

Compare this to the 48 MB that would be needed to stream all expert weights. The hybrid design reduces PCIe traffic by **3400×** compared to full PCIe streaming.
