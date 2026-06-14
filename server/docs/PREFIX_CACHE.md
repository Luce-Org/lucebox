# Prefix Cache Design

## Overview

The prefix cache accelerates multi-turn conversations by snapshotting the
KV cache state at turn boundaries. On subsequent requests that share the
same system prompt / early conversation, the server restores from a snapshot
instead of recomputing the full prefill — saving both latency and compute.

```
Request 1: [system + user1 + assistant1 + user2]
                    ↑ boundary — snapshot here
Request 2: [system + user1 + assistant1 + user2 + assistant2 + user3]
            └── restore snapshot ──────────┘      └── diff-prefill ──┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Server                           │
│  • Tokenizes prompt                                     │
│  • Calls PrefixCache for lookup / prepare               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              PrefixCache  (LRU logic)                    │
│  • detect chat boundaries via ChatMarkers               │
│  • SHA-1 hash prefix at each boundary                   │
│  • LRU eviction when cap is reached                     │
│  • Two RAM pools: inline prefix + exact prefill         │
└────────────────────────┬────────────────────────────────┘
                         │ snapshot_save / restore_and_generate
                         ▼
┌─────────────────────────────────────────────────────────┐
│             ModelBackend (per-arch)                      │
│  • snapshot_save(slot): copy KV cache → snap_backend_   │
│  • restore_and_generate(slot, req): snap → KV + decode  │
│  • snapshot_free(slot): release snapshot memory          │
└─────────────────────────────────────────────────────────┘
```

## Cache Pools

### Tier 1: Inline Prefix Cache

Caches KV state at **turn boundaries** within a conversation. The boundary
detector uses `ChatMarkers` to find end-of-message + start-of-next-role
token sequences.

- **lookup()**: Finds the longest cached prefix matching the current prompt.
- **prepare_inline_snap()**: Selects a slot and cut-point for snapshotting
  after the current prefill completes.
- **confirm_inline_snap()**: Commits the entry after successful save.
- LRU eviction: oldest entry's slot is reused when capacity is reached.

### Tier 2: Exact Prefill Cache

Caches the **entire post-compression KV state**, keyed on the raw
(pre-compression) prompt. Hits skip both PFlash compression and prefill
entirely -- the fastest path for repeated prompts.

- Separate slot pool starting at `cap` (inline slots are `[0, cap)`).
- Keyed on SHA-1 of the full raw prompt (not just a prefix).

The two cache kinds are intentionally separate:

- `prefix` controls turn-boundary prefix snapshots.
- `prefill` controls exact full-prompt snapshots.

Each kind can have an independent RAM and disk budget. Snapshot handles are
still used internally by the backend, but operators configure byte budgets
rather than handle counts.

## Snapshot Memory Management

### Problem: VRAM Pressure on Discrete GPUs

Naive snapshots stored **full max_ctx KV tensors** regardless of actual
cache occupancy. For short prefixes this was extremely wasteful:

| Model | max_ctx | Full snapshot | Right-sized (29 tokens) |
|-------|---------|--------------|------------------------|
| Qwen3.5-27B (Q8_0 KV) | 64000 | ~1.5 GB | ~0.17 MB |
| Gemma4 | 32768 | ~0.5 GB | ~0.05 MB |
| Laguna | 16384 | ~0.3 GB | ~0.03 MB |

With 3 inline snapshots at cur_pos=29,138,265 the old code allocated ~4.5 GB
of GPU memory (on a 22 GB card already holding ~17 GB for model+cache),
causing spill to system RAM and 5× decode slowdowns.

### Solution: Right-Sized Snapshots + Platform-Aware Backend

Two complementary fixes:

1. **Right-sized allocation**: KV tensors are allocated as
   `[head_dim, cur_pos, n_head_kv]` instead of `[head_dim, max_ctx, n_head_kv]`.
   This reduces per-snapshot memory from ~1.5 GB to a few MB for typical
   prefix lengths.

2. **Platform-aware backend**: Snapshots are stored on system RAM (CPU backend)
   for discrete GPUs, keeping VRAM free for model weights and active cache.

```cpp
// Right-sized KV allocation in snapshot_target_cache():
ggml_tensor * K = ggml_new_tensor_3d(snap.ctx, sk->type,
                                      sk->ne[0], snap_pos, sk->ne[2]);
// snap_pos = cache.cur_pos (e.g., 29 instead of 64000)
```

**Buffer reuse**: When the same slot is saved at the same `cur_pos`, the
existing buffer is reused (no free+alloc). Only when `cur_pos` changes is
the buffer freed and a new (still tiny) one allocated.

**Strip copy**: Since right-sized KV tensors have different ne[1] than the
full-size cache, save/restore uses per-head strip copies via
`ggml_backend_tensor_get/set` — the same pattern as thin snapshots.

### Platform-Aware Snapshot Backend

Snapshots are stored on a **snapshot backend** selected at init time based
on the compute backend's memory characteristics:

```cpp
// common/snapshot_backend.h

ggml_backend_t create_snapshot_backend(ggml_backend_t compute_backend);
void free_snapshot_backend(ggml_backend_t snap, ggml_backend_t compute);
```

**Decision logic:**

```
compute_backend's default buffer type is host-accessible?
├── YES (unified memory: Metal, AMD iGPU/HALO)
│   └── return compute_backend  (no copy needed, same physical RAM)
└── NO  (discrete VRAM: CUDA, HIP)
    └── return ggml_backend_cpu_init()  (system RAM, off-GPU)
```

**Platform behavior:**

| Platform | GPU type | Snapshot storage | Copy cost |
|----------|----------|-----------------|-----------|
| CUDA (discrete) | RTX 2080 Ti, etc. | System RAM (CPU backend) | GPU↔CPU via PCIe at save/restore |
| Metal (Apple Silicon) | Unified | Same as compute (no-op) | Zero (same memory) |
| AMD HALO / iGPU | Unified | Same as compute (no-op) | Zero (same memory) |
| HIP (discrete) | RX 7900, etc. | System RAM (CPU backend) | GPU↔CPU via PCIe at save/restore |

### Cross-Backend Transfer

Right-sized snapshots use `ggml_backend_tensor_get/set` with explicit
offsets for KV tensors (since source and destination have different shapes).
SSM/conv state (fixed-size) uses `ggml_backend_tensor_copy()` directly.

### Integration Pattern (per-backend)

Each backend adds a single member and three code points:

```cpp
// Header:
ggml_backend_t snap_backend_ = nullptr;

// init():
snap_backend_ = create_snapshot_backend(compute_backend_);

// snapshot_save(): right-sized alloc + partial copy
//   KV: [head_dim, cur_pos, n_head_kv] on snap_backend_
//   SSM/conv: full-size on snap_backend_
//   target_feat: [fc_in, min(cur_pos, cap)] on snap_backend_

// shutdown(): free in correct order
for (auto & s : snapshots_) free_snapshot(s);  // free tensors first
free_snapshot_backend(snap_backend_, compute_backend_);  // then backend
```

## Configuration

| Server flag | Default | Description |
|-------------|---------|-------------|
| `--cache-prefix-ram SIZE` | 2GiB | RAM budget for turn-boundary prefix snapshots |
| `--cache-prefill-ram SIZE` | 0 | RAM budget for exact full-prompt snapshots |
| `--kv-cache-dir PATH` | unset | Root directory for disk cache pools |
| `--cache-prefix-disk SIZE` | 4GiB | Disk budget for turn-boundary prefix snapshots |
| `--cache-prefill-disk SIZE` | 0 | Disk budget for exact full-prompt snapshots |
| `--skip-park` | false | Skip parking draft model during compress |

Deprecated compatibility aliases:

| Server flag | Replacement |
|-------------|-------------|
| `--prefix-cache-slots N` | `--cache-prefix-ram N*64MiB` |
| `--prefill-cache-slots N` | `--cache-prefill-ram N*64MiB` |
| `--kv-cache-budget N` | `--cache-prefix-disk NMiB` |

### Choosing RAM Budgets

With right-sized, CPU-resident snapshots the limiting resource is **system RAM**,
not VRAM. Cache budgets are enforced against the actual snapshot byte size
reported by the backend after save. If a snapshot is larger than its configured
pool, it is evicted immediately and the next identical request will not hit.

Inline prefix snapshots are usually small because they are taken at turn
boundaries. Exact prefill snapshots cover the full effective prompt and can be
hundreds of MiB for multi-thousand-token Qwen3.6 prompts, so size the prefill
pool for the largest repeated prompt you want to retain.

| Scenario | Typical prefix length | Recommended prefix RAM |
|----------|----------------------|------------------------|
| Single-user chat | 200–2000 tokens | 512MiB-2GiB |
| Multi-session agent | 500–5000 tokens | 2GiB-4GiB |
| Batch / benchmark | N/A (cold starts) | 256MiB |

| Scenario | Typical full prompt | Recommended prefill RAM |
|----------|---------------------|-------------------------|
| Disabled exact-cache path | N/A | 0 |
| Qwen3.6 repeated-prompt proof | ~5k tokens | 1GiB |
| Large prompt replay | 10k+ tokens | 2GiB+ |

The backend still has a finite snapshot-handle table. The server derives
internal handle counts from RAM budgets using a conservative 64MiB-per-handle
estimate, then enforces the configured byte budget with LRU eviction after
snapshot save. Slot 63 is reserved for disk-cache staging.

## File Map

| File | Role |
|------|------|
| `server/prefix_cache.{h,cpp}` | LRU logic, boundary detection, hashing |
| `common/snapshot_backend.h` | Platform-aware snapshot backend selection |
| `common/model_backend.h` | `snapshot_save/free/used/cur_pos` interface |
| `qwen35/qwen35_target_graph.cpp` | `snapshot_target_cache()`, `restore_target_cache()` |
| `laguna/laguna_target_graph.cpp` | `laguna_snapshot_alloc/save/restore()` |
| `gemma4/gemma4_backend.cpp` | Inline snapshot allocation + copy |

## Performance Characteristics

### Save (prefill → snapshot)
- **Unified memory**: Near-instant (just memcpy, no PCIe)
- **Discrete GPU**: PCIe transfer (right-sized: typically < 1ms for short prefixes)
- Amortized over the full prefill time (typically seconds for long prompts)

### Restore (snapshot → KV cache)
- **Unified memory**: Near-instant
- **Discrete GPU**: PCIe transfer (e.g., 4096 tokens → ~6ms)
- Always faster than re-computing prefill (which takes seconds)

### Net Impact
With right-sized snapshots, typical save/restore transfers are small:
- cur_pos=265 → ~4.6 MB → < 1ms over PCIe
- cur_pos=4096 → ~70 MB → ~6ms over PCIe

The old full-size approach (1.5 GB per snapshot) is eliminated. System RAM
footprint is proportional to actual token count, enabling many more cached
prefixes with no VRAM cost.
