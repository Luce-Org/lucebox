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
token sequences. DeepSeek's template closes only assistant turns with an end
marker (the system text and user turns end where the next role starts), so
for that family every role marker is a boundary: the system text, each
completed turn, and the generation prompt. A first turn therefore snapshots
its system prompt, and a new session on the same system prompt restores it.

For tool-using chat templates, tool definitions are rendered in the system
prefix. The first safe boundary therefore includes the complete tool schema:
the first request pays that prefill once, and later requests with the same
tools restore it through the normal native cache. See
[`TOOL_PREFIX_CACHE.md`](TOOL_PREFIX_CACHE.md) for the correctness contract and
benchmark.

- **lookup()**: Finds the longest cached prefix matching the current prompt.
- **prepare_inline_snap()**: Selects a slot and cut-point for snapshotting
  after the current prefill completes.
- **confirm_inline_snap()**: Commits the entry after successful save.
- LRU eviction: oldest entry's slot is reused when capacity is reached.

### Tier 2: Exact Prefill Cache

Caches the **entire post-compression KV state** in RAM, keyed on the raw
(pre-compression) prompt. Hits skip both PFlash compression and prefill
entirely -- the fastest path for repeated prompts.

- Separate slot pool starting at `cap` (inline slots are `[0, cap)`).
- Keyed on SHA-1 of the full raw prompt (not just a prefix).

The two flags are intentionally separate:

- `--prefix-cache-slots` controls turn-boundary prefix snapshots.
- `--prefill-cache-slots` controls exact full-prompt snapshots.

The backend can save one snapshot during a generation. For requests carrying
tools, the reusable inline system/tool boundary takes priority over an exact
full-prompt snapshot; for ordinary requests, the exact cache keeps priority.
If the preferred cache has no useful boundary or available slot, the server
falls back to the other tier. A restore only reserves a new inline snapshot
when the selected boundary advances beyond the restored prefix.

The disk prefix cache is a separate persistence/overflow layer for token-keyed
snapshots. Today it is integrated with the inline/effective-prompt path; exact
prefill snapshots are kept in RAM unless a dedicated raw-prompt disk path is
added. With the default `full` policy a lookup probes the whole prompt and
then every chat boundary, deepest first, so a restart recovers the inline
snapshots the previous process persisted (probes are index lookups; only a
hit reads a file).

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
| `--prefix-cache-slots N` | 32 | Max turn-boundary prefix cache slots |
| `--prefix-cache-max-mib auto\|N` | auto | Resident RAM limit for single-sequence prefix snapshots; `auto` keeps room for three snapshots at `--max-ctx`, capped at a quarter of the memory available at startup (a readable cgroup v2 container limit is included; otherwise total physical RAM is used); `0` is unlimited. Enforced for Qwen and DeepSeek4 (not DeepSeek4 mixed-backend splits): other backends cannot size snapshots, so `auto` stays unlimited there and an explicit limit is rejected at startup |
| `--concurrent-prefix-cache-max-mib N` | 4096 | Resident RAM limit for copied concurrent paged checkpoints; `0` is unlimited |
| `--prefill-cache-slots N` | 0 | Max exact full-prompt prefill cache slots |
| `--skip-park` | false | Skip parking draft model during compress |

### Choosing `--prefix-cache-slots`

With right-sized, CPU-resident snapshots the limiting resource is **system RAM**,
not VRAM. A snapshot of a hybrid model also copies the full recurrent state, so
the fixed part dominates short prefixes. For Qwen3.5/3.6-27B with q8_0 KV and a
DFlash draft, one snapshot is about 34 KiB per token plus about 350 MiB fixed:
0.7 GiB at 10K tokens, 1.0 GiB at 20K and 3.5 GiB at 94K.

Every chat boundary (and every extra cut a cache may take) starts a prefill
chunk, in a cold prefill and after a restore alike (`restore_points`). A
snapshot therefore lands exactly on the boundary it was requested at, and a
restored prefix plus the suffix prefill reproduces a cold prefill. Apart from
the last chunk of the prompt, Qwen never ends a chunk within 16 tokens of its
start, or 64 after restoring a generated-turn checkpoint, so the cache does not
request a snapshot that close to the restored prefix. The
generation prompt's own boundary starts a chunk only when the request
snapshots there, since no saved state lies past it otherwise.

After a tool-call turn the server also keeps the state the generation left
behind, keyed by the prompt plus the generated tokens the next request renders
identically (see [Agent Turn Cache](API.md#agent-turn-cache)). Qwen defers that
snapshot: while the next request continues the conversation it is never copied
to system RAM, and it is copied out only before other work would overwrite the
live state, or when the disk cache persists it at shutdown. Restoring the
snapshot the live state already holds copies nothing back either.

Coding agents grow one conversation turn by turn, and each turn's snapshot is a
strict prefix of the next. After committing a snapshot, the single-sequence
path frees the ancestors it supersedes, keeping only the shallowest one (the
shared system/tools head) and protected tool pins. A conversation therefore
holds about two snapshots instead of one per turn. Chains that are abandoned,
for example after a client compacts its context, are bounded by
`--prefix-cache-max-mib`: when a new snapshot would not fit, the cache replaces
the oldest leaf first and skips the capture if nothing can make room. Snapshots
the new one supersedes (including the one it restored from) count as freed,
because they are pruned as soon as it is committed; a failed capture keeps them.
After each commit the cache also evicts, oldest leaf first, until the committed
snapshots fit the limit again (never evicting the new snapshot or a protected
tools pin), so a capture that lands shorter than it reserved does not leave the
cache over budget. The limit covers committed snapshots: the
new snapshot is allocated before the ones it replaces are freed, so while a
request runs, process memory can exceed the limit by up to one snapshot.

Concurrent paged serving measures the exact backend allocation required for
each checkpoint before copying it. The cache keeps committed checkpoints under
`--concurrent-prefix-cache-max-mib`: when necessary it replaces one eligible least-recently-used
entry, and if no single eligible entry can make enough room it skips the new
checkpoint without disturbing the committed cache. The configured limit covers
resident committed checkpoint buffers. During an atomic replacement, the new
buffer and the selected victim can coexist briefly, so transient process memory
can exceed the limit by up to one checkpoint.

| Scenario | Typical prefix length | Recommended cap |
|----------|----------------------|-----------------|
| Single-user chat | 200–2000 tokens | 16–32 |
| Multi-session agent | 500–5000 tokens | 32–63 |
| Batch / benchmark | N/A (cold starts) | 4 |

The in-memory cache limit is 63 slots. Backend slot 63 is reserved for disk
cache staging, preventing disk loads from overwriting a live in-memory entry.

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
