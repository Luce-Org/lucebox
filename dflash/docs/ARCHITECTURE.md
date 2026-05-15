# DFlash Multi-Model Architecture

## Overview

DFlash supports multiple LLM architectures through a plugin-style backend
system. Each model family (qwen35, qwen3, gemma4, laguna) implements a
`ModelBackend` interface, and a single generic daemon loop drives them all
using a uniform stdin/stdout protocol.

```
┌─────────────────────────────────────────────────────────┐
│                   test_dflash / run.py                   │
│              (CLI dispatch, benchmarks, tests)           │
└────────────────────────┬────────────────────────────────┘
                         │ selects backend by GGUF arch
                         ▼
┌─────────────────────────────────────────────────────────┐
│               daemon_loop.cpp  (generic)                │
│  stdin/stdout protocol: generate, park, snapshot, ...   │
└────────────────────────┬────────────────────────────────┘
                         │ calls ModelBackend vtable
          ┌──────────────┼──────────────┬──────────────┐
          ▼              ▼              ▼              ▼
   ┌────────────┐ ┌───────────┐ ┌───────────┐ ┌────────────┐
   │ Qwen35     │ │ Qwen3     │ │ Gemma4    │ │ Laguna     │
   │ Backend    │ │ Backend   │ │ Backend   │ │ Backend    │
   │ (spec dec) │ │ (AR only) │ │ (AR only) │ │ (AR only)  │
   └──────┬─────┘ └───────────┘ └───────────┘ └────────────┘
          │
          ▼
   ┌──────────────┐
   │ DFlashTarget │  (opt-in interface for speculative decode)
   └──────────────┘
```

## Directory Structure

```
dflash/src/
├── common/                 # Shared infrastructure (all backends)
│   ├── model_backend.h     # ModelBackend abstract interface
│   ├── dflash_target.h     # DFlashTarget interface (spec decode)
│   ├── daemon_loop.{h,cpp} # Generic stdin/stdout daemon loop
│   ├── device_placement.h  # Multi-GPU placement config
│   ├── gguf_inspect.{h,cpp}# Read arch + layer count from GGUF
│   ├── layer_split_utils.{h,cpp}  # compute_layer_ranges()
│   ├── dflash_layer_split_runtime.h  # LayerSplitRuntimeConfig + ActivationPair
│   ├── dflash_feature_ring.{h,cpp}   # DraftFeatureMirror + ring copy helpers
│   ├── dflash_capture.{h,cpp}        # target_capture_index() helper
│   ├── dflash_draft_ipc.{h,cpp}      # DFlash draft IPC client + remote copy
│   ├── dflash_draft_ipc_daemon.cpp   # Generic DFlash draft IPC daemon body
│   ├── dflash_draft_graph.{h,cpp}    # Universal build_draft_step (DFlash draft graph)
│   ├── dflash_spec_decode.{h,cpp}    # Generic spec-decode loop over DFlashTarget
│   ├── ddtree.{h,cpp}     # Dynamic Draft Tree algorithm
│   ├── attn_masks.h        # Causal + tree attention mask builders
│   ├── peer_access.{h,cpp} # CUDA peer-access utilities
│   ├── step_graph.h        # ggml graph container (StepGraph)
│   ├── io_utils.h          # Binary I/O helpers (read_int32_file, etc.)
│   ├── sampler.{h,cpp}     # Token sampling (greedy, top-k, top-p)
│   └── gpu_runtime_compat.h
│
├── qwen35/                 # Qwen3.5 hybrid (attn + DeltaNet/SSM)
│   ├── qwen35_backend.{h,cpp}     # Full backend with spec decode
│   ├── qwen35_daemon.{h,cpp}      # Thin daemon entry point
│   ├── qwen35_dflash_target.{h,cpp}  # DFlashTarget adapter (single-GPU)
│   ├── qwen35_layer_split_dflash_target.{h,cpp}  # DFlashTarget adapter (multi-GPU)
│   ├── graph_builders.{h,cpp}     # Build ggml graphs for qwen35 target + lm_head
│   ├── layer_split_types.h        # qwen35 TargetLayerSplitShard
│   ├── layer_split_*.{h,cpp}      # Multi-GPU layer-split daemon
│   └── ...
│
├── qwen3/                  # Qwen3 (standard transformer)
│   ├── qwen3_backend.{h,cpp}
│   ├── qwen3_daemon.{h,cpp}
│   └── qwen3_loader.cpp
│
├── gemma4/                 # Gemma4 (iSWA + MoE)
│   ├── gemma4_backend.{h,cpp}
│   ├── gemma4_daemon.{h,cpp}
│   ├── gemma4_graph.cpp
│   ├── gemma4_loader.cpp
│   └── gemma4_internal.h
│
└── laguna/                 # Poolside Laguna
    ├── laguna_backend.{h,cpp}
    └── laguna_daemon.{h,cpp}
```

## Core Interfaces

### 1. ModelBackend (`common/model_backend.h`)

The central abstraction. Every model backend implements this interface:

```cpp
struct ModelBackend {
    // Lifecycle
    virtual void print_ready_banner() const = 0;
    virtual void shutdown() = 0;

    // Generation — backend owns the strategy (AR, spec decode, etc.)
    virtual GenerateResult generate(const GenerateRequest & req,
                                     const DaemonIO & io) = 0;

    // Park/Unpark — release/restore GPU resources on demand
    virtual bool park(const std::string & what) = 0;
    virtual bool unpark(const std::string & what) = 0;
    virtual bool is_target_parked() const = 0;

    // Prefix cache snapshots (up to 8 slots)
    virtual bool snapshot_save(int slot) = 0;
    virtual void snapshot_free(int slot) = 0;
    virtual bool snapshot_used(int slot) const = 0;
    virtual int  snapshot_cur_pos(int slot) const = 0;
    virtual GenerateResult restore_and_generate(int slot,
                                                 const GenerateRequest & req,
                                                 const DaemonIO & io) = 0;

    // PFlash compression
    virtual bool handle_compress(const std::string & line,
                                  const DaemonIO & io) = 0;
    virtual void free_drafter() = 0;

    // Optional: DFlash speculative decode support
    virtual bool supports_dflash_spec_decode() const { return false; }
    virtual DFlashTarget * dflash_target() { return nullptr; }

    // Optional: arch-specific commands
    virtual bool try_handle_command(const std::string & line,
                                     const DaemonIO & io) { return false; }
};
```

**Key design decisions:**
- `generate()` is the single entry point — the backend decides internally
  whether to use autoregressive, speculative decode, DDTree, etc.
- Snapshots and compress are part of the base interface because the daemon
  protocol exposes them for all architectures.
- `try_handle_command()` is the escape hatch for arch-specific extensions.

### 2. DFlashTarget (`common/dflash_target.h`)

Optional interface for backends that want speculative decoding with the
universal DFlash draft model:

```cpp
struct DFlashTarget {
    // Run a verify batch, return argmax tokens
    virtual bool verify_batch(const std::vector<int32_t> & tokens,
                              int base_pos, int & last_tok,
                              std::vector<int32_t> * all_argmax) = 0;

    // KV cache snapshot/restore for rollback on rejected tokens
    virtual bool snapshot_kv() = 0;
    virtual bool restore_kv() = 0;

    // Token utilities
    virtual bool is_eos(int token) const = 0;
    virtual bool embed_tokens(const int32_t * tokens, int n, float * out) const = 0;

    // Project draft hidden states → token IDs via target's lm_head
    virtual bool project_hidden_to_tokens(const float * hidden, int n_tokens,
                                          std::vector<int32_t> & tokens_out) = 0;

    // Configuration the draft model needs
    virtual int hidden_size() const = 0;
    virtual int mask_token_id() const = 0;
    virtual const std::vector<int> & capture_layer_ids() const = 0;
};
```

**How it works:** The DFlash draft model is architecture-agnostic — it
cross-attends to intermediate features captured during the target's forward
pass. The target only needs to:
1. Capture activations at specified layers during `verify_batch()`
2. Support KV snapshot/restore for rollback
3. Project draft hidden states through its own lm_head

### 3. DevicePlacement (`common/device_placement.h`)

Uniform GPU configuration for all backends:

```cpp
struct DevicePlacement {
    int gpu = 0;                              // primary GPU
    std::vector<int>    layer_split_gpus;     // multi-GPU sharding
    std::vector<double> layer_split_weights;  // proportional distribution
    bool peer_access = false;
    int  max_ctx     = 8192;

    bool is_layer_split() const;
    int  primary_gpu() const;
};
```

### 4. Daemon Loop (`common/daemon_loop.h`)

The generic daemon loop handles the stdin/stdout protocol and dispatches to
the backend:

```cpp
int run_daemon(ModelBackend & backend, const DaemonLoopArgs & args);
```

Protocol commands handled generically: `quit`, `park`, `unpark`, `compress`,
`SNAPSHOT`, `RESTORE`, `FREE_SNAPSHOT`, `LIST_SLOTS`, generate (prompt file
or inline). Everything else falls through to `backend.try_handle_command()`.

### 5. Supporting Utilities

| Header | Purpose |
|--------|---------|
| `ddtree.{h,cpp}` | DDTree algorithm — build best-first spec decode trees from draft top-K |
| `attn_masks.h` | Build causal masks (chain) and tree-structured masks (DDTree verify) |
| `step_graph.h` | `StepGraph` — per-forward-call ggml graph container with persistent allocator |
| `peer_access.{h,cpp}` | CUDA peer access setup between GPUs |
| `gguf_inspect.{h,cpp}` | Read arch + layer count from GGUF without loading weights |
| `layer_split_utils.{h,cpp}` | `compute_layer_ranges()` for multi-GPU sharding |
| `io_utils.h` | `read_int32_file()`, `stream_emit()`, and other I/O helpers |

## Existing Backend Implementations

### Qwen35Backend (full-featured reference)

The most complete backend — implements everything including DFlash speculative
decode, DDTree mode, multi-GPU layer-split, SSM state rollback, and prefix
cache snapshots. Use this as a reference when implementing a new backend.

Key components:
- **Graph builders**: Per-target `graph_builders.{h,cpp}` build the
  ggml compute graphs for target forward and lm_head projection. The
  universal DFlash draft graph (`build_draft_step`) lives in
  `common/dflash_draft_graph.{h,cpp}` and is shared across all targets.
- **Spec decode** (`common/dflash_spec_decode.{h,cpp}`): The generic
  draft→verify→accept loop, typed against `DFlashTarget`. Reusable by every
  backend.
- **DFlashTarget adapters** (`qwen35_dflash_target.{h,cpp}` for single-GPU,
  `qwen35_layer_split_dflash_target.{h,cpp}` for multi-GPU layer split):
  Bridge qwen35 internals (`TargetWeights`, `TargetCache`,
  `TargetLayerSplitShard`) to the generic `DFlashTarget` interface so the
  shared spec-decode loop can drive verification.
- **Feature transfer + draft daemon** (`common/dflash_feature_ring.{h,cpp}`,
  `common/dflash_capture.{h,cpp}`, `common/dflash_draft_ipc.{h,cpp}`,
  `common/dflash_draft_ipc_daemon.cpp`):
  Move captured target activations into the draft-side ring buffer
  (`DraftFeatureMirror`) and ship them across processes/GPUs. The IPC
  client, parent-side feature-slice helper, and the daemon body itself
  all live in `common/` and are reusable by any DFlash target architecture
  (the DFlash draft model is a single universal Qwen3-style network shared
  across every target).

### Qwen3Backend, Gemma4Backend, LagunaBackend

AR-only backends. They implement the full ModelBackend interface but use
simple autoregressive decode in `generate()`. They do NOT implement
`DFlashTarget` (yet).

---

## Guide: Adding DFlash Speculative Decode to Gemma4

This section walks through what's needed to give Gemma4 the same ~2x speedup
that qwen35 gets from DFlash speculative decoding.

### Prerequisites

- A working `Gemma4Backend` with AR-only `generate()` (already exists)
- The universal DFlash draft model weights (same `draft-*.safetensors` used
  for all architectures)
- Understanding of which Gemma4 layers to capture features from

### Step 1: Implement `Gemma4DFlashTarget`

Create `src/gemma4/gemma4_dflash_target.{h,cpp}` implementing `DFlashTarget`:

```cpp
class Gemma4DFlashTarget : public DFlashTarget {
public:
    Gemma4DFlashTarget(Gemma4Weights & w, Gemma4Cache & cache,
                       ggml_backend_t backend, /* ... */);

    // Must implement all 9 methods:

    bool verify_batch(...) override;
    //   1. Build a ggml graph for batch-verify (multiple tokens at once)
    //   2. During forward, capture activations at capture_layer_ids()
    //   3. Copy captured features into the draft model's feature ring
    //   4. Run argmax on output logits, fill all_argmax if requested

    bool snapshot_kv() override;
    bool restore_kv() override;
    //   Save/restore KV cache tensors. For Gemma4's iSWA architecture,
    //   this means snapshotting both global-attention and sliding-window
    //   KV caches.

    bool is_eos(int token) const override;
    //   Check against Gemma4's EOS token(s).

    bool embed_tokens(const int32_t * tokens, int n, float * out) const override;
    //   Look up embeddings in w_.token_embd (Gemma4 uses per-layer
    //   embedding scaling, but the raw embedding is what the draft needs).

    bool project_hidden_to_tokens(const float * hidden, int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;
    //   Build a small ggml graph: out_norm → lm_head matmul → argmax.
    //   Gemma4 applies logit softcapping here.

    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override { /* Gemma4's mask token */ }
    const std::vector<int> & capture_layer_ids() const override;
    //   Return the layer indices where features should be captured.
    //   Typically evenly spaced (e.g., layers 4, 9, 14, 19 for a
    //   26-layer model). The draft model's fc layer expects exactly
    //   this many feature slices.
};
```

### Step 2: Add Feature Capture to the Gemma4 Forward Pass

During `verify_batch()`, after each transformer layer's output, check if
that layer is in `capture_layer_ids()`. If so, copy the hidden states to
the draft model's feature ring buffer.

For single-GPU (target and draft on same device):
```cpp
// After layer i's output (hidden_states tensor on GPU):
if (is_capture_layer(i)) {
    // Direct GPU→GPU copy into draft feature ring
    ggml_backend_tensor_copy(hidden_states, draft_feature_slot[capture_idx]);
}
```

For multi-GPU, use `DraftFeatureMirror` (from `common/`) to handle cross-GPU
transfer via staging buffers.

### Step 3: Add KV Snapshot/Restore

Gemma4's iSWA architecture has two KV cache types:
- **Global attention layers**: standard KV cache (full context)
- **Sliding window layers**: windowed KV cache

Both must be snapshotted before speculative verify and restored on rejection:

```cpp
bool Gemma4DFlashTarget::snapshot_kv() {
    // Save cur_pos and a copy of KV tensors for both cache types
    saved_pos_ = cache_.cur_pos;
    // cudaMemcpy each KV tensor to snapshot buffer
}

bool Gemma4DFlashTarget::restore_kv() {
    cache_.cur_pos = saved_pos_;
    // cudaMemcpy snapshot buffers back to KV cache tensors
}
```

### Step 4: Wire Into Gemma4Backend

Update `Gemma4Backend` to create the adapter and enable spec decode:

```cpp
// In gemma4_backend.h:
bool supports_dflash_spec_decode() const override { return true; }
DFlashTarget * dflash_target() override;

// In gemma4_backend.cpp:
DFlashTarget * Gemma4Backend::dflash_target() {
    if (!dflash_target_) {
        dflash_target_ = std::make_unique<Gemma4DFlashTarget>(
            w_, cache_, backend_, /* sg_, kq_stride_pad, ... */);
    }
    return dflash_target_.get();
}
```

Update `generate()` to use speculative decode when a draft model is available:

```cpp
GenerateResult Gemma4Backend::generate(const GenerateRequest & req,
                                        const DaemonIO & io) {
    int committed = do_prefill(req.prompt, io);

    if (draft_loaded_ && supports_dflash_spec_decode()) {
        return do_spec_decode(committed, req.n_gen, ...);
    } else {
        return do_decode(committed, req.n_gen, ...);  // AR fallback
    }
}
```

### Step 5: Add Draft Model Loading

The DFlash draft model is a single universal architecture (Qwen3-style) that
works with any target. Load it using the existing `DraftWeights` loader:

```cpp
// In Gemma4Backend::init() or lazy on first spec-decode request:
DraftWeights dw;
if (!load_draft_weights(cfg_.draft_path, draft_backend_, dw)) { ... }
```

The draft model doesn't need Gemma4-specific code — it's the same model used
by Qwen35Backend.

### Step 6: Build and Test

```bash
cd dflash/build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(nproc)

# AR baseline
./test_dflash daemon --target ../../models/gemma4.gguf

# With speculative decode
DFLASH_DRAFT=../../models/draft/draft-Gemma4.safetensors \
    ./test_dflash daemon --target ../../models/gemma4.gguf
```

### Gemma4-Specific Considerations

1. **iSWA dual caches**: KV snapshot/restore is more complex than standard
   transformers because you have two cache types with different sizes
2. **Logit softcapping**: `project_hidden_to_tokens()` must apply Gemma4's
   `tanh(logit / cap) * cap` before argmax
3. **MoE routing**: Feature capture happens AFTER the MoE layer (at the
   residual stream), not inside individual experts
4. **Per-layer embedding scaling**: The draft model sees raw hidden states;
   any layer-specific scaling must be undone before capture

### Checklist

- [ ] Create `gemma4_dflash_target.{h,cpp}`
- [ ] Implement all 9 `DFlashTarget` methods
- [ ] Add KV snapshot/restore for both global and sliding-window caches
- [ ] Add feature capture hooks in the forward pass
- [ ] Wire `supports_dflash_spec_decode()` and `dflash_target()` in backend
- [ ] Load draft model weights
- [ ] Update `generate()` to dispatch to spec decode when draft is available
- [ ] Determine optimal `capture_layer_ids()` for the Gemma4 layer count
- [ ] Add `Gemma4DaemonArgs.draft_path` and CLI wiring
- [ ] Update `CMakeLists.txt` with new source files
- [ ] Build and verify AR baseline unchanged
- [ ] Benchmark spec decode speedup
