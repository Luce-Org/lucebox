# Laguna + Gemma4 Unimplemented Stubs & Fix Plan

---

## Laguna (`dflash/src/laguna/`)

**Status:** Phase 2 complete (forward graph, loader, cache, snapshots, daemon, PFlash compress).
Phase 3/4 work (sparse prefill, spec decode) not yet started.

### L1. MoE Shared-Only Stub (debug — intentional, no fix needed)
- **File:** `laguna_target_graph.cpp:287-303`
- **What:** Env `DFLASH_LAGUNA_MOE_STUB=1` → shared expert only. Full MoE IS the default.

### L2. Dead Code: `build_laguna_moe_block_legacy` (cleanup)
- **File:** `laguna_target_graph.cpp:386-468`
- **What:** An alternative MoE implementation never called from anywhere.
- **Fix:** Delete or gate behind compile flag.

### L3. SWA Ring-Buffer KV Optimization (deferred)
- **File:** `laguna_target_graph.cpp:544`
- **What:** All 40 layers allocate full `max_ctx` KV cache. SWA layers only need
  `sliding_window=512` tokens. Wastes ~2.5 GiB at 32K context.
- **Fix:** Ring-buffer for SWA layers with capacity = 512.

### L4. PFlash Sparse Prefill (Phase 3 — not started)
- **File:** Not yet created
- **What:** Laguna uses `ggml_flash_attn_ext` (dense) for all attention. Missing
  block-sparse FA via `flash_prefill_forward_bf16`/`f16`/`q8` on full-attention layers.
- **Impact:** 2-4× TTFT regression vs what PFlash would give on long contexts.
- **Fix:** Layer-segmented prefill path (pattern in `qwen3_graph.cpp:488-558`).

### L5. DFlash Speculative Decode (Phase 4 — not started)
- **File:** `laguna_backend.h` — does NOT override `supports_dflash_spec_decode()`
- **What:** No `LagunaDFlashTarget` class. No spec decode acceleration.
- **Fix:** Implement `DFlashTarget` interface (`verify_batch`, `snapshot_kv`,
  `restore_kv`, `is_eos`, `embed_tokens`, `project_hidden_to_tokens`, etc.).

### L6. C++ Cross-Tokenizer Mapping (Phase 4 — Python only)
- **File:** `bench_laguna_pflash.cpp:88` — uses `fake_l = 1972` placeholder
- **What:** Qwen3→Laguna vocab mapping only in Python (`laguna_pflash_niah.py:394`).
- **Fix:** Port `cross_tok_compressed` logic to C++.

---

## Gemma4 (`dflash/src/gemma4/`)

**Status:** Early integration. Loader, cache, graph, backend, daemon all exist and compile.
Forward graph is functional but several features are stubbed or missing.

### G1. Per-Layer Embedding Injection NOT Wired (functional gap)
- **File:** `gemma4_graph.cpp:352-365`
- **What:** Two explicit TODOs: `gemma4_step()` does not pass token IDs needed for
  per-layer embeddings. `per_layer_input` is always `nullptr` in the layer loop.
  The architecture uses per-layer token embeddings gated into the residual stream —
  this is skipped entirely during forward.
- **Impact:** Numerically wrong output on models that rely on per-layer embeddings
  (architecture defines `per_layer_tok_embd`, `per_layer_inp_gate`, `per_layer_proj`).
- **Fix:** Add a `const int32_t * token_ids` parameter to `gemma4_step()`, look up
  per-layer embeddings via the global `per_layer_tok_embd` tensor (sliced per layer),
  project through `per_layer_model_proj` + `per_layer_proj_norm`, and pass result
  to each layer's `build_gemma4_layer()`.

### G2. Park/Unpark Are No-Ops (functional gap)
- **File:** `gemma4_backend.cpp:70-81`
- **What:** `park()` sets a flag but does NOT free weights/cache. `unpark()` clears
  the flag but does NOT reload. Unlike Laguna which frees/reloads VRAM.
- **Impact:** Park command doesn't actually release GPU memory for pflash drafter.
- **Fix:** Mirror Laguna's pattern: `park()` → `free_gemma4_cache` + `free_gemma4_weights`;
  `unpark()` → `load_gemma4_gguf` + `create_gemma4_cache`.

### G3. PFlash Compress Not Supported (stub)
- **File:** `gemma4_backend.cpp:342-349`
- **What:** `handle_compress()` prints "not supported" and returns. No drafter integration.
- **Impact:** No PFlash compression for Gemma4 targets (longer TTFT on long contexts).
- **Fix:** Port Laguna's compress path (lazy-load Qwen3-0.6B drafter, score+compress,
  emit surviving tokens). Needs cross-tokenizer mapping if Gemma4 vocab differs.

### G4. DFlash Speculative Decode Not Supported
- **File:** `gemma4_backend.h` — does NOT override `supports_dflash_spec_decode()`
- **What:** No `Gemma4DFlashTarget`. Same gap as Laguna.
- **Fix:** Implement `DFlashTarget` interface for Gemma4.

### G5. SWA Ring-Buffer KV Optimization (same as Laguna)
- **File:** `gemma4_loader.cpp:334` — SWA layers allocate full `max_ctx` KV
- **What:** KV cache for SWA layers is full-sized. Only need `sliding_window` tokens.
- **Fix:** Ring-buffer for SWA layers.

### G6. PFlash Sparse Prefill Not Implemented
- **File:** `gemma4_graph.cpp` — only uses `ggml_flash_attn_ext`
- **What:** Same as Laguna — no block-sparse FA for full-attention layers.
- **Fix:** Layer-segmented prefill with `flash_prefill_forward_*`.

### G7. `restore_and_generate` Resets Cache Position (logic issue)
- **File:** `gemma4_backend.cpp:245-267`
- **What:** After restoring snapshot, calls `generate()` which sets `cache_.cur_pos = 0`
  at line 178, discarding the restored position. This means the diff-prefill
  optimization (only re-prefilling tokens after the snapshot) is NOT happening.
- **Impact:** Every restore+generate does a full re-prefill from token 0.
- **Fix:** Don't reset `cur_pos` in the restore path; only prefill the diff
  (tokens from `snap.cur_pos` onward). Follow Laguna's `restore_and_generate` pattern.

---

## Combined Summary Table

| ID | Backend | Stub | File | Severity | Effort |
|----|---------|------|------|----------|--------|
| G1 | Gemma4 | Per-layer embedding injection not wired (always nullptr) | `gemma4_graph.cpp:352-365` | **Critical** | 1-2d |
| G7 | Gemma4 | `restore_and_generate` resets cache to 0 (no diff-prefill) | `gemma4_backend.cpp:178,245` | **High** | 0.5d |
| G2 | Gemma4 | Park/unpark are no-ops (don't free/reload VRAM) | `gemma4_backend.cpp:70-81` | **High** | 0.5d |
| G3 | Gemma4 | `handle_compress()` prints "not supported" | `gemma4_backend.cpp:342-349` | High | 1d |
| G6 | Gemma4 | No sparse prefill (dense ggml FA only) | `gemma4_graph.cpp` | High | 2-3d |
| G4 | Gemma4 | No DFlash speculative decode | `gemma4_backend.h` | High | 3-5d |
| G5 | Gemma4 | SWA layers allocate full max_ctx KV (no ring buffer) | `gemma4_loader.cpp:334` | Medium | 1-2d |
| L4 | Laguna | No sparse prefill (dense ggml FA only) | not yet created | **High** | 2-3d |
| L5 | Laguna | No DFlash speculative decode | `laguna_backend.h` | High | 3-5d |
| L6 | Laguna | C++ cross-tokenizer mapping missing (Python only) | `bench_laguna_pflash.cpp:88` | Medium | 1-2d |
| L3 | Laguna | SWA layers allocate full max_ctx KV (no ring buffer) | `laguna_target_graph.cpp:544` | Medium | 1-2d |
| L2 | Laguna | Dead `build_laguna_moe_block_legacy` never called | `laguna_target_graph.cpp:386-468` | Low | 1h |
| L1 | Laguna | MoE shared-only stub (env debug switch) | `laguna_target_graph.cpp:287` | None | — |

---

## Recommended Fix Order

1. **G1** — Per-layer embeddings (Gemma4 produces wrong output without this)
2. **G7** — restore_and_generate cache reset bug (correctness)
3. **G2** — Park/unpark no-ops (VRAM management)
4. **L4 + G6** — PFlash sparse prefill (both archs, biggest perf win)
5. **G3** — PFlash compress for Gemma4
6. **L5 + G4** — DFlash spec decode (both archs)
7. **L6** — C++ cross-tokenizer
8. **L3 + G5** — SWA ring-buffer (VRAM optimization)
9. **L2** — Dead code cleanup
