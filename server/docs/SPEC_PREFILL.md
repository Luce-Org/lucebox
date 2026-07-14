# dflash spec-prefill, daemon-side build & tunables

In-process speculative-prefill + speculative-decode daemon (C++/CUDA only,
no Python, no Triton, no PyTorch at runtime).

This doc is the build / runtime / tunables reference for the C++ daemon
path described in [`optimizations/pflash/README.md`](../../optimizations/pflash/README.md) and on the
[blog post](https://lucebox.com/blog/pflash):

- **Drafter** (Qwen3-0.6B) loaded via a custom forward (`qwen3_*`)
  with the FlashPrefill block-sparse attention kernel for long-context
  scoring.
- **Target** (Qwen3.6-27B Q4_K_M) loaded directly via ggml.
- **Speculative decode** between draft + target with rollback / DDTree.

Both models live in the same process, the same ggml allocator, on a
single RTX 3090 (24 GB). No PyTorch at runtime.

## Build

```
git submodule update --init --recursive
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=86 -DDFLASH27B_ENABLE_BSA=ON ..
cmake --build . --target test_dflash test_flashprefill_kernels -- -j8
```

Required:
- CUDA Toolkit 12.0+ (sm_80+ for BSA path; sm_86 RTX 3090 is the
  reference target).
- `git submodule update --init --recursive` to pull
  `deps/Block-Sparse-Attention` (with cutlass). `deps/llama.cpp` is vendored
  directly and keeps only the `ggml` subtree.

CMake options:
- `DFLASH27B_ENABLE_BSA=ON` (default) — build the Block-Sparse-Attention
  kernel for sparse FA forward. Required for the long-context perf claim.
  Turn OFF only on sm<80.
- `DFLASH27B_FA_ALL_QUANTS=ON` (default) — compile ggml-cuda fattn for
  all KV-quant pairs (needed for asymmetric Q4_0 K + Q8_0 V cache). Off
  cuts build time ~3x but breaks the 128K target gen path.

## Runtime tunables

```
DFLASH_FP_USE_BSA=1    # dispatch sparse FA forward through BSA (sm_80+)
DFLASH_FP_ALPHA=0.85   # block-selection threshold (default 0.12);
                       # higher = stricter = fewer K-blocks per Q-row.
DFLASH_FP_PROFILE=1    # log mean/score/select/forward stage timings
```

See `src/flashprefill.h` for the full list and defaults.

## Mixed backend placement

CUDA/HIP mixed backend placement uses separate CUDA and HIP builds at the
PFlash phase or DFlash draft-process boundary. See
[`MIXED_BACKEND.md`](MIXED_BACKEND.md) for the build and harness instructions.

## Performance

NIAH single-needle end-to-end on RTX 3090 (Qwen3.6-27B Q4_K_M target,
Qwen3-0.6B drafter, in-process daemon, `DFLASH_FP_USE_BSA=1`,
`DFLASH_FP_ALPHA=0.85`, `keep_ratio=0.05`):

| Source S | dflash TTFT | llama.cpp baseline | Speedup | NIAH |
|----------|------------:|-------------------:|--------:|:----:|
| 64K      | **13.5 s**  | 134.95 s (FA off, dense) | **10.0×** | ✅ |
| 128K     | **24.8 s**  | ~257 s (FA on, Q4_0 KV)  | **~10.4×** | ✅ |

NIAH needle retrieved (accuracy 1/1) at every measured context. The
runtime is C++/CUDA only — the headline number is the dflash binary on
its own, no Python or Triton in the loop.

## Repo layout

```
src/
  flashprefill.{h,cpp}         FlashPrefill C++ entry + dispatcher
  flashprefill_kernels.cu       4 CUDA kernels (mean_K, score, select, sparse_fwd)
  flashprefill_select.cpp       Host fallback for block_select (rarely used)
  bsa_launcher.cu               BSA launcher: blockmask conversion + Flash_fwd_params
  bsa_fwd_inst.cu               Single-TU instantiation of BSA's hdim128 kernel
  qwen3/                   Qwen3-0.6B drafter model code
    qwen3_loader.cpp       GGUF → Qwen3-0.6B BF16 weight tensors
    qwen3_graph.cpp        Custom Qwen3-0.6B forward (per-layer A/FP/B graphs)
    qwen3_drafter.{h,cpp}       drafter_score_and_compress() entry point
  qwen35/                       Qwen3.5/3.6 target + DFlash draft model code
    qwen35_target_graph.cpp     Qwen3.5/3.6 target graph (ggml)
    gguf_target_loader.cpp      Qwen3.5 target GGUF loader
  draft/                        Special DFlash draft model code
    draft_graph.cpp             DFlash speculative draft head
    draft_gguf_loader.cpp       Draft GGUF loader
    draft_safetensors_loader.cpp Draft safetensors loader
  laguna/                       Laguna target + daemon model code
    laguna_target_loader.cpp    Laguna GGUF loader
    laguna_target_graph.cpp     Laguna forward graph
    laguna_daemon.{h,cpp}       Laguna daemon protocol/runtime
  common/                       Shared runtime helpers
    sampler.{h,cpp}             Shared CPU sampler chain
  kv_cache.cpp / kv_quant.cpp   Q4_0 KV cache + asymmetric quant
test/
  test_dflash.cpp               daemon executable; supports
                                  `compress / generate / park / unpark / free drafter`
  test_flashprefill_kernels.cpp parity tests for the 4 FP kernels
  smoke_qwen3_forward.cpp  drafter forward smoke at S=8K-128K
deps/
  llama.cpp/                    vendored ggml snapshot + extracted helpers (libllama not built)
  Block-Sparse-Attention/       submodule (BSA + cutlass)
  bsa_stubs/                    PyTorch ATen/c10 header shims (see its README)
```
