# bsa_stubs

Header shims that let `mit-han-lab/Block-Sparse-Attention` (BSA) compile
without depending on PyTorch's `libtorch`.

BSA was originally built as a PyTorch C++ extension and pulls in
`<ATen/...>` and `<c10/...>` headers. We don't link `libtorch` in the
dflash daemon, so this directory provides minimal stand-ins that satisfy
the references BSA actually uses:

- `c10/cuda/CUDAException.h` — `C10_CUDA_CHECK`, `C10_CUDA_KERNEL_LAUNCH_CHECK`
  macros (forward to `cudaPeekAtLastError`).
- `ATen/cuda/CUDAGeneratorImpl.h` — `at::PhiloxCudaState` POD struct (only
  used by BSA's dropout path, which we never enable).
- `ATen/cuda/CUDAGraphsUtils.cuh` — `at::cuda::philox::unpack` no-op
  returning `{seed, offset}` from the stub state.

These headers are placed FIRST on the BSA include path
(`server/CMakeLists.txt`, gated on `DFLASH27B_ENABLE_BSA`). When BSA's
generated CUDA includes `<c10/cuda/CUDAException.h>`, the compiler picks up
this stub instead of trying to find PyTorch.

Because we always build BSA with `FLASHATTENTION_DISABLE_DROPOUT`, the
philox / generator stubs are never exercised — they exist only to satisfy
declarations.

If a future BSA upgrade pulls in additional ATen/c10 headers, add a
matching stub here rather than vendoring all of libtorch.
