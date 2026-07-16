# ROCmFPx formats

This directory contains the lower- and higher-bit siblings of ROCmFP4. They
remain separate from `rocmfp4/` so the promoted 4-bit layouts can evolve
without changing these experimental formats.

All layouts store 32 weights per block and use finite unsigned UE4M3 scale
bytes.

| GGML type | Quantized payload | Scales | Block size | Bits/weight |
|---|---:|---:|---:|---:|
| `Q2_0_ROCMFP2` | 8 bytes | 2 | 10 bytes | 2.50 |
| `Q3_0_ROCMFPX` | 12 bytes | 2 | 14 bytes | 3.50 |
| `Q6_0_ROCMFPX` | 24 bytes | 2 | 26 bytes | 6.50 |
| `Q8_0_ROCMFPX` | 32 bytes | 1 | 33 bytes | 8.25 |

ROCmFP2 uses the integer levels `-1, 0, 1, 2`; ROCmFP3 uses
`0, +/-1, +/-2, +/-4`; ROCmFP6 uses signed-magnitude levels up to 31; and
ROCmFP8 uses signed integer levels clamped to `[-127, 127]`.

`rocmfpx.c` provides deterministic CPU reference quantization,
dequantization, validation, and vector-dot functions. Backend dispatch is
integrated in ggml's HIP source tree. Vulkan kernels are not implemented in
this vendored build.

With `DFLASH27B_TESTS=ON`, `test_rocmfpx` checks reference round trips,
expected error ordering, weighted quantization, validation, and saturation of
large finite FP6/FP8 inputs.
