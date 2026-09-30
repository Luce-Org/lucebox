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

ROCmFP2 scale bytes carry one more bit ("fp2s"). UE4M3 scales never exceed
`0x7e`, so bit 7 is free: when set it negates the codebook of that half-block
(`1, 0, -1, -2` instead of `-1, 0, 1, 2`), which lets the encoder put the
asymmetric level on whichever side the weights need. Files written without
the bit decode exactly as before. The affine ROCmFP2 build does not use it.

`rocmfpx.c` provides deterministic CPU reference quantization,
dequantization, validation, and vector-dot functions. Backend dispatch is
integrated in ggml's HIP source tree. Vulkan kernels are not implemented in
this vendored build.

With `LUCE_TESTS=ON`, `test_rocmfpx` checks reference round trips,
expected error ordering, weighted quantization, validation, and saturation of
large finite FP6/FP8 inputs.
