# ROCmFP4

This directory implements two GGUF weight formats used by the ROCm/HIP
backend:

| GGML type | Block layout | Block size | Bits/weight |
|---|---|---:|---:|
| `Q4_0_ROCMFP4` | 32 packed 4-bit codes and one UE4M3 scale per 16 weights | 18 bytes | 4.50 |
| `Q4_0_ROCMFP4_FAST` | 32 packed 4-bit codes and one UE4M3 scale per 32 weights | 17 bytes | 4.25 |

Both layouts use the signed Codebook10 levels
`0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10`. Scale bytes are finite
unsigned UE4M3 values; validation rejects `0x7f` and values with the sign bit
set.

## Implementation

- `rocmfp4.c` contains deterministic CPU reference quantization,
  dequantization, validation, and vector-dot functions.
- `rocmfp4_hip_codebook.cuh` and `rocmfp4_hip_scale.cuh` contain device
  helpers shared by the ggml HIP kernels.
- Backend dispatch is integrated in `ggml-cuda`, which is also the source tree
  used by ggml's HIP build.

The quantizer selects UE4M3 scales by minimizing reconstruction error. When an
importance matrix is supplied, finite positive weights are used in the error
metric. Non-finite source values quantize to zero and non-finite or non-positive
importance weights are ignored.

Vulkan kernels are not implemented in this vendored build.

## Tests

With `DFLASH27B_TESTS=ON`:

- `test_rocmfp4` checks CPU reference round trips, validation, and imatrix
  handling.
- `test_rocmfp4_hip_tail` compares HIP conversion with the CPU reference
  byte-for-byte and checks output bounds for row sizes that are and are not
  multiples of 256.
