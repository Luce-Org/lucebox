# Pascal (sm_61) Megakernel

Fused single-kernel decode for Qwen3.5-0.8B, adapted for NVIDIA Pascal GPUs.

## Hardware reference

| GPU | SMs | FP16 TFLOPS | Expected tok/s (tg128) |
|-----|-----|-------------|------------------------|
| GTX 1080 Ti | 28 | 8.9 | ~5-15 |
| GTX 1080 | 28 | 8.2 | ~5-12 |
| GTX 1070 Ti | 24 | 6.5 | ~3-8 |
| GTX 1070 | 24 | 6.5 | ~3-8 |
| GTX 1060 6GB | 16 | 4.4 | ~2-5 |
| GTX 1060 3GB | 16 | 4.4 | ~1-4 (OOM on full bf16) |

## Pascal adaptations vs main branch

| Feature | Main (sm_86) | Pascal (sm_61) |
|---------|-------------|----------------|
| Half precision | BF16 native | FP16 (`__half`) |
| Tensor cores | WMMA m16n16k16 | **Scalar F16×F16→F32** |
| Async copy | `cp.async` | **Cooperative shared loads** |
| Warp shuffle | `__shfl_down_sync(mask, val, off)` | `__shfl_down(val, off)` |
| Broadcast | `__shfl_sync(mask, val, src)` | `__shfl(val, src)` |
| Xor shuffle | `__shfl_xor_sync(mask, val, off)` | `__shfl_xor(val, off)` |
| Memory fence | `fence.acq_rel.gpu` | `membar.gl` |
| Global load | `ld.global.cg.v4.b32` | `__ldg(ptr)` (regular cached) |
| Block count | 82 (one per 3090 SM) | 28 (one per 1080 Ti SM) |
| LM head blocks | 512 | 256 |
| WMMA prefill | m16n16k16 tensor cores | **Scalar only** |

## Build

```bash
cd megakernel
python -m venv .venv && source .venv/bin/activate
pip install torch
pip install -e . --no-build-isolation
```

The `setup.py` auto-detects Pascal via `torch.cuda.get_device_capability()` and sets `TARGET_SM=61`. Override with:

```bash
MEGAKERNEL_CUDA_ARCH=sm_61 pip install -e . --no-build-isolation
```

## Run

```bash
python final_bench.py --backend bf16
```

> **Note:** Pascal has no BF16 hardware support. Despite the `--backend bf16` name, the kernel uses FP16 (`__half`) on Pascal. The `half_type.h` header handles this transparently via the `TARGET_SM < 80` guard.

## DFlash on Pascal

The dflash build also supports Pascal. Add `-DCMAKE_CUDA_ARCHITECTURES=61` to your cmake command:

```bash
cd dflash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build --target test_dflash -j
```

On Pascal, the FlashPrefill drafter path falls back to `flashprefill_scalar.cu` (scalar F16, no tensor cores). BSA (Block-Sparse Attention) is auto-disabled — the build will warn about it. The drafter scoring will use the WMMA-free scalar kernel.

## Limitations

- **No tensor cores:** All GEMM-equivalent work (matvec, attention QK/PV) uses scalar F16×F16→F32. Expect 5–10× lower throughput than RTX 3090.
- **No async copy:** All shared-memory loads are synchronous cooperative loads. This eliminates the load-latency-hiding pipeline that Ampere+ kernels use.
- **No BF16:** Weights and activations use FP16. All math is done in F32 accumulators, so precision is equivalent.
- **No BSA:** Block-Sparse Attention requires CUTLASS tensor-core kernels (sm_80+). PFlash speculative prefill falls back to the scalar attention kernel on Pascal.
- **No WMMA prefill:** The prefill chunk-parallel DeltaNet uses scalar math only. The WMMA phase2 path is excluded by `TARGET_SM >= 80`.
- **Smaller block counts:** Pascal GPUs have 16–28 SMs vs 82 on RTX 3090. Block counts are scaled down accordingly (28 decode blocks, 256 LM blocks).

## Why this exists

Pascal is still widely deployed in education, embedded, and older consumer systems. This branch proves that the megakernel approach can target any GPU architecture — it's just slower on older silicon. The code structure is identical to the main branch; only the low-level intrinsics differ.
