# Qwen3.8-Flash-Next (`qwen4exp`) backend

Hand-written ggml graph and CUDA/HIP kernels for `Qwen/Qwen3.8-Flash-Next`, the
experimental hybrid-attention MoE. Runs on the existing `dflash_server`
(`--target-device hip:0`), alongside the DFlash/PFlash backends.

## Model

- 48 layers, hidden 2560; layout `12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))`.
- MoE: 512 experts, top-10 routed + 1 shared; expert intermediate 640.
- QSA (full-attention layers): 24 Q / 2 KV heads, head_dim 256, RoPE dim 64;
  indexer = MQA with 4 Q heads / 1 KV head, head_dim 128, budget 512 blocks (2048 tokens).
- Hyper-connections: 4 streams, bottleneck rank 320. Per-layer n-gram embeddings (PLE).
- 125B params / 6B active, plus a 51B n-gram embedding table and a 4B MTP head.
  Context 262144 native.

## Layout

- `server/src/qwen4exp/` — loader (sharded GGUF, lazy PLE reader), backend/daemon
  wiring, KV + recurrent + indexer cache, CPU embedding, and the forward graph.
- `server/deps/llama.cpp/ggml/src/ggml-cuda/` — QSA selected-attention kernels
  (`qsa*.cu[h]`), the fused DS4 indexer, the bf16-dequant GEMM family (`mmb*`),
  fused hyper-connection / MoE / GDN / PLE-conv kernels, and packed `get_rows`.

## Measured (gfx1151, Radeon 8060S, 3-shard IQ4_NL, planted-correct)

Prefill is reported as **min-of-N prompt-processing-only** (single-run
end-to-end timings include decode + warmup and hide 1-2% changes).

| prompt | prefill tok/s | notes |
|---|---|---|
| 13,664 | ~960 | single chunk |
| 16,366 | **1002.2** | RMS fusion + opt-in last-token FFN; median 994.0, slowest 989.5; [experiment](performance/qwen4exp-prefill-20260920.md) |
| 32,792 | ~893 | chunked QSA |
| 65,528 | ~926 | `--chunk 16384`, chunked QSA |

Decode ~29 tok/s (autoregressive; MTP is not wired yet).

## Enablement / env

Best-known IQ4_NL prefill config on gfx1151:
`QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5 DFLASH_MMB_SHADOW=1 LLAMA_MMB_HC16=2 QWEN4EXP_LAST_TOKEN_FFN=1`.
Use `GGML_CUDA_MMB=1`, `GGML_DS4_INDEXER_M32_CACHE_B=0`, and `--chunk 16384`.
The September 20 measurements use minimum TTFT of eight (best throughput),
not minimum throughput; see the linked experiment for all samples and gates.

- `QWEN4EXP_QSA` — QSA selected attention (chunked prefill via the indexer-K cache).
  This switch is presence-based: unset it to disable; `=0` still enables QSA.
- `QWEN4EXP_RMS_SCALE_FUSED` — defaults on for eligible RDNA3.5 F32 width-128
  RMS_NORM + SCALE pairs; preserves the original intermediate rounding. Set `0`
  for the two-kernel fallback.
- `QWEN4EXP_LAST_TOKEN_FFN=1` — evaluate only the final output row in the last
  layer's HC/FFN after attention/cache updates. Opt-in because GEMM dispatch and
  rounding change. IQ4_NL quality was validated; other fast model profiles were
  not. The upstream reference profile already performs this row selection.
- `QWEN4EXP_MMB_CUBLAS` — route quantized GEMM shapes to cuBLAS/hipBLASLt via a
  bf16 weight shadow (1/3/5 add ssm_out and HC down/up).
- `DFLASH_MMB_SHADOW` — bf16 weight shadow mode (1 = IQ4_NL/Q5_K, 2 = Q6_K).
- `LLAMA_MMB_HC16` — keep the hyper-connection normalized stream bf16-only.
- `QWEN4EXP_CHUNK` — prefill chunk size (16384 recommended; smaller sizes still
  populate the indexer cache).

Upstream comparison profile (opt-in): `QWEN4EXP_UPSTREAM=1`. It uses padded
256-row dense K/V and masks, upstream F32 MRoPE angles, unfused HC/MoE, and
last-token-only final FFN evaluation, and upstream vector-dot dispatch. It
preserves the shipped defaults. Use `GGML_CUDA_MMB=0` for bitwise comparisons
and start with the QSA/shadow performance flags unset.
For individual A/Bs, `QWEN4EXP_FA_PAD256=1` enables padding and
`QWEN4EXP_ROPE_F32=1` selects F32 MRoPE. Other RoPE modes retain FP64 angles.
The RDNA3.5 head-256 MMA selector uses upstream's GQA divisor (4 for 24/2
heads) and 64-column prefill configuration. It is enabled by either profile/
padding flag above, or an explicit `DFLASH27B_FA256_MMA=1`; otherwise existing
RDNA3.5 QSA/padded callers retain their shipped selector.

## Benchmarking and correctness

- **Correctness gate:** `client_test_runner.py bench --suite recall` (planted-fact
  long-context recall at 13.6k/24k); the FA launch counter
  (`QWEN4EXP_FA_TELEMETRY=1`) asserts QSA engaged.
- **Quality:** `client_test_runner.py bench --suite he,gsm,math` (HumanEval scored by
  executing gold tests).
- **Kernel profiling:** `rocprofv3 --kernel-trace`; `GGML_CUDA_OP_PROF=1` for
  per-shape `mul_mat` timings.
- **Kernel differentials:** `server/test/` (mmb-vs-cuBLAS, DS4 mmid regression,
  forward smoke).

## Not included

Speculative decoding (the model's MTP head) and QSA decode attention are planned
follow-ups; decode is currently dense autoregressive.

Run the differential harness on the GPU box with
`python3 server/scripts/qwen4exp_upstream_diff.py --model MODEL --seq 16 --reference --output-dir /tmp/qwen-reference`.
`--reference` checks every expert ID, including the last-token-only final layer;
`--output-dir` retains both raw node logs. Scalar equality alone does not prove
bitwise tensor equality; use the existing binary dump environment variables
for that comparison.

For the real-prompt regression, set `QWEN4EXP_TOKEN_FILE` to a whitespace-separated
file of token IDs and pass its length as `--seq`. The pinned-host BF16 input
regression is also covered by `test_mul_mat_host_input` (widths 16, 17, 32).
