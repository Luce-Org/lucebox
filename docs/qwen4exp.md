# Qwen3.8-Flash-Next (`qwen4exp`) backend

Hand-written ggml graph and CUDA/HIP kernels for `Qwen/Qwen3.8-Flash-Next`, the
experimental hybrid-attention MoE. Runs on the existing `luce_server`
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

## Serving and environment

The backend is single-sequence. It does not implement paged attention, a shared
KV pool, concurrent decode slots, speculative decoding, layer split, or remote
drafting. Use `--chunk 16384` for long prefill; there is no `QWEN4EXP_CHUNK`
environment variable.

The measured gfx1151 IQ4_NL configuration is:

```
LUCE_HIP_NO_AUTO_UMA=1 GGML_CUDA_MMB=1 QWEN4EXP_QSA=1 \
QWEN4EXP_MMB_CUBLAS=5 LUCE_MMB_SHADOW=1 LLAMA_MMB_HC16=2
```

The following variables are supported serving controls or temporary burn-in
kill switches:

| Variable | Default | Purpose |
|---|---:|---|
| `LUCE_HIP_NO_AUTO_UMA` | unset | Disable automatic managed-memory selection. Set to `1` on unified-memory systems when explicit placement is required. |
| `LUCE_HIP_UMA_MIN_FRAC` | `0.45` | Model-allocation fraction of system RAM above which integrated GPUs select managed memory. |
| `QWEN4EXP_QSA` | `0` | `1` enables sparse selected attention for eligible prefill chunks. Decode remains dense. |
| `QWEN4EXP_MMB_CUBLAS` | `0` | Select validated bf16-shadow rocBLAS routes: `1`, `3`, and `5` progressively add dense shapes. Mode `2` is a diagnostic broad route and is not safe for serving. |
| `LUCE_MMB_SHADOW` | `2` | Weight-shadow policy: `0` off, `1` IQ4_NL/Q5_K, `2` Q6_K. |
| `LUCE_MMB_SHADOW_CAP_MB` | `40960` | Process-wide cap for bf16 weight shadows. |
| `LLAMA_MMB_HC16` | `0` | `2` keeps eligible hyper-connection streams in bf16 between consumers. |
| `QWEN4EXP_DENSE_TABLE` | `1` | Kill switch for the gfx1151, exactly-16366-token measured MMQ dispatch table. Disabled by `QWEN4EXP_UPSTREAM`. |
| `QWEN4EXP_HC_TILE16` | `1` | Kill switch for the measured IQ4_NL 16-row hyper-connection tile. Disabled by `QWEN4EXP_UPSTREAM`. |
| `QWEN4EXP_RMS_SCALE_FUSED` | `1` | Kill switch for eligible RDNA3.5 F32 width-128 RMS-Norm-plus-scale fusion. |
| `QWEN4EXP_LAST_TOKEN_FFN` | `1` | Kill switch for final-row-only evaluation after the last layer has completed its state writes. |
| `QWEN4EXP_DECODE_REUSE` | `1` | Kill switch for reuse of the T=1 ggml context and graph allocator. |
| `QWEN4EXP_DECODE_STABLEGRAPH` | `1` | Kill switch for bucketed, pointer-stable T=1 graphs. HIP graph capture itself still depends on a `GGML_HIP_GRAPHS` build. |
| `QWEN4EXP_BATCHED_DECODE` | `0` | Experimental opt-in for the exact-width independent-slot decode entry point. Excluded under `QWEN4EXP_UPSTREAM=1`; requires one shared `Qwen4ExpBatchedDecodeWorkspace`. |
| `LUCE_QWEN4EXP_SEQ_ENGINE` | unset | Experimental full-cache sequence engine. Requires `QWEN4EXP_BATCHED_DECODE=1`, `--max-concurrency=2..4`, and `--max-ctx=32768`; reserves four F16 full caches at N=4 and does not use paging. |

These variables are diagnostics and differential-test controls; they are not
production tuning requirements:

| Variable(s) | Purpose |
|---|---|
| `QWEN4EXP_UPSTREAM` | Select reference-compatible graph, attention, RoPE, and dispatch paths. Optimizations that can change numerics exclude themselves under this profile. |
| `QWEN4EXP_HC_UNFUSED`, `QWEN4EXP_MOE_UNFUSED`, `QWEN4EXP_GDNL2_LEGACY` | Restore individual unfused or legacy numerical paths for differential bisection. |
| `QWEN4EXP_FA_PAD256`, `QWEN4EXP_ROPE_F32` | Isolate the reference padded-attention and F32-angle RoPE behavior. |
| `QWEN4EXP_DENSE_PROBE` | Benchmark candidate dense GEMM routes on real tensors. It changes execution timing and must not be used for serving. |
| `QWEN4EXP_HC_TILE_CHECK` | Byte-compare the 16-row HC tile against the original tile and abort on mismatch. |
| `QWEN4EXP_PROF`, `QWEN4EXP_FA_TELEMETRY`, `QWEN4EXP_STABLEGRAPH_TELEMETRY` | Print graph phase, attention route, or stable-graph telemetry. |
| `QWEN4EXP_MM_LOG`, `QWEN4EXP_CUBLAS_LOG`, `LUCE_MMB_TELEMETRY` | Print matrix shape and dispatch telemetry. |
| `QWEN4EXP_DUMP`, `QWEN4EXP_DUMP_BIN` | Materialize and dump internal graph activations for the differential harness. |
| `QWEN4EXP_LAYER_TRACE`, `QWEN4EXP_TRACE_TAG` | Batched-path diagnostic: dump selected per-layer tensors (tagged by `QWEN4EXP_TRACE_TAG`) to `/tmp/q4trace_*` for layer-by-layer diffing. No effect when unset. |
| `LUCE_HIP_NO_PINNED_STAGE`, `LUCE_HIP_NO_UMA_RING` | Disable pinned staging or the qwen4exp pinned input ring for diagnosis. |
| `LUCE_GDN_NO_TILED`, `LUCE_GDN_FORCE_GROUPED_COLS`, `LUCE_GDN_NO_GROUPED_COLS` | Override GDN kernel dispatch for profiling and bisection. |

The differential tools additionally use `QWEN4EXP_LLAMA_TREE`,
`QWEN4EXP_TOKEN_FILE`, and `QWEN4EXP_UP_DUMP_BIN`. The benchmark harness accepts
`QWEN4EXP_SERVER`, `QWEN4EXP_MODEL`, and `QWEN4EXP_DIFF_PORT`; these do not
change backend execution.

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

## Concurrent serving (opt-in, experimental)

Up to four concurrent decode slots are supported at `--max-ctx 32768` by a
full-cache sequence engine (no paged attention). It is **default-off** and must
be requested explicitly, e.g.

```
LUCE_QWEN4EXP_SEQ_ENGINE=1 QWEN4EXP_BATCHED_DECODE=1 \
    luce_server MODEL --max-ctx 32768 --max-concurrency 4 --chunk 16384
```

- **Envelope:** `--max-concurrency 2..4`, `--max-ctx 32768` exactly, one local
  target device, no `--kv-pool-tokens`, no layer split / tensor parallel. Outside
  it the feature gate keeps the normal "--paged-attention required" rejection.
- **Design:** one full F16 `Qwen4ExpCache` per slot (~929 MiB each at 32k), one
  shared batched-decode workspace, one FIFO prefill owner (512-token slices).
  `qwen4exp_forward_batched()` runs the shared embedding/projection/MoE/HC ops as
  one row batch and the per-slot attention/GDN/PLE against each slot's own cache.
- **Verified behavior:** four simultaneous distinct requests are correct with no
  cross-talk; the quality suite (HE/GSM/Math/recall) is unchanged; a fifth request
  defers cleanly; an over-context request is a per-request `400`; a mid-stream
  client disconnect does not disturb other slots. The SeqEngine contract and a
  randomized admit/retire soak pass at N=2/4, and peak GTT (~75 GiB) returns to
  idle with no leak. Evidence: `performance/qwen4exp-concurrency-{p1,p2,p3}/`.
- **Goodput — not a throughput multiplier:** the model is MoE (512 experts,
  top-10), so each concurrent row streams *different* experts and the expert
  weight traffic scales with N. Measured aggregate decode is ~**1.08x** at N=4
  (per-request decode 27.9 -> 7.5 t/s). Concurrency serves multiple users at once;
  it does not increase total throughput.
- **Numerics:** batched (T=4) and single-stream (T=1) greedy output can differ at
  near ties (measured epsilon up to ~2; flips occur where the solo top-2 margin is
  small). This is the well-documented batch-size floating-point non-associativity
  (see arXiv 2506.09501), amplified here by MoE routing: near-tie expert flips
  cascade and expert sets first diverge around layer 12. A tunable component comes
  from the batch-dependent dense dispatch (T=1 MMVQ vs T=4 MMQ); forcing MMVQ at
  T=4 lowers epsilon but does not restore exact greedy parity. Keep concurrent
  serving opt-in where single-stream reproducibility is required. Evidence:
  `performance/qwen4exp-concurrency-numerics/`.

## Not included

Concurrent serving is opt-in (above), and speculative decoding (the model's MTP
head) and QSA decode attention are planned follow-ups; default decode is dense
autoregressive and single-stream.

Run the differential harness on the GPU box with
`python3 server/scripts/qwen4exp_upstream_diff.py --model MODEL --seq 16 --reference --output-dir /tmp/qwen-reference`.
`--reference` checks every expert ID, including the last-token-only final layer;
`--output-dir` retains both raw node logs. Scalar equality alone does not prove
bitwise tensor equality; use the existing binary dump environment variables
for that comparison.

For the real-prompt regression, set `QWEN4EXP_TOKEN_FILE` to a whitespace-separated
file of token IDs and pass its length as `--seq`. The pinned-host BF16 input
regression is also covered by `test_mul_mat_host_input` (widths 16, 17, 32).
