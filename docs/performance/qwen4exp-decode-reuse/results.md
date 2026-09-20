# qwen4exp T=1 decode workspace reuse

Date: 2026-09-20. Model: IQ4_NL. Device: gfx1151, ROCm 7.2.2. All GPU
runs held `/tmp/qwen-perf/gpu.lock`; only one server was resident. The ten
delivered performance variables were asserted from the live server process.

## Change

`QWEN4EXP_DECODE_REUSE=1` retains a T=1 decode metadata context and graph
allocator in `Qwen4ExpCache`. Each token resets the metadata arena, rebuilds
the graph, remeasures allocator assignments, and reuses the allocator backing
buffers. The remeasure is required: reusing the previous index-wise assignment
without it changed output. The workspace is freed with the cache.

The path is default-off, applies only to T=1 with graph dumps disabled, and is
excluded under `QWEN4EXP_UPSTREAM=1`.

## Decode A/B

Greedy, 128 output tokens, cache off, one warm-up then N=8 measured requests at
each depth. Token rates below are derived from unrounded `decode_ms`; brackets
are min-max across the eight requests. Candidate ran second, so its win is not
from a colder machine.

| Effective prompt | Control median t/s [min-max] | Reuse median t/s [min-max] | Median delta | Best control -> reuse |
|---:|---:|---:|---:|---:|
| 1,996 | 27.87 [27.81-28.03] | 28.55 [28.32-28.77] | +2.43% | 28.03 -> 28.77 |
| 16,348 | 24.62 [24.37-24.85] | 25.34 [25.13-26.30] | +2.93% | 24.85 -> 26.30 |

Phase instrumentation across 508 T=1 calls reduced median `build+alloc` from
1.3 to 1.0 ms at 2k and from 1.7 to 1.3 ms at 16k. This explains only part of
the total host-side gap: graph construction and roughly 2,800 eager kernel
dispatches still occur on every token.

Raw data: `decode-reuse-{control,candidate}.json`, `ab-driver.log`, and
`decode-reuse-phase-{control,candidate}.server.log` in this directory.

## Correctness and regression gates

| Gate | Result |
|---|---|
| HIP build | `server/build-hip/dflash_server` rebuilt successfully; final cache has `GGML_HIP_GRAPHS=OFF` |
| Greedy identity, ~2k | Exact generated text `QUINCE-AMBER-7731`; 12 vs 12 completion tokens |
| Greedy identity, ~16k | Exact generated text `QUINCE-AMBER-7731`; 12 vs 12 completion tokens |
| Upstream reference differential | Exit 0; every aligned activation ratio 1.0000; expert IDs 0 mismatches at all 48 layers |
| Default quality suite | HE 10/10, GSM 10/10, Math 9/10, recall 2/2, equal to baseline |
| Prefill, 16,366 tokens, N=8 | Control best 15.22 s / 1,075.30 t/s; reuse best 15.28 s / 1,071.07 t/s (-0.39%) |

The full prefill spreads were control median 15.37 s [15.22-17.59] and
candidate median 15.79 s [15.28-17.64]. The candidate ran after the complete
control series; the min-of-8 result is within 0.4% and shows no material
regression in a path the flag does not enter.

Evidence: `identity.json`, `reference-diff.log`, `reference/`,
`quality-harness.log`, `quality.json`, and `prefill-{control,candidate}.txt`.

A free-form numbered-list prompt was not a valid strict identity oracle: two
fresh control servers also diverged near token ties. The reported identity gate
therefore uses a constrained planted-key answer; exact output text plus the
same 12-token completion implies the same tokenization. The layer-by-layer
reference differential and full quality suite provide the broader correctness
checks.

## Why HIP graph replay was not selected

The production build initially had `GGML_HIP_GRAPHS=OFF`, so no HIP graph
capture code was present. A temporary build with it enabled proved that merely
turning it on does not help this graph:

| Probe | T=1 graphs | Captures | Replays | Eager |
|---|---:|---:|---:|---:|
| Normal pinned input ring | 510 | 0 | 0 | 510 |
| `DFLASH_HIP_NO_UMA_RING=1` | 508 | 0 | 0 | 508 |

With the ring, source data alternates between two pinned input slots. Without
the ring, advancing KV view addresses and rebuilt `RESHAPE` tensor metadata
still fail `ggml_cuda_graph_check_compability`. Capture/replay therefore needs
a larger stable-graph design with fixed or bucketed KV views and dynamic kernel
parameter updates. `ScopedCudaGraphOverrides` is not used by the qwen4exp
path, so an override was not the blocker. The build was restored to
`GGML_HIP_GRAPHS=OFF` after the probe.

Evidence: `decode-hipgraph-probe.server.log` and
`decode-hipgraph-noring.server.log`.

## Decision

Accept as an experimental, default-off decode optimization. It is correct and
gives a repeatable 2.4-2.9% median gain, but it removes only 0.3-0.4 ms of
construction/allocation work per token. The remaining high-yield work requires
stable graph capture/replay or kernel fusion to reduce eager dispatch count.
The retained allocator buffer lives until cache destruction, so the tradeoff is
persistent scratch residency rather than per-token allocation churn.
